"""Generic log-space Hidden Markov Model inference (an adapter over the vendored markovlib).

This module is the *discrete shadow* of the hybrid dynamical system of
THEORY.md §5: the hybrid system's continuous flows-within-a-mode and
discrete jumps-between-modes are discretized per frame into an HMM whose hidden
states are modes, whose transition prior encodes "the tendency to persist," and
whose emissions are the per-state likelihoods of §4. Running the standard
inference here (forward-backward and Viterbi) is what "replaces all three cleanup
heuristics at once" — persistence is now a probabilistic prior rather than a
hand-tuned post-process.

Deliberately *contact-free*: there is no notion of gap, twist, or mode in here.
It is a clean, reusable HMM over abstract states `0..S-1` so that the contact
layer can sit entirely on top of it. The only inputs are log-space quantities;
everything is computed in log-space because likelihoods over a whole trajectory
underflow catastrophically if multiplied raw (THEORY.md §4 & §5 work in
log-likelihood-ratio space for exactly this reason).

Conventions
-----------
* `T` = number of time steps, `S` = number of hidden states.
* `log_emission[t, s]` = log p(observation at t | state = s).
* `log_init[s]`        = log p(state_0 = s)            (the initial prior).
* `log_trans`          = log P(next state | current state). Either
      time-homogeneous, shape `(S, S)`, used at every step; or
      time-varying, shape `(T, S, S)`, where `log_trans[t]` is the transition
      from step `t` to step `t+1` (steps `0 .. T-2` are consumed; `log_trans[T-1]`
      is ignored). Time-varying transitions are what THEORY.md §5 calls a
      *state-dependent guard* — e.g. free->contact rising as the gap nears zero.
* Each row of a transition matrix is a distribution over the *next* state, so
      `logsumexp(log_trans[..., s, :]) == 0` for a proper (normalized) matrix.
      We never require normalization here, but document the convention.

Public API
----------
* `logsumexp(a, axis=None)`                       -- numerically stable log-sum-exp.
* `forward_backward(log_emission, log_trans, log_init)` -> (gamma, total_loglik).
* `viterbi(log_emission, log_trans, log_init)`    -> MAP state path, int array (T,).
"""

from __future__ import annotations

from typing import Protocol

import markovlib as _markovlib
import numpy as np
from scipy.special import logsumexp

__all__ = ["logsumexp", "forward_backward", "viterbi", "TemporalSmoother", "HMM"]


def _broadcast_trans(log_trans: np.ndarray, T: int, S: int) -> np.ndarray:
    """Return transitions as a ``(T-1, S, S)`` stack regardless of input layout.

    Accepts the time-homogeneous ``(S, S)`` form (tiled across all `T-1` steps)
    or the time-varying ``(T, S, S)`` form (steps ``0 .. T-2`` are the inter-step
    transitions; the trailing slice is dropped). This single normalization lets
    the recursions below be written once for both cases.
    """
    log_trans = np.asarray(log_trans, dtype=float)
    if log_trans.shape == (S, S):
        return np.broadcast_to(log_trans, (max(T - 1, 0), S, S))
    if log_trans.shape == (T, S, S):
        return log_trans[: T - 1]
    raise ValueError(
        f"log_trans must have shape (S, S)=({S}, {S}) or (T, S, S)=({T}, {S}, {S}); "
        f"got {log_trans.shape}"
    )


def _validate(
    log_emission: np.ndarray, log_init: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Coerce, shape-check, and return ``(log_emission, log_init, T, S)``."""
    log_emission = np.asarray(log_emission, dtype=float)
    if log_emission.ndim != 2:
        raise ValueError(f"log_emission must be 2-D (T, S); got shape {log_emission.shape}")
    T, S = log_emission.shape
    if T == 0 or S == 0:
        raise ValueError(f"log_emission must be non-empty; got shape {log_emission.shape}")
    log_init = np.asarray(log_init, dtype=float)
    if log_init.shape != (S,):
        raise ValueError(f"log_init must have shape (S,)=({S},); got {log_init.shape}")
    return log_emission, log_init, T, S


def forward_backward(
    log_emission: np.ndarray,
    log_trans: np.ndarray,
    log_init: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Smoothed state posterior via the forward-backward algorithm (log-space).

    The two classic recursions, in log-space so trajectory-length products never
    underflow — written out here as the *specification*; ``markovlib.smooth``
    performs them (this adapter's job is the contract: accept both transition
    layouts, validate shapes, return ``(gamma, loglik)``). This is the *smoothing*
    inference of THEORY.md §5: it conditions each frame on the entire
    record (past and future), which is exactly why a real-time causal detector is
    necessarily less certain than this offline one (§6, the
    latency-accuracy tradeoff).

    Forward (alpha):  alpha[t, s] = log p(o_0..o_t, state_t = s)

        alpha[0, s]   = log_init[s] + log_emission[0, s]
        alpha[t, s]   = log_emission[t, s]
                        + logsumexp_j( alpha[t-1, j] + log_trans[t-1, j, s] )

    Backward (beta):  beta[t, s] = log p(o_{t+1}..o_{T-1} | state_t = s)

        beta[T-1, s]  = 0                                     (log 1)
        beta[t, s]    = logsumexp_j( log_trans[t, s, j]
                                     + log_emission[t+1, j] + beta[t+1, j] )

    The total data log-likelihood is read straight off the end of the forward
    pass, ``logsumexp_s alpha[T-1, s]`` = log p(o_0..o_{T-1}). The posterior is
    ``gamma[t, s] = p(state_t = s | all observations)`` obtained by normalizing
    ``alpha + beta`` per row in log-space and exponentiating.

    Parameters
    ----------
    log_emission:
        ``(T, S)`` log-emission likelihoods.
    log_trans:
        ``(S, S)`` time-homogeneous or ``(T, S, S)`` time-varying log-transitions
        (see module docstring for the time-varying step convention).
    log_init:
        ``(S,)`` log initial-state prior.

    Returns
    -------
    gamma:
        ``(T, S)`` posterior probabilities, already exponentiated and row-
        normalized so each row sums to 1.
    total_loglik:
        Scalar ``log p(observations)`` from the forward pass.
    """
    log_emission, log_init, T, S = _validate(log_emission, log_init)
    log_A = _broadcast_trans(log_trans, T, S)  # (T-1, S, S)

    # Delegate the recursion to markovlib (the vendored general engine), whose categorical
    # forward-backward is this exact log-space recursion (verified bit-for-bit in
    # ``verify_markovlib.py``). markovlib uses the same ``(T-1, S, S)`` transition layout.
    result = _markovlib.smooth(_markovlib.DiscreteChain(log_init=log_init, log_trans=log_A), log_emission)
    gamma = np.asarray(result.gamma, dtype=float)
    # The same final guard as before: renormalize so rows sum to exactly 1 (a ~1e-15 effect).
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, result.loglik


def viterbi(
    log_emission: np.ndarray,
    log_trans: np.ndarray,
    log_init: np.ndarray,
) -> np.ndarray:
    """Maximum-a-posteriori state path via the Viterbi algorithm (log-space).

    Where forward-backward gives the *per-frame* posterior, Viterbi gives the
    single most likely *contiguous* state sequence — THEORY.md §5's "clean
    boolean segmentation." Replacing every product with a sum in log-space and
    every marginal sum with a max turns the forward recursion into a max-product
    (shortest-path) recursion — the specification of what ``markovlib.decode``
    computes for us:

        delta[0, s] = log_init[s] + log_emission[0, s]
        delta[t, s] = log_emission[t, s]
                      + max_j( delta[t-1, j] + log_trans[t-1, j, s] )
        psi[t, s]   = argmax_j( delta[t-1, j] + log_trans[t-1, j, s] )   (backpointer)

    The path is recovered by taking ``argmax_s delta[T-1, s]`` and following the
    backpointers ``psi`` from ``T-1`` to ``0``. ``delta[t, s]`` is the log-
    probability of the best path that ends in state ``s`` at time ``t``.

    Parameters
    ----------
    log_emission:
        ``(T, S)`` log-emission likelihoods.
    log_trans:
        ``(S, S)`` time-homogeneous or ``(T, S, S)`` time-varying log-transitions.
    log_init:
        ``(S,)`` log initial-state prior.

    Returns
    -------
    np.ndarray
        The MAP state path as an ``int`` array of shape ``(T,)``.
    """
    log_emission, log_init, T, S = _validate(log_emission, log_init)
    log_A = _broadcast_trans(log_trans, T, S)  # (T-1, S, S)

    # Delegate to markovlib's Viterbi (the identical max-plus recursion + backtrace; verified
    # bit-for-bit in ``verify_markovlib.py``).
    path = _markovlib.decode(_markovlib.DiscreteChain(log_init=log_init, log_trans=log_A), log_emission)
    return path.astype(int)


# ======================================================================================
# Object interface: a temporal model bundles its prior so emissions are the only per-call
# input. The contact layer plugs its per-mode emissions into one of these; the engine
# itself stays contact-free (THEORY.md §5).
# ======================================================================================


class TemporalSmoother(Protocol):
    """A latent-state temporal model over abstract states ``0..S-1``.

    It turns a per-frame log-emission matrix ``(T, S)`` into a smoothed posterior and a MAP
    path. :class:`HMM` and :class:`contact.hsmm.SemiMarkovHMM` both satisfy this interface,
    so the detector can swap the plain Markov prior for the explicit-duration one without
    changing how it consumes the result.
    """

    def posterior(self, log_emission: np.ndarray) -> tuple[np.ndarray, float]:
        """``((T, S) gamma, total_loglik)`` -- the smoothed per-frame state posterior."""
        ...

    def map_path(self, log_emission: np.ndarray) -> np.ndarray:
        """``(T,)`` MAP state path -- the clean contiguous segmentation."""
        ...


class HMM:
    """Hidden Markov Model over modes -- the discrete shadow of the hybrid system (§5).

    Bundles the temporal prior (initial distribution + transitions, either a homogeneous
    ``(S, S)`` matrix or a gap-gated time-varying ``(T, S, S)`` stack) so emissions are the
    only per-call input. :meth:`posterior` is forward-backward (smoothing); :meth:`map_path`
    is Viterbi (the MAP segmentation). Both run in log-space on the module functions above.
    """

    def __init__(self, log_trans: np.ndarray, log_init: np.ndarray) -> None:
        self.log_trans = log_trans
        self.log_init = log_init

    def posterior(self, log_emission: np.ndarray) -> tuple[np.ndarray, float]:
        return forward_backward(log_emission, self.log_trans, self.log_init)

    def map_path(self, log_emission: np.ndarray) -> np.ndarray:
        return viterbi(log_emission, self.log_trans, self.log_init)
