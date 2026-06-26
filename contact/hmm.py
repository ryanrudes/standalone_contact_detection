"""Generic log-space Hidden Markov Model inference (numpy only).

This module is the *discrete shadow* of the hybrid dynamical system of
THEORY.md section 5: the hybrid system's continuous flows-within-a-mode and
discrete jumps-between-modes are discretized per frame into an HMM whose hidden
states are modes, whose transition prior encodes "the tendency to persist," and
whose emissions are the per-state likelihoods of section 4. Running the standard
inference here (forward-backward and Viterbi) is what "replaces all three cleanup
heuristics at once" — persistence is now a probabilistic prior rather than a
hand-tuned post-process.

Deliberately *contact-free*: there is no notion of gap, twist, or mode in here.
It is a clean, reusable HMM over abstract states `0..S-1` so that the contact
layer can sit entirely on top of it. The only inputs are log-space quantities;
everything is computed in log-space because likelihoods over a whole trajectory
underflow catastrophically if multiplied raw (THEORY.md sections 4 & 5 work in
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
      is ignored). Time-varying transitions are what THEORY.md section 5 calls a
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

import numpy as np

__all__ = ["logsumexp", "forward_backward", "viterbi"]

# A finite stand-in for log(0). Using -inf directly is correct but litters the
# arithmetic with nan from (-inf) + (+inf) style cancellations; this sentinel is
# small enough to behave like zero probability while staying finite.
_LOG_ZERO = -1e30


def logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable ``log(sum(exp(a)))`` along ``axis``.

    Computes ``m + log(sum(exp(a - m)))`` with ``m = max(a)``, the standard
    max-shift trick: factoring out the largest term keeps every exponential in
    ``(0, 1]`` so the sum cannot overflow, and underflow of the small terms is
    harmless. This is the single primitive that lets the whole HMM stay in
    log-space (THEORY.md section 5 — never multiply raw likelihoods).

    Parameters
    ----------
    a:
        Input array of log-values.
    axis:
        Axis (or None for the whole array) to reduce over.

    Returns
    -------
    np.ndarray
        The reduced log-sum-exp, with ``axis`` removed (a 0-d array / scalar
        when ``axis is None``).
    """
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    # Where an entire slice is -inf, a_max is -inf; replace with 0 so the shift
    # `a - a_max` yields -inf - 0 = -inf (and exp -> 0), not the invalid -inf -(-inf).
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    s = np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True)
    # A fully -inf slice has sum 0; log(0) = -inf is the correct result there, so
    # silence the (expected) divide-by-zero warning rather than special-casing it.
    with np.errstate(divide="ignore"):
        out = np.log(s) + a_max_safe
    return np.squeeze(out, axis=axis) if axis is not None else out.reshape(())


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

    Implements the two classic recursions, both in log-space using ``logsumexp``
    so trajectory-length products never underflow. This is the *smoothing*
    inference of THEORY.md section 5: it conditions each frame on the entire
    record (past and future), which is exactly why a real-time causal detector is
    necessarily less certain than this offline one (section 6, the
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

    # --- Forward pass -----------------------------------------------------------
    log_alpha = np.empty((T, S), dtype=float)
    log_alpha[0] = log_init + log_emission[0]
    for t in range(1, T):
        # alpha[t, s] = emit + logsumexp_j(alpha[t-1, j] + A[t-1, j, s]).
        # log_alpha[t-1] is indexed by j (rows); broadcast against A[t-1]'s rows.
        prev = log_alpha[t - 1][:, None] + log_A[t - 1]  # (S_j, S_s)
        log_alpha[t] = log_emission[t] + logsumexp(prev, axis=0)

    total_loglik = float(logsumexp(log_alpha[T - 1]))

    # --- Backward pass ----------------------------------------------------------
    log_beta = np.empty((T, S), dtype=float)
    log_beta[T - 1] = 0.0  # log 1
    for t in range(T - 2, -1, -1):
        # beta[t, s] = logsumexp_j(A[t, s, j] + emit[t+1, j] + beta[t+1, j]).
        nxt = log_A[t] + (log_emission[t + 1] + log_beta[t + 1])[None, :]  # (S_s, S_j)
        log_beta[t] = logsumexp(nxt, axis=1)

    # --- Combine into the smoothed posterior -----------------------------------
    log_gamma = log_alpha + log_beta
    # Row-normalize in log-space (logsumexp returns shape (T,); add a trailing axis).
    log_gamma -= logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)
    # Guard against tiny numerical drift so rows sum to exactly 1.
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, total_loglik


def viterbi(
    log_emission: np.ndarray,
    log_trans: np.ndarray,
    log_init: np.ndarray,
) -> np.ndarray:
    """Maximum-a-posteriori state path via the Viterbi algorithm (log-space).

    Where forward-backward gives the *per-frame* posterior, Viterbi gives the
    single most likely *contiguous* state sequence — THEORY.md section 5's "clean
    boolean segmentation." Replacing every product with a sum in log-space and
    every marginal sum with a max turns the forward recursion into a max-product
    (shortest-path) recursion:

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

    log_delta = np.empty((T, S), dtype=float)
    psi = np.zeros((T, S), dtype=np.intp)  # backpointers; psi[0] unused
    log_delta[0] = log_init + log_emission[0]

    for t in range(1, T):
        # scores[j, s] = delta[t-1, j] + A[t-1, j, s]; maximize over j (rows).
        scores = log_delta[t - 1][:, None] + log_A[t - 1]  # (S_j, S_s)
        psi[t] = np.argmax(scores, axis=0)
        log_delta[t] = log_emission[t] + np.max(scores, axis=0)

    # Backtrace from the best terminal state.
    path = np.empty(T, dtype=np.intp)
    path[T - 1] = int(np.argmax(log_delta[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path.astype(int)
