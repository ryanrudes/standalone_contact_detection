"""Scalable posterior over the active-set sequence of a contact graph (THEORY.md §8).

THEORY.md §8 makes "richer contact information" precise: the hidden thing we
infer over a contact graph is not a bit but a *structure* -- **which set of edges is
simultaneously active**, over time, as a Bayesian posterior. §8 also names the
tractability fork explicitly: inference is "an HMM/particle filter over the discrete
structure", and §10 spells out the consequence -- exact ``2**E`` enumeration is
the correct, preferred choice for the small graphs the package ships with, but large
``E`` is exponential and needs a sampling method that never materializes the ``2**E``
alphabet.

This module owns *both* ends of that fork, as a clean, **contact-free** library over the
abstract active-set problem (so it is reusable for any structure-posterior task, exactly
as :mod:`contact.hmm` is a contact-free HMM):

* :func:`exact_active_sets` -- the **reference** estimator. It builds the ``2**E`` subset
  alphabet, runs forward-backward + Viterbi (reusing :mod:`contact.hmm`), and marginalizes
  to per-edge active probabilities. Exact, and the ground truth the particle filter is
  validated against. Tractable for small ``E`` (``2**E`` states).

* :func:`particle_filter_active_sets` -- a Rao-Blackwellized bootstrap particle filter
  over the active-set *sequence* that scales to large ``E`` **without enumerating**
  ``2**E``. Each particle carries one current active set (an ``E``-bit boolean vector);
  particles are propagated with a proposal consistent with the dwell prior, weighted by
  the per-edge evidence, and resampled (systematic). The Rao-Blackwellization marginalizes
  each particle's *active probabilities* analytically rather than reporting a hard 0/1 set,
  which sharply reduces Monte-Carlo variance of the per-edge marginals.

The temporal model
------------------
Both estimators use the **same** structure-level temporal prior, the one
:mod:`contact.graph` uses (THEORY.md §5 lifted to the structure level, §8): a Markov
chain on the subset alphabet with self-stay probability ``p_stay = exp(log_dwell_stay)``
and the remaining mass ``1 - p_stay`` split *uniformly* over the other ``2**E - 1``
subsets. Using the identical model on both sides is what lets the particle filter match
the exact reference in the large-particle limit (it is not approximating a *different*
prior). The key observation that makes the filter scale is that a uniform draw over the
``2**E - 1`` non-self subsets needs no enumeration: it is ``E`` independent fair-coin
flips of the current set, rejecting the (vanishingly likely for large ``E``) no-change
outcome -- see :func:`_propose_next` for the exact, normalization-correct proposal.

The evidence
------------
Both take ``log_evidence`` of shape ``(T, E, 2)``: ``log_evidence[t, e, 0]`` is the
per-frame log-likelihood that edge ``e`` is *inactive* and ``[t, e, 1]`` that it is
*active*. Edges are treated as conditionally independent given the active set (they
observe disjoint body-pairs, THEORY.md §4/§8), so a subset's emission log-likelihood is
the per-edge sum -- exactly :mod:`contact.graph`'s factorization, but here abstracted away
from contacts. The two columns need not be normalized; only their *difference* (the
log-evidence ratio active-vs-inactive) affects the posterior, since a common per-edge
offset cancels out of every subset's emission identically.

Determinism
-----------
Everything is in log-space and properly normalized. The particle filter takes a ``seed``
(default ``0``) and uses an isolated :class:`numpy.random.Generator`, so results are
reproducible and never touch global RNG state.
"""

from __future__ import annotations

import markovlib as _markovlib
import numpy as np

from .hmm import forward_backward, viterbi

__all__ = ["exact_active_sets", "particle_filter_active_sets", "StructurePosterior"]

# Clip the dwell self-stay probability strictly inside (0, 1). A p_stay of exactly 1 makes
# the chain reducible (it can never switch sets) and exactly 0 forbids persistence; both
# break the log-transition and the proposal. THEORY.md §5: the temporal prior is *soft*.
_P_EPS = 1e-9


# ======================================================================================
# Shared helpers (the subset alphabet and the structure-level temporal prior).
# ======================================================================================


def _validate_evidence(log_evidence: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Coerce ``log_evidence`` to ``(T, E, 2)`` float and return ``(arr, T, E)``."""
    arr = np.asarray(log_evidence, dtype=float)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(
            f"log_evidence must have shape (T, E, 2); got {arr.shape}"
        )
    T, E, _ = arr.shape
    if T == 0 or E == 0:
        raise ValueError(f"log_evidence must be non-empty in T and E; got {arr.shape}")
    return arr, int(T), int(E)


def _p_stay(log_dwell_stay: float) -> float:
    """The self-stay probability ``exp(log_dwell_stay)``, clipped into ``(0, 1)``.

    ``log_dwell_stay`` is the log self-transition probability (THEORY.md §5: the
    discretized continuous-time Markov jump gives ``p_stay = exp(-dt/dwell)``, whose log is
    a natural log-domain knob). Clipped off the endpoints so the chain stays irreducible.
    """
    p = float(np.exp(float(log_dwell_stay)))
    return float(np.clip(p, _P_EPS, 1.0 - _P_EPS))


def _subset_log_transition(n_subsets: int, p_stay: float) -> np.ndarray:
    """``(2**E, 2**E)`` log-transition: self-stay ``p_stay``, rest uniform over the others.

    The structure-level temporal prior of THEORY.md §5 (lifted to §8), identical to
    :func:`contact.graph._subset_log_transition`::

        P(k -> k)  = p_stay
        P(k -> k') = (1 - p_stay) / (n_subsets - 1)    for k' != k.

    With a single subset (``E == 0`` would be the only way, excluded by validation) the
    chain self-loops. The particle-filter proposal of :func:`_propose_next` samples from
    exactly this distribution without forming the matrix.
    """
    if n_subsets <= 1:
        return np.zeros((1, 1), dtype=float)
    p_switch = (1.0 - p_stay) / (n_subsets - 1)
    A = np.full((n_subsets, n_subsets), p_switch, dtype=float)
    np.fill_diagonal(A, p_stay)
    A /= A.sum(axis=1, keepdims=True)  # guard tiny drift; rows are distributions
    return np.log(A)


# ======================================================================================
# (A) Exact estimator -- the reference (THEORY.md §8 / §10: 2**E enumeration).
# ======================================================================================


def exact_active_sets(
    log_evidence: np.ndarray,
    log_dwell_stay: float,
    seed: int = 0,
) -> tuple[np.ndarray, list[frozenset[int]]]:
    """Exact per-edge active-set posterior by ``2**E`` subset enumeration (THEORY.md §8).

    The reference estimator. Builds the ``2**E`` subset alphabet, assembles the joint
    log-emission (per-edge active/inactive sum, conditionally independent given the set),
    runs forward-backward + Viterbi over that alphabet (:mod:`contact.hmm`) under the
    structure-level dwell prior, and marginalizes the subset posterior to the per-edge
    active probability. Exact for any ``E`` but ``O(T * 4**E)`` in time and ``O(2**E)`` in
    state -- correct and preferred for the small graphs the package ships with (§10),
    and the ground truth :func:`particle_filter_active_sets` is validated against.

    Parameters
    ----------
    log_evidence:
        ``(T, E, 2)`` per-frame per-edge log-likelihood of ``[inactive, active]``. The two
        columns need not be normalized; only their difference affects the posterior.
    log_dwell_stay:
        Log self-transition probability of the structure-level Markov chain (THEORY.md
        §5); ``p_stay = exp(log_dwell_stay)``, the leftover mass split uniformly over the
        other subsets.
    seed:
        Accepted for API symmetry with :func:`particle_filter_active_sets` and
        determinism; the exact path is deterministic and does not use it.

    Returns
    -------
    active_posterior:
        ``(T, E)`` marginal ``P(edge e active at frame t)``.
    map_sets:
        Length-``T`` list of :class:`frozenset` of active edge indices -- the MAP
        (Viterbi) active set per frame.
    """
    del seed  # deterministic; present only for a uniform signature with the PF
    evidence, T, E = _validate_evidence(log_evidence)

    active_mask = _markovlib.product_membership(E, 2)          # (2**E, E) factor membership
    n_subsets = active_mask.shape[0]
    log_emission = _markovlib.product_log_emission(evidence, active_mask)  # (T, 2**E)
    log_trans = _subset_log_transition(n_subsets, _p_stay(log_dwell_stay))

    # Initial prior over subsets: favour the empty set (a record usually begins with no
    # contacts made), the rest uniform -- soft, so emissions dominate after frame 0. This
    # matches contact.graph.detect_scene's FREE-favouring init (THEORY.md §5).
    init = np.full(n_subsets, 0.5 / (n_subsets - 1) if n_subsets > 1 else 1.0, dtype=float)
    init[0] = 0.5 if n_subsets > 1 else 1.0
    init /= init.sum()
    log_init = np.log(init)

    gamma, _ = forward_backward(log_emission, log_trans, log_init)  # (T, 2**E)
    active_posterior = np.clip(gamma @ active_mask.astype(float), 0.0, 1.0)  # (T, E)

    map_path = viterbi(log_emission, log_trans, log_init)  # (T,) subset indices
    map_sets: list[frozenset[int]] = [
        frozenset(np.flatnonzero(active_mask[int(k)]).tolist()) for k in map_path
    ]
    return active_posterior, map_sets


# ======================================================================================
# (B) Particle smoother -- scales to large E without enumerating 2**E (THEORY.md §8).
#
# The exact reference (A) is a *smoother* (forward-backward conditions each frame on the
# whole record). A plain bootstrap particle *filter* is causal and would estimate a
# different quantity (the filtering marginal), so to match (A) we run a Rao-Blackwellized
# Forward-Filter / Backward-Smoother (FFBS): a forward bootstrap pass discovers, per frame,
# the small set of active-sets the data actually supports (never the full 2**E), and a
# backward pass reweights those *visited* sets into the smoothing marginal using the
# analytic dwell transition. Both passes are O(E) per particle and only ever touch visited
# sets, so the cost is governed by the particle count, not by 2**E (THEORY.md §8/§10).
# ======================================================================================


def _initial_particles(
    n_particles: int, E: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_particles, E)`` boolean initial particle sets matching the exact-path init.

    The exact estimator's initial prior puts mass 0.5 on the empty set and spreads the
    other 0.5 uniformly over the ``2**E - 1`` non-empty subsets. We sample that exact
    mixture without enumeration: half the particles (in expectation) are the empty set;
    the rest are a *uniform non-empty* subset, drawn as ``E`` fair coins with the all-zero
    outcome rejected (resampled) so the support is the non-empty subsets, uniformly. This
    keeps the filter's prior identical to the reference's (THEORY.md §5, soft init).
    """
    parts = np.zeros((n_particles, E), dtype=bool)
    pick_nonempty = rng.random(n_particles) >= 0.5
    for i in np.flatnonzero(pick_nonempty):
        s = rng.random(E) < 0.5
        while not s.any():  # reject the empty draw so this branch is uniform over non-empty
            s = rng.random(E) < 0.5
        parts[i] = s
    return parts


def _propose_next(parts: np.ndarray, p_stay: float, rng: np.random.Generator) -> np.ndarray:
    """Sample each particle's next set from the structure-level dwell transition.

    Draws from exactly the :func:`_subset_log_transition` law without forming the ``2**E``
    matrix: with probability ``p_stay`` a particle keeps its current set; otherwise it
    jumps to a set drawn *uniformly over the other ``2**E - 1`` subsets*. A uniform draw
    over all ``2**E`` subsets is just ``E`` fair coins; conditioning it on "not the current
    set" (rejection) yields the uniform-over-others jump. This is ``O(E)`` per particle and
    never enumerates -- the property that makes the filter scale (THEORY.md §8/§10).
    """
    n, E = parts.shape
    out = parts.copy()
    jump = rng.random(n) >= p_stay  # particles that switch this step
    for i in np.flatnonzero(jump):
        cand = rng.random(E) < 0.5
        while np.array_equal(cand, parts[i]):  # reject self => uniform over the *other* subsets
            cand = rng.random(E) < 0.5
        out[i] = cand
    return out


def particle_filter_active_sets(
    log_evidence: np.ndarray,
    log_dwell_stay: float,
    n_particles: int = 256,
    seed: int = 0,
) -> tuple[np.ndarray, list[frozenset[int]]]:
    """Rao-Blackwellized particle SMOOTHER over the active-set sequence (THEORY.md §8).

    Scales the structure posterior to large ``E`` **without enumerating** ``2**E`` (the
    THEORY.md §8 "particle filter over the discrete structure"; §10's large-``E`` rung)
    while estimating the *same* smoothing marginal as the exact reference
    :func:`exact_active_sets`, so the two agree in the large-particle limit. Each particle
    is one active set (an ``E``-bit boolean vector). The method is a Rao-Blackwellized
    Forward-Filter / Backward-Smoother (FFBS):

    **Forward (bootstrap filter).** Per frame ``t``:

    1. **Propose** (:func:`_propose_next`): each particle keeps its set with probability
       ``p_stay = exp(log_dwell_stay)`` else jumps to a uniform *other* subset -- sampled in
       ``O(E)`` as fair coins with self rejected, i.e. exactly the structure-level dwell
       transition that :func:`exact_active_sets` uses, never forming its ``2**E`` matrix.
    2. **Weight** by the frame's emission: a particle's log-weight is the per-edge evidence
       sum ``sum_e log_evidence[t, e, active_e]`` for its set (the same conditional-
       independence factorization as the exact path).
    3. **Resample** systematically.

    The forward pass *also* records, per frame, the distinct sets the resampled cloud
    occupies and their (pre-resampling) forward weights -- the empirical forward
    distribution on the small, data-supported support the particles discovered.

    **Backward (marginal smoother).** A standard FFBS backward recursion turns the stored
    forward weights into smoothing weights, using the analytic dwell transition
    (``O(E)`` per pair) evaluated *only between the visited sets* of adjacent frames --
    never over the ``2**E`` alphabet. The Rao-Blackwellized per-edge
    marginal is then the smoothing-weight-average of set membership (a conditional
    expectation, lower variance than raw particle frequencies by the Rao-Blackwell theorem),
    which is what lets a moderate particle count meet the exact reference.

    Because both passes only ever touch the handful of sets the cloud visits (which
    concentrates on data-supported sets regardless of ``E``), the cost is governed by the
    particle count, not by ``2**E`` -- the scalability the large-``E`` rung needs.

    The MAP set sequence is decoded per frame from the smoothed marginal (edge active iff
    ``P(edge active) > 0.5``): a tractable, well-defined per-frame decode (not a jointly-
    optimal Viterbi over ``2**E``, which is the honest scope of a particle method).

    Parameters
    ----------
    log_evidence:
        ``(T, E, 2)`` per-frame per-edge log-likelihood of ``[inactive, active]``. The two
        columns need not be normalized; only their difference affects the posterior.
    log_dwell_stay:
        Log self-transition probability of the structure-level dwell prior (as in
        :func:`exact_active_sets`).
    n_particles:
        Number of particles. More particles -> lower Monte-Carlo error and closer match to
        :func:`exact_active_sets` (validated: mean-abs-diff < 0.05 at ``>= 512`` on ``E=3``).
    seed:
        Seed for an isolated :class:`numpy.random.Generator` (default ``0``); results are
        deterministic and do not touch global RNG state.

    Returns
    -------
    active_posterior:
        ``(T, E)`` Rao-Blackwellized smoothing marginal ``P(edge e active at frame t)``.
    map_sets:
        Length-``T`` list of :class:`frozenset` of active edge indices (per-frame MAP).
    """
    evidence, T, E = _validate_evidence(log_evidence)
    n_particles = max(int(n_particles), 1)
    p_stay = _p_stay(log_dwell_stay)
    n_subsets = 1 << E
    log_stay = float(np.log(p_stay))
    log_switch = float(np.log((1.0 - p_stay) / (n_subsets - 1)))

    # The active-set chain expressed as a particle model: each particle is one active set (an E-bit
    # vector, carried as float 0/1 rows so np.unique / weighted-averaging are exact). markovlib runs the
    # generic Rao-Blackwellized particle smoother -- a forward bootstrap pass that records the visited
    # supports, then a backward FFBS over them; what is contact-specific is the *model*, supplied as the
    # four callables below: the soft init, the dwell-law proposal, the factored per-edge emission, and
    # the analytic dwell transition evaluated only between visited sets.
    def sample_prior(rng: np.random.Generator, n: int) -> np.ndarray:
        return _initial_particles(n, E, rng).astype(np.float64)

    def propagate(rng: np.random.Generator, parts: np.ndarray) -> np.ndarray:
        return _propose_next(parts.astype(bool), p_stay, rng).astype(np.float64)

    def log_likelihood(ev_t: np.ndarray, parts: np.ndarray) -> np.ndarray:
        # sum_e log_evidence[t, e, active_e] per particle -- the same conditional-independence
        # factorization the exact path uses (ev_t is (E, 2); parts is (N, E)). The frame-0 prior is
        # already carried by the sampled cloud, so the bare emission is the correct weight everywhere.
        active = parts.astype(bool)
        return np.where(active, ev_t[:, 1][None, :], ev_t[:, 0][None, :]).sum(axis=1)

    def log_transition(sets_from: np.ndarray, sets_to: np.ndarray) -> np.ndarray:
        # (U_from, U_to) dwell law: p_stay on equal sets, (1 - p_stay)/(2**E - 1) off -- the closed
        # form _subset_log_transition tabulates, evaluated only between the (few) visited sets.
        same = (sets_from.astype(bool)[:, None, :] == sets_to.astype(bool)[None, :, :]).all(axis=2)
        return np.where(same, log_stay, log_switch)

    model = _markovlib.StateSpaceModel(
        sample_prior=sample_prior, propagate=propagate, log_likelihood=log_likelihood
    )
    active_posterior = _markovlib.particle_smooth(
        model, evidence, log_transition, n_particles=n_particles, seed=int(seed)
    )
    active_posterior = np.clip(active_posterior, 0.0, 1.0)

    map_sets: list[frozenset[int]] = [
        frozenset(np.flatnonzero(active_posterior[t] > 0.5).tolist()) for t in range(T)
    ]
    return active_posterior, map_sets


# ======================================================================================
# Object interface: the multi-body engine, mirroring contact.hmm.HMM for one pair.
# ======================================================================================


class StructurePosterior:
    """The active-contact-STRUCTURE posterior over a graph's ``E`` candidate edges (§8).

    The multi-body analog of :class:`contact.hmm.HMM`. Where the HMM infers the latent MODE
    per frame for ONE body pair, this infers which SET of edges is simultaneously active over
    time -- a posterior over the ``2**E`` structures. It bundles the structure-level temporal
    prior (the self-stay log-probability ``log_dwell_stay``, THEORY.md §5 lifted to the
    structure level), so the per-edge log-evidence ``(T, E, 2)`` is the only per-call input.
    Two estimators of the SAME posterior, each returning ``(per-edge active posterior (T, E),
    MAP active-set sequence)``:

      * :meth:`exact`  -- enumerate the ``2**E`` subset alphabet and run the HMM
        forward-backward / Viterbi over it (the reference; preferred for small ``E``, §10).
      * :meth:`filter` -- a Rao-Blackwellized particle smoother that scales to large ``E``
        without ever materializing the ``2**E`` alphabet.
    """

    def __init__(self, log_dwell_stay: float, seed: int = 0) -> None:
        self.log_dwell_stay = float(log_dwell_stay)
        self.seed = int(seed)

    def exact(self, log_evidence: np.ndarray) -> tuple[np.ndarray, list[frozenset[int]]]:
        return exact_active_sets(log_evidence, self.log_dwell_stay, seed=self.seed)

    def filter(
        self, log_evidence: np.ndarray, n_particles: int = 256
    ) -> tuple[np.ndarray, list[frozenset[int]]]:
        return particle_filter_active_sets(
            log_evidence, self.log_dwell_stay, n_particles=n_particles, seed=self.seed
        )
