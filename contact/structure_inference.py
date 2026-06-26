"""Scalable posterior over the active-set sequence of a contact graph (THEORY.md s.8).

THEORY.md section 8 makes "richer contact information" precise: the hidden thing we
infer over a contact graph is not a bit but a *structure* -- **which set of edges is
simultaneously active**, over time, as a Bayesian posterior. Section 8 also names the
tractability fork explicitly: inference is "an HMM/particle filter over the discrete
structure", and section 10 spells out the consequence -- exact ``2**E`` enumeration is
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
:mod:`contact.graph` uses (THEORY.md s.5 lifted to the structure level, s.8): a Markov
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
observe disjoint body-pairs, THEORY.md s.4/s.8), so a subset's emission log-likelihood is
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

import numpy as np

from .hmm import forward_backward, logsumexp, viterbi

__all__ = ["exact_active_sets", "particle_filter_active_sets"]

# Clip the dwell self-stay probability strictly inside (0, 1). A p_stay of exactly 1 makes
# the chain reducible (it can never switch sets) and exactly 0 forbids persistence; both
# break the log-transition and the proposal. THEORY.md s.5: the temporal prior is *soft*.
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

    ``log_dwell_stay`` is the log self-transition probability (THEORY.md s.5: the
    discretized continuous-time Markov jump gives ``p_stay = exp(-dt/dwell)``, whose log is
    a natural log-domain knob). Clipped off the endpoints so the chain stays irreducible.
    """
    p = float(np.exp(float(log_dwell_stay)))
    return float(np.clip(p, _P_EPS, 1.0 - _P_EPS))


def _subset_active_mask(num_edges: int) -> np.ndarray:
    """``(2**E, E)`` boolean membership matrix; row ``k``'s bit ``e`` is edge ``e`` active.

    Mirrors :func:`contact.graph._subset_active_mask` (the same bitmask convention: state
    ``k`` is the subset ``{e : bit e of k is set}``, state 0 is the empty set) so the
    exact estimator here and the contact graph layer agree on the subset ordering. Only
    used by the *exact* path -- the particle filter never materializes this.
    """
    n_subsets = 1 << num_edges
    edges = np.arange(num_edges)
    ks = np.arange(n_subsets)[:, None]
    return ((ks >> edges[None, :]) & 1).astype(bool)


def _subset_log_emission(log_evidence: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    """``(T, 2**E)`` joint log-emission: per-subset sum of per-edge active/inactive evidence.

    For subset ``k`` (membership ``active_mask[k]``) and frame ``t``::

        log_emission[t, k] = sum_e ( log_evidence[t, e, 1]   if e in k
                                     else log_evidence[t, e, 0] ).

    Edges observe disjoint body-pairs => conditionally independent given the active set
    (THEORY.md s.4/s.8), hence the per-edge sum. Vectorized: pick the active or inactive
    column per (subset, edge) and sum over edges.
    """
    li = log_evidence[:, None, :, 0]   # (T, 1, E)  inactive
    la = log_evidence[:, None, :, 1]   # (T, 1, E)  active
    m = active_mask[None, :, :]        # (1, 2**E, E)
    per_edge = np.where(m, la, li)     # (T, 2**E, E)
    return per_edge.sum(axis=2)        # (T, 2**E)


def _subset_log_transition(n_subsets: int, p_stay: float) -> np.ndarray:
    """``(2**E, 2**E)`` log-transition: self-stay ``p_stay``, rest uniform over the others.

    The structure-level temporal prior of THEORY.md s.5 (lifted to s.8), identical to
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
# (A) Exact estimator -- the reference (THEORY.md s.8 / s.10: 2**E enumeration).
# ======================================================================================


def exact_active_sets(
    log_evidence: np.ndarray,
    log_dwell_stay: float,
    seed: int = 0,
) -> tuple[np.ndarray, list[frozenset[int]]]:
    """Exact per-edge active-set posterior by ``2**E`` subset enumeration (THEORY.md s.8).

    The reference estimator. Builds the ``2**E`` subset alphabet, assembles the joint
    log-emission (per-edge active/inactive sum, conditionally independent given the set),
    runs forward-backward + Viterbi over that alphabet (:mod:`contact.hmm`) under the
    structure-level dwell prior, and marginalizes the subset posterior to the per-edge
    active probability. Exact for any ``E`` but ``O(T * 4**E)`` in time and ``O(2**E)`` in
    state -- correct and preferred for the small graphs the package ships with (s.10),
    and the ground truth :func:`particle_filter_active_sets` is validated against.

    Parameters
    ----------
    log_evidence:
        ``(T, E, 2)`` per-frame per-edge log-likelihood of ``[inactive, active]``. The two
        columns need not be normalized; only their difference affects the posterior.
    log_dwell_stay:
        Log self-transition probability of the structure-level Markov chain (THEORY.md
        s.5); ``p_stay = exp(log_dwell_stay)``, the leftover mass split uniformly over the
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

    active_mask = _subset_active_mask(E)                       # (2**E, E)
    n_subsets = active_mask.shape[0]
    log_emission = _subset_log_emission(evidence, active_mask)  # (T, 2**E)
    log_trans = _subset_log_transition(n_subsets, _p_stay(log_dwell_stay))

    # Initial prior over subsets: favour the empty set (a record usually begins with no
    # contacts made), the rest uniform -- soft, so emissions dominate after frame 0. This
    # matches contact.graph.detect_scene's FREE-favouring init (THEORY.md s.5).
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
# (B) Particle smoother -- scales to large E without enumerating 2**E (THEORY.md s.8).
#
# The exact reference (A) is a *smoother* (forward-backward conditions each frame on the
# whole record). A plain bootstrap particle *filter* is causal and would estimate a
# different quantity (the filtering marginal), so to match (A) we run a Rao-Blackwellized
# Forward-Filter / Backward-Smoother (FFBS): a forward bootstrap pass discovers, per frame,
# the small set of active-sets the data actually supports (never the full 2**E), and a
# backward pass reweights those *visited* sets into the smoothing marginal using the
# analytic dwell transition. Both passes are O(E) per particle and only ever touch visited
# sets, so the cost is governed by the particle count, not by 2**E (THEORY.md s.8/s.10).
# ======================================================================================


def _log_dwell_pair(
    set_to: np.ndarray, set_from: np.ndarray, p_stay: float, n_subsets: int
) -> float:
    """log P(set_from -> set_to) under the structure-level dwell transition.

    The closed form of the same chain :func:`_subset_log_transition` tabulates -- evaluated
    for a *single* pair of sets in O(E), so the backward pass never forms the ``2**E`` x
    ``2**E`` matrix. ``p_stay`` on the diagonal, ``(1 - p_stay) / (n_subsets - 1)`` off it.
    """
    if np.array_equal(set_to, set_from):
        return float(np.log(p_stay))
    return float(np.log((1.0 - p_stay) / (n_subsets - 1)))


def _initial_particles(
    n_particles: int, E: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_particles, E)`` boolean initial particle sets matching the exact-path init.

    The exact estimator's initial prior puts mass 0.5 on the empty set and spreads the
    other 0.5 uniformly over the ``2**E - 1`` non-empty subsets. We sample that exact
    mixture without enumeration: half the particles (in expectation) are the empty set;
    the rest are a *uniform non-empty* subset, drawn as ``E`` fair coins with the all-zero
    outcome rejected (resampled) so the support is the non-empty subsets, uniformly. This
    keeps the filter's prior identical to the reference's (THEORY.md s.5, soft init).
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
    never enumerates -- the property that makes the filter scale (THEORY.md s.8/s.10).
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


def _systematic_resample(log_w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling indices from log weights ``log_w`` (``(N,)``).

    Systematic (a.k.a. stratified-with-one-uniform) resampling: one uniform draw
    ``u ~ U[0, 1/N)`` defines ``N`` equally spaced pointers ``(u + i)/N`` into the
    cumulative weight, giving lower-variance, ``O(N)`` resampling than multinomial. Weights
    are exponentiated from a max-shifted log to avoid underflow.
    """
    w = np.exp(log_w - np.max(log_w))
    w /= w.sum()
    N = w.shape[0]
    positions = (rng.random() + np.arange(N)) / N
    return np.searchsorted(np.cumsum(w), positions, side="left").clip(max=N - 1)


def _unique_sets(parts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a particle cloud to its distinct sets and an inverse index.

    Returns ``(uniq, inv)`` where ``uniq`` is ``(U, E)`` of the distinct sets present and
    ``inv`` maps each particle to its row in ``uniq``. The Rao-Blackwellized smoother works
    on these *distinct* sets (a handful, even for large E, since the cloud concentrates on
    data-supported sets), which is what keeps the backward pass off the ``2**E`` alphabet.
    """
    uniq, inv = np.unique(parts, axis=0, return_inverse=True)
    return uniq, inv.ravel()


def particle_filter_active_sets(
    log_evidence: np.ndarray,
    log_dwell_stay: float,
    n_particles: int = 256,
    seed: int = 0,
) -> tuple[np.ndarray, list[frozenset[int]]]:
    """Rao-Blackwellized particle SMOOTHER over the active-set sequence (THEORY.md s.8).

    Scales the structure posterior to large ``E`` **without enumerating** ``2**E`` (the
    THEORY.md s.8 "particle filter over the discrete structure"; s.10's large-``E`` rung)
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
    3. **Resample** systematically (:func:`_systematic_resample`).

    The forward pass *also* records, per frame, the distinct sets the resampled cloud
    occupies and their (pre-resampling) forward weights -- the empirical forward
    distribution on the small, data-supported support the particles discovered.

    **Backward (marginal smoother).** A standard FFBS backward recursion turns the stored
    forward weights into smoothing weights, using the analytic dwell transition
    (:func:`_log_dwell_pair`, ``O(E)`` per pair) evaluated *only between the visited sets*
    of adjacent frames -- never over the ``2**E`` alphabet. The Rao-Blackwellized per-edge
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
    rng = np.random.default_rng(int(seed))
    p_stay = _p_stay(log_dwell_stay)
    n_subsets = 1 << E

    # --- Forward bootstrap pass; record per-frame distinct sets + forward log-weights. --
    # `fwd_sets[t]`    : (U_t, E) distinct sets after frame t's resample.
    # `fwd_logw[t]`    : (U_t,)   forward log-weight mass on each distinct set (filtering
    #                              distribution p(A_t | o_{0:t}), in log, up to a constant).
    fwd_sets: list[np.ndarray] = []
    fwd_logw: list[np.ndarray] = []

    parts = _initial_particles(n_particles, E, rng)  # (N, E) bool
    for t in range(T):
        if t > 0:
            parts = _propose_next(parts, p_stay, rng)

        ev_t = evidence[t]  # (E, 2)
        emit = np.where(parts, ev_t[:, 1][None, :], ev_t[:, 0][None, :]).sum(axis=1)  # (N,)
        # The frame-0 cloud is already SAMPLED from the exact-path initial prior by
        # _initial_particles (each set appears with frequency proportional to init(k)), so
        # the prior is represented once, in the particle frequencies. The bare emission is
        # therefore the correct weight at every frame; adding log init(k) here too would
        # multiply by the prior a SECOND time and bias the frame-0 marginal toward
        # init(k)**2 * emit(k) -- an inconsistency that does not vanish as N -> infinity.
        # Later frames absorb the transition prior through the proposal in _propose_next.

        # Collapse to distinct sets and accumulate each set's forward log-mass (logsumexp of
        # the particle weights that landed on it). Equal proposal multiplicity is implicit in
        # how many particles occupy a set, so summing their weights is the empirical mass.
        uniq, inv = _unique_sets(parts)
        logw = np.full(uniq.shape[0], -np.inf)
        for k in range(uniq.shape[0]):
            members = emit[inv == k]
            logw[k] = logsumexp(members) if members.size else -np.inf
        fwd_sets.append(uniq)
        fwd_logw.append(logw)

        idx = _systematic_resample(emit, rng)
        parts = parts[idx]

    # --- Backward FFBS pass over the *visited* sets only (never the 2**E alphabet). -----
    # smoothing log-weight s_t(i) = f_t(i) + logsumexp_j [ log T(set_i -> set_j)
    #                                                      + s_{t+1}(j) - pred_{t+1}(j) ]
    # where f_t is the (normalized) forward log-weight and pred_{t+1}(j) = logsumexp_i
    # [ f_t(i) + log T(set_i -> set_j) ] is the one-step predictive at the visited set j.
    smooth_logw: list[np.ndarray] = [np.zeros(0) for _ in range(T)]
    f_last = fwd_logw[T - 1] - logsumexp(fwd_logw[T - 1])
    smooth_logw[T - 1] = f_last
    for t in range(T - 2, -1, -1):
        sets_t = fwd_sets[t]
        sets_n = fwd_sets[t + 1]
        f_t = fwd_logw[t] - logsumexp(fwd_logw[t])  # normalized forward at t
        # Transition log-matrix between the (small) visited supports of t and t+1.
        log_T = np.empty((sets_t.shape[0], sets_n.shape[0]))
        for i in range(sets_t.shape[0]):
            for j in range(sets_n.shape[0]):
                log_T[i, j] = _log_dwell_pair(sets_n[j], sets_t[i], p_stay, n_subsets)
        # Predictive at each next-set j: logsumexp_i (f_t(i) + log T(i->j)).
        pred_n = logsumexp(f_t[:, None] + log_T, axis=0)  # (U_{t+1},)
        s_next = smooth_logw[t + 1]
        ratio = s_next - pred_n  # safe: every visited set has nonzero predictive mass
        # s_t(i) = f_t(i) + logsumexp_j (log T(i->j) + ratio(j)).
        back = logsumexp(log_T + ratio[None, :], axis=1)  # (U_t,)
        s_t = f_t + back
        smooth_logw[t] = s_t - logsumexp(s_t)

    # --- Rao-Blackwellized per-edge smoothing marginal. ---------------------------------
    active_posterior = np.empty((T, E), dtype=float)
    for t in range(T):
        w = np.exp(smooth_logw[t] - logsumexp(smooth_logw[t]))  # (U_t,)
        active_posterior[t] = w @ fwd_sets[t].astype(float)
    active_posterior = np.clip(active_posterior, 0.0, 1.0)

    map_sets: list[frozenset[int]] = [
        frozenset(np.flatnonzero(active_posterior[t] > 0.5).tolist()) for t in range(T)
    ]
    return active_posterior, map_sets
