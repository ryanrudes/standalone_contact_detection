"""Multi-body contact-graph detector — the active-set structure posterior (THEORY.md s.8).

This is rung 5 of the pragmatic ladder (THEORY.md s.10): the single-pair estimator of
:class:`contact.model.ContactDetector` is lifted from one body-pair to a whole *contact
graph* whose nodes are bodies and whose edges are candidate body-pair contacts
(person<->deck, deck<->ground, hand<->rail). The hidden thing we infer is no longer a
bit per edge but a *structure*: **which set of edges is simultaneously active**, over
time, as a Bayesian posterior (THEORY.md s.8, first paragraph).

The object assembled here is exactly the s.8 sentence, restricted to the existence
layer of the structure (each edge's mode/loading is already carried inside its per-edge
:class:`~contact.types.DetectionResult`):

  * the **contact graph** with proximity as a broad-phase filter so we don't test every
    pair (:func:`build_candidate_edges`);
  * a **Bayesian posterior over active-constraint structures** — the joint active set —
    obtained by running the s.5 HMM machinery over the alphabet of the ``2**E`` subsets
    of edges (:func:`detect_scene`);
  * the **Signorini complementarity** of s.2 as the legality prior (each subset is a
    legal active set; per-edge Signorini is already enforced inside each edge's HMM);
  * the **hybrid temporal prior** of s.5 (the active set is temporally coherent — it
    persists, modelled by a Markov self-transition on the subset alphabet);
  * optional **energy/dissipation** and **balance** consistency factors that couple the
    edges globally (THEORY.md s.8: "an energy/dissipation budget ... is a global
    consistency check linking all contacts"), supplied by :mod:`contact.consistency`.

Tractability (THEORY.md s.8). For ``E`` edges there are ``2**E`` candidate active sets.
We **enumerate them exactly**; this is correct and preferred for the small graphs here
(``E <= 4`` => ``<= 16`` states). For large ``E`` the exact enumeration is exponential
and one would instead use RJMCMC / particle methods over the structure (THEORY.md s.8,
"an HMM/particle filter over the discrete structure"); that is out of scope here.

The support body of an edge **may itself be moving** — a foot on a skateboard deck whose
own contact with the ground is another edge. Everything is measured support-relative via
:func:`contact.geometry.observe`, which is the s.1 relative-frame payoff: the same
machinery handles moving-on-moving with no special case.

This module owns *only* :func:`build_candidate_edges` and :func:`detect_scene`. It
reuses :class:`contact.model.ContactDetector` per edge, :func:`contact.geometry.observe`
for the support-relative channels, :mod:`contact.hmm` for the joint inference, and
(optionally, if present) :mod:`contact.consistency` for the global soft factors.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import markovlib as _markovlib
import numpy as np

from . import geometry, hmm, structure_inference
from .config import DetectorConfig
from .model import ContactDetector
from .types import ContactEdge, GraphDetectionResult, MultiBodyScene, PoseTrajectory

__all__ = ["build_candidate_edges", "detect_scene"]


# Conventional name of the implicit, infinite-mass static support (THEORY.md s.1). A
# scene's `bodies` dict never contains "world" (the scene generators say so explicitly);
# an edge whose `support_body` is "world" is a contact against the fixed ground plane, the
# degenerate s.1 support that never moves. We synthesize an identity PoseTrajectory for it
# on demand so the same support-relative `geometry.observe` path handles a static floor and
# a moving deck with no special case.
_WORLD = "world"


def _resolve_support(
    scene: MultiBodyScene, support_body: str, like: PoseTrajectory | None
) -> PoseTrajectory | None:
    """Return the support body's PoseTrajectory, synthesizing a static "world" identity.

    THEORY.md s.1: the static floor is the special case of an infinite-mass support whose
    pose is identity for all time. The scene `bodies` dict omits it, so when an edge's
    ``support_body`` is ``"world"`` (or is otherwise absent but we have a reference body to
    borrow a timebase from) we build an identity trajectory: zero position, unit (w=1)
    quaternion, sharing ``like``'s clock. A real body is returned as-is; an unresolvable
    support (no reference timebase) returns ``None``.
    """
    body = scene.bodies.get(support_body)
    if body is not None:
        return body
    if support_body != _WORLD or like is None:
        return None
    t = np.asarray(like.t, dtype=float).ravel()
    T = int(t.shape[0])
    position = np.zeros((T, 3), dtype=float)
    quat = np.zeros((T, 4), dtype=float)
    quat[:, 0] = 1.0  # identity (w, x, y, z) = (1, 0, 0, 0)
    return PoseTrajectory(t=t, position=position, quat=quat)


# A clip on the per-edge active/inactive probability before taking a log, so an edge
# whose per-frame posterior pins at exactly 0 or 1 contributes a large-but-finite
# log-evidence rather than -inf (which would make whole subsets impossible for the wrong
# reason and inject NaNs into forward-backward). THEORY.md s.4: posteriors are soft.
_PROB_EPS = 1e-6


# --------------------------------------------------------------------------------------
# Broad phase (THEORY.md s.8: "proximity used as a broad-phase filter so we don't test
# every pair"). The contact graph would otherwise have O(N^2) edges; most are physically
# impossible (bodies that never come near each other). We prune an edge unless its two
# bodies come within `proximity_gap` at *some* frame.
# --------------------------------------------------------------------------------------


def build_candidate_edges(
    scene: MultiBodyScene, params=None
) -> list[ContactEdge]:
    """Broad-phase prune of the scene's candidate edges (THEORY.md s.8).

    Keep an edge only if, at *some* frame, the moving body's tracked material point comes
    within ``params.proximity_gap`` of the support body's surface — the cheap proximity
    test that stops us from running the (relatively expensive) per-edge HMM on body pairs
    that can never touch. The gap used here is exactly the support-relative signed
    distance :func:`contact.geometry.observe` computes (s.1), so the broad-phase metric
    and the fine-phase metric agree.

    The scenes we build already carry only *plausible* edges (the simulator emits the
    candidate graph it knows about), so on those this function is the **identity** — it
    is a guard for generality and for hand-assembled scenes, not a behaviour change on
    the standard scenarios. An edge is kept on any error evaluating its proximity (we
    never silently drop a candidate the simulator vouched for).

    Parameters
    ----------
    scene:
        The :class:`~contact.types.MultiBodyScene` whose ``edges`` are filtered.
    params:
        A :class:`~contact.config.GraphParams` (or anything with a ``proximity_gap``
        float). ``None`` uses the default :class:`~contact.config.GraphParams`.

    Returns
    -------
    list[ContactEdge]
        The surviving subset of ``scene.edges`` (order preserved).
    """
    if params is None:
        params = DetectorConfig().graph
    gap_thresh = float(getattr(params, "proximity_gap", 0.05))

    kept: list[ContactEdge] = []
    for edge in scene.edges:
        moving = scene.bodies.get(edge.moving_body)
        # The support may be the implicit static "world" floor (THEORY.md s.1), which the
        # scene's `bodies` dict omits; synthesize an identity trajectory on the moving
        # body's clock so a ground-contact edge is not mistaken for malformed.
        support = _resolve_support(scene, edge.support_body, moving)
        if moving is None or support is None:
            # A malformed edge references a body not in the scene (and not "world"); drop
            # it (it cannot be detected anyway). This is the one unambiguous prune case.
            continue
        try:
            obs = geometry.observe(
                moving, support, edge.surface, edge.contact_point_local
            )
            min_gap = float(np.nanmin(np.asarray(obs.gap, dtype=float)))
        except Exception:
            # Could not evaluate proximity (e.g. degenerate poses); keep the edge rather
            # than discard a candidate the scene vouched for.
            kept.append(edge)
            continue
        if min_gap <= gap_thresh:
            kept.append(edge)
    return kept


# --------------------------------------------------------------------------------------
# The subset alphabet (THEORY.md s.8: the hidden structure is *which edges are active*).
# We label each of the 2**E subsets by an integer in [0, 2**E); bit `e` of the integer is
# 1 iff edge `e` is in the active set. State 0 is the empty set (no contact anywhere).
# --------------------------------------------------------------------------------------


def _enumerate_subsets(num_edges: int) -> list[tuple[int, ...]]:
    """All ``2**E`` active sets as tuples of edge indices, ordered by the bitmask integer.

    State index ``k`` corresponds to the subset ``{e : bit e of k is set}``. Index 0 is
    the empty set. THEORY.md s.8: this exact enumeration is the tractable choice for the
    small graphs here (``E <= 4``); large ``E`` needs RJMCMC/particle methods.
    """
    subsets: list[tuple[int, ...]] = []
    for k in range(1 << num_edges):
        subsets.append(tuple(e for e in range(num_edges) if (k >> e) & 1))
    return subsets


def _subset_active_mask(num_edges: int) -> np.ndarray:
    """``(2**E, E)`` factor membership (int ``{0, 1}``); ``[k, e] == 1`` iff edge ``e`` is in subset ``k``.

    The hand-rolled bitmask logic now lives in :func:`markovlib.product_membership` (the library's
    factored/product-alphabet support); this is the graph layer's thin adapter over it.
    """
    return _markovlib.product_membership(num_edges, 2)


# --------------------------------------------------------------------------------------
# Joint emission (THEORY.md s.8 & s.4). For each frame t and each subset A we need
# log p(observations_t | active set = A). The per-edge detectors already gave us, per
# frame, P(edge active) = res.contact_posterior (a calibrated s.4 posterior). Treating the
# edges' *evidence* as conditionally independent given the active set (the edges observe
# disjoint body-pairs), the subset emission is the product over edges of the per-edge
# active/inactive likelihood — a sum in log-space:
#
#     log p(o_t | A) = sum_{e in A} log P(edge e active)_t
#                    + sum_{e not in A} log P(edge e inactive)_t
#
# The optional consistency factors (energy/balance) multiply this (add in log-space),
# coupling the edges globally where pure per-edge evidence cannot.
# --------------------------------------------------------------------------------------


def _per_edge_log_evidence(
    per_edge_posterior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame log P(active) and log P(inactive) for every edge, clipped off {0,1}.

    Parameters
    ----------
    per_edge_posterior:
        ``(T, E)`` array whose column ``e`` is edge ``e``'s ``contact_posterior``
        (P(edge active) per frame from its single-pair HMM, THEORY.md s.4/s.5).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(log_active, log_inactive)``, each ``(T, E)``. Probabilities are clipped to
        ``[_PROB_EPS, 1 - _PROB_EPS]`` first so a saturated edge gives a large-but-finite
        log-evidence rather than ``-inf`` (THEORY.md s.4: evidence is soft).
    """
    p = np.clip(np.asarray(per_edge_posterior, dtype=float), _PROB_EPS, 1.0 - _PROB_EPS)
    return np.log(p), np.log1p(-p)


def _subset_log_emission(
    log_active: np.ndarray, log_inactive: np.ndarray, active_mask: np.ndarray
) -> np.ndarray:
    """``(T, 2**E)`` joint log-emission: per-subset sum of per-edge active/inactive evidence.

    Edges observe disjoint body-pairs, so their evidence is conditionally independent given the active
    set (THEORY.md s.4/s.8) -- the per-subset sum. The construction now lives in
    :func:`markovlib.product_log_emission`; here we stack the per-edge evidence as ``(T, E, 2)``
    (``[..., 0]`` inactive, ``[..., 1]`` active) and delegate.
    """
    log_evidence = np.stack([log_inactive, log_active], axis=2)
    return _markovlib.product_log_emission(log_evidence, active_mask)


# --------------------------------------------------------------------------------------
# Temporal prior over the subset sequence (THEORY.md s.5 lifted to s.8). The active set
# is temporally coherent — the s.8 structure persists. We put a Markov chain on the 2**E
# subsets with a self-transition probability set by `active_set_dwell_time` (the same
# continuous-time-Markov-jump-discretized-per-frame device as the single-pair HMM): the
# leftover mass is spread uniformly over the OTHER subsets. This is deliberately a plain
# (memoryless) Markov prior on the structure; per-edge persistence already lives inside
# each edge's own HMM, so this layer only needs to keep the *joint* set coherent.
# --------------------------------------------------------------------------------------


def _subset_log_transition(
    n_subsets: int, dt: float, dwell_time: float
) -> np.ndarray:
    """``(n_subsets, n_subsets)`` log-transition for the active-set Markov chain.

    ``P(stay) = exp(-dt / dwell_time)`` (the discretized continuous-time Markov jump of
    THEORY.md s.5), with the complementary mass split uniformly over the other subsets:

        P(k -> k)  = p_stay
        P(k -> k') = (1 - p_stay) / (n_subsets - 1)   for k' != k.

    With a single subset (``n_subsets == 1``, the empty graph) the chain is trivially
    self-looping. Returns the elementwise log; all entries are strictly positive so the
    log is finite (THEORY.md s.5, the prior is soft).
    """
    if n_subsets <= 1:
        return np.zeros((1, 1), dtype=float)  # log P(0->0) = log 1 = 0
    dt = max(float(dt), 1e-9)
    dwell = max(float(dwell_time), 1e-9)
    p_stay = float(np.exp(-dt / dwell))
    p_switch = (1.0 - p_stay) / (n_subsets - 1)
    # Guard the floor so no entry is exactly zero (keeps the log finite, s.5).
    p_switch = max(p_switch, _PROB_EPS / n_subsets)
    A = np.full((n_subsets, n_subsets), p_switch, dtype=float)
    np.fill_diagonal(A, p_stay)
    # Renormalize rows (the floor may perturb the sum by a hair).
    A /= A.sum(axis=1, keepdims=True)
    return np.log(A)


def _median_dt(t: np.ndarray) -> float:
    """Median sampling interval of ``t`` (s); robust representative frame period.

    Mirrors :func:`contact.model._median_dt` so the active-set chain is discretized on
    the same clock the per-edge HMMs use (THEORY.md s.5).
    """
    t = np.asarray(t, dtype=float).ravel()
    if t.shape[0] < 2:
        return 1.0
    dts = np.diff(t)
    dts = dts[dts > 0.0]
    if dts.size == 0:
        return 1.0
    return float(np.median(dts))


# --------------------------------------------------------------------------------------
# Optional global consistency factors (THEORY.md s.8). Built in parallel in
# `contact.consistency`; we code to the pinned signatures and degrade to no-ops if the
# module is unavailable or the priors are disabled. The factors couple the edges
# *globally* — something the per-edge product emission structurally cannot do.
#
#   energy_log_factor(scene, edges, subset_index_per_state, masses_or_none) -> (n_subsets,) or (T, n_subsets)
#   balance_log_factor(scene, edges, subset_index_per_state)                -> (n_subsets,) or (T, n_subsets)
#
# Both return additive log-factors (0.0 == disabled/unknown), so we add them straight onto
# the joint log-emission. `subset_index_per_state` is our `subsets`: the list of
# active-edge-INDEX tuples (e.g. (), (0,), (1,), (0, 1)), telling consistency which edges
# each HMM state turns on. consistency resolves those integer indices against the *same*
# ordered `edges` list we pass alongside (its `_edge_ids(edges)` order), so the index ->
# edge-id mapping is exactly this module's `edge_ids` ordering — do NOT pre-convert here.
# --------------------------------------------------------------------------------------


def _consistency_factors(
    scene: MultiBodyScene,
    edges: list[ContactEdge],
    subsets: list[tuple[int, ...]],
    config: DetectorConfig,
    T: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Fetch the optional energy/balance log-factors, shaped to ``(T, n_subsets)``.

    Returns ``(energy, balance, diagnostics)`` where each factor is ``None`` when its
    prior is disabled or :mod:`contact.consistency` is unavailable, otherwise a
    ``(T, n_subsets)`` array (a ``(n_subsets,)`` return from consistency is broadcast
    across time). ``diagnostics`` records what was computed for ``GraphDetectionResult.meta``.

    THEORY.md s.8: these are *soft* global checks. If consistency hands back all-zeros
    (its documented "disabled/unknown" sentinel) we record the factor as inactive so the
    joint inference is unchanged — :func:`detect_scene` must work with or without them.
    """
    diagnostics: dict = {
        "energy_prior_active": False,
        "balance_prior_active": False,
    }
    try:  # consistency is built in parallel; tolerate its absence entirely.
        from . import consistency as _consistency
    except Exception:
        diagnostics["consistency_available"] = False
        return None, None, diagnostics
    diagnostics["consistency_available"] = True

    n_subsets = len(subsets)
    masses = scene.meta.get("masses") if isinstance(scene.meta, dict) else None

    def _shape(factor) -> np.ndarray | None:
        """Coerce a consistency return to (T, n_subsets); None if it is all-zero/empty."""
        if factor is None:
            return None
        arr = np.asarray(factor, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            if arr.shape[0] != n_subsets:
                return None
            arr = np.broadcast_to(arr, (T, n_subsets))
        elif arr.ndim == 2:
            if arr.shape != (T, n_subsets):
                return None
        else:
            return None
        # All-zero is consistency's "disabled/unknown" sentinel -> treat as inactive.
        if not np.any(arr):
            return None
        return np.ascontiguousarray(arr)

    energy = None
    balance = None
    if getattr(config.graph, "use_energy_prior", False):
        try:
            energy = _shape(
                _consistency.energy_log_factor(scene, edges, subsets, masses)
            )
        except Exception as exc:  # pragma: no cover - consistency is optional
            diagnostics["energy_error"] = repr(exc)
            energy = None
        diagnostics["energy_prior_active"] = energy is not None
    if getattr(config.graph, "use_balance_prior", False):
        try:
            balance = _shape(_consistency.balance_log_factor(scene, edges, subsets))
        except Exception as exc:  # pragma: no cover - consistency is optional
            diagnostics["balance_error"] = repr(exc)
            balance = None
        diagnostics["balance_prior_active"] = balance is not None
    return energy, balance, diagnostics


# --------------------------------------------------------------------------------------
# The graph detector (THEORY.md s.8): the whole pipeline end to end.
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Subset emission factors: the (T, 2^E) grid as a SUM of SubsetFactors (THEORY.md s.8)
# --------------------------------------------------------------------------------------
#
# Symmetric with the single-pair EmissionFactor sum (contact.emissions), one grid up: the subset
# emission is the always-present per-edge evidence plus the optional global consistency factors
# (energy / balance), each a no-op (the ZERO identity) when its capability is off. A distinct
# family from the (T, S) EmissionFactor -- different grid and inputs -- sharing the additive shape.

_SUBSET_ZERO = 0.0  # additive identity of the (T, 2^E) subset-emission grid


class SubsetFactor(Protocol):
    def contribute(self) -> np.ndarray | float:
        """A (T, 2^E) log-contribution over the active-set alphabet, or ``_SUBSET_ZERO`` when off."""
        ...


class SubsetEvidenceFactor:
    """Per-edge log-evidence summed over each active set (always present)."""

    def __init__(self, log_active: np.ndarray, log_inactive: np.ndarray, active_mask: np.ndarray) -> None:
        self.log_active = log_active
        self.log_inactive = log_inactive
        self.active_mask = active_mask

    def contribute(self) -> np.ndarray:
        return _subset_log_emission(self.log_active, self.log_inactive, self.active_mask)


class _ArraySubsetFactor:
    """Wraps a precomputed (T, n_subsets) log-factor (energy / balance from :mod:`contact.consistency`)."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def contribute(self) -> np.ndarray:
        return self.array


def detect_scene(
    scene: MultiBodyScene,
    config: DetectorConfig | None = None,
    edge_forces: dict[str, np.ndarray] | None = None,
) -> GraphDetectionResult:
    """Infer the joint active-set posterior over a multi-body contact graph (THEORY.md s.8).

    Pipeline:

    1. **Per-edge detection.** For every candidate edge run the single-pair estimator:
       ``obs = geometry.observe(moving, support, edge.surface, edge.contact_point_local,
       config.vel_smooth_time)`` then ``ContactDetector(config).detect(obs)``. The support
       body may itself be moving — the relative-frame payoff of s.1 (a foot on a deck
       whose own ground-contact is another edge). Collected into ``per_edge[edge_id]``.

    2. **Joint active-set inference.** With ``E`` edges, enumerate the ``2**E`` active
       sets (exact enumeration is correct and preferred for ``E <= 4``; large ``E`` would
       need RJMCMC/particle methods, s.8). For each frame and subset, the joint emission
       log-likelihood is the per-edge sum ``sum_{e in A} log P(e active) + sum_{e not in
       A} log P(e inactive)`` (edges observe disjoint body-pairs => conditionally
       independent given the set, s.4/s.8), evidence drawn from each edge's
       ``contact_posterior`` (clipped off {0,1}). Optional **energy** and **balance**
       consistency factors (THEORY.md s.8, the global dissipation/CoM checks) are added
       in log-space when enabled. A **temporal prior** — a Markov chain on the subsets
       with self-transition from ``config.graph.active_set_dwell_time`` — makes the active
       set temporally coherent (the s.8 structure persists). Forward-backward over the
       subset alphabet (:func:`contact.hmm.forward_backward`, the subsets as the state
       alphabet) gives the joint posterior, marginalized to ``active_posterior[t, e] =
       P(edge e active)``; Viterbi (:func:`contact.hmm.viterbi`) gives the MAP subset per
       frame, decoded to ``map_active_set``.

    3. **Return** a :class:`~contact.types.GraphDetectionResult` carrying the per-edge
       results, the per-edge active marginals, the MAP active sets, and energy/balance
       diagnostics in ``meta``.

    The candidate edges are taken as given on ``scene.edges`` — the simulator emits the
    plausible graph and :func:`build_candidate_edges` (the broad-phase) is the identity on
    those (call it first for hand-assembled scenes). With **zero edges** the result is the
    empty structure (a single empty active set for all frames).

    Parameters
    ----------
    scene:
        The :class:`~contact.types.MultiBodyScene` (bodies + candidate edges, shared timebase).
    config:
        A :class:`~contact.config.DetectorConfig` (fresh defaults if ``None``); its
        ``graph`` block controls the active-set dwell prior and the energy/balance factors,
        and the rest configures every per-edge :class:`~contact.model.ContactDetector`.
    edge_forces:
        Optional ``{edge_id: (T,) normal force}`` MEASURED-force streams (DESIGN.md PART II.A;
        PHASE 4a). When given, each edge's observations are augmented with its force (via
        ``dataclasses.replace``) before per-edge detection, so the gated per-state force
        emission sharpens free/contact/impact on that edge; an edge missing from the dict (or
        ``edge_forces is None``, the default) carries no force factor and is unchanged.

    Returns
    -------
    GraphDetectionResult
        ``t``, ``edges`` (column order of ``active_posterior``), ``per_edge``,
        ``active_posterior`` ``(T, E)``, ``map_active_set`` (length-T list of active edge-id
        lists), and ``meta`` (diagnostics: which factors were active, the joint
        log-likelihood, the subset alphabet size).
    """
    cfg = config if config is not None else DetectorConfig()
    edges = list(scene.edges)
    edge_ids = [e.edge_id for e in edges]
    E = len(edges)

    # Establish the timebase. The authoritative time vector comes from the bodies (all
    # share a timebase, s.8); fall back to a single-frame stub for an empty scene.
    any_body = next(iter(scene.bodies.values())) if scene.bodies else None
    t = (
        np.asarray(any_body.t, dtype=float).ravel()
        if any_body is not None
        else np.zeros(1, dtype=float)
    )
    T = int(t.shape[0])

    # --- (1) Per-edge single-pair detection (THEORY.md s.1 + s.4-s.7). --------------
    # Each edge gets its OWN emission scaling, fit to THAT edge's tangential motion. A scene
    # mixes slow and fast edges (a deck sliding ~1 m/s, a ball-ball surface slipping ~10 m/s);
    # a single scene-wide speed lets the fastest edge poison the slowest -- inflating the
    # sliding scale so a slow edge's real sliding reads FREE. Per-edge keeps each calibrated.
    per_edge: dict[str, "object"] = {}
    per_edge_posterior = np.zeros((T, max(E, 0)), dtype=float)
    for j, edge in enumerate(edges):
        moving = scene.bodies[edge.moving_body]
        # The support may itself be moving (s.1, a deck under a rider) OR the implicit
        # static "world" floor that the scene's `bodies` dict omits -- synthesize an
        # identity PoseTrajectory for the latter (THEORY.md s.1, infinite-mass support).
        support = _resolve_support(scene, edge.support_body, moving)
        if support is None:
            raise KeyError(
                f"edge {edge.edge_id!r}: support body {edge.support_body!r} is not in "
                f"scene.bodies and is not the implicit 'world' floor"
            )
        obs = geometry.observe(
            moving,
            support,
            edge.surface,
            edge.contact_point_local,
            cfg.vel_smooth_time,
            geometry=edge.geometry,
        )
        # Optional MEASURED-force channel (DESIGN.md PART II.A; PHASE 4a). When the caller
        # supplies per-edge normal forces, attach this edge's stream to its observations; an
        # edge absent from the dict gets None and so contributes no force factor. With
        # `edge_forces is None` (the default) the observations are untouched -- today's flow.
        if edge_forces is not None:
            obs = dataclasses.replace(obs, normal_force=edge_forces.get(edge.edge_id))
        # ContactDetector self-scales its emission to each obs's own tangential motion, so a
        # fast edge can no longer poison a slow one (the detector is applied per edge here).
        res = ContactDetector(cfg).detect(obs)
        per_edge[edge.edge_id] = res
        # The per-edge active probability per frame is the calibrated contact posterior.
        cp = np.asarray(res.contact_posterior, dtype=float).ravel()
        # Align to the scene timebase length (per-edge obs share it; guard a length skew).
        if cp.shape[0] != T:
            t = np.asarray(res.t, dtype=float).ravel()
            T = int(t.shape[0])
            per_edge_posterior = np.zeros((T, E), dtype=float)
            # Re-fill any earlier columns at the new length is unnecessary here because
            # all edges share the scene clock; this branch only triggers for E==1 skews.
        per_edge_posterior[:, j] = cp

    # --- Empty graph: no edges -> the structure is the single empty active set. -----
    if E == 0:
        return GraphDetectionResult(
            t=t,
            edges=[],
            per_edge={},
            active_posterior=np.zeros((T, 0), dtype=float),
            map_active_set=[[] for _ in range(T)],
            meta={
                "num_edges": 0,
                "num_subsets": 1,
                "energy_prior_active": False,
                "balance_prior_active": False,
            },
        )

    # --- (2) Joint active-set inference over the active-set sequence. ----------------
    # The per-edge active/inactive log-evidence (THEORY.md s.4/s.8) is the common input to
    # BOTH the exact enumeration and the large-graph particle smoother, so we build it once
    # here. `log_active`/`log_inactive` are each (T, E); the (T, E, 2) stack is exactly the
    # `[inactive, active]` evidence `contact.structure_inference` consumes.
    log_active, log_inactive = _per_edge_log_evidence(per_edge_posterior)

    # Tractability fork (THEORY.md s.8/s.10, governed by config.inference.enumerate_max_edges):
    #   * E <= enumerate_max_edges -> EXACT 2**E enumeration (correct and preferred for the
    #     small graphs the package ships with; this branch is byte-for-byte the prior
    #     behaviour, including the optional energy/balance consistency factors).
    #   * E >  enumerate_max_edges -> a Rao-Blackwellized particle SMOOTHER over the
    #     active-set sequence (contact.structure_inference.particle_filter_active_sets) that
    #     never materializes the 2**E alphabet, so it stays tractable as the graph grows.
    enumerate_max = int(getattr(cfg.inference, "enumerate_max_edges", 4))
    dt = _median_dt(t)

    if E <= enumerate_max:
        subsets = _enumerate_subsets(E)          # list of edge-index tuples; index 0 = {}
        active_mask = _subset_active_mask(E)     # (n_subsets, E) membership
        n_subsets = len(subsets)

        # Subset emission = a SUM of SubsetFactors: the always-present per-edge evidence + the
        # optional global consistency factors (energy / balance), each a no-op when off (s.8).
        energy, balance, diag = _consistency_factors(scene, edges, subsets, cfg, T)
        factors: list[SubsetFactor] = [SubsetEvidenceFactor(log_active, log_inactive, active_mask)]
        if energy is not None:
            factors.append(_ArraySubsetFactor(energy))
        if balance is not None:
            factors.append(_ArraySubsetFactor(balance))
        log_emission = factors[0].contribute()  # (T, n_subsets)
        for f in factors[1:]:
            log_emission = log_emission + f.contribute()

        # Temporal prior on the subset sequence: a Markov chain with the active-set dwell
        # self-transition (THEORY.md s.5 lifted to the structure level, s.8).
        log_trans = _subset_log_transition(
            n_subsets, dt, cfg.graph.active_set_dwell_time
        )

        # Initial prior over subsets: favour the empty set (a scene usually begins with the
        # contacts not yet made), the rest uniform — soft, so the emissions dominate after
        # the first frame (mirrors the single-pair FREE-favouring init, s.5).
        init = np.full(n_subsets, 0.5 / (n_subsets - 1) if n_subsets > 1 else 1.0, dtype=float)
        init[0] = 0.5 if n_subsets > 1 else 1.0
        init /= init.sum()
        log_init = np.log(init)

        # Joint posterior over subsets (forward-backward, the subsets as the state alphabet).
        gamma, total_loglik = hmm.forward_backward(log_emission, log_trans, log_init)  # (T, n_subsets)

        # Marginalize the subset posterior to the per-edge active marginal: P(edge e active)_t
        # = sum over subsets containing e of gamma[t, subset]. active_mask picks those subsets.
        active_posterior = gamma @ active_mask.astype(float)  # (T, n_subsets) @ (n_subsets, E)
        active_posterior = np.clip(active_posterior, 0.0, 1.0)

        # MAP active-set sequence (Viterbi over the subset alphabet) -> per-frame edge-id lists.
        map_path = hmm.viterbi(log_emission, log_trans, log_init)  # (T,) subset indices
        map_active_set: list[list[str]] = [
            [edge_ids[e] for e in subsets[int(k)]] for k in map_path
        ]

        meta = {
            "num_edges": E,
            "num_subsets": n_subsets,
            "inference": "exact",
            "joint_loglik": float(total_loglik),
            "active_set_dwell_time": float(cfg.graph.active_set_dwell_time),
            **diag,
        }
    else:
        # Large graph: enumerating 2**E is intractable. Run the Rao-Blackwellized particle
        # smoother on the per-edge evidence (THEORY.md s.8 "particle filter over the
        # discrete structure"; s.10's large-E rung). It uses the SAME structure-level dwell
        # prior as the exact path: p_stay = exp(-dt/active_set_dwell_time), so its log
        # self-transition is `-dt/active_set_dwell_time`. The per-edge active marginal it
        # returns is exactly `active_posterior[t, e]`.
        #
        # Honest scope of this branch (documented, not hidden): the particle smoother
        # works off the per-edge evidence and the dwell prior only -- it does NOT carry the
        # optional global energy/balance consistency factors (those are subset-level
        # couplings the per-edge-factored proposal cannot represent without re-introducing
        # the 2**E alphabet). For the small graphs the package ships with, E never exceeds
        # the default enumerate_max_edges=4, so this branch is dormant on the validated
        # scenes and the exact (factor-carrying) path is always taken there.
        dwell = max(float(cfg.graph.active_set_dwell_time), 1e-9)
        log_dwell_stay = -max(float(dt), 1e-9) / dwell
        log_evidence = np.stack([log_inactive, log_active], axis=-1)  # (T, E, 2)
        posterior = structure_inference.StructurePosterior(log_dwell_stay, seed=0)
        active_posterior, map_sets = posterior.filter(
            log_evidence, n_particles=int(getattr(cfg.inference, "n_particles", 256))
        )
        active_posterior = np.clip(np.asarray(active_posterior, dtype=float), 0.0, 1.0)
        map_active_set = [
            [edge_ids[e] for e in sorted(s)] for s in map_sets
        ]
        meta = {
            "num_edges": E,
            "num_subsets": 1 << E,
            "inference": "particle_filter",
            "n_particles": int(getattr(cfg.inference, "n_particles", 256)),
            "active_set_dwell_time": float(cfg.graph.active_set_dwell_time),
            "energy_prior_active": False,
            "balance_prior_active": False,
            "consistency_note": (
                "global energy/balance factors are not applied on the particle-filter "
                "(large-E) path; they are exact-enumeration-only"
            ),
        }

    return GraphDetectionResult(
        t=t,
        edges=edge_ids,
        per_edge=per_edge,
        active_posterior=np.asarray(active_posterior, dtype=float),
        map_active_set=map_active_set,
        meta=meta,
    )
