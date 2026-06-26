"""Unit tests for the multi-body contact-graph machinery (THEORY.md section 8).

These exercise rung 5 of the pragmatic ladder (THEORY.md s.10) — the active-set
structure posterior over a contact graph — in isolation, on small *hand-built*
synthetic scenes whose ground truth is known by construction:

* :func:`contact.graph.build_candidate_edges` — the s.8 broad-phase: prune a
  body-pair that never comes within ``proximity_gap``, keep one that touches.
* :func:`contact.graph.detect_scene` — the joint active-set inference: with two
  edges (A in contact throughout, B in contact only in the middle third) the
  per-edge active marginals and the MAP active set must recover that structure,
  the posterior columns must align with the ``edges`` list, and every per-frame
  subset distribution must be a valid (non-negative, normalized) distribution.
* :func:`contact.consistency.energy_log_factor` — the s.8 global energy/dissipation
  factor: a no-op (all-zeros) when it cannot be computed, otherwise a finite,
  NaN/inf-free, per-frame *relative* (mean-centred) preference.
* Exactness of the joint inference for ``E == 2``: the ``2**E`` enumeration of
  active sets is *exact*, so with a deliberately weak temporal prior the MAP
  active set equals the brute-force per-frame argmax over the 4 subsets (THEORY.md
  s.8: exact enumeration is correct/preferred for the small E here; large E would
  need RJMCMC/particle methods).

The scenes are built so the *gap* — the only channel that decides existence at
rest — is directly authored: every body is a point at the world origin offset in
z, with identity orientation, contacting a static ``"world"`` floor at ``z = 0``
with outward normal ``+z``. Then ``gap(t) == z(t)`` exactly, so "in contact"
means "z near 0" and we can place contact on whichever frames we like.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable regardless of how pytest is invoked (so this file
# collects whether run as ``uv run pytest``, ``pytest tests/``, or from inside
# ``tests/``). Mirrors the shim in the sibling suites (test_integration.py,
# test_scene_integration.py); without it ``from contact import ...`` errors at
# collection when the repo root is not already on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contact import consistency, graph
from contact.config import DetectorConfig, GraphParams
from contact.types import (
    ContactEdge,
    MultiBodyScene,
    PoseTrajectory,
    SupportSurface,
)

# --------------------------------------------------------------------------------------
# Scene construction helpers
# --------------------------------------------------------------------------------------

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

#: A static floor at z = 0 with outward normal +z, in the (identity) world frame.
FLOOR = SupportSurface(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))


def _time(T: int, fps: float = 200.0) -> np.ndarray:
    """A uniform time vector of ``T`` samples at ``fps`` Hz (s)."""
    return np.arange(T, dtype=float) / fps


def _vertical_body(t: np.ndarray, z: np.ndarray, x: float = 0.0, y: float = 0.0) -> PoseTrajectory:
    """A point body at fixed (x, y), identity orientation, with prescribed height ``z(t)``.

    Because the tracked contact point is the body origin and orientation is identity,
    the support-relative gap against the ``z = 0`` floor is exactly ``z(t)`` — so the
    caller authors the gap channel directly.
    """
    T = t.shape[0]
    position = np.zeros((T, 3), dtype=float)
    position[:, 0] = x
    position[:, 1] = y
    position[:, 2] = np.asarray(z, dtype=float)
    quat = np.tile(IDENTITY_QUAT, (T, 1))
    return PoseTrajectory(t=t, position=position, quat=quat)


def _edge(edge_id: str, moving_body: str, support_body: str = "world") -> ContactEdge:
    """A candidate edge of ``moving_body`` against the static ``"world"`` floor."""
    return ContactEdge(
        edge_id=edge_id,
        moving_body=moving_body,
        support_body=support_body,
        surface=FLOOR,
        contact_point_local=np.zeros(3),
    )


def _two_edge_scene(T: int = 240) -> MultiBodyScene:
    """A scene with two edges (A, B) whose ground-truth active structure is known.

    * Body ``a`` rests on the floor for the *entire* clip (z == 0)  -> edge A active throughout.
    * Body ``b`` is lifted well clear (z == 0.5 m) early and late, and rests *flat on the
      floor* (z == 0, genuinely at rest) through the middle third -> edge B active mid-clip.

    The descent/ascent of body ``b`` are *smooth, quick ramps* placed just outside the
    central resting window, so during that window ``b`` is truly stationary at gap == 0
    (the contact emission of THEORY.md s.4 wants both ``gap ~ 0`` and twist ~ 0). A hard
    z-step would inject a velocity transient that the s.4 pre-differentiation smoothing
    bleeds across the whole window, defeating the contact peak — so we author a settled
    rest, exactly as a real landing would look after touchdown.

    The two bodies sit at different (x, y) so their contact points are distinct (this also
    gives a non-degenerate support polygon for the balance factor, though we don't use it
    here).
    """
    t = _time(T)

    za = np.zeros(T, dtype=float)  # A always touching

    lo, hi = T // 3, 2 * T // 3
    ramp = max(12, T // 8)  # gentle transition width (frames), inside the outer thirds
    high = 0.2              # modest lift: a small velocity transient that settles quickly
    zb = np.full(T, high, dtype=float)
    # Smooth descent into the rest window (a cosine ease), flat rest, smooth ascent out.
    down = np.arange(lo - ramp, lo)
    up = np.arange(hi, hi + ramp)
    ease_down = 0.5 * (1.0 + np.cos(np.pi * (down - (lo - ramp)) / ramp))  # 1 -> 0
    ease_up = 0.5 * (1.0 - np.cos(np.pi * (up - hi) / ramp))               # 0 -> 1
    zb[down] = high * ease_down
    zb[lo:hi] = 0.0
    zb[up] = high * ease_up

    bodies = {
        "a": _vertical_body(t, za, x=0.0, y=0.0),
        "b": _vertical_body(t, zb, x=0.3, y=0.0),
    }
    edges = [_edge("A", "a"), _edge("B", "b")]
    scene = MultiBodyScene(name="two_edge", bodies=bodies, edges=edges, truth={})
    # Record the truth windows for the assertions (not consumed by the detector). ``ramp``
    # is how far the descent/ascent of B intrudes into the outer thirds; assertions skip
    # those transition frames (B is genuinely mid-air/touching only outside them).
    scene.meta["truth_window"] = {"A": (0, T), "B": (lo, hi), "ramp": ramp}
    return scene


# --------------------------------------------------------------------------------------
# build_candidate_edges — the s.8 broad-phase proximity prune
# --------------------------------------------------------------------------------------


def test_build_candidate_edges_prunes_far_keeps_near():
    """An edge whose bodies never come within ``proximity_gap`` is pruned; a touching one kept."""
    T = 60
    t = _time(T)
    # Edge "near": body rests on the floor (gap == 0) -> well within any proximity gap.
    # Edge "far":  body hovers 1 m above the floor for the whole clip -> never near.
    bodies = {
        "near_body": _vertical_body(t, np.zeros(T), x=0.0),
        "far_body": _vertical_body(t, np.full(T, 1.0), x=1.0),
    }
    edges = [_edge("near", "near_body"), _edge("far", "far_body")]
    scene = MultiBodyScene(name="prune", bodies=bodies, edges=edges, truth={})

    params = GraphParams(proximity_gap=0.05)
    kept = graph.build_candidate_edges(scene, params)
    kept_ids = [e.edge_id for e in kept]

    assert "near" in kept_ids, "a touching edge must survive the broad phase"
    assert "far" not in kept_ids, "an edge that never comes within proximity_gap must be pruned"
    # Order of the survivors is preserved (here only one survives).
    assert kept_ids == ["near"]


def test_build_candidate_edges_keeps_edge_that_touches_only_briefly():
    """Proximity is over the *whole* clip: an edge that touches at any frame is kept."""
    T = 90
    t = _time(T)
    z = np.full(T, 0.5)
    z[40:45] = 0.0  # comes down to touch for a handful of frames
    bodies = {"brief": _vertical_body(t, z)}
    scene = MultiBodyScene(
        name="brief", bodies=bodies, edges=[_edge("brief_edge", "brief")], truth={}
    )
    kept = graph.build_candidate_edges(scene, GraphParams(proximity_gap=0.05))
    assert [e.edge_id for e in kept] == ["brief_edge"]


def test_build_candidate_edges_drops_unknown_body():
    """An edge referencing a body absent from the scene (and not 'world') is dropped."""
    T = 30
    t = _time(T)
    bodies = {"a": _vertical_body(t, np.zeros(T))}
    edges = [_edge("good", "a"), _edge("bad", "ghost_body")]
    scene = MultiBodyScene(name="bad_ref", bodies=bodies, edges=edges, truth={})
    kept = graph.build_candidate_edges(scene, GraphParams(proximity_gap=0.05))
    assert [e.edge_id for e in kept] == ["good"]


# --------------------------------------------------------------------------------------
# detect_scene — the joint active-set posterior on a known two-edge scene
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_edge_result():
    """Run ``detect_scene`` once on the canonical two-edge scene (reused across tests)."""
    scene = _two_edge_scene(T=240)
    result = graph.detect_scene(scene, DetectorConfig())
    return scene, result


def test_detect_scene_columns_align_with_edges(two_edge_result):
    """``edges`` lists the column order of ``active_posterior``; shapes are consistent."""
    scene, result = two_edge_result
    T = scene.bodies["a"].t.shape[0]

    assert result.edges == ["A", "B"], "edge id order must match the scene's edge order"
    assert result.active_posterior.shape == (T, 2)
    assert set(result.per_edge.keys()) == {"A", "B"}
    assert len(result.map_active_set) == T
    # The MAP set per frame only ever names known edges.
    for active in result.map_active_set:
        assert set(active) <= {"A", "B"}


def test_detect_scene_recovers_per_edge_structure(two_edge_result):
    """A active throughout; B active only in the middle third (per-edge marginals)."""
    scene, result = two_edge_result
    T = scene.bodies["a"].t.shape[0]
    tw = scene.meta["truth_window"]
    lo, hi = tw["B"]
    ramp = tw["ramp"]

    col_A = result.active_posterior[:, result.edges.index("A")]
    col_B = result.active_posterior[:, result.edges.index("B")]

    # Stay away from the very first/last few frames (smoothing/derivative edge effects).
    pad = 5
    interior = slice(pad, T - pad)

    # Edge A: in contact throughout -> high marginal everywhere in the interior.
    assert np.all(col_A[interior] > 0.8), "edge A should be confidently active throughout"

    # Edge B: high only in the middle third, low in the outer thirds. The descent/ascent
    # ramps intrude ``ramp`` frames into the outer thirds, so skip those transition bands.
    mid = slice(lo + pad, hi - pad)
    first_third = slice(pad, lo - ramp - pad)
    last_third = slice(hi + ramp + pad, T - pad)
    assert np.all(col_B[mid] > 0.8), "edge B should be active in the middle third"
    assert np.all(col_B[first_third] < 0.2), "edge B should be inactive in the first third"
    assert np.all(col_B[last_third] < 0.2), "edge B should be inactive in the last third"


def test_detect_scene_map_active_set_transitions(two_edge_result):
    """The MAP active set transitions {A} -> {A, B} -> {A} across the clip."""
    scene, result = two_edge_result
    T = scene.bodies["a"].t.shape[0]
    tw = scene.meta["truth_window"]
    lo, hi = tw["B"]
    ramp = tw["ramp"]
    pad = 5

    def _set(k: int) -> set[str]:
        return set(result.map_active_set[k])

    # First third (before B's descent ramp): only A.
    for k in range(pad, lo - ramp - pad):
        assert _set(k) == {"A"}, f"frame {k}: expected MAP set {{A}}, got {_set(k)}"
    # Middle third (B settled on the floor): both A and B.
    for k in range(lo + pad, hi - pad):
        assert _set(k) == {"A", "B"}, f"frame {k}: expected MAP set {{A, B}}, got {_set(k)}"
    # Last third (after B's ascent ramp): back to A only.
    for k in range(hi + ramp + pad, T - pad):
        assert _set(k) == {"A"}, f"frame {k}: expected MAP set {{A}}, got {_set(k)}"


def test_detect_scene_marginals_are_valid_probabilities(two_edge_result):
    """Every per-edge marginal is a probability in [0, 1]."""
    _scene, result = two_edge_result
    ap = result.active_posterior
    assert np.all(np.isfinite(ap))
    assert np.all(ap >= -1e-9)
    assert np.all(ap <= 1.0 + 1e-9)


def test_detect_scene_subset_posterior_rows_are_distributions():
    """Every row of the joint subset posterior is a valid distribution (>=0, sums to 1).

    ``detect_scene`` marginalizes the subset posterior away into per-edge columns, so we
    re-run the joint forward-backward over the ``2**E`` subset alphabet exactly as the
    detector does and check the *full* subset posterior gamma is a proper per-frame
    distribution (THEORY.md s.4: posteriors are normalized).
    """
    from contact import hmm

    scene = _two_edge_scene(T=90)
    cfg = DetectorConfig()
    result = graph.detect_scene(scene, cfg)

    # Rebuild the joint emission/transition the same way detect_scene does, then run
    # forward-backward and verify gamma is a row-stochastic matrix.
    E = len(result.edges)
    T = result.t.shape[0]
    per_edge_post = np.column_stack(
        [result.per_edge[eid].contact_posterior for eid in result.edges]
    )
    subsets = graph._enumerate_subsets(E)
    active_mask = graph._subset_active_mask(E)
    log_active, log_inactive = graph._per_edge_log_evidence(per_edge_post)
    log_em = graph._subset_log_emission(log_active, log_inactive, active_mask)
    n_subsets = len(subsets)
    dt = graph._median_dt(result.t)
    log_trans = graph._subset_log_transition(n_subsets, dt, cfg.graph.active_set_dwell_time)
    init = np.full(n_subsets, 0.5 / (n_subsets - 1))
    init[0] = 0.5
    init /= init.sum()
    gamma, _ll = hmm.forward_backward(log_em, log_trans, np.log(init))

    assert gamma.shape == (T, n_subsets)
    assert np.all(np.isfinite(gamma))
    assert np.all(gamma >= -1e-9), "subset posterior must be non-negative"
    row_sums = gamma.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)


def test_detect_scene_empty_graph():
    """A scene with zero edges yields the single empty active set for all frames."""
    T = 20
    t = _time(T)
    scene = MultiBodyScene(
        name="empty", bodies={"a": _vertical_body(t, np.zeros(T))}, edges=[], truth={}
    )
    result = graph.detect_scene(scene, DetectorConfig())
    assert result.edges == []
    assert result.active_posterior.shape == (T, 0)
    assert result.map_active_set == [[] for _ in range(T)]
    assert result.meta["num_subsets"] == 1


# --------------------------------------------------------------------------------------
# Exactness of the joint inference (THEORY.md s.8: 2**E enumeration is EXACT)
# --------------------------------------------------------------------------------------


def test_joint_inference_is_exact_enumeration_for_E2():
    """With a weak temporal prior the MAP active set == brute-force per-frame argmax.

    The ``2**E`` subset enumeration is an *exact* joint model (no approximation), so when
    the temporal coupling is made negligible (a very long active-set dwell time => the
    self-transition advantage vanishes and every subset is ~equally reachable), Viterbi
    decouples across frames and must agree, frame by frame, with the independent argmax
    over the 4 subsets of the (emission-only) joint log-likelihood. We reproduce the
    emission exactly as ``detect_scene`` builds it and compare.
    """
    from contact import hmm

    scene = _two_edge_scene(T=100)
    # Make the temporal prior negligible: an enormous dwell time => P(stay) ~ P(switch),
    # so the Markov chain on subsets contributes ~no temporal coupling and the MAP path
    # is driven purely by the per-frame emission (exactly the brute-force regime).
    cfg = DetectorConfig()
    cfg.graph.active_set_dwell_time = 1e9
    # Disable the (otherwise tie-breaking) global consistency factors so the comparison is
    # against the pure per-edge emission argmax.
    cfg.graph.use_energy_prior = False
    cfg.graph.use_balance_prior = False

    result = graph.detect_scene(scene, cfg)
    E = len(result.edges)
    per_edge_post = np.column_stack(
        [result.per_edge[eid].contact_posterior for eid in result.edges]
    )
    subsets = graph._enumerate_subsets(E)
    active_mask = graph._subset_active_mask(E)
    log_active, log_inactive = graph._per_edge_log_evidence(per_edge_post)
    log_em = graph._subset_log_emission(log_active, log_inactive, active_mask)  # (T, 4)

    # Brute force: independent per-frame argmax over the 4 subset emissions.
    brute_path = np.argmax(log_em, axis=1)
    brute_sets = [set(subsets[int(k)]) for k in brute_path]

    n_subsets = len(subsets)
    dt = graph._median_dt(result.t)
    log_trans = graph._subset_log_transition(n_subsets, dt, cfg.graph.active_set_dwell_time)
    init = np.full(n_subsets, 0.5 / (n_subsets - 1))
    init[0] = 0.5
    init /= init.sum()
    viterbi_path = hmm.viterbi(log_em, log_trans, np.log(init))
    viterbi_sets = [set(subsets[int(k)]) for k in viterbi_path]

    # With the temporal prior switched off, the exact-enumeration MAP path equals the
    # independent per-frame argmax everywhere except possibly a transition frame where two
    # subsets tie. Require agreement on at least the overwhelming majority of frames.
    agree = sum(1 for a, b in zip(viterbi_sets, brute_sets) if a == b)
    assert agree >= len(brute_sets) - 2, (
        f"exact enumeration should match brute-force argmax with a weak prior; "
        f"agreed on {agree}/{len(brute_sets)} frames"
    )

    # And the detector's own reported MAP sets must agree with that Viterbi path too. The
    # detector reports edge *ids*; our local subsets are edge *indices* -> map to ids.
    edge_ids = result.edges
    viterbi_id_sets = [{edge_ids[e] for e in subsets[int(k)]} for k in viterbi_path]
    detector_sets = [set(s) for s in result.map_active_set]
    assert detector_sets == viterbi_id_sets


# --------------------------------------------------------------------------------------
# consistency.energy_log_factor — the s.8 global energy/dissipation factor
# --------------------------------------------------------------------------------------


def _subsets_for(scene: MultiBodyScene):
    """The list of active-edge-id tuples (the subset alphabet) for a scene's edges."""
    E = len(scene.edges)
    ids = [e.edge_id for e in scene.edges]
    return [tuple(ids[e] for e in idxs) for idxs in graph._enumerate_subsets(E)]


def test_energy_log_factor_finite_and_meancentred():
    """With moving bodies the energy factor is finite, NaN/inf-free, and mean-centred."""
    scene = _two_edge_scene(T=120)
    subsets = _subsets_for(scene)
    factor = consistency.energy_log_factor(scene, scene.edges, subsets, masses=None)

    n_subsets = len(subsets)
    T = scene.bodies["a"].t.shape[0]
    assert factor.shape == (T, n_subsets)
    assert np.all(np.isfinite(factor)), "energy factor must never be NaN/inf"
    # It is a *relative* per-frame preference: each row is mean-centred across states.
    np.testing.assert_allclose(factor.mean(axis=1), 0.0, atol=1e-9)


def test_energy_log_factor_noop_when_energy_flat():
    """A scene whose total mechanical energy never changes => the factor is a no-op (zeros).

    Every body is perfectly static at a fixed height, so KE == 0 and PE is constant: dE is
    identically zero, leaving the dissipation factor nothing to arbitrate. The documented
    contract is then all-zeros (THEORY.md s.8: a soft factor that abstains when it cannot
    be evaluated honestly).
    """
    T = 80
    t = _time(T)
    # Two bodies both pinned at constant heights (no motion => flat energy).
    bodies = {
        "a": _vertical_body(t, np.zeros(T), x=0.0),
        "b": _vertical_body(t, np.full(T, 0.2), x=0.3),
    }
    edges = [_edge("A", "a"), _edge("B", "b")]
    scene = MultiBodyScene(name="flat", bodies=bodies, edges=edges, truth={})
    subsets = _subsets_for(scene)
    factor = consistency.energy_log_factor(scene, scene.edges, subsets, masses=None)
    assert np.all(np.isfinite(factor))
    assert np.allclose(factor, 0.0), "flat-energy scene must yield a no-op energy factor"


def test_energy_log_factor_noop_on_empty_inputs():
    """No states / no bodies => an all-zero (degenerate-shape) factor, never NaN."""
    # Empty subset alphabet.
    scene = _two_edge_scene(T=40)
    empty = consistency.energy_log_factor(scene, scene.edges, [], masses=None)
    assert empty.size == 0 or np.allclose(empty, 0.0)

    # No bodies at all (T == 0): the factor returns a 0-row array, finite by construction.
    bare = MultiBodyScene(name="bare", bodies={}, edges=[], truth={})
    factor = consistency.energy_log_factor(bare, [], [], masses=None)
    assert np.all(np.isfinite(factor)) if factor.size else True
    assert not np.any(np.isnan(factor))


def test_energy_log_factor_masses_none_vs_given_both_finite():
    """The factor is finite/NaN-free whether masses are None (relative) or provided (Joule)."""
    scene = _two_edge_scene(T=100)
    subsets = _subsets_for(scene)

    f_none = consistency.energy_log_factor(scene, scene.edges, subsets, masses=None)
    f_mass = consistency.energy_log_factor(
        scene, scene.edges, subsets, masses={"a": 70.0, "b": 2.0}
    )
    for f in (f_none, f_mass):
        assert np.all(np.isfinite(f))
        np.testing.assert_allclose(f.mean(axis=1), 0.0, atol=1e-9)


# --------------------------------------------------------------------------------------
# Regression: the consistency factors must accept the *exact* encoding the production
# caller (graph.detect_scene) sends -- tuples of integer edge INDICES, not edge-id
# strings. A prior contract mismatch silently collapsed every active set to the empty set,
# turning both factors into permanent no-ops despite use_energy_prior defaulting True.
# --------------------------------------------------------------------------------------


def test_consistency_factors_accept_integer_index_subsets():
    """energy/balance factors are non-zero on the *raw* ``_enumerate_subsets`` index tuples.

    ``graph._enumerate_subsets(E)`` yields integer-index tuples ``(), (0,), (1,), (0, 1)``
    and ``detect_scene`` passes those straight through. The factors must resolve those
    indices against the ordered ``edges`` and produce a real (non-zero, mean-centred,
    finite) preference -- the production code path, not the id-string path the other tests
    use.
    """
    scene = _two_edge_scene(T=120)
    E = len(scene.edges)
    subsets_idx = graph._enumerate_subsets(E)  # integer-index tuples, the real encoding
    assert subsets_idx == [(), (0,), (1,), (0, 1)]

    ef = consistency.energy_log_factor(scene, scene.edges, subsets_idx, masses=None)
    bf = consistency.balance_log_factor(scene, scene.edges, subsets_idx)
    for f in (ef, bf):
        assert np.all(np.isfinite(f)), "factor must never be NaN/inf"
        np.testing.assert_allclose(f.mean(axis=1), 0.0, atol=1e-9)
    assert np.max(np.abs(ef)) > 1e-6, (
        "energy factor must be a non-zero preference on a moving scene when fed the "
        "integer-index subsets that graph.detect_scene actually emits"
    )
    assert np.max(np.abs(bf)) > 1e-6, (
        "balance factor must be a non-zero preference when fed integer-index subsets"
    )

    # And the index encoding must agree, factor-for-factor, with the equivalent edge-id
    # encoding (both are valid contracts; they must not diverge).
    subsets_ids = _subsets_for(scene)
    ef_ids = consistency.energy_log_factor(scene, scene.edges, subsets_ids, masses=None)
    bf_ids = consistency.balance_log_factor(scene, scene.edges, subsets_ids)
    np.testing.assert_allclose(ef, ef_ids)
    np.testing.assert_allclose(bf, bf_ids)


def test_detect_scene_energy_prior_active_and_shifts_posterior():
    """The s.8 energy prior is on by default and demonstrably couples the joint inference.

    Two assertions, both of which the old no-op bug would fail:
      1. ``meta['energy_prior_active']`` is ``True`` for a moving two-edge scene under the
         DEFAULT config (``use_energy_prior`` defaults True).
      2. Enabling vs. disabling the energy prior actually changes the joint posterior
         (the factor is no longer a silent no-op).
    """
    scene = _two_edge_scene(T=120)

    cfg_on = DetectorConfig()
    assert cfg_on.graph.use_energy_prior is True, "energy prior is meant to default on"
    res_on = graph.detect_scene(scene, cfg_on)
    assert res_on.meta["energy_prior_active"] is True

    cfg_off = DetectorConfig()
    cfg_off.graph.use_energy_prior = False
    cfg_off.graph.use_balance_prior = False
    res_off = graph.detect_scene(scene, cfg_off)
    assert res_off.meta["energy_prior_active"] is False

    # The energy prior must leave a fingerprint on the joint posterior. (It is a gentle
    # sub-nat nudge, so we only require a measurable, non-trivial difference somewhere.)
    diff = np.max(np.abs(res_on.active_posterior - res_off.active_posterior))
    assert diff > 1e-6, (
        "enabling the energy prior must shift the joint active-set posterior; a zero "
        "shift means the consistency factor is a silent no-op again"
    )


def test_detect_scene_balance_prior_active_when_enabled():
    """``use_balance_prior=True`` makes the balance factor actually engage end-to-end."""
    scene = _two_edge_scene(T=120)
    cfg = DetectorConfig()
    cfg.graph.use_balance_prior = True
    res = graph.detect_scene(scene, cfg)
    assert res.meta["balance_prior_active"] is True
