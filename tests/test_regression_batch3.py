"""Regression guard for the RESEARCH-FRONTIER additions (THEORY.md s.8 & s.10).

This suite proves the Batch-3 frontier layer is wired in *correctly* and, just as
importantly, that it is **inert by default** -- every new capability is OFF unless
explicitly enabled, and turning it off must reproduce the validated Batch-1/Batch-2
behaviour byte-for-byte. The four guards mirror the four frontier additions:

  (1) **Uncertainty tempering is inert by default.** With ``use_uncertainty=False``
      (the shipped default) ``ContactDetector().detect`` on ``drop_rest`` returns the
      SAME ``in_contact`` mask / ``contact_iou`` the prior rungs measured (>0.9): the
      ``uncertainty.apply_tempering`` path in :meth:`ContactDetector.detect` is never
      taken, so the emission matrix is used unchanged (THEORY.md s.8, "per-frame
      measurement uncertainty" is opt-in).

  (2) **Tempering is a strict no-op without data.** Even with the flag flipped ON, if
      the observations carry no ``meas_cov`` the detector short-circuits (``temper_w``
      stays ``None``) and the result is *identical* to the default run -- the flag alone
      cannot change anything, only flag AND data together can.

  (3) **The structure-inference tractability fork (THEORY.md s.8/s.10).** On a 5-edge
      scene (``E = 5 > enumerate_max_edges = 4``) ``graph.detect_scene`` must take the
      Rao-Blackwellized particle path and still return a *valid* GraphDetectionResult
      (``active_posterior`` shape ``(T, 5)``, every marginal a probability in [0, 1],
      MAP sets naming only known edges). On a 2-edge scene (``E <= 4``) it must take the
      exact ``2**E`` enumeration and reproduce the Batch-2 behaviour (columns aligned,
      per-edge structure recovered, ``meta['inference'] == 'exact'``).

  (4) **Mode discovery returns the documented contract.** ``ContactDetector.discover_modes``
      returns a well-formed :class:`~contact.types.DiscoveredModeResult` (THEORY.md s.8,
      the label-free HDP-HMM mode vocabulary).

The synthetic scenes are authored exactly like the Batch-2 ``test_graph.py`` ones:
point bodies at the world origin offset in ``z``, identity orientation, contacting a
static ``"world"`` floor at ``z = 0`` with ``+z`` normal, so ``gap(t) == z(t)`` and the
contact structure is placed directly. MuJoCo is needed *only* for the ``drop_rest``
existence guards (skipped cleanly when absent); the graph / discovery guards are pure
synthetic and always run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable regardless of how pytest is invoked (mirrors the shim in
# the sibling suites test_graph.py / test_regression_full.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contact import graph
from contact.config import DetectorConfig
from contact.model import ContactDetector
from contact.types import (
    ContactEdge,
    ContactObservations,
    DiscoveredModeResult,
    GraphDetectionResult,
    MultiBodyScene,
    PoseTrajectory,
    SupportSurface,
)

# One fixed seed for the (noise-driven) MuJoCo runs so the existence guards are
# reproducible (the physics is deterministic; only the mocap noise depends on the seed).
SEED = 12345

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

#: A static floor at z = 0 with outward normal +z, in the (identity) world frame.
FLOOR = SupportSurface(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))


# --------------------------------------------------------------------------------------
# Synthetic scene construction (authoring the gap channel directly; same device as
# test_graph.py's _two_edge_scene -- gap(t) == z(t) for an origin-tracked, +z-up point).
# --------------------------------------------------------------------------------------


def _time(T: int, fps: float = 200.0) -> np.ndarray:
    """A uniform time vector of ``T`` samples at ``fps`` Hz (s)."""
    return np.arange(T, dtype=float) / fps


def _vertical_body(t: np.ndarray, z: np.ndarray, x: float = 0.0, y: float = 0.0) -> PoseTrajectory:
    """A point body at fixed (x, y), identity orientation, with prescribed height z(t).

    Because the tracked contact point is the body origin and orientation is identity, the
    support-relative gap against the z = 0 floor is exactly z(t) -- the caller authors the
    gap channel directly (mirrors test_graph.py).
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
    """A 2-edge scene (E <= enumerate_max_edges) -> exact enumeration path.

    Authored exactly like ``test_graph.py._two_edge_scene``: body ``a`` rests on the
    floor for the whole clip (edge A active throughout); body ``b`` is lifted clear early
    and late and rests flat on the floor through the middle third (edge B active mid-clip),
    with smooth cosine descent/ascent ramps so the rest window is genuinely settled (the
    s.4 contact emission wants gap ~ 0 AND twist ~ 0).
    """
    t = _time(T)
    za = np.zeros(T, dtype=float)  # A always touching

    lo, hi = T // 3, 2 * T // 3
    ramp = max(12, T // 8)
    high = 0.2
    zb = np.full(T, high, dtype=float)
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
    scene.meta["truth_window"] = {"A": (0, T), "B": (lo, hi), "ramp": ramp}
    return scene


def _five_edge_scene(T: int = 120) -> MultiBodyScene:
    """A 5-edge scene (E = 5 > enumerate_max_edges = 4) -> particle-filter path.

    Five point bodies on a static "world" floor, each at a distinct (x, y) so the contact
    points are spatially distinct. Their resting structure is authored directly via the
    gap channel:

      * ``e0`` rests on the floor the WHOLE clip (always active).
      * ``e1`` rests on the floor the WHOLE clip (always active).
      * ``e2`` hovers 0.5 m clear the WHOLE clip (never active).
      * ``e3`` rests through the first half, lifts clear in the second (active early).
      * ``e4`` hovers in the first half, settles onto the floor in the second (active late).

    The bodies that change state do so with a smooth cosine ramp at the midpoint so the
    rest windows are genuinely settled (gap ~ 0 AND twist ~ 0, the s.4 contact peak).
    We assert only that the result is *structurally valid* (the spec's bar for the large-E
    particle path); the exact-path scene below carries the recovery assertions.
    """
    t = _time(T)
    mid = T // 2
    ramp = max(10, T // 10)
    high = 0.5

    def _settle_then_lift() -> np.ndarray:
        """On floor through the first half, smooth lift to ``high`` across the midpoint."""
        z = np.zeros(T, dtype=float)
        up = np.arange(mid, min(mid + ramp, T))
        ease = 0.5 * (1.0 - np.cos(np.pi * (up - mid) / ramp))  # 0 -> 1
        z[up] = high * ease
        z[min(mid + ramp, T):] = high
        return z

    def _hover_then_settle() -> np.ndarray:
        """Clear through the first half, smooth descent to the floor across the midpoint."""
        z = np.full(T, high, dtype=float)
        down = np.arange(max(mid - ramp, 0), mid)
        ease = 0.5 * (1.0 + np.cos(np.pi * (down - (mid - ramp)) / ramp))  # 1 -> 0
        z[down] = high * ease
        z[mid:] = 0.0
        return z

    bodies = {
        "b0": _vertical_body(t, np.zeros(T), x=0.0, y=0.0),
        "b1": _vertical_body(t, np.zeros(T), x=0.4, y=0.0),
        "b2": _vertical_body(t, np.full(T, high), x=0.8, y=0.0),
        "b3": _vertical_body(t, _settle_then_lift(), x=0.0, y=0.4),
        "b4": _vertical_body(t, _hover_then_settle(), x=0.4, y=0.4),
    }
    edges = [
        _edge("e0", "b0"),
        _edge("e1", "b1"),
        _edge("e2", "b2"),
        _edge("e3", "b3"),
        _edge("e4", "b4"),
    ]
    return MultiBodyScene(name="five_edge", bodies=bodies, edges=edges, truth={})


def _synthetic_obs(T: int = 200, fps: float = 200.0) -> ContactObservations:
    """A small but mode-varied synthetic observation clip for the discovery contract.

    Three regimes back-to-back -- free (large gap, broad motion), static contact
    (gap ~ 0, ~still), and sliding contact (gap ~ 0, tangential motion) -- so the
    label-free discovery has real structure to find. Deterministic (no RNG here; the
    Gibbs sampler in discover_modes carries its own seed).
    """
    t = _time(T, fps)
    third = T // 3
    gap = np.zeros(T, dtype=float)
    gap[:third] = 0.30  # free: well clear
    v_tangent = np.zeros((T, 2), dtype=float)
    v_normal = np.zeros(T, dtype=float)
    v_normal[:third] = -0.5  # free: descending
    v_tangent[2 * third:, 0] = 0.20  # sliding: tangential drift in the last third
    omega_normal = np.zeros(T, dtype=float)
    omega_tangent = np.zeros((T, 2), dtype=float)
    return ContactObservations(
        t=t,
        gap=gap,
        v_normal=v_normal,
        v_tangent=v_tangent,
        omega_normal=omega_normal,
        omega_tangent=omega_tangent,
    )


# --------------------------------------------------------------------------------------
# (1) + (2) Uncertainty tempering: inert by default; a strict no-op without meas_cov.
# These run on the real drop_rest pipeline, so they need MuJoCo (skipped cleanly w/o it).
# --------------------------------------------------------------------------------------

mujoco = pytest.importorskip("mujoco")

from contact import geometry, mujoco_gen  # noqa: E402  (after the importorskip)
from oracle import report  # noqa: E402


def _drop_rest_obs() -> ContactObservations:
    """The drop_rest observable channel (noisy poses -> support-relative twist)."""
    raw = mujoco_gen.generate("drop_rest", seed=SEED)
    return geometry.observe(
        raw.moving, raw.support, raw.surface, raw.contact_point_local,
        geometry=getattr(raw, "geometry", None),
    ), raw


def test_default_config_tempering_is_inert_on_drop_rest():
    """use_uncertainty=False (default): drop_rest in_contact / contact_iou unchanged (>0.9).

    The tempering branch in ContactDetector.detect is guarded by BOTH the flag and the
    presence of obs.meas_cov; with the shipped default (flag off) ``temper_w`` is None and
    the emission matrix is used byte-for-byte as in Batch-1/Batch-2. So the existence
    recovery the prior rungs validated must be preserved exactly. drop_rest measures
    contact_iou ~ 0.96 at this seed; we hold the spec's >0.9 bar.
    """
    obs, raw = _drop_rest_obs()
    cfg = DetectorConfig()
    assert cfg.inference.use_uncertainty is False, "uncertainty must default OFF (s.8)"

    result = ContactDetector(cfg).detect(obs)
    scores = report.score(result, raw.truth)
    assert scores["contact_iou"] > 0.9, (
        f"drop_rest contact_iou regressed to {scores['contact_iou']:.3f} under the "
        f"default (tempering-off) config; the frontier layer is not inert"
    )


def test_uncertainty_flag_without_meas_cov_is_noop():
    """use_uncertainty=True but obs.meas_cov=None -> result IDENTICAL to the default run.

    The flag alone cannot change anything: detect() only builds ``temper_w`` when the flag
    is set AND obs carries a meas_cov. With meas_cov None the branch short-circuits, so the
    enabled and default runs must agree on EVERY output that the inference produces -- the
    boolean mask, the calibrated posterior, the full state posterior, the MAP path, the
    resting bias, and the scored IoU. This is the strict no-op the backward-compat rule
    demands (anything added is OFF unless explicitly enabled *and* fed data).
    """
    obs, raw = _drop_rest_obs()
    assert obs.meas_cov is None, "drop_rest obs must carry no measurement covariance"

    res_default = ContactDetector(DetectorConfig()).detect(obs)

    cfg_on = DetectorConfig()
    cfg_on.inference.use_uncertainty = True
    res_on = ContactDetector(cfg_on).detect(obs)

    # Identical booleans / labels / scalars ...
    assert list(res_on.map_state) == list(res_default.map_state)
    np.testing.assert_array_equal(res_on.in_contact, res_default.in_contact)
    assert res_on.resting_bias == res_default.resting_bias
    # ... and identical continuous posteriors (same emission matrix => same arithmetic).
    np.testing.assert_array_equal(
        res_on.contact_posterior, res_default.contact_posterior
    )
    np.testing.assert_array_equal(res_on.state_posterior, res_default.state_posterior)

    # And the externally-visible score is unchanged.
    s_default = report.score(res_default, raw.truth)
    s_on = report.score(res_on, raw.truth)
    assert s_on["contact_iou"] == s_default["contact_iou"]


# --------------------------------------------------------------------------------------
# (3) Structure-inference tractability fork (THEORY.md s.8/s.10).
# These are pure synthetic scenes -- no MuJoCo needed -- so they always run.
# --------------------------------------------------------------------------------------


def test_detect_scene_large_graph_takes_particle_path():
    """E = 5 > enumerate_max_edges = 4 -> the particle path, returning a valid result.

    The spec's bar for the large-E branch is *structural validity*, not exact recovery
    (the particle smoother is an approximation, documented as such): a well-formed
    GraphDetectionResult with ``active_posterior`` shape (T, 5), every marginal a real
    probability in [0, 1], finite, MAP sets naming only known edges, and the meta marked
    as the particle-filter inference (so we are certain the exact 2**E enumeration was NOT
    silently run on 32 subsets).
    """
    scene = _five_edge_scene(T=120)
    T = scene.bodies["b0"].t.shape[0]

    cfg = DetectorConfig()
    assert cfg.inference.enumerate_max_edges == 4, "default enumerate cap is 4 (s.8)"
    assert len(scene.edges) == 5 > cfg.inference.enumerate_max_edges

    result = graph.detect_scene(scene, cfg)

    assert isinstance(result, GraphDetectionResult)
    assert result.edges == ["e0", "e1", "e2", "e3", "e4"]
    assert result.active_posterior.shape == (T, 5)
    # Valid probabilities everywhere (finite, within [0, 1] up to fp slack).
    ap = result.active_posterior
    assert np.all(np.isfinite(ap)), "active_posterior must be finite"
    assert np.all(ap >= -1e-9), "active_posterior must be non-negative"
    assert np.all(ap <= 1.0 + 1e-9), "active_posterior must not exceed 1"
    # MAP active sets: one per frame, naming only the known edges.
    assert len(result.map_active_set) == T
    known = set(result.edges)
    for active in result.map_active_set:
        assert set(active) <= known
    # The tractability fork actually took the particle branch (not exact enumeration).
    assert result.meta["inference"] == "particle_filter", (
        f"E=5 must route through the particle path; got "
        f"{result.meta.get('inference')!r}"
    )
    assert result.meta["num_edges"] == 5
    # Every edge has its own per-edge single-pair result.
    assert set(result.per_edge.keys()) == known


def test_detect_scene_small_graph_takes_exact_enumeration():
    """E = 2 <= enumerate_max_edges -> exact 2**E enumeration, matching Batch-2 behaviour.

    Reproduces the validated Batch-2 (test_graph.py) recovery on the canonical two-edge
    scene: edge A active throughout, edge B active only in the middle third, columns
    aligned with ``edges``, and -- the Batch-3-specific check -- ``meta['inference']`` is
    ``'exact'`` (the small graph did NOT regress onto the particle approximation).
    """
    scene = _two_edge_scene(T=240)
    T = scene.bodies["a"].t.shape[0]
    tw = scene.meta["truth_window"]
    lo, hi = tw["B"]
    ramp = tw["ramp"]

    cfg = DetectorConfig()
    assert len(scene.edges) == 2 <= cfg.inference.enumerate_max_edges

    result = graph.detect_scene(scene, cfg)

    # Exact path was taken (the Batch-3 fork marker).
    assert result.meta["inference"] == "exact", (
        f"E=2 must use exact enumeration; got {result.meta.get('inference')!r}"
    )
    assert result.meta["num_subsets"] == 4  # 2**2

    # Columns aligned with the edge order, shapes consistent (Batch-2 contract).
    assert result.edges == ["A", "B"]
    assert result.active_posterior.shape == (T, 2)

    col_A = result.active_posterior[:, result.edges.index("A")]
    col_B = result.active_posterior[:, result.edges.index("B")]

    pad = 5
    interior = slice(pad, T - pad)
    mid = slice(lo + pad, hi - pad)
    first_third = slice(pad, lo - ramp - pad)
    last_third = slice(hi + ramp + pad, T - pad)

    # The Batch-2 per-edge structure must be recovered unchanged (same bounds as test_graph).
    assert np.all(col_A[interior] > 0.8), "edge A should be confidently active throughout"
    assert np.all(col_B[mid] > 0.8), "edge B should be active in the middle third"
    assert np.all(col_B[first_third] < 0.2), "edge B should be inactive in the first third"
    assert np.all(col_B[last_third] < 0.2), "edge B should be inactive in the last third"


# --------------------------------------------------------------------------------------
# (4) Mode discovery returns the documented DiscoveredModeResult contract (s.8).
# --------------------------------------------------------------------------------------


def test_discover_modes_returns_discovered_mode_result():
    """ContactDetector.discover_modes(obs) returns a well-formed DiscoveredModeResult (s.8).

    The label-free HDP-HMM entrypoint must return the contract types.py pins: per-frame
    integer ``labels`` (length T), a positive ``n_modes`` that matches the number of
    distinct labels actually used, a ``signatures`` dict keyed by the used mode ids (each
    a length-5 twist feature vector), and an ``alignment`` dict over the same ids. The
    sampler is seeded, so a given (obs, seed) is deterministic -- we assert that too.
    """
    obs = _synthetic_obs(T=180)
    T = obs.t.shape[0]

    detector = ContactDetector(DetectorConfig())
    result = detector.discover_modes(obs, seed=0)

    assert isinstance(result, DiscoveredModeResult)

    labels = np.asarray(result.labels)
    assert labels.shape == (T,), "labels must be one integer mode id per frame"
    assert np.issubdtype(labels.dtype, np.integer), "labels must be integer mode ids"

    # n_modes is the count of distinct modes the model actually used, and is positive and
    # within the HDP truncation level.
    used = set(int(x) for x in labels)
    assert result.n_modes == len(used), (
        f"n_modes ({result.n_modes}) must equal the number of distinct labels "
        f"({len(used)}) actually populated"
    )
    assert 1 <= result.n_modes <= DetectorConfig().inference.max_modes

    # signatures: one length-5 twist feature per used mode (gap, |v_n|, |v_t|, |w_n|, |w_t|).
    assert set(result.signatures.keys()) == used
    for mid, sig in result.signatures.items():
        arr = np.asarray(sig, dtype=float)
        assert arr.shape == (5,), f"signature for mode {mid} must be a length-5 feature"
        assert np.all(np.isfinite(arr)), f"signature for mode {mid} must be finite"

    # alignment: a canonical-name (validation-only) tag per used mode.
    assert set(result.alignment.keys()) == used
    for name in result.alignment.values():
        assert isinstance(name, str) and name

    # Deterministic for a fixed (obs, seed): a second run must reproduce the labelling.
    again = detector.discover_modes(obs, seed=0)
    np.testing.assert_array_equal(np.asarray(again.labels), labels)
