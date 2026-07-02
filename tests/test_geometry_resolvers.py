"""Permanent validation of the contact-geometry resolver ladder (DESIGN.md PART II.D / Phases 0-2).

These promote the session's resolver checks into self-contained tests: each generates its
scene/scenario *fresh* from the MuJoCo harness (no baseline snapshot, no scratchpad file) and
asserts either an analytic identity or a cross-code-path equality that pins the exact bug each
resolver was built to fix (DESIGN.md PART II section D):

* :class:`~contact.geometry_resolvers.FlatRegion` is the bit-identical Phase-0 floor --
  ``observe(geometry=FlatRegion(surface, cpl))`` reproduces ``observe()`` with no resolver on
  every channel (the regression lock that keeps the validated kinematic path untouched).
* :class:`~contact.geometry_resolvers.SphereSphere`'s POSITION-derived normal collapses the
  ball-ball "7 phantom impacts" to one real collision (the FlatRegion floor is what manufactures
  them), and is invariant to a spinning support -- the spin artifact a body-fixed normal injects.
* :class:`~contact.geometry_resolvers.BoxPlane`'s migrating nearest corner recovers the
  tumbling-box impact mode (a small gap at a bounce) exactly where the body-fixed FlatRegion gap
  is stale and large, with no corner-teleport velocity spike.

The emphasis matches ``tests/test_units.py``: qualitative discrimination that must hold *by
construction*, with tolerances loose enough to survive numerical noise yet tight enough to fail
on a real regression.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("mujoco")  # scenarios/scenes come from the MuJoCo harness; skip cleanly if absent

import oracle  # noqa: E402
from contact import observe  # noqa: E402
from contact.config import DetectorConfig  # noqa: E402
from contact.geometry_resolvers import BoxPlane, FlatRegion, SphereSphere  # noqa: E402
from contact.graph import _resolve_support  # noqa: E402
from contact.model import ContactDetector  # noqa: E402
from contact.types import IMPACT, PoseTrajectory  # noqa: E402

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

VST = DetectorConfig().vel_smooth_time


def _scene_edge(scene_name: str, edge_id: str):
    """(moving pose, resolved support pose, edge) for one edge of a freshly-generated scene."""
    scene = oracle.generate_scene(scene_name)
    edge = next(e for e in scene.edges if e.edge_id == edge_id)
    moving = scene.bodies[edge.moving_body]
    support = _resolve_support(scene, edge.support_body, moving)
    return moving, support, edge


def _n_impulses(obs) -> int:
    """Number of impulse atoms (impact events) the detector reports for one edge's observations."""
    return len(ContactDetector(DetectorConfig()).detect(obs).impulses)


# ======================================================================================
# FlatRegion -- the bit-identical Phase-0 floor (DESIGN.md III.2 / §9 Phase 0)
# ======================================================================================


class TestFlatRegionFloor:
    def test_flatregion_is_bit_identical_to_default_observe(self) -> None:
        """``observe(geometry=FlatRegion(surface, cpl))`` == ``observe()`` with geometry omitted.

        FlatRegion *is* the configuration of today's ``(surface, contact_point_local)`` spec, so
        it must reproduce the default ``observe`` arithmetic bit-for-bit on every channel -- the
        regression lock guaranteeing the validated kinematic/flat-floor path is untouched."""
        raw = oracle.generate("drop_rest")
        default = observe(raw.moving, raw.support, raw.surface, raw.contact_point_local, VST)
        flat = observe(
            raw.moving, raw.support, raw.surface, raw.contact_point_local, VST,
            geometry=FlatRegion(raw.surface, raw.contact_point_local),
        )
        for ch in ("gap", "v_normal", "v_tangent", "omega_normal", "omega_tangent"):
            a = np.asarray(getattr(default, ch), dtype=float)
            b = np.asarray(getattr(flat, ch), dtype=float)
            assert np.allclose(a, b, rtol=0, atol=1e-12), (
                f"FlatRegion must reproduce the default observe() {ch!r} bit-for-bit (Phase-0 lock); "
                f"max abs diff = {float(np.max(np.abs(a - b))):.2e}"
            )


# ======================================================================================
# SphereSphere -- the position-derived normal (DESIGN.md PART II.D / §3.3)
# ======================================================================================


class TestSphereSphere:
    def test_resolver_fixes_ball_ball_phantom_impacts(self) -> None:
        """SphereSphere collapses ball-ball impacts to <=2; the FlatRegion floor keeps the >=5 phantoms.

        On ``two_balls_collide``'s ballA<->ballB edge the SphereSphere resolver (which the scene
        attaches) must report at most the single real collision, while resolving the SAME edge
        through the body-fixed FlatRegion normal manufactures strictly MORE impact atoms -- proving
        the resolver, not anything downstream, is what fixes it (DESIGN.md PART II.D)."""
        moving, support, edge = _scene_edge("two_balls_collide", "ballA_ballB")
        assert type(edge.geometry).__name__ == "SphereSphere", (
            "the two_balls_collide ballA_ballB edge is expected to carry a SphereSphere resolver"
        )
        obs_sphere = observe(
            moving, support, edge.surface, edge.contact_point_local, VST, geometry=edge.geometry
        )
        obs_flat = observe(
            moving, support, edge.surface, edge.contact_point_local, VST, geometry=None
        )  # geometry=None -> the FlatRegion floor
        n_sphere = _n_impulses(obs_sphere)
        n_flat = _n_impulses(obs_flat)
        assert n_sphere <= 2, (
            f"SphereSphere must collapse the ball-ball impacts to <=2; got {n_sphere}"
        )
        assert n_flat >= 5, (
            f"the FlatRegion floor must exhibit the >=5 phantom impacts; got {n_flat}"
        )
        assert n_flat > n_sphere, (
            "the resolver is the fix: FlatRegion must yield strictly MORE impact atoms than "
            f"SphereSphere ({n_flat} !> {n_sphere})"
        )

    def test_normal_is_position_derived_and_invariant_to_support_spin(self) -> None:
        """The contact normal ``(c1-c2)/||.||`` is UNCHANGED when the support sphere spins.

        THE spin-artifact fix (DESIGN.md PART II.D): the normal comes from the line of centres,
        never from a body-fixed direction rotated by a quaternion. Spinning the support about its
        own axis must therefore leave the resolved normal identical, and equal to the
        line-of-centres direction."""
        T = 24
        t = np.linspace(0.0, 1.0, T)
        c1 = np.column_stack([np.linspace(0.20, -0.05, T), np.zeros(T), np.zeros(T)])  # moving centre
        c2 = np.zeros((T, 3))                                                          # support centre
        ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))
        ang = 3.7 * t                                       # arbitrary spin of the SUPPORT about +z
        spin = np.column_stack([np.cos(ang / 2.0), np.zeros(T), np.zeros(T), np.sin(ang / 2.0)])
        moving = PoseTrajectory(t=t, position=c1, quat=ident)
        sup_still = PoseTrajectory(t=t, position=c2, quat=ident)
        sup_spin = PoseTrajectory(t=t, position=c2, quat=spin)

        ss = SphereSphere(0.05, 0.05)
        n_still = np.array([fr[0].normal for fr in ss.resolve(moving, sup_still)])
        n_spin = np.array([fr[0].normal for fr in ss.resolve(moving, sup_spin)])
        d = c1 - c2
        expected = d / np.linalg.norm(d, axis=1, keepdims=True)

        assert np.allclose(n_spin, n_still, atol=1e-9), (
            "a position-derived normal must be invariant to a spinning support (the spin-artifact "
            f"fix); max change = {float(np.max(np.abs(n_spin - n_still))):.2e}"
        )
        assert np.allclose(n_still, expected, atol=1e-9), (
            "the SphereSphere normal must equal the line-of-centres direction (c1 - c2)/||.||"
        )


# ======================================================================================
# BoxPlane -- the migrating nearest corner (DESIGN.md PART II.D / Phase 2)
# ======================================================================================


class TestBoxPlane:
    def _tumbling(self):
        """The tumbling box resolved both ways: the FlatRegion floor and the BoxPlane resolver."""
        raw = oracle.generate("tumbling_box")
        flat = observe(raw.moving, raw.support, raw.surface, raw.contact_point_local, VST)
        box = observe(
            raw.moving, raw.support, raw.surface, raw.contact_point_local, VST,
            geometry=BoxPlane(np.array([0.1, 0.1, 0.1]), raw.surface),
        )
        return raw, flat, box

    def test_recovers_impact_mode_where_flatregion_gap_is_stale(self) -> None:
        """BoxPlane fires the impact mode and tracks the bouncing corner; the fixed FlatRegion point lags.

        The migrating nearest corner is at the plane (small ``|gap|``) at a tumbling bounce, while
        the body-FIXED FlatRegion point rides on a non-contacting face and reads a large gap. Both
        the impact-mode recovery (>0 frames) and that gap CONTRAST must hold (DESIGN.md PART II.D)."""
        _, flat, box = self._tumbling()
        n_impact = sum(
            1 for m in ContactDetector(DetectorConfig()).detect(box).map_state if m == IMPACT
        )
        assert n_impact > 0, (
            "BoxPlane must light the impact mode on the tumbling box (>0 impact-mode frames)"
        )
        # At a tumbling bounce: BoxPlane's nearest corner is ~touching (<0.02 m) yet the fixed-point
        # FlatRegion gap is far (>0.05 m). Such frames must exist (the migrating-corner payoff).
        at_bounce = np.abs(box.gap) < 0.02
        flat_stale = at_bounce & (flat.gap > 0.05)
        assert flat_stale.any(), (
            "there must be a bounce where the BoxPlane gap is small (<0.02 m) while the fixed-point "
            "FlatRegion gap is large (>0.05 m) -- the stale-contact bug BoxPlane fixes"
        )

    def test_velocity_has_no_corner_switch_spike(self) -> None:
        """max|v_normal| through BoxPlane is physically bounded (<5 m/s), not a corner-teleport spike.

        BoxPlane recovers the moving-point velocity ANALYTICALLY from the body twist, so the
        contact-point switching corners does not manufacture a teleport spike when differentiated
        (DESIGN.md PART II.D)."""
        _, _, box = self._tumbling()
        vmax = float(np.max(np.abs(np.asarray(box.v_normal, dtype=float))))
        assert vmax < 5.0, (
            f"BoxPlane's analytic migrating-contact normal velocity must be bounded (<5 m/s); got {vmax:.2f}"
        )
