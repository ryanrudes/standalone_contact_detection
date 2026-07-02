"""Permanent validation of the capability registry + value-of-information (DESIGN.md Phase 5).

These promote the session's capability checks into self-contained tests. Each builds its scene
*fresh* and asserts the registry's contract (DESIGN.md PART I §4-§8):

* an empty :class:`~contact.capabilities.Capabilities` reproduces the validated kinematic/flat
  floor bit-for-bit (the no-op-when-absent guarantee);
* declaring a ``shape`` swaps in the higher-fidelity resolver -- ``sphere_sphere`` collapses the
  ball-ball phantom impacts the FlatRegion floor manufactures;
* a ``measured`` force channel recovers cradle clacks that are unobservable from kinematics;
* :func:`~contact.capabilities.value_of_information` ranks the force channel TOP for the cradle
  (MAP-change gain > 0) and emits the force-transfer guidance;
* the bare-pair API refuses ``force='inferred'`` (a whole-body quantity) with NotImplementedError.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("mujoco")  # the demos come from the MuJoCo harness; skip cleanly if absent

import oracle  # noqa: E402
from contact import observe  # noqa: E402
from contact.capabilities import Capabilities, detect_pair, value_of_information  # noqa: E402
from contact.config import DetectorConfig  # noqa: E402
from contact.graph import _resolve_support  # noqa: E402
from contact.detector import ContactDetector  # noqa: E402
from contact.types import IMPACT  # noqa: E402

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

CFG = DetectorConfig()


def _scene_pair(scene_name: str, edge_id: str):
    """(moving, support, surface, contact_point_local, truth_force) for one edge of a fresh scene."""
    scene = oracle.generate_scene(scene_name)
    edge = next(e for e in scene.edges if e.edge_id == edge_id)
    moving = scene.bodies[edge.moving_body]
    support = _resolve_support(scene, edge.support_body, moving)
    truth_force = np.asarray(scene.truth[edge_id].normal_force, dtype=float)
    return moving, support, edge.surface, edge.contact_point_local, truth_force


def _impact_frames(result) -> int:
    """Number of frames whose MAP mode is IMPACT."""
    return sum(1 for m in result.map_state if m == IMPACT)


# ======================================================================================
# The floor: an empty Capabilities() == today's kinematic/flat pipeline (DESIGN.md §7 inv. 1)
# ======================================================================================


class TestCapabilitiesFloor:
    def test_empty_capabilities_equals_kinematic_floor(self) -> None:
        """``detect_pair(Capabilities())`` == ``ContactDetector().detect(observe(...))`` on a demo.

        An empty declaration selects exactly the validated floor (FlatRegion geometry, no force
        factor), so its MAP path and contact posterior must match the bare pipeline bit-for-bit."""
        raw = oracle.generate("drop_rest")
        via_caps = detect_pair(
            raw.moving, raw.support, raw.surface, raw.contact_point_local, Capabilities(), CFG
        )
        obs = observe(raw.moving, raw.support, raw.surface, raw.contact_point_local, CFG.vel_smooth_time)
        floor = ContactDetector(CFG).detect(obs)
        assert via_caps.map_state == floor.map_state, (
            "an empty Capabilities() must reproduce the floor's MAP mode path exactly"
        )
        assert np.allclose(via_caps.contact_posterior, floor.contact_posterior, rtol=0, atol=1e-12), (
            "an empty Capabilities() must reproduce the floor's contact posterior bit-for-bit"
        )


# ======================================================================================
# Declaring a shape selects the richer resolver (DESIGN.md §4 / Phase 1)
# ======================================================================================


class TestShapeSelectsResolver:
    def test_sphere_sphere_shape_fixes_ball_ball_impacts(self) -> None:
        """Capabilities(shape='sphere_sphere') selects SphereSphere => <=2 impacts; the floor shows more.

        The same ballA<->ballB pair, detected through ``detect_pair``: declaring the sphere shape
        routes through the position-derived SphereSphere normal (<=2 impact atoms), while the bare
        ``Capabilities()`` (FlatRegion) reports strictly more (the phantom impacts)."""
        moving, support, surface, cpl, _ = _scene_pair("two_balls_collide", "ballA_ballB")
        r_floor = detect_pair(moving, support, surface, cpl, Capabilities(), CFG)
        r_shape = detect_pair(
            moving, support, surface, cpl,
            Capabilities(shape="sphere_sphere", params={"r_moving": 0.05, "r_support": 0.05}), CFG,
        )
        n_floor = len(r_floor.impulses)
        n_shape = len(r_shape.impulses)
        assert n_shape <= 2, (
            f"declaring the sphere shape must collapse the ball-ball impacts to <=2; got {n_shape}"
        )
        assert n_floor > n_shape, (
            "the FlatRegion floor must report strictly MORE impact atoms than the sphere resolver "
            f"({n_floor} !> {n_shape})"
        )


# ======================================================================================
# The force channel (DESIGN.md §6 / Phase 4a)
# ======================================================================================


class TestForceChannel:
    def test_measured_force_helps_the_cradle(self) -> None:
        """force='measured' + truth_force yields MORE impact-mode frames on the cradle than force='none'.

        The cradle's b3<->b4 momentum transfer is invisible to kinematics (no relative motion); a
        measured force pulse is what lights the IMPACT mode, so the measured-force run must show
        strictly more impact-mode frames than the kinematics-only run (shape held fixed)."""
        moving, support, surface, cpl, truth_force = _scene_pair("newtons_cradle", "b3_b4")
        shape = dict(shape="sphere_sphere", params={"r_moving": 0.035, "r_support": 0.035})
        r_none = detect_pair(moving, support, surface, cpl, Capabilities(force="none", **shape), CFG)
        r_force = detect_pair(
            moving, support, surface, cpl, Capabilities(force="measured", **shape), CFG,
            truth_force=truth_force,
        )
        assert _impact_frames(r_force) > _impact_frames(r_none), (
            "a measured force channel must recover cradle clacks invisible to kinematics "
            f"(impact-mode frames {_impact_frames(r_force)} !> {_impact_frames(r_none)})"
        )

    def test_inferred_force_unsupported_on_bare_pair(self) -> None:
        """force='inferred' is a whole-body quantity => the bare-pair API raises NotImplementedError.

        Inferred force is recovered at the scene/body level from mass/inertia + candidate geometry,
        inputs the bare ``(moving, support, surface, cpl)`` pair does not carry (DESIGN.md II.B)."""
        moving, support, surface, cpl, _ = _scene_pair("newtons_cradle", "b3_b4")
        with pytest.raises(NotImplementedError):
            detect_pair(moving, support, surface, cpl, Capabilities(force="inferred"), CFG)


# ======================================================================================
# Value of information (DESIGN.md §8): what should the user provide next?
# ======================================================================================


class TestValueOfInformation:
    def test_voi_ranks_force_top_for_the_cradle(self) -> None:
        """VoI ranks the force channel first (gain>0) and emits the force-transfer guidance.

        Given a sphere-shape base, the cradle's force-transfer signature means a measured force
        channel changes the MAP path while a no-op candidate does not; VoI (ranking by MAP-change,
        not entropy) must put ``force`` first with positive gain and carry the force-transfer note."""
        moving, support, surface, cpl, truth_force = _scene_pair("newtons_cradle", "b3_b4")
        voi = value_of_information(
            moving, support, surface, cpl,
            base=Capabilities(shape="sphere_sphere", params={"r_moving": 0.035, "r_support": 0.035}),
            candidates={"force": Capabilities(force="measured"), "none": Capabilities()},
            config=CFG, truth_force=truth_force,
        )
        assert len(voi) > 0, "VoI must return a non-empty ranking"
        assert voi[0][0] == "force", (
            f"the force channel must rank first for the cradle; got {voi[0][0]!r}"
        )
        assert voi[0][1] > 0.0, (
            f"the top capability must have strictly positive MAP-change gain; got {voi[0][1]}"
        )
        assert voi.guidance, (
            "VoI must emit the force-transfer ('unobservable from kinematics') guidance note"
        )
