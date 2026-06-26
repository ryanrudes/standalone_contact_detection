"""Single-pair contact-detection demos showcasing IMPACTS (THEORY.md s.6 / s.9).

These are additional SCENARIO builders for the MuJoCo truth factory, focused on the
*impact* regime: a touchdown is a near-instantaneous reset of the relative NORMAL velocity
(``v+ = -e v-`` with restitution ``e``), which the labeler (``mujoco_gen._classify_mode``)
flags as IMPACT whenever ``|relative normal closing speed|`` exceeds its threshold. Each
demo shapes the physics so a particular impact STRUCTURE dominates:

* ``hard_drop``         : a dense box dropped from height onto a STIFF floor -> one sharp
  FREE -> IMPACT -> STATIC (a single decisive touchdown atom, then a quiet rest).
* ``restitution_bounce``: a bouncy ball with high restitution -> a DECAYING TRAIN of
  IMPACT atoms separated by shrinking FREE flight arcs (the s.6 reset map iterated).
* ``angled_impact``     : a ball hurled at the floor along a diagonal -> a FREE approach,
  an IMPACT, then a tangential departure (the normal velocity flips, the tangential
  velocity survives -> the post-impact motion slides/rolls away).
* ``drop_on_incline``   : a ball dropped onto a TILTED plane -> an IMPACT against a
  NON-vertical contact normal, then a deflected bounce (the gap/impact channel tested
  against a tilted support, like ``demos_motion.incline_slide`` but for the impact mode).

Self-contained by contract: this module imports ONLY ``mujoco`` and ``numpy``. It defines
its own tiny <option>/id helpers so it never has to import ``mujoco_gen`` (which would
create an import cycle, since ``mujoco_gen`` imports THIS file at its end). The generic
simulate/label/observe path in ``mujoco_gen`` does everything else; a builder only returns
``(model, build_dict)``.

The builder contract (single-contact-pair scenarios), reproduced here for clarity:
    build["moving_body"], build["moving_geom"], build["support_geom"]   (names)
    build["support_body"]                                               ("world" or a name)
    build["surface_point_local"], build["surface_normal_local"]         (3,) on the support
    build["contact_point_local"]                                        (3,) on the moving body
    build["shape"]                              "box" | "sphere" | "cylinder" | "capsule"
    build["duration"]                           seconds to simulate
    build["init"]      optional callable(model, data) -> None  (one-time, e.g. set qvel)
    build["forcing"]   optional callable(model, data) -> None  (each substep)
    build["record_hz"] optional float: a per-scenario recording-rate floor. The truth labeler
                       samples the active set only on recorded frames (every 1/hz s), so a
                       brief energetic touchdown can be sub-frame at the default 100 Hz and
                       never register as the named IMPACT; a builder whose phenomenon needs a
                       finer cadence pins it here. mujoco_gen.generate takes max(caller hz,
                       record_hz), so it never narrows a caller asking for an even higher rate.
We deliberately do NOT set ``box_corners_local`` (that triggers the inverse-dynamics
metadata path reserved for the box-on-plane scenarios in ``mujoco_gen``).
"""

from __future__ import annotations

import numpy as np

import mujoco

# NOTE on self-containment (the import contract): this module imports ONLY ``mujoco`` and
# ``numpy``. The mode strings it ultimately produces (free/impact/static/sliding/rolling)
# are defined in ``contact.types`` and emitted by ``mujoco_gen._classify_mode``; we never
# need them here, so we deliberately do NOT import any ``contact`` submodule -- ``mujoco_gen``
# imports THIS file at its end, and importing it back would create a cycle.


# --------------------------------------------------------------------------------------
# Tiny self-contained helpers (intentionally NOT imported from mujoco_gen -- see module
# docstring on the import cycle).
# --------------------------------------------------------------------------------------

def _common_options() -> str:
    """MuJoCo <option> block shared by these demos.

    Gravity -9.81 z, a small timestep for clean impacts, and a pyramidal friction cone
    (matching the rest of the truth factory). The small timestep matters for the impact
    demos: a sharp touchdown otherwise smears across the recorded frames.
    """
    return (
        '<option timestep="0.0005" gravity="0 0 -9.81" '
        'integrator="implicitfast" cone="pyramidal"/>'
    )


def _id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    """Resolve a named MuJoCo object to its integer id (raises if absent)."""
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise KeyError(f"no {objtype!r} named {name!r} in model")
    return i


def _free_dofadr(model: mujoco.MjModel, body_id: int) -> int:
    """Index into ``data.qvel`` of the first DOF of ``body_id``'s (free)joint.

    For a freejoint body the six dofs are [vx, vy, vz, wx, wy, wz] (linear then angular,
    all WORLD-frame for a freejoint), so ``qvel[adr+0:adr+3]`` is the COM linear velocity
    and ``qvel[adr+3:adr+6]`` is the angular velocity.
    """
    return int(model.jnt_dofadr[model.body_jntadr[body_id]])


# Shared geometry constants (kept consistent so contact_point_local / surface lines up).
_BOX_HALF = 0.10        # box half-extent (m); bottom-center material point is [0,0,-_BOX_HALF]
_BALL_R = 0.05          # sphere radius (m) -- MUST match the detector's roll_radius (0.05) or
                        # the no-slip constraint |v_t| = R*|omega| no longer holds and ROLLING
                        # stops being detected. Ball visibility comes from tighter framing
                        # (shorter travel), not a larger radius.


# --------------------------------------------------------------------------------------
# Scenario builders
# --------------------------------------------------------------------------------------

def _build_hard_drop() -> tuple[mujoco.MjModel, dict]:
    """A dense box dropped from height onto a STIFF floor: one decisive touchdown.

    Physics (THEORY.md s.6, the impact reset map). The box free-falls from ~0.6 m, so at
    touchdown it carries a large downward speed (``v = sqrt(2 g h)`` ~ 3.4 m/s) -- well above
    the impact normal-velocity threshold, so the touchdown frame labels IMPACT. The floor is
    made STIFF and well-damped (``solref`` with a short time constant and near-critical
    damping) and the box is DENSE, so the touchdown is a single sharp arrest with essentially
    no bounce: the sequence is FREE -> a brief IMPACT atom -> a long sustained STATIC rest.
    This is the cleanest single-impact demo -- the s.6 "force atom" in isolation.
    """
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"
          friction="0.8 0.01 0.001" solref="0.004 1"/>
    <body name="box" pos="0 0 0.60">
      <freejoint name="boxj"/>
      <geom name="boxg" type="box" size="{_BOX_HALF} {_BOX_HALF} {_BOX_HALF}"
            density="2000" friction="0.8 0.01 0.001" solref="0.004 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    build = {
        "moving_body": "box",
        "moving_geom": "boxg",
        "support_body": "world",
        "support_geom": "floor",
        "surface_point_local": np.zeros(3),
        "surface_normal_local": np.array([0.0, 0.0, 1.0]),
        "contact_point_local": np.array([0.0, 0.0, -_BOX_HALF]),
        "shape": "box",
        "duration": 1.2,
    }
    return model, build


def _build_restitution_bounce() -> tuple[mujoco.MjModel, dict]:
    """A bouncy ball with high restitution: a decaying TRAIN of impact atoms.

    Physics (THEORY.md s.6): each touchdown applies the reset map ``v+ = -e v-`` with ``e``
    the coefficient of restitution; the damping ratio in ``solref`` sets ``e`` (a LOWER ratio
    returns more impact energy -> a bouncier ball). The result is a sequence of sharp IMPACT
    atoms (each touchdown) separated by FREE flight arcs of decreasing length, settling toward
    STATIC as the energy bleeds away. Distinct from ``demos_motion`` (which has no impact demo)
    and from the chain scenes (which collide bodies, not body-vs-floor).

    Tuning so the TRAIN is OBSERVABLE (the recording-cadence constraint). The truth labeler
    scans the simulator's active set only on RECORDED frames (every ``1/hz`` s), so each
    energetic touchdown -- a handful of 0.5 ms substeps -- must be SAMPLED or it is invisible.
    At the default 100 Hz every bouncy touchdown of this scenario falls between recorded
    frames, so the recorder caught exactly ONE impact and then a smear of low-speed Zeno
    chatter, NOT the named train (the very under-sampling failure warned about above, biting
    even at a moderately bouncy ``solref``). Rather than over-soften the contact into a single
    dead squish (which would erase the decaying TRAIN itself), we keep a genuinely BOUNCY,
    well-shaped contact (``solref="0.005 0.2"``, restitution in (0,1)) dropped from a modest
    0.25 m -> a clean decaying train of ~5 rebounds, and instead pin this scenario's recording
    rate via ``build["record_hz"] = 250``: fast enough that the first several distinct
    touchdowns are each sampled at their (high, then decreasing) normal closing speed, so the
    recorded truth shows the decaying TRAIN of several IMPACT atoms at decreasing speeds
    (~2.2, ~1.1, ~0.6, ... m/s) before the ball settles to STATIC. The drop is lowered from the
    old 0.6 m so the first touchdown is slow enough to be resolved, yielding more sampled
    rebounds. The physics (the THEORY.md s.6 reset map iterated) is unchanged; only the drop
    height and the observation cadence are.
    """
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"
          solref="0.005 0.2" solimp="0.95 0.99 0.001"/>
    <body name="ball" pos="0 0 0.30">
      <freejoint name="ballj"/>
      <geom name="ballg" type="sphere" size="{_BALL_R}" density="800"
            solref="0.005 0.2" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    build = {
        "moving_body": "ball",
        "moving_geom": "ballg",
        "support_body": "world",
        "support_geom": "floor",
        # Raise the observation-side plane by one radius so the tracked sphere CENTER's signed
        # distance reads ~0 at contact (the sphere convention: the surface absorbs the radius;
        # see mujoco_gen's bouncing_ball). Truth labels are geom-based and unaffected.
        "surface_point_local": np.array([0.0, 0.0, _BALL_R]),
        "surface_normal_local": np.array([0.0, 0.0, 1.0]),
        "contact_point_local": np.array([0.0, 0.0, 0.0]),
        "shape": "sphere",
        "duration": 3.0,
        # Sample fast enough that each of the first several distinct, energetic touchdowns is
        # caught at its normal closing speed (the decaying TRAIN); at the default 100 Hz they
        # are all sub-frame and only a single impact + Zeno chatter survives. See the docstring.
        "record_hz": 250.0,
    }
    return model, build


def _build_angled_impact() -> tuple[mujoco.MjModel, dict]:
    """A ball hurled at the floor along a diagonal: impact, then a tangential departure.

    Physics (THEORY.md s.6 / s.3). The ball starts in the air and is launched DOWN AND
    FORWARD (large -z and +x velocity). The approach is FREE; at the floor the NORMAL
    component of the velocity is reset (``v_n+ = -e v_n-``) while the TANGENTIAL component
    largely survives -- so the touchdown labels IMPACT (large normal closing speed) and the
    post-impact motion carries on along +x, sliding/rolling away across the floor. This is
    the canonical "oblique bounce" that separates the normal (impact) channel from the
    tangential (slide/roll) channel of the relative twist. A moderately bouncy floor gives
    one clear primary IMPACT and possibly a smaller secondary one before the ball departs.
    """
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="40 40 0.1" pos="0 0 0"
          friction="0.6 0.01 0.001" solref="0.003 0.4"/>
    <body name="ball" pos="-0.5 0 0.45">
      <freejoint name="ballj"/>
      <geom name="ballg" type="sphere" size="{_BALL_R}" density="800"
            friction="0.6 0.01 0.001" solref="0.003 0.4"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    ball_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    def init(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Hurl the ball forward (+x) and DOWN (-z) so it strikes the floor at an angle: the
        # approach is FREE, the touchdown is a sharp IMPACT (large -z closing speed), and the
        # surviving +x velocity carries the ball tangentially away after the bounce.
        adr = _free_dofadr(m, ball_id)
        d.qvel[adr + 0] = 1.1    # +x linear (forward; survives the bounce). Kept modest so the
        d.qvel[adr + 2] = -3.0   # ball does not travel far -> the camera frames it tight and big.

    build = {
        "moving_body": "ball",
        "moving_geom": "ballg",
        "support_body": "world",
        "support_geom": "floor",
        # Sphere convention: raise the observation plane by one radius (see restitution_bounce).
        "surface_point_local": np.array([0.0, 0.0, _BALL_R]),
        "surface_normal_local": np.array([0.0, 0.0, 1.0]),
        "contact_point_local": np.array([0.0, 0.0, 0.0]),
        "shape": "sphere",
        "duration": 1.1,         # shorter -> less travel -> the ball stays large in the frame
        "init": init,
    }
    return model, build


def _build_drop_on_incline() -> tuple[mujoco.MjModel, dict]:
    """A ball dropped onto a TILTED plane: an impact against a NON-vertical normal.

    Physics (THEORY.md s.6, impacts; s.1, the support-relative gap with a non-vertical
    normal). The plane is tilted by ``theta`` about +y, so its outward normal is
    ``n = (sin theta, 0, cos theta)`` -- NOT +z. The ball free-falls and strikes the incline:
    the touchdown labels IMPACT (the relative normal closing speed, taken along the TILTED
    normal, is large), then the ball deflects DOWN-slope and bounces away. This stresses the
    impact channel against a non-vertical support, the impact-mode analogue of
    ``demos_motion.incline_slide``.

    We mirror ``incline_slide``'s exact-incline geometry so the OBSERVED support-relative gap
    reads ~0 at the strike (a wrong surface point would leave a spurious cm-scale standoff in
    the observable channel). The ball is dropped from above the incline's top-face center.

    Tuning the contact so the IMPACT is OBSERVABLE (the recording-cadence constraint). The
    truth labeler scans the simulator's active set only on RECORDED frames (every ``1/hz`` s).
    A stiff bouncy contact (the old ``solref="0.003 0.4"``) makes the first strike a handful of
    0.5 ms substeps that the 100 Hz recorder never lands on, AND rebounds the ball back into
    free flight before the next recorded frame -- so the recorder only catches the ball once it
    is already steadily ROLLING down-slope and the named IMPACT never appears in the truth. We
    use a SOFTER, critically-damped contact (``solref="0.02 1"``): a 20 ms time constant
    lengthens the arrest so the touchdown spans several recorded frames while the normal
    closing speed is still well above the labeler's ``_IMPACT_VN`` threshold (so several
    frames label IMPACT), and the near-critical damping deflects the ball cleanly down-slope
    into a ROLLING departure instead of an energetic bounce. The sequence is now the named
    FREE -> IMPACT (against the tilted normal) -> ROLLING, observable at the default 100 Hz.
    """
    theta = np.deg2rad(18.0)
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    theta_deg = 18.0

    # The ramp is a large BOX geom tilted by +theta about +y; its TOP face is the incline.
    # Geometry: the ramp box has half-z = 0.5 and origin at z = -0.5, so its top-face center
    # in the ramp-local frame is (0, 0, 0.5). Rotating by R(theta about +y) and translating
    # gives the incline surface point and its outward normal in the world.
    R = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])  # rot about +y by +theta
    ramp_half_z = 0.5
    ramp_origin = np.array([0.0, 0.0, -ramp_half_z])
    top_world = ramp_origin + R @ np.array([0.0, 0.0, ramp_half_z])   # incline surface point
    n_world = R @ np.array([0.0, 0.0, 1.0])                           # incline outward normal
    # Drop the ball's CENTER from ~0.4 m above the incline surface point (along the world +z,
    # so it genuinely free-falls and strikes the tilted face).
    ball_origin = top_world + np.array([0.0, 0.0, 0.40])
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="ramp" type="box" size="3 3 {ramp_half_z}" pos="0 0 -{ramp_half_z}"
          euler="0 {theta_deg} 0" friction="0.6 0.01 0.001" solref="0.02 1"/>
    <body name="ball" pos="{ball_origin[0]} {ball_origin[1]} {ball_origin[2]}">
      <freejoint name="ballj"/>
      <geom name="ballg" type="sphere" size="{_BALL_R}" density="800"
            friction="0.6 0.01 0.001" solref="0.02 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    build = {
        "moving_body": "ball",
        "moving_geom": "ballg",
        "support_body": "world",
        "support_geom": "ramp",  # noqa: (duration shortened below to keep the ball framed large)
        # Surface point + TILTED normal in the support (world) frame -- the incline top face.
        # We push the observation plane out by one radius along the tilted normal so the
        # tracked sphere CENTER's signed distance reads ~0 at the strike (the sphere
        # convention, generalized to a non-vertical normal). Truth labels are geom-based.
        "surface_point_local": top_world + n_world * _BALL_R,
        "surface_normal_local": n_world,
        "contact_point_local": np.array([0.0, 0.0, 0.0]),
        "shape": "sphere",
        "duration": 1.0,  # shorter roll-down -> tighter framing so the small ball reads clearly
    }
    return model, build


# --------------------------------------------------------------------------------------
# Registries (the required module-level dicts). SCENE_BUILDERS is intentionally empty:
# every demo here is a single contact PAIR (a moving body vs one support).
# --------------------------------------------------------------------------------------

SCENARIO_BUILDERS: dict[str, callable] = {
    "hard_drop": _build_hard_drop,
    "restitution_bounce": _build_restitution_bounce,
    "angled_impact": _build_angled_impact,
    "drop_on_incline": _build_drop_on_incline,
}

SCENE_BUILDERS: dict[str, callable] = {}
