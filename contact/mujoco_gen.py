"""MuJoCo as the ground-truth oracle (THEORY.md section 9).

This module is the *truth factory* for the whole package. The theory makes claims
about quantities that are, by design, hard to observe from a mocap rig (the active
set, the contact mode, the normal force, the penetration). A physics simulator
makes those quantities visible to *us* — the experimenter — while still letting us
hand the detector only the "observable" channel (noisy body poses). That is exactly
the workflow of THEORY.md s.9:

    simulate -> record the full physical truth -> expose only noisy pose streams
    -> later score the inferred posterior against the withheld truth.

Everything here is headless physics only: we step `mujoco.mj_step` and read state
arrays. There is NO rendering and no GL context.

What we extract every recorded frame (THEORY.md s.9 bullet 1):

* body poses              : ``data.xpos`` / ``data.xquat`` for the moving body and
                            the support body (the cart for ``moving_support``,
                            otherwise an identity/world trajectory).
* the true active set     : iterate ``data.contact[:data.ncon]`` and keep the
                            contact between the moving geom and the support geom.
* penetration             : ``max(0, -contact.dist)`` (THEORY.md s.2: rigid bodies
                            cannot truly interpenetrate, so this is the simulator's
                            compliant squish, our calibrated force gauge in s.7).
* normal force            : ``mj_contactForce(...)[0]`` — the normal component in
                            the contact frame (THEORY.md s.7: the Lagrange
                            multiplier that pure kinematics cannot recover).
* mode                    : from the *relative* twist of the material contact point
                            (THEORY.md s.3: a mode is the twist subspace the relative
                            motion lives in). Documented thresholds below.

The detector never sees this module's truth labels — only ``RawScenario.moving`` /
``.support`` / ``.surface`` (the noisy observable channel) flow into
``geometry.observe``. The labels are withheld for scoring.

One disciplined caveat (THEORY.md s.9): MuJoCo's truth is truth *for MuJoCo's
contact model* (soft convex constraints, a pyramidal friction cone). It validates
the estimator's logic and identifiability, not absolute physical fidelity.
"""

from __future__ import annotations

import numpy as np

import mujoco

from contact.geometry import plane_gap, quat_to_matrix
from contact.types import (
    FREE,
    IMPACT,
    PIVOTING,
    ROLLING,
    SLIDING,
    STATIC,
    ContactEdge,
    GroundTruth,
    MultiBodyScene,
    PoseTrajectory,
    RawScenario,
    SupportSurface,
)

# --------------------------------------------------------------------------------------
# Mode-labeling thresholds (THEORY.md section 3).
#
# A contact mode is *which subspace of the 6D relative twist the motion lives in*. We
# classify each in-contact frame by the relative twist of the MATERIAL point currently
# at the contact (v = v_com + omega x r), measured in the support's contact frame.
# These thresholds are deliberately generous: the simulator's clean state is far less
# noisy than mocap, so the boundaries only need to separate qualitatively distinct
# regimes, not survive differentiation noise.
# --------------------------------------------------------------------------------------

_SLIP_EPS = 0.01        # m/s : tangential slip of the material contact point below this => no sliding
_SPIN_EPS = 0.30        # rad/s: relative spin about the normal below this => no pivoting
_IMPACT_VN = 0.20       # m/s : |relative normal closing speed| above this on a contact frame => IMPACT
_ROLL_VTAN = 0.05       # m/s : a sphere's COM tangential speed above this (with ~0 slip) => ROLLING


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def _id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    """Resolve a named MuJoCo object to its integer id (raises if absent)."""
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise KeyError(f"no {objtype!r} named {name!r} in model")
    return i


def _object_twist_world(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    """World-frame spatial velocity of a body: (omega (3,), v_com (3,)).

    ``mj_objectVelocity`` returns the 6-vector [angular(3); linear(3)] of the body's
    *origin/com* expressed in the world frame (flg_local=0). THEORY.md s.3: the right
    feature is the relative twist, kept as a vector so channel correlations survive.
    """
    buf = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, buf, 0)
    omega = buf[:3].copy()
    v_lin = buf[3:].copy()
    return omega, v_lin


def _material_point_velocity(
    omega: np.ndarray, v_lin: np.ndarray, body_pos: np.ndarray, contact_pos: np.ndarray
) -> np.ndarray:
    """World velocity of the body's material point currently at ``contact_pos``.

    v_point = v_com + omega x (contact_pos - body_pos). THEORY.md s.3: rolling vs.
    sliding is separated by the velocity of the *material point at the contact*, not
    the COM velocity — a rolling wheel's COM moves while its contact point is
    instantaneously at rest.
    """
    r = contact_pos - body_pos
    return v_lin + np.cross(omega, r)


def _body_inertial(model: mujoco.MjModel, body_id: int) -> dict:
    """Inertial properties of a body in its OWN (body-origin) frame (THEORY.md s.8).

    Contact-implicit inverse dynamics needs the Newton-Euler mass matrix: the scalar mass,
    the 3x3 rotational inertia, and the center of mass. MuJoCo stores these w.r.t. the
    *principal* inertial frame, so we reconstruct the body-frame tensor:

    * ``model.body_mass[id]``    : scalar mass (kg).
    * ``model.body_inertia[id]`` : the DIAGONAL principal moments, in the principal frame.
    * ``model.body_iquat[id]``   : rotation principal-frame -> body frame (scalar-first).
    * ``model.body_ipos[id]``    : the com offset from the body origin, in the body frame.

    The body-frame inertia tensor about the com is therefore ``I_body = R diag(I_p) R^T``
    with ``R = R(iquat)`` (principal -> body). For the plain box (drop_rest etc.) ``iquat``
    is identity and ``ipos`` is zero, so this collapses to ``diag(body_inertia)`` and a zero
    com; for the off-center rig the lump tilts the principal frame and shifts the com, and
    this reconstruction recovers the full body-frame tensor honestly.
    """
    mass = float(model.body_mass[body_id])
    principal = np.asarray(model.body_inertia[body_id], dtype=float)  # (3,) diagonal moments
    R = quat_to_matrix(np.asarray(model.body_iquat[body_id], dtype=float))  # principal -> body
    inertia = R @ np.diag(principal) @ R.T                            # (3,3) body-frame tensor
    com_local = np.asarray(model.body_ipos[body_id], dtype=float).copy()  # (3,) com offset
    return {"mass": mass, "inertia": inertia, "com_local": com_local}


def _world_to_body_local(
    point_world: np.ndarray, body_pos: np.ndarray, body_quat: np.ndarray
) -> np.ndarray:
    """Express a world-frame point in a body's local frame.

    p_local = R(quat)^T @ (p_world - body_pos), with the scalar-first quaternion rotated
    by ``mju_rotVecQuat`` after conjugation. Used to match each multi-contact point to
    the box corner it sits under (THEORY.md s.7 / s.8: a contact's material location on
    the body is how we attribute a per-corner load to a per-corner penetration).
    """
    rel = np.asarray(point_world, dtype=float) - np.asarray(body_pos, dtype=float)
    q = np.asarray(body_quat, dtype=float)
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])  # inverse rotation (unit quaternion)
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, rel, q_conj)
    return out


# --------------------------------------------------------------------------------------
# Scenario builders.
#
# Each returns (model, build) where `build` is a dict describing the named entities and
# any per-step forcing. The named-entity contract every scenario must satisfy:
#   build["moving_body"], build["moving_geom"], build["support_geom"]  (names)
#   build["support_body"]                                              (name or "world")
#   build["surface_point_local"], build["surface_normal_local"]        (3,) on support
#   build["contact_point_local"]                                       (3,) on moving body
#   build["shape"]                                                     "box" | "sphere"
#   build["duration"]                                                  seconds to simulate
#   build["forcing"]  optional callable(model, data) -> None, applied each substep
# --------------------------------------------------------------------------------------

# Shared geometry constants (kept consistent so contact_point_local etc. line up).
_BOX_HALF = 0.10        # box half-extent (m); bottom-center material point is [0,0,-_BOX_HALF]
_BALL_R = 0.05          # sphere radius (m)

# The four BOTTOM-FACE corners of a (_BOX_HALF)-cube, in the box body-local frame. These
# are the K candidate point-contacts that contact-implicit inverse dynamics (THEORY.md s.8,
# the north star) reasons over: each is a place the box can push against a plane, and the
# Signorini complementarity (s.2) decides which actually carry force. The same four corners
# are also the s.7 statically-indeterminate load-split unknowns (see `_RIG_CORNERS_LOCAL`).
_BOX_BOTTOM_CORNERS_LOCAL = np.array(
    [
        [-_BOX_HALF, -_BOX_HALF, -_BOX_HALF],
        [+_BOX_HALF, -_BOX_HALF, -_BOX_HALF],
        [-_BOX_HALF, +_BOX_HALF, -_BOX_HALF],
        [+_BOX_HALF, +_BOX_HALF, -_BOX_HALF],
    ],
    dtype=float,
)

# Indeterminate-rig compliance (THEORY.md s.7). We deliberately make the contact a
# CONSTANT-impedance linear spring: MuJoCo's `solref="-k -b"` selects a direct
# stiffness/damping, and a CONSTANT `solimp` (dmin == dmax) keeps the constraint
# impedance from drifting with penetration depth, so f = k_eff * penetration holds
# *exactly* and *identically* at every corner. That linearity is the whole point: it
# is the compliance that collapses the indeterminate null space (s.7), turning each
# corner's measurable squish into an individually-identifiable force gauge. The nominal
# `-k` here is not the effective stiffness (MuJoCo's impedance rescales it); we MEASURE
# the realized k_eff = f/penetration from the settled sim and report THAT as the truth.
_RIG_SOLREF = "-2000 -120"          # direct (stiffness, damping); negative => bypass time-const form
_RIG_SOLIMP = "0.2 0.2 0.001"       # constant impedance (dmin == dmax) => a clean linear spring


def _common_options() -> str:
    """MuJoCo <option> block shared by the scenarios (explicit Euler-ish defaults)."""
    return (
        '<option timestep="0.0005" gravity="0 0 -9.81" '
        'integrator="implicitfast" cone="pyramidal"/>'
    )


def _build_drop_rest() -> tuple[mujoco.MjModel, dict]:
    """Box free-falls onto a static plane and rests.

    Physics: gravity pulls the box down; at touchdown the normal velocity is arrested
    almost discontinuously (an impact, THEORY.md s.6), then the contact settles into
    sustained STATIC contact (zero relative twist, THEORY.md s.3). Exercises existence
    plus a single touchdown impact.
    """
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"
          friction="0.8 0.01 0.001" solref="0.005 1"/>
    <body name="box" pos="0 0 0.40">
      <freejoint name="boxj"/>
      <geom name="boxg" type="box" size="{_BOX_HALF} {_BOX_HALF} {_BOX_HALF}"
            density="500" friction="0.8 0.01 0.001" solref="0.005 1"/>
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
        "duration": 1.5,
        # Candidate contact corners for contact-implicit inverse dynamics (THEORY.md s.8).
        "box_corners_local": _BOX_BOTTOM_CORNERS_LOCAL,
    }
    return model, build


def _build_drop_rest_liftoff() -> tuple[mujoco.MjModel, dict]:
    """Box drops, rests, then is lifted back off via ``data.xfrc_applied``.

    Physics: as drop_rest until rest, then we apply a vertical world force larger than
    weight to the box, peeling it off the plane. This produces both guards of THEORY.md
    s.5: free->contact (gap reaches 0) and contact->free (the normal force lambda
    reaches 0). Exercises touchdown and liftoff.
    """
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"
          friction="0.8 0.01 0.001" solref="0.005 1"/>
    <body name="box" pos="0 0 0.40">
      <freejoint name="boxj"/>
      <geom name="boxg" type="box" size="{_BOX_HALF} {_BOX_HALF} {_BOX_HALF}"
            density="500" friction="0.8 0.01 0.001" solref="0.005 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    box_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    weight = float(model.body_mass[box_id]) * 9.81

    def forcing(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # After the box has had time to land and settle (~0.9 s), pull straight up with
        # ~1.8x its weight so the contact unloads to lambda=0 and the box lifts off.
        d.xfrc_applied[box_id, :] = 0.0
        if d.time > 0.9:
            d.xfrc_applied[box_id, 2] = 1.8 * weight

    build = {
        "moving_body": "box",
        "moving_geom": "boxg",
        "support_body": "world",
        "support_geom": "floor",
        "surface_point_local": np.zeros(3),
        "surface_normal_local": np.array([0.0, 0.0, 1.0]),
        "contact_point_local": np.array([0.0, 0.0, -_BOX_HALF]),
        "shape": "box",
        "duration": 1.6,
        "forcing": forcing,
        "box_corners_local": _BOX_BOTTOM_CORNERS_LOCAL,
    }
    return model, build


def _build_push_to_slide() -> tuple[mujoco.MjModel, dict]:
    """Box resting on a plane; a ramped horizontal force eventually breaks friction.

    Physics (THEORY.md s.7, the stick->slip guard): while sticking, the tangential
    force lives inside the friction cone ||f_t|| <= mu*f_n; sliding begins exactly when
    the applied force reaches the cone boundary. We ramp a +x world force from 0; below
    mu*weight the box is STATIC, above it the box SLIDES. Exercises static -> sliding.
    """
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" friction="0.5 0.005 0.0001"/>
    <body name="box" pos="0 0 {_BOX_HALF}">
      <freejoint name="boxj"/>
      <geom name="boxg" type="box" size="{_BOX_HALF} {_BOX_HALF} {_BOX_HALF}"
            density="500" friction="0.5 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    box_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    weight = float(model.body_mass[box_id]) * 9.81

    def forcing(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Settle for 0.3 s, then ramp the horizontal push toward the friction limit
        # (mu*weight, mu=0.5) and CAP it just above. Capping matters: an unbounded ramp
        # eventually flings the box off the plane (the contact would then flicker), but
        # a force only slightly above the cone boundary makes the box slide steadily
        # while staying seated -> a clean STATIC segment followed by a sustained SLIDING
        # segment (the stick->slip guard of THEORY.md s.7).
        d.xfrc_applied[box_id, :] = 0.0
        if d.time > 0.3:
            ramp = 1.5 * weight * (d.time - 0.3)
            d.xfrc_applied[box_id, 0] = min(ramp, 0.58 * weight)

    build = {
        "moving_body": "box",
        "moving_geom": "boxg",
        "support_body": "world",
        "support_geom": "floor",
        "surface_point_local": np.zeros(3),
        "surface_normal_local": np.array([0.0, 0.0, 1.0]),
        "contact_point_local": np.array([0.0, 0.0, -_BOX_HALF]),
        "shape": "box",
        # Keep the window short enough to stay in the clean static->sliding regime
        # before the box reaches a high enough speed to skip on the soft-constraint plane.
        "duration": 1.0,
        "forcing": forcing,
        "box_corners_local": _BOX_BOTTOM_CORNERS_LOCAL,
    }
    return model, build


def _build_rolling_ball() -> tuple[mujoco.MjModel, dict]:
    """Sphere given v and omega satisfying rolling-without-slip on a plane.

    Physics (THEORY.md s.3, the rolling mode): rolling is the *curved* twist subspace
    where tangential linear and tangential angular velocity are locked by v = omega x r.
    We initialize the ball so the material contact point has ~zero slip while the COM
    translates. Exercises ROLLING. (High friction + tiny rolling friction keeps it
    rolling rather than slipping or stopping immediately.)
    """
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" pos="0 0 0" friction="1.0 0.005 0.0001"/>
    <body name="ball" pos="0 0 {_BALL_R}">
      <freejoint name="ballj"/>
      <geom name="ballg" type="sphere" size="{_BALL_R}" density="800"
            friction="1.0 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    ball_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    def init(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Rolling-without-slip along +x. Let r be the vector from the COM to the bottom
        # contact point, r = (0, 0, -R). No-slip pins the material contact-point velocity
        # to zero: v_com + omega x r = 0, i.e. v_com = -(omega x r). With omega = (0, +w, 0)
        # we get omega x r = (0,+w,0) x (0,0,-R) = (-w*R, 0, 0), so v_com = (+w*R, 0, 0):
        # the ball translates toward +x while spinning about +y, and the contact point is
        # instantaneously stationary -> pure ROLLING from t=0. (Using -w instead would
        # leave the contact point moving at +2*w*R, i.e. maximal slip.)
        w = 12.0
        v = w * _BALL_R
        jadr = m.jnt_dofadr[m.body_jntadr[ball_id]]
        d.qvel[jadr + 0] = v       # linear x (world): v_com = +w*R toward +x
        d.qvel[jadr + 4] = +w      # angular y (about +y -> rolls toward +x without slip)

    build = {
        "moving_body": "ball",
        "moving_geom": "ballg",
        "support_body": "world",
        "support_geom": "floor",
        # Sphere "material point" tracked is the center; the radius is the gap/offset
        # handled by the surface. We raise the (observation-side) plane by one radius so
        # the tracked CENTER's signed distance to it reads ~0 at contact, just like a
        # box's tracked bottom-face point. This is the surface absorbing the radius
        # exactly as documented for the sphere (THEORY.md s.3 note on tracking a single
        # material point). It changes only the OBSERVABLE gap; the MuJoCo truth labels
        # come from the geom-level contact and are unaffected.
        "surface_point_local": np.array([0.0, 0.0, _BALL_R]),
        "surface_normal_local": np.array([0.0, 0.0, 1.0]),
        "contact_point_local": np.array([0.0, 0.0, 0.0]),
        "shape": "sphere",
        "duration": 1.0,
        "init": init,
    }
    return model, build


def _build_bouncing_ball() -> tuple[mujoco.MjModel, dict]:
    """Sphere with restitution dropped to bounce several times.

    Physics (THEORY.md s.6): each touchdown is a reset map v+ = -e*v- with e the
    coefficient of restitution. We get repeated impacts (force atoms) separated by
    flight (FREE). Restitution is tuned via ``solref`` (negative second entry selects a
    direct stiffness/damping; low damping ratio => bouncy). Exercises repeated impacts.
    """
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"
          solref="0.002 0.2" solimp="0.95 0.99 0.001"/>
    <body name="ball" pos="0 0 0.50">
      <freejoint name="ballj"/>
      <geom name="ballg" type="sphere" size="{_BALL_R}" density="800"
            solref="0.002 0.2" solimp="0.95 0.99 0.001"/>
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
        # Raise the observation-side plane by one radius so the tracked sphere CENTER's
        # signed distance reads ~0 at contact (see _build_rolling_ball for the rationale);
        # truth labels are geom-based and unaffected.
        "surface_point_local": np.array([0.0, 0.0, _BALL_R]),
        "surface_normal_local": np.array([0.0, 0.0, 1.0]),
        "contact_point_local": np.array([0.0, 0.0, 0.0]),
        "shape": "sphere",
        "duration": 2.5,
    }
    return model, build


def _build_moving_support() -> tuple[mujoco.MjModel, dict]:
    """Box rests on a CART that slides horizontally; the box rides along.

    Physics (THEORY.md s.1, the central principle): contact is *relative*. The cart and
    box both have large world velocity, yet the box-on-cart contact is unambiguously
    STATIC because their *relative* twist is ~0. The support is the cart body, NOT the
    world. A position actuator drives the cart along a rail (slide joint); friction
    carries the box with it. This is the moving-on-moving edge case of THEORY.md s.9.
    """
    cart_half = np.array([0.30, 0.30, 0.05])  # cart deck half-extents
    deck_top = cart_half[2]                    # top face z in cart-local frame
    box_z = 2.0 * cart_half[2] + _BOX_HALF + 0.001  # box world z resting on the deck top
    # Both the cart and the box must be TOP-LEVEL bodies (a freejoint cannot be a child).
    # The cart slides on a rail (slide joint, driven by a position actuator); the box is
    # a free body that rides on the deck purely through friction.
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" pos="0 0 0"/>
    <body name="cart" pos="0 0 {cart_half[2]}">
      <joint name="cartx" type="slide" axis="1 0 0" damping="0"/>
      <geom name="cartg" type="box" size="{cart_half[0]} {cart_half[1]} {cart_half[2]}"
            density="2000" friction="1.0 0.01 0.001"/>
    </body>
    <body name="box" pos="0 0 {box_z}">
      <freejoint name="boxj"/>
      <geom name="boxg" type="box" size="{_BOX_HALF} {_BOX_HALF} {_BOX_HALF}"
            density="400" friction="1.0 0.01 0.001"/>
    </body>
  </worldbody>
  <actuator>
    <position name="cartdrive" joint="cartx" kp="800" kv="60"/>
  </actuator>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)

    def forcing(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Drive the cart smoothly to x = 1.5 m after a short settle; the box rides along
        # on friction. The box's WORLD velocity is large, but its velocity RELATIVE to
        # the cart deck is ~0 -> STATIC support-relative contact.
        if d.time < 0.3:
            d.ctrl[0] = 0.0
        else:
            d.ctrl[0] = min(1.5, 0.8 * (d.time - 0.3))

    build = {
        "moving_body": "box",
        "moving_geom": "boxg",
        "support_body": "cart",
        "support_geom": "cartg",
        # Surface in the CART's local frame: the deck top face.
        "surface_point_local": np.array([0.0, 0.0, deck_top]),
        "surface_normal_local": np.array([0.0, 0.0, 1.0]),
        "contact_point_local": np.array([0.0, 0.0, -_BOX_HALF]),
        "shape": "box",
        "duration": 2.0,
        "forcing": forcing,
        # The deck is the (moving) support; the box's four bottom corners are the
        # candidates. Gaps/forces are computed support-relative against the deck top.
        "box_corners_local": _BOX_BOTTOM_CORNERS_LOCAL,
    }
    return model, build


# Local corner positions of the rig box's bottom face, in the box body frame. These are
# the K candidate point contacts whose individual loads s.7 says are unobservable from
# kinematics yet identifiable from per-corner penetration once compliance is known. They
# are exactly the shared box bottom-face corners (the same K=4 candidates the inverse-
# dynamics layer of s.8 reasons over), so we alias the one constant.
_RIG_CORNERS_LOCAL = _BOX_BOTTOM_CORNERS_LOCAL


def _build_indeterminate_rig() -> tuple[mujoco.MjModel, dict]:
    """A statically-indeterminate rig resting on a plane — the s.7 observability demo.

    Physics (THEORY.md s.7, the deepest result). A rigid box rests on a plane making
    K = 4 simultaneous corner contacts. The unknowns are the K vertical corner forces
    (f1..f4); static balance gives only THREE scalar equations -- one vertical force
    balance (sum fi = m g) and two moment balances (sum fi * x_i = 0, sum fi * y_i = 0).
    With K = 4 > 3 the system is *statically indeterminate*: an entire one-parameter
    family of corner-force distributions is consistent with the identical rigid-body
    equilibrium, so the load split is UNRECOVERABLE from kinematics alone -- exactly the
    s.7 theorem.

    The resolution, also from s.7: make the contacts slightly COMPLIANT. We tune
    `solref`/`solimp` so each corner is a linear spring f = k_eff * penetration with a
    constant, identical k_eff (see `_RIG_SOLREF`/`_RIG_SOLIMP`). The instant the bodies
    are compliant the indeterminate null space collapses -- each corner's force is pinned
    by its OWN measurable penetration, so the load split becomes individually
    identifiable. To make that split non-trivial (and so the demo visible), we hang a
    dense off-center lump inside the box, shoving the center of mass toward the +x/+y
    corner; that corner then both carries more load AND penetrates deeper, monotonically,
    which is the whole point.

    The box just rests (static) -- no forcing. The normal `RawScenario` fields track the
    box center against the plane as usual; the per-corner truth lives in
    ``meta["contact_points"]`` (assembled in :func:`generate`).
    """
    # Off-center dense lump: shoves the COM toward the (+x, +y) corner so the four corner
    # loads differ (a non-trivial, observable load split). The lump itself does not
    # collide (contype/conaffinity = 0); it only adds inertia/weight.
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"
          friction="0.8 0.01 0.001" solref="{_RIG_SOLREF}" solimp="{_RIG_SOLIMP}"/>
    <body name="box" pos="0 0 {_BOX_HALF + 0.02}">
      <freejoint name="boxj"/>
      <geom name="boxg" type="box" size="{_BOX_HALF} {_BOX_HALF} {_BOX_HALF}"
            density="600" friction="0.8 0.01 0.001"
            solref="{_RIG_SOLREF}" solimp="{_RIG_SOLIMP}"/>
      <geom name="lump" type="box" size="0.04 0.04 0.04" pos="0.06 0.06 0.0"
            density="4000" contype="0" conaffinity="0"/>
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
        # Tracked center-of-bottom-face point, as for the other resting box scenarios.
        "contact_point_local": np.array([0.0, 0.0, -_BOX_HALF]),
        "shape": "box",
        # Long enough to drop, dissipate the touchdown transient, and settle into a quiet
        # static multi-contact equilibrium before we read off the per-corner truth.
        "duration": 4.0,
        # Marks this builder so `generate` knows to harvest the K-corner contact_points.
        "is_indeterminate_rig": True,
        "corners_local": _RIG_CORNERS_LOCAL,
        # Also expose the unified inverse-dynamics candidate view (signed gap + per-corner
        # force + active flags) consistent with the new schema; these are the same K=4
        # corners, so meta["candidates"] aliases meta["contact_points"] downstream (s.8).
        "box_corners_local": _RIG_CORNERS_LOCAL,
    }
    return model, build


#: Registry mapping scenario name -> builder. THEORY.md s.9: "the entire edge-case
#: taxonomy on demand."
_BUILDERS = {
    "drop_rest": _build_drop_rest,
    "drop_rest_liftoff": _build_drop_rest_liftoff,
    "push_to_slide": _build_push_to_slide,
    "rolling_ball": _build_rolling_ball,
    "bouncing_ball": _build_bouncing_ball,
    "moving_support": _build_moving_support,
    "indeterminate_rig": _build_indeterminate_rig,
}

#: The available scenario names (public).
SCENARIOS: list[str] = list(_BUILDERS.keys())


# --------------------------------------------------------------------------------------
# The simulation + extraction loop
# --------------------------------------------------------------------------------------

def _classify_mode(
    in_contact: bool,
    v_normal: float,
    slip_tan: float,
    spin_normal: float,
    com_tan_speed: float,
    shape: str,
) -> str:
    """Label one frame's contact mode from its relative twist (THEORY.md section 3).

    A mode is the subspace of the 6D relative twist the motion lives in:

    * not in contact                              -> FREE.
    * |relative normal velocity| large            -> IMPACT (transient, THEORY.md s.6).
    * sphere, ~0 material-point slip but COM moves -> ROLLING (v coupled to omega).
    * tangential slip of the material point large  -> SLIDING.
    * spin about the normal dominant               -> PIVOTING.
    * otherwise                                    -> STATIC (twist ~ 0).

    ``slip_tan`` is the tangential speed of the *material contact point* (the rigorous
    rolling/sliding discriminator of s.3), while ``com_tan_speed`` is the COM tangential
    speed used only to recognize that a low-slip sphere is actually rolling (not at rest).
    """
    if not in_contact:
        return FREE
    if abs(v_normal) > _IMPACT_VN:
        return IMPACT
    if shape == "sphere" and slip_tan < _SLIP_EPS and com_tan_speed > _ROLL_VTAN:
        return ROLLING
    if slip_tan > _SLIP_EPS:
        return SLIDING
    if spin_normal > _SPIN_EPS:
        return PIVOTING
    return STATIC


def _simulate(model: mujoco.MjModel, build: dict, hz: float) -> dict:
    """Run the headless sim, subsampling to ``hz``, and return clean recorded arrays.

    Returns a dict of stacked per-recorded-frame arrays (no noise yet): times, moving &
    support poses, in_contact, mode, normal_force, penetration.
    """
    data = mujoco.MjData(model)

    moving_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, build["moving_body"])
    moving_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, build["moving_geom"])
    support_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, build["support_geom"])
    if build["support_body"] == "world":
        support_body = 0  # MuJoCo's world body id is 0; its pose is identity for all t.
    else:
        support_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, build["support_body"])

    # Optional one-time initialization (e.g. rolling-ball initial velocities).
    init = build.get("init")
    forcing = build.get("forcing")

    mujoco.mj_forward(model, data)
    if init is not None:
        init(model, data)
        mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    sub = max(1, int(round((1.0 / hz) / dt)))  # physics substeps per recorded frame
    n_frames = int(round(build["duration"] * hz))

    shape = build["shape"]

    # --- multi-contact (indeterminate rig) per-corner harvesting setup (THEORY.md s.7) ---
    # When this scenario is the statically-indeterminate rig, we additionally record, for
    # each of the K box corners, its penetration and normal force every frame. We match a
    # contact to a corner by transforming contact.pos into the BOX local frame and taking
    # the nearest listed corner (in the tangent plane). Defaults to 0 on frames where a
    # corner happens to carry no contact, so the arrays are always dense (K, T).
    rig_corners_local = (
        np.asarray(build["corners_local"], dtype=float)
        if build.get("is_indeterminate_rig")
        else None
    )
    n_corners = 0 if rig_corners_local is None else int(rig_corners_local.shape[0])
    corner_pen: list[np.ndarray] = []   # per frame: (K,) penetration
    corner_fn: list[np.ndarray] = []    # per frame: (K,) normal force

    # --- contact-implicit inverse-dynamics candidate harvesting (THEORY.md s.8 north star) ---
    # For a single rigid box contacting a plane we expose, per box-bottom corner, the data the
    # inverse-dynamics layer needs to recover the per-corner forces and check them against the
    # MuJoCo truth: the SIGNED support-relative gap (s.1/s.2 -- distinct from the rig's >=0
    # penetration), the TRUE per-corner normal force (matched to the nearest corner), and an
    # ACTIVE flag (Signorini: a corner carries force only where its gap is closed, s.2). Gaps
    # are computed support-relative against the (possibly moving) plane via `plane_gap`, exactly
    # the gap channel the detector itself sees (geometry.observe). The contact normal is the
    # plane outward normal carried into the box-local frame each frame (documented choice).
    box_corners_local = (
        np.asarray(build["box_corners_local"], dtype=float)
        if build.get("box_corners_local") is not None
        else None
    )
    n_box_corners = 0 if box_corners_local is None else int(box_corners_local.shape[0])
    surf_pt_local = np.asarray(build["surface_point_local"], dtype=float)
    surf_n_local = np.asarray(build["surface_normal_local"], dtype=float)
    surf_n_local = surf_n_local / np.linalg.norm(surf_n_local)
    cand_gap: list[np.ndarray] = []     # per frame: (K,) SIGNED gap (m)
    cand_fn: list[np.ndarray] = []      # per frame: (K,) true normal force (N)
    cand_active: list[np.ndarray] = []  # per frame: (K,) bool true contact
    cand_normal_local: list[np.ndarray] = []  # per frame: (K,3) plane normal in box-local frame
    # Active iff a corner carries more than this fraction of the box weight (a clean Signorini
    # threshold robust to tiny solver residual forces on barely-touching corners).
    _cand_active_floor = 1e-3 * float(model.body_mass[moving_body]) * 9.81

    t_rec: list[float] = []
    mov_pos: list[np.ndarray] = []
    mov_quat: list[np.ndarray] = []
    sup_pos: list[np.ndarray] = []
    sup_quat: list[np.ndarray] = []
    in_contact: list[bool] = []
    mode: list[str] = []
    normal_force: list[float] = []
    penetration: list[float] = []

    buf6 = np.zeros(6)

    for _ in range(n_frames):
        # --- advance the physics `sub` substeps, applying any forcing each substep ---
        for _ in range(sub):
            if forcing is not None:
                forcing(model, data)
            mujoco.mj_step(model, data)

        # --- record CLEAN poses (truth) for the moving and support bodies ---
        t_rec.append(float(data.time))
        mov_pos.append(data.xpos[moving_body].copy())
        mov_quat.append(data.xquat[moving_body].copy())
        sup_pos.append(data.xpos[support_body].copy())
        sup_quat.append(data.xquat[support_body].copy())

        # --- scan the true active set for the moving<->support contact (THEORY.md s.9) ---
        found = False
        f_n = 0.0
        pen = 0.0
        m_mode = FREE
        # Per-frame per-corner accumulators (only populated for the indeterminate rig).
        frame_corner_pen = np.zeros(n_corners)
        frame_corner_fn = np.zeros(n_corners)
        # Per-frame per-candidate-corner true normal force (the inverse-dynamics view).
        frame_cand_fn = np.zeros(n_box_corners)
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            is_pair = (g1 == moving_geom and g2 == support_geom) or (
                g1 == support_geom and g2 == moving_geom
            )
            if not is_pair:
                continue
            found = True
            c_pen = max(0.0, -float(c.dist))
            pen = max(pen, c_pen)
            mujoco.mj_contactForce(model, data, ci, buf6)
            c_fn = float(buf6[0])  # normal force in the contact frame
            f_n += c_fn

            # Attribute this point contact to its box corner (indeterminate rig only):
            # transform contact.pos into the box local frame and pick the nearest listed
            # corner in the tangent (x,y) plane (THEORY.md s.7: the per-corner penetration
            # is the gauge that pins the per-corner force the kinematics cannot give us).
            if rig_corners_local is not None:
                p_local = _world_to_body_local(
                    np.array(c.pos), data.xpos[moving_body], data.xquat[moving_body]
                )
                d2 = np.sum((rig_corners_local[:, :2] - p_local[:2]) ** 2, axis=1)
                k = int(np.argmin(d2))
                # If two sub-contacts land on the same corner, accumulate (sum the force,
                # keep the deepest penetration) so a corner's total load is well-defined.
                frame_corner_fn[k] += c_fn
                frame_corner_pen[k] = max(frame_corner_pen[k], c_pen)

            # Attribute this point contact's TRUE normal force to its nearest candidate
            # corner for the inverse-dynamics view (THEORY.md s.8): same box-local nearest-
            # corner match as above, but kept independent so it runs for every box-on-plane
            # scenario (not just the rig). Sub-contacts on the same corner accumulate.
            if box_corners_local is not None:
                p_local_b = _world_to_body_local(
                    np.array(c.pos), data.xpos[moving_body], data.xquat[moving_body]
                )
                d2b = np.sum((box_corners_local[:, :2] - p_local_b[:2]) ** 2, axis=1)
                frame_cand_fn[int(np.argmin(d2b))] += c_fn

            # Contact frame: rows of c.frame are (normal, tangent1, tangent2). The
            # relative twist is measured in THIS support-attached frame (types.py /
            # THEORY.md s.1).
            cframe = np.array(c.frame).reshape(3, 3)
            n_hat = cframe[0]
            t1, t2 = cframe[1], cframe[2]
            cpos = np.array(c.pos)

            # Relative twist of the moving body's material point w.r.t. the support's.
            om_m, vlin_m = _object_twist_world(model, data, moving_body)
            vp_m = _material_point_velocity(om_m, vlin_m, data.xpos[moving_body], cpos)
            if support_body == 0:
                om_s = np.zeros(3)
                vp_s = np.zeros(3)
            else:
                om_s, vlin_s = _object_twist_world(model, data, support_body)
                vp_s = _material_point_velocity(om_s, vlin_s, data.xpos[support_body], cpos)

            v_rel = vp_m - vp_s            # relative material-point velocity (world)
            om_rel = om_m - om_s           # relative angular velocity (world)

            v_normal = float(v_rel @ n_hat)
            slip_tan = float(np.hypot(v_rel @ t1, v_rel @ t2))
            spin_normal = abs(float(om_rel @ n_hat))

            # COM tangential speed (relative), to recognize a rolling sphere.
            v_com_rel = vlin_m - (np.zeros(3) if support_body == 0 else vlin_s)
            com_tan_speed = float(np.hypot(v_com_rel @ t1, v_com_rel @ t2))

            m_mode = _classify_mode(
                True, v_normal, slip_tan, spin_normal, com_tan_speed, shape
            )

        if not found:
            # No force-active contact in the solver list -- fall back to true proximity so a
            # fast roller riding at ~0 penetration is not mislabeled FREE (see _proximity_mode).
            found, m_mode = _proximity_mode(
                model, data, {moving_geom}, {support_geom}, support_body, None, shape
            )

        in_contact.append(found)
        mode.append(m_mode if found else FREE)
        normal_force.append(f_n)
        penetration.append(pen)
        if rig_corners_local is not None:
            corner_pen.append(frame_corner_pen)
            corner_fn.append(frame_corner_fn)

        # --- candidate-corner signed gap + normal in box-local frame (THEORY.md s.8) ---
        if box_corners_local is not None:
            # World position of each corner this frame: box origin + R(box) @ corner_local.
            R_box = quat_to_matrix(data.xquat[moving_body])          # (3,3) box-local -> world
            corners_world = data.xpos[moving_body] + box_corners_local @ R_box.T  # (K,3)
            # Plane carried into the world via the (possibly moving) support pose, then the
            # SIGNED gap of each corner -- exactly the support-relative gap of s.1/s.2.
            R_sup = quat_to_matrix(data.xquat[support_body])         # (3,3) support-local -> world
            plane_pt_w = data.xpos[support_body] + R_sup @ surf_pt_local   # (3,)
            normal_w = R_sup @ surf_n_local                          # (3,) world plane normal
            gaps = plane_gap(corners_world, plane_pt_w, normal_w)    # (K,)
            cand_gap.append(gaps)
            cand_fn.append(frame_cand_fn.copy())
            cand_active.append(frame_cand_fn > _cand_active_floor)
            # Document the per-frame contact normal in the BOX-local frame: R(box)^T @ n_world,
            # the same outward plane normal each corner pushes along, expressed body-locally.
            cand_normal_local.append(np.tile(R_box.T @ normal_w, (n_box_corners, 1)))

    out = {
        "t": np.asarray(t_rec, dtype=float),
        "mov_pos": np.asarray(mov_pos, dtype=float),
        "mov_quat": np.asarray(mov_quat, dtype=float),
        "sup_pos": np.asarray(sup_pos, dtype=float),
        "sup_quat": np.asarray(sup_quat, dtype=float),
        "in_contact": np.asarray(in_contact, dtype=bool),
        "mode": mode,
        "normal_force": np.asarray(normal_force, dtype=float),
        "penetration": np.asarray(penetration, dtype=float),
    }
    if rig_corners_local is not None:
        # Stack to (K, T): K corners (rows) over T recorded frames (columns), exactly the
        # shape meta["contact_points"] promises (THEORY.md s.7 observability arrays).
        out["corner_penetration"] = np.asarray(corner_pen, dtype=float).T  # (K, T)
        out["corner_normal_force"] = np.asarray(corner_fn, dtype=float).T  # (K, T)
        out["corners_local"] = rig_corners_local                          # (K, 3)
    if box_corners_local is not None:
        # Inverse-dynamics candidate arrays, all (K, T) over the recorded frames (THEORY.md
        # s.8): SIGNED gap, true per-corner normal force, and the Signorini active flag, plus
        # the static (K,3) candidate points and their (K,T,3) box-local contact normals.
        out["cand_points_local"] = box_corners_local                       # (K, 3)
        out["cand_gap"] = np.asarray(cand_gap, dtype=float).T               # (K, T)
        out["cand_normal_force"] = np.asarray(cand_fn, dtype=float).T       # (K, T)
        out["cand_active"] = np.asarray(cand_active, dtype=bool).T          # (K, T)
        # (T,K,3) -> (K,T,3): the world plane normal expressed in the box-local frame per frame.
        out["cand_normals_local"] = np.asarray(cand_normal_local, dtype=float).transpose(1, 0, 2)
    return out


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def generate(
    name: str, seed: int = 0, hz: float = 100.0, noise_m: float = 5e-4
) -> RawScenario:
    """Build, simulate, and label one scenario (THEORY.md section 9).

    Parameters
    ----------
    name:
        One of :data:`SCENARIOS`.
    seed:
        RNG seed for the additive mocap noise (reproducible labels).
    hz:
        Recording rate; the physics runs at the model timestep and is subsampled to
        the nearest multiple of ``1/hz``.
    noise_m:
        Standard deviation (m) of the i.i.d. Gaussian position noise added to the
        RECORDED moving-body positions, emulating optical mocap. THEORY.md s.4/s.9:
        the detector only ever sees this noisy "observable channel"; the truth labels
        come from the CLEAN simulator state, never from the noised poses.

    Returns
    -------
    RawScenario
        ``moving``/``support`` pose trajectories (moving positions noised), the support
        ``surface`` in the support's local frame, the tracked ``contact_point_local`` on
        the moving body, and the withheld ``truth`` labels.

    Note
    ----
    Headless physics only — no rendering. See THEORY.md s.9 on the simulate -> record
    truth -> expose only noisy poses workflow.
    """
    if name not in _BUILDERS:
        raise KeyError(f"unknown scenario {name!r}; available: {SCENARIOS}")

    model, build = _BUILDERS[name]()
    # A builder may pin its own recording rate via build["record_hz"] (the recording-cadence
    # override): some impact-regime demos contain energetic, brief touchdowns that are
    # sub-frame at the default 100 Hz and only become observable -- the named IMPACT appearing
    # in the truth -- when sampled faster. This is per-scenario and never narrows the caller's
    # request (we take the MAX), so a caller asking for a higher hz still gets it.
    rec_hz = max(float(hz), float(build.get("record_hz") or hz))
    rec = _simulate(model, build, rec_hz)

    # --- emulate mocap: additive Gaussian noise on the RECORDED moving positions only.
    # (THEORY.md s.4: we observe noisy marker positions; velocities come from
    #  differentiating these, which is why the detector must reason probabilistically.)
    rng = np.random.default_rng(seed)
    noisy_mov_pos = rec["mov_pos"] + rng.normal(0.0, noise_m, size=rec["mov_pos"].shape)

    moving = PoseTrajectory(
        t=rec["t"], position=noisy_mov_pos, quat=rec["mov_quat"]
    )
    support = PoseTrajectory(
        t=rec["t"], position=rec["sup_pos"], quat=rec["sup_quat"]
    )
    surface = SupportSurface(
        point=np.asarray(build["surface_point_local"], dtype=float),
        normal=np.asarray(build["surface_normal_local"], dtype=float),
    )
    truth = GroundTruth(
        t=rec["t"],
        in_contact=rec["in_contact"],
        mode=rec["mode"],
        normal_force=rec["normal_force"],
        penetration=rec["penetration"],
    )

    meta = {
        "scenario": name,
        "seed": seed,
        "hz": rec_hz,
        "noise_m": noise_m,
        "shape": build["shape"],
        "timestep": float(model.opt.timestep),
        "moving_body": build["moving_body"],
        "support_body": build["support_body"],
        "mode_thresholds": {
            "slip_eps": _SLIP_EPS,
            "spin_eps": _SPIN_EPS,
            "impact_vn": _IMPACT_VN,
            "roll_vtan": _ROLL_VTAN,
        },
        "note": (
            "MuJoCo truth is truth for MuJoCo's soft-constraint contact model "
            "(THEORY.md s.9). Truth labels come from clean sim state; only moving "
            "positions are noised to emulate mocap."
        ),
    }

    # --- contact-implicit inverse-dynamics metadata (THEORY.md s.8, the north star) ---
    # For a single rigid box contacting a plane we expose the inertial mass matrix and the
    # candidate point-contacts (the box corners) so dynamics_id can recover the per-corner
    # forces under Newton-Euler + Signorini (s.2) + the friction cone (s.7) and compare to
    # the MuJoCo truth. Only emitted for the box-on-plane scenarios that carry corner data.
    if build.get("box_corners_local") is not None:
        moving_body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, build["moving_body"])
        meta["inertial"] = _body_inertial(model, moving_body_id)
        meta["gravity"] = 9.81  # m/s^2 (matches `_common_options` gravity = -9.81 z)
        meta["candidates"] = {
            # K box-bottom corners in the box body-local frame -- the contact candidates.
            "points_local": rec["cand_points_local"],     # (K, 3)
            # The plane outward normal expressed in the box-local frame, PER FRAME (our
            # documented choice; for the static-floor scenarios this is ~+z constant, for
            # moving_support it tracks the (level) cart deck). Shape (K, T, 3).
            "normals_local": rec["cand_normals_local"],   # (K, T, 3)
            # SIGNED support-relative distance of each corner to the plane (m); >0 separation,
            # <0 penetration (THEORY.md s.1/s.2) -- distinct from the rig's >=0 penetration.
            "gap": rec["cand_gap"],                        # (K, T)
            # TRUE per-corner normal force from MuJoCo, each sub-contact matched to its nearest
            # corner (the truth the recovered forces are scored against). Summed over corners
            # equals the box weight m*g at rest.
            "normal_force": rec["cand_normal_force"],      # (K, T)
            # Signorini active set: a corner carries force only where its gap is closed (s.2).
            "active": rec["cand_active"],                  # (K, T) bool
        }

    # --- statically-indeterminate rig: expose the per-corner observability arrays (s.7) ---
    if build.get("is_indeterminate_rig"):
        pen_kt = rec["corner_penetration"]   # (K, T) penetration per corner per frame
        fn_kt = rec["corner_normal_force"]   # (K, T) normal force per corner per frame
        corners = rec["corners_local"]       # (K, 3) corner positions in box-local frame

        # Effective contact stiffness k_eff (N/m). Per s.7 the penetration depth is a
        # calibrated force gauge f = k * delta; we IDENTIFY k by the least-squares slope
        # of force vs. penetration, i.e. exactly the "trace the penetration-force slope"
        # persistent-excitation reading of s.7. We fit over the QUIET, SETTLED tail (the
        # last quarter of the run) where the velocity-dependent damper term b*delta_dot
        # has died out, so the measured force is the pure spring f = k*delta -- otherwise
        # the touchdown transient (where f leads delta) corrupts the slope. The
        # constant-impedance contact (`_RIG_SOLIMP` with dmin == dmax) makes this slope a
        # single number shared by all corners; we report it as meta["stiffness"].
        settle0 = (3 * pen_kt.shape[1]) // 4
        d = pen_kt[:, settle0:]
        f = fn_kt[:, settle0:]
        loaded = d > 1e-9
        if np.any(loaded):
            stiffness = float(np.dot(d[loaded], f[loaded]) / np.dot(d[loaded], d[loaded]))
        else:
            stiffness = float("nan")

        meta["contact_points"] = {
            "penetration": pen_kt,          # (K, T) >= 0
            "normal_force": fn_kt,          # (K, T) >= 0
            "corners_local": corners,       # (K, 3) which corner each row is
            "n_corners": int(pen_kt.shape[0]),
        }
        meta["stiffness"] = stiffness       # effective contact stiffness k_eff (N/m)
        meta["indeterminacy"] = (
            "K=%d vertical corner-force unknowns vs 3 static balance equations "
            "(sum F_z, sum M_x, sum M_y) => statically indeterminate; load split "
            "unobservable from kinematics, identifiable only via per-corner penetration "
            "under known compliance (THEORY.md s.7)." % int(pen_kt.shape[0])
        )

    # --- optional per-scenario contact-geometry resolver (DESIGN.md III.1 / PHASE 2) ---
    # A builder may attach a "geometry" spec; we construct the matching resolver against THIS
    # scenario's `surface` so the plane lines up exactly. With no spec the resolver is None, so
    # `observe` wraps `surface` + `contact_point_local` in a FlatRegion -- today's bit-identical
    # path. Only the tumbling box ships a spec: a BoxPlane whose 8 corners give the migrating
    # nearest-corner contact, so the per-bounce IMPACT fires (the fixed bottom-face point, ~225
    # mm up when a corner strikes, never reads gap ~0 at the bounce).
    raw_geometry = None
    geom_spec = build.get("geometry")
    if geom_spec is not None:
        kind = geom_spec.get("kind")
        if kind == "box_plane":
            from contact.geometry_resolvers import BoxPlane

            raw_geometry = BoxPlane(
                np.asarray(geom_spec["half_extents"], dtype=float), surface
            )
        else:
            raise ValueError(
                f"unknown geometry spec kind {kind!r} in scenario {name!r}"
            )

    return RawScenario(
        name=name,
        moving=moving,
        support=support,
        surface=surface,
        contact_point_local=np.asarray(build["contact_point_local"], dtype=float),
        truth=truth,
        meta=meta,
        geometry=raw_geometry,
    )


# ======================================================================================
# Multi-body SCENES (THEORY.md section 8: the contact graph + active-set structure).
#
# Everything above produces a single body-PAIR (`RawScenario`, one `GroundTruth`). The
# theory's final object (s.8) is richer: the hidden thing we infer is a *structure* over
# a CONTACT GRAPH whose nodes are bodies and whose edges are candidate body-pair contacts
# (person<->deck, deck<->ground, hand<->rail), and we want a posterior over *which* edges
# are active. A scene therefore carries SEVERAL bodies sharing one time base and a LIST of
# candidate edges, with a separate per-edge `GroundTruth` so the graph layer can be scored
# edge by edge and as a joint active set.
#
# These generators reuse the same headless simulate->extract path as the single-pair ones
# (THEORY.md s.9): we step the physics, record CLEAN poses for every body, scan the true
# active set per edge (the relevant geom pair[s]), and label each frame's per-edge
# existence / penetration / normal force / mode exactly as `_simulate` does for one pair.
# Only the moving body's recorded positions are noised downstream-free here (the scene
# carries clean truth; a separate observable-channel noising is the integrator's job, as
# for `generate`). The contracts (ContactEdge / MultiBodyScene) live in contact.types.
#
# Tractability note (THEORY.md s.8): exact joint inference enumerates the 2^E active sets;
# these scenes keep E <= 2, so that is trivially exact. Large E would need RJMCMC/particle
# methods -- not these generators' concern, but stated for honesty.
# ======================================================================================

# Shared skateboard geometry (kept as named constants so the surface point/normal, the
# tracked material points, and the wheel placement all line up). The deck is a thin box;
# the person is a tall box PROXY (we document below that this is a contact-detection test,
# NOT human dynamics). Four hinge-jointed cylinder wheels make the board<->ground contact
# genuinely ROLLING (the wheels spin freely about the board's lateral axis), so the scene
# exercises s.3's rolling mode on one edge while the other edge is static.
_DECK_HALF = np.array([0.35, 0.12, 0.015])  # deck half-extents (m)
_DECK_LIFT = 0.045                           # deck center height above the board origin (m)
_WHEEL_R = 0.04                              # wheel radius (m)
_WHEEL_HALF_W = 0.02                         # wheel half-width (m)
_WHEEL_DX = 0.25                             # wheel longitudinal offset from board origin (m)
_WHEEL_DY = 0.10                             # wheel lateral offset from board origin (m)
_PERSON_HALF = np.array([0.08, 0.08, 0.28])  # person-proxy box half-extents (m)


def _build_person_on_skateboard() -> tuple[mujoco.MjModel, dict]:
    """A person PROXY rides a wheeled board that rolls across the ground (THEORY.md s.1/s.8).

    Bodies: ``person`` (a tall BOX PROXY -- this is a contact-detection test, NOT human
    dynamics; the box stands in for a foot/leg planted on the deck) and ``board`` (a thin
    deck on four hinge-jointed cylinder WHEELS). The ground is the world (an identity pose
    for all t). The board is launched with a !actuated initial horizontal velocity along
    +x; the freely-spinning wheels make it ROLL, and friction carries the person along.

    Two candidate edges (THEORY.md s.8 contact graph):

    * ``person_board`` : moving = person, support = board, surface = the deck TOP face in
      the BOARD-local frame. THE KEY POINT (THEORY.md s.1): this is a STATIC contact even
      though both the person and the board have a large WORLD velocity -- their *relative*
      twist is ~0. A world-frame "speed" test would wrongly call it "moving, not in
      contact"; the support-relative frame is the whole payoff.
    * ``board_ground`` : moving = board, support = world, surface = the z = 0 ground plane.
      The board touches the ground through its freely-spinning WHEELS, but the detector
      tracks a BOARD-fixed point (the board origin projected to the ground -- the board is
      the only body recorded for this edge). A board-fixed point does NOT spin with the
      wheels; it translates with the deck. So this edge is observed as SLIDING (the tracked
      point slips over the ground at the board's travel speed), and its truth mode is
      classified from the board too (``truth_mode_body="board"``) so truth and observation
      agree. The wheels' instantaneous near-zero slip (true ROLLING of the wheel material
      point) is a separate, un-tracked fact -- THEORY.md s.3: rolling vs sliding depends on
      which material point you follow.

    Per-edge truth is harvested from the relevant geom pairs: person_board from
    person-geom <-> deck-geom contacts; board_ground from wheel-geom <-> floor contacts
    (aggregated across the four wheels), with its MODE classified from the board-fixed
    tracked point (see the edge's ``truth_mode_body``).
    """
    dxh, dyh, dzh = _DECK_HALF
    # Board origin height so the wheels (radius _WHEEL_R, centered at the board origin
    # plane z=0) just touch the ground: board origin sits one wheel-radius above z=0.
    board_z = _WHEEL_R
    deck_top_local = _DECK_LIFT + dzh                  # deck top face z in board-local frame
    # Person rests on the deck top: world z of the person origin = board_z + deck_top + person_half.
    person_z = board_z + deck_top_local + _PERSON_HALF[2] + 0.001

    def _wheel(name: str, x: float, y: float) -> str:
        # A child body with a single hinge about the board's lateral (y) axis: the wheel
        # spins freely so the board<->ground contact ROLLS rather than slides. The cylinder
        # is rotated (euler 90 about x) so its circular face axis aligns with y.
        return (
            f'<body name="{name}" pos="{x} {y} 0">'
            f'  <joint name="{name}_j" type="hinge" axis="0 1 0" damping="0.0001"/>'
            f'  <geom name="{name}_g" type="cylinder" size="{_WHEEL_R} {_WHEEL_HALF_W}" '
            f'        euler="90 0 0" density="700" friction="1.0 0.01 0.001"/>'
            f'</body>'
        )

    wheels = "\n      ".join(
        [
            _wheel("w_fl", +_WHEEL_DX, +_WHEEL_DY),
            _wheel("w_fr", +_WHEEL_DX, -_WHEEL_DY),
            _wheel("w_bl", -_WHEEL_DX, +_WHEEL_DY),
            _wheel("w_br", -_WHEEL_DX, -_WHEEL_DY),
        ]
    )
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="80 80 0.1" pos="0 0 0" friction="1.0 0.01 0.001"/>
    <body name="board" pos="0 0 {board_z}">
      <freejoint name="boardj"/>
      <geom name="deckg" type="box" size="{dxh} {dyh} {dzh}" pos="0 0 {_DECK_LIFT}"
            density="300" friction="1.5 0.01 0.001"/>
      {wheels}
    </body>
    <body name="person" pos="0 0 {person_z}">
      <freejoint name="personj"/>
      <geom name="persong" type="box"
            size="{_PERSON_HALF[0]} {_PERSON_HALF[1]} {_PERSON_HALF[2]}"
            density="250" friction="1.5 0.01 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    board_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "board")
    person_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "person")

    # Settle phase length (seconds) before the launch: let the person seat on the deck and
    # the wheels seat on the ground so the launch starts from a clean static equilibrium.
    settle = 0.30
    launch_v = 1.2  # m/s, !actuated initial horizontal velocity given to BOTH bodies.

    def init_after_settle(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Give the board AND the person the same +x world velocity at launch. We hand the
        # person the same velocity (rather than relying on friction to drag it up to speed)
        # so the person_board contact stays cleanly STATIC from the launch instant -- a slow
        # friction spin-up would otherwise read as a transient SLIDING smear at the start.
        bj = m.jnt_dofadr[m.body_jntadr[board_id]]
        pj = m.jnt_dofadr[m.body_jntadr[person_id]]
        d.qvel[bj + 0] = launch_v
        d.qvel[pj + 0] = launch_v

    build = {
        "bodies": ["board", "person"],   # recorded pose trajectories (world is implicit)
        "settle": settle,
        "launch": init_after_settle,     # applied once, right after the settle phase
        "duration": 2.0,
        "edges": [
            {
                "edge_id": "person_board",
                "moving_body": "person",
                "support_body": "board",
                "moving_geoms": ["persong"],
                "support_geoms": ["deckg"],
                # Surface = deck TOP face, in the BOARD-local frame (s.8: the surface is
                # carried by the support body, which itself moves).
                "surface_point_local": np.array([0.0, 0.0, deck_top_local]),
                "surface_normal_local": np.array([0.0, 0.0, 1.0]),
                # Tracked material point on the person: its bottom-face center.
                "contact_point_local": np.array([0.0, 0.0, -_PERSON_HALF[2]]),
                "shape": "box",          # mode hint: a box rider -> static/sliding, never rolling
            },
            {
                "edge_id": "board_ground",
                "moving_body": "board",
                "support_body": "world",
                # The board touches the ground through its WHEELS, so the relevant geom
                # pairs are wheel-geoms <-> floor (aggregated across the four wheels).
                "moving_geoms": ["w_fl_g", "w_fr_g", "w_bl_g", "w_br_g"],
                "support_geoms": ["floor"],
                # Ground plane z = 0 in the world frame.
                "surface_point_local": np.array([0.0, 0.0, 0.0]),
                "surface_normal_local": np.array([0.0, 0.0, 1.0]),
                # Tracked material point on the board: its origin projected to the ground
                # (the board origin sits one wheel-radius up, so the contact is one radius
                # below it).
                "contact_point_local": np.array([0.0, 0.0, -board_z]),
                # The DETECTOR observes this edge from the recorded BOARD pose (the only body
                # recorded for this edge), tracking the board-fixed point above. A board-fixed
                # point does not spin with the wheels -- it translates with the deck at the
                # board's travel speed -- so what the detector sees is SLIDING, not rolling
                # (THEORY.md s.3: rolling vs sliding is a property of the tracked material
                # point). We therefore (a) classify the truth mode from the BOARD body too
                # (truth_mode_body), so truth and observation describe the same point and
                # agree, and (b) hint shape="box" so the truth mode is classified as
                # static/sliding (the wheels' rolling is a separate, un-tracked fact).
                "truth_mode_body": "board",
                "shape": "box",
            },
        ],
    }
    return model, build


def _build_box_on_two_blocks() -> tuple[mujoco.MjModel, dict]:
    """A box bridges two separated blocks; one support is removed mid-run (THEORY.md s.8).

    Bodies: ``box`` (a long thin plank), ``blockL`` (a fixed wide support), ``blockR`` (a
    support on a vertical slide joint, driven DOWN partway through). The box's center of
    mass is shoved toward the LEFT block by an off-center dense lump (a non-colliding
    inertial lump, contype/conaffinity = 0), so blockL alone fully supports it.

    Two candidate edges:

    * ``box_blockL`` : box <-> blockL -- active for the WHOLE run.
    * ``box_blockR`` : box <-> blockR -- active at first, then DEACTIVATES when blockR is
      lowered out of reach.

    This is the changing-active-set test of THEORY.md s.8: the true active structure is
    ``{box_blockL, box_blockR}`` early and ``{box_blockL}`` after blockR drops. The box
    never tips (blockL is wide enough to hold it level with the CoM over it), so the
    box_blockL edge is a clean sustained STATIC contact throughout while box_blockR's
    existence flips off -- exactly the structure-inference signal the graph layer must
    recover. Both contacts are STATIC (no relative motion) when active; the interesting
    variable is the *active set*, not the per-edge mode.
    """
    box_half = np.array([0.32, 0.10, 0.025])  # plank half-extents (m)
    block_half = np.array([0.10, 0.12, 0.10])  # block half-extents (m) (blockL wide)
    blockR_half = np.array([0.06, 0.12, 0.10])
    block_top = 2.0 * block_half[2]            # block top face world z (block origin at z=block_half[2])
    box_z = block_top + box_half[2]            # box rests level on the block tops
    blockL_x = -0.18
    blockR_x = +0.22
    box_x = -0.02                              # box centered slightly toward blockL
    # blockR lowers by this much after the drop time -> the box loses contact with it.
    blockR_drop = 0.30
    drop_time = 1.5
    # blockR rests on a slide joint and droops slightly under its own weight + the box's
    # light load before being held by the servo; raise its rest height by this much so the
    # pre-drop contact is SNUG (a real ~12 N resting touch, gap ~ 0) rather than a few-mm
    # standoff. Without this the box never actually rests on blockR and the truth labels it
    # never-in-contact, defeating the active-set-change test.
    blockR_lift = 0.006

    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <!-- Collision groups: floor + blockL + box live in group bit 1; blockR lives in
         bit 2 ONLY (so blockR does NOT collide with the floor). The box carries both
         bits (3 = 1|2) so it rests on blockL (bit 1) AND blockR (bit 2). This is the
         crux of making the active-set change OBSERVABLE: blockR's geom bottom is at
         world z=0 (origin z=block_half, half-height=block_half), so if it collided
         with the floor it could only descend ~contact-compliance (sub-mm) and the
         scene's advertised both-supports to blockL-only change would have NO
         kinematic trace (force drops but the bodies never separate -- the s.7
         observability trap). With blockR free of the floor it drops the full commanded
         distance into the well below, opening a ~0.30 m gap the detector can see. -->
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0" friction="1.0 0.01 0.001"
          contype="1" conaffinity="1"/>
    <body name="blockL" pos="{blockL_x} 0 {block_half[2]}">
      <geom name="blockLg" type="box" size="{block_half[0]} {block_half[1]} {block_half[2]}"
            density="4000" friction="1.0 0.01 0.001" contype="1" conaffinity="1"/>
    </body>
    <body name="blockR" pos="{blockR_x} 0 {block_half[2] + blockR_lift}">
      <joint name="jR" type="slide" axis="0 0 1" damping="40"/>
      <geom name="blockRg" type="box" size="{blockR_half[0]} {blockR_half[1]} {blockR_half[2]}"
            density="4000" friction="1.0 0.01 0.001" contype="2" conaffinity="2"/>
    </body>
    <body name="box" pos="{box_x} 0 {box_z}">
      <freejoint name="boxj"/>
      <geom name="boxg" type="box" size="{box_half[0]} {box_half[1]} {box_half[2]}"
            density="300" friction="1.0 0.01 0.001" contype="3" conaffinity="3"/>
      <geom name="lump" type="box" size="0.06 0.06 0.02" pos="-0.16 0 0.0"
            density="9000" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="driveR" joint="jR" kp="40000" kv="600"/>
  </actuator>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)

    def forcing(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Hold blockR at its rest height until `drop_time`, then command it DOWN by
        # `blockR_drop` so the box separates from it (the box stays seated on the wide
        # blockL). This deactivates the box_blockR edge mid-run -> a changing active set.
        d.ctrl[0] = 0.0 if d.time < drop_time else -blockR_drop

    # Surfaces are the block TOP faces, in each block's local frame (block origin is at the
    # block center, so the top face is at +block_half[2]). The tracked material point on the
    # box for each edge is the box bottom-face point above that block (in box-local x).
    surf_L = np.array([0.0, 0.0, block_half[2]])
    surf_R = np.array([0.0, 0.0, blockR_half[2]])
    cp_L = np.array([blockL_x - box_x, 0.0, -box_half[2]])  # bottom point over blockL
    cp_R = np.array([blockR_x - box_x, 0.0, -box_half[2]])  # bottom point over blockR

    build = {
        "bodies": ["box", "blockL", "blockR"],
        # Let the box seat onto the (snug, ~12 N) blockR touch and onto blockL before
        # recording, so the recorded window opens from a clean static equilibrium instead
        # of a 2-frame settling bounce in the truth labels. The clock is reset to 0 after
        # settle, so `drop_time` below is measured from the start of the recorded window.
        "settle": 0.4,
        "duration": 3.0,
        "forcing": forcing,
        "edges": [
            {
                "edge_id": "box_blockL",
                "moving_body": "box",
                "support_body": "blockL",
                "moving_geoms": ["boxg"],
                "support_geoms": ["blockLg"],
                "surface_point_local": surf_L,
                "surface_normal_local": np.array([0.0, 0.0, 1.0]),
                "contact_point_local": cp_L,
                "shape": "box",
            },
            {
                "edge_id": "box_blockR",
                "moving_body": "box",
                "support_body": "blockR",
                "moving_geoms": ["boxg"],
                "support_geoms": ["blockRg"],
                "surface_point_local": surf_R,
                "surface_normal_local": np.array([0.0, 0.0, 1.0]),
                "contact_point_local": cp_R,
                "shape": "box",
            },
        ],
        "meta": {
            "active_set_change": (
                "{box_blockL, box_blockR} -> {box_blockL} at t=%.2fs (blockR lowered "
                "%.2fm)." % (drop_time, blockR_drop)
            ),
        },
    }
    return model, build


#: Registry mapping scene name -> builder (THEORY.md s.8 multi-body contact graphs).
_SCENE_BUILDERS = {
    "person_on_skateboard": _build_person_on_skateboard,
    "box_on_two_blocks": _build_box_on_two_blocks,
}

#: The available multi-body scene names (public).
SCENES: list[str] = list(_SCENE_BUILDERS.keys())


def _geom_ids(model: mujoco.MjModel, names: list[str]) -> set[int]:
    """Resolve a list of geom names to a set of integer ids."""
    return {_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names}


#: Truth-contact proximity threshold (m). MuJoCo's force-active contact list (``data.contact``)
#: only includes contacts the solver is pushing on -- with the default zero margin, that means
#: only *penetrating* pairs. A fast roller/slider rides at ~0 penetration, so the solver list
#: intermittently drops it and the GROUND TRUTH would flicker FREE even though the bodies are
#: physically touching. We treat a true closest-approach within this distance as contact.
_PROX_THRESH = 0.0015


def _proximity_mode(model, data, moving_geoms, support_geoms, support_body_id, mode_body_id,
                    shape, threshold=_PROX_THRESH):
    """Fallback truth contact when the solver lists none: use the actual geom distance.

    Returns ``(in_contact, mode)``. Computes the closest approach between the moving and
    support geoms (``mj_geomDistance``); within ``threshold`` it counts as contact and
    classifies the mode from the closest-point geometry, exactly as the active-set scan does.
    """
    fromto = np.zeros(6)
    best = None
    for mg in moving_geoms:
        for sg in support_geoms:
            d = float(mujoco.mj_geomDistance(model, data, int(mg), int(sg), 2.0 * threshold, fromto))
            if d < threshold and (best is None or d < best[0]):
                best = (d, int(mg), fromto.copy())
    if best is None:
        return False, FREE
    _d, mg, ft = best
    p1, p2 = ft[:3], ft[3:]
    n_hat = p2 - p1
    nn = float(np.linalg.norm(n_hat))
    n_hat = n_hat / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
    cpos = 0.5 * (p1 + p2)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(n_hat, ref)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(n_hat, t1)
    mov_id = int(model.geom_bodyid[mg]) if mode_body_id is None else mode_body_id
    om_m, vlin_m = _object_twist_world(model, data, mov_id)
    vp_m = _material_point_velocity(om_m, vlin_m, data.xpos[mov_id], cpos)
    if support_body_id == 0:
        om_s = np.zeros(3)
        vp_s = np.zeros(3)
        vlin_s = np.zeros(3)
    else:
        om_s, vlin_s = _object_twist_world(model, data, support_body_id)
        vp_s = _material_point_velocity(om_s, vlin_s, data.xpos[support_body_id], cpos)
    v_rel = vp_m - vp_s
    om_rel = om_m - om_s
    v_normal = float(v_rel @ n_hat)
    slip_tan = float(np.hypot(v_rel @ t1, v_rel @ t2))
    spin_normal = abs(float(om_rel @ n_hat))
    v_com_rel = vlin_m - (np.zeros(3) if support_body_id == 0 else vlin_s)
    com_tan = float(np.hypot(v_com_rel @ t1, v_com_rel @ t2))
    return True, _classify_mode(True, v_normal, slip_tan, spin_normal, com_tan, shape)


def _edge_frame_truth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    support_body_id: int,
    moving_geoms: set[int],
    support_geoms: set[int],
    shape: str,
    buf6: np.ndarray,
    mode_body_id: int | None = None,
) -> tuple[bool, float, float, str]:
    """Extract one edge's per-frame truth by scanning the active set (THEORY.md s.9).

    Mirrors the single-pair scan inside `_simulate`, but matched against a SET of moving
    geoms vs a SET of support geoms (so e.g. the board<->ground edge aggregates all four
    wheel-floor contacts). Returns ``(in_contact, normal_force, penetration, mode)`` for
    this edge at the current ``data`` state. The mode is classified from the RELATIVE
    material-point twist in the contact frame, exactly as `_classify_mode` expects.

    Which body's material point is used for the mode classification matters (THEORY.md s.3:
    rolling vs sliding is a property of the *tracked material point*, not the contact):

    * ``mode_body_id is None`` (default): use the body that OWNS the contacting moving geom.
      For a hinge-jointed wheel this captures the wheel's own spin, so a rolling wheel reads
      as ROLLING (its material contact point is ~stationary) even though the parent body
      merely translates. This is the right choice when the detector tracks a point on the
      contacting sub-body.
    * ``mode_body_id`` given: classify from THAT body's material point instead. This is for
      edges whose *observation* tracks a point on a parent body rather than the contacting
      sub-body -- e.g. the ``board_ground`` edge tracks the board origin (a board-fixed,
      non-spinning point that genuinely SLIDES over the ground at the board's travel speed),
      so its truth mode must be classified from the board too, or truth (wheel = rolling) and
      observation (board = sliding) would describe different bodies and disagree by
      construction. Tracking a board-fixed point IS sliding; the wheels' rolling is a
      separate, un-tracked fact (THEORY.md s.3).
    """
    found = False
    f_n = 0.0
    pen = 0.0
    m_mode = FREE
    for ci in range(data.ncon):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        is_pair = (g1 in moving_geoms and g2 in support_geoms) or (
            g1 in support_geoms and g2 in moving_geoms
        )
        if not is_pair:
            continue
        found = True
        c_pen = max(0.0, -float(c.dist))
        pen = max(pen, c_pen)
        mujoco.mj_contactForce(model, data, ci, buf6)
        f_n += float(buf6[0])  # normal component in the contact frame

        # Contact frame rows are (normal, tangent1, tangent2); the relative twist is
        # measured in this support-attached frame (THEORY.md s.1).
        cframe = np.array(c.frame).reshape(3, 3)
        n_hat = cframe[0]
        t1, t2 = cframe[1], cframe[2]
        cpos = np.array(c.pos)

        # Relative twist of the moving body's material point w.r.t. the support's, taken at
        # the contact location (THEORY.md s.3: rolling vs sliding needs the velocity of the
        # MATERIAL point at the contact, not the COM). Use the body that actually OWNS the
        # contacting moving geom -- not the nominal edge moving body -- because the contact
        # may live on a freely-moving SUB-body (e.g. a skateboard's hinge-jointed wheel
        # spins about its own axle: its material contact point is ~stationary -> ROLLING,
        # while the parent board body merely translates -> would falsely read as SLIDING).
        mg = g1 if g1 in moving_geoms else g2
        mov_id = int(model.geom_bodyid[mg]) if mode_body_id is None else mode_body_id
        om_m, vlin_m = _object_twist_world(model, data, mov_id)
        vp_m = _material_point_velocity(om_m, vlin_m, data.xpos[mov_id], cpos)
        if support_body_id == 0:
            om_s = np.zeros(3)
            vp_s = np.zeros(3)
        else:
            om_s, vlin_s = _object_twist_world(model, data, support_body_id)
            vp_s = _material_point_velocity(om_s, vlin_s, data.xpos[support_body_id], cpos)

        v_rel = vp_m - vp_s
        om_rel = om_m - om_s
        v_normal = float(v_rel @ n_hat)
        slip_tan = float(np.hypot(v_rel @ t1, v_rel @ t2))
        spin_normal = abs(float(om_rel @ n_hat))

        v_com_rel = vlin_m - (np.zeros(3) if support_body_id == 0 else vlin_s)
        com_tan_speed = float(np.hypot(v_com_rel @ t1, v_com_rel @ t2))

        m_mode = _classify_mode(True, v_normal, slip_tan, spin_normal, com_tan_speed, shape)

    if not found:
        # Solver listed no force-active contact -- check true proximity (a fast roller riding
        # at ~0 penetration is physically touching but absent from data.contact).
        found, m_mode = _proximity_mode(model, data, moving_geoms, support_geoms,
                                        support_body_id, mode_body_id, shape)

    return found, f_n, pen, (m_mode if found else FREE)


def _simulate_scene(model: mujoco.MjModel, build: dict, hz: float) -> dict:
    """Run the headless multi-body sim and return clean per-body poses + per-edge truth.

    Same simulate->record loop as `_simulate` (THEORY.md s.9), generalized to N bodies and
    a LIST of edges. Records, per recorded frame: each named body's clean pose, and for
    every edge its existence / normal force / penetration / mode (from the active-set scan).
    Supports an optional ``settle`` phase followed by a one-shot ``launch`` (used to give
    the skateboard its !actuated initial velocity after the bodies have seated), plus an
    optional per-substep ``forcing`` (used to lower a support).
    """
    data = mujoco.MjData(model)

    body_names = list(build["bodies"])
    body_ids = {n: _id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names}

    edges = build["edges"]
    # Pre-resolve each edge's geom-id sets and support body id (world body id is 0, an
    # identity pose). The moving body is resolved PER CONTACT inside `_edge_frame_truth`
    # from the contacting geom, so a sub-body wheel's own spin is captured.
    edge_rt = []
    for e in edges:
        support_id = (
            0 if e["support_body"] == "world"
            else _id(model, mujoco.mjtObj.mjOBJ_BODY, e["support_body"])
        )
        # Optional: classify the truth MODE from a specific body's material point rather than
        # the contacting sub-body's (see _edge_frame_truth). Used by board_ground so the truth
        # mode (board-fixed point = sliding) matches what the detector observes.
        mode_body = e.get("truth_mode_body")
        mode_body_id = (
            None if mode_body is None
            else _id(model, mujoco.mjtObj.mjOBJ_BODY, mode_body)
        )
        edge_rt.append(
            {
                "edge_id": e["edge_id"],
                "support_id": support_id,
                "moving_geoms": _geom_ids(model, e["moving_geoms"]),
                "support_geoms": _geom_ids(model, e["support_geoms"]),
                "shape": e["shape"],
                "mode_body_id": mode_body_id,
            }
        )

    forcing = build.get("forcing")
    launch = build.get("launch")
    settle = float(build.get("settle", 0.0))

    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    sub = max(1, int(round((1.0 / hz) / dt)))  # physics substeps per recorded frame
    n_frames = int(round(build["duration"] * hz))

    # --- optional settle + one-shot launch (the skateboard's !actuated initial velocity) ---
    # Run the settle phase WITHOUT recording, then apply the launch impulse once, so the
    # recorded window starts at the launch with the bodies already seated.
    if settle > 0.0:
        n_settle = int(round(settle / dt))
        for _ in range(n_settle):
            if forcing is not None:
                forcing(model, data)
            mujoco.mj_step(model, data)
        data.time = 0.0  # reset the clock so the recorded time base starts at 0
    if launch is not None:
        launch(model, data)
        mujoco.mj_forward(model, data)

    # --- recording buffers ---
    t_rec: list[float] = []
    pos = {n: [] for n in body_names}
    quat = {n: [] for n in body_names}
    e_contact = {e["edge_id"]: [] for e in edges}
    e_force = {e["edge_id"]: [] for e in edges}
    e_pen = {e["edge_id"]: [] for e in edges}
    e_mode = {e["edge_id"]: [] for e in edges}

    buf6 = np.zeros(6)

    for _ in range(n_frames):
        for _ in range(sub):
            if forcing is not None:
                forcing(model, data)
            mujoco.mj_step(model, data)

        t_rec.append(float(data.time))
        for n in body_names:
            pos[n].append(data.xpos[body_ids[n]].copy())
            quat[n].append(data.xquat[body_ids[n]].copy())

        for ert in edge_rt:
            ok, fn, pen, mode = _edge_frame_truth(
                model,
                data,
                ert["support_id"],
                ert["moving_geoms"],
                ert["support_geoms"],
                ert["shape"],
                buf6,
                ert["mode_body_id"],
            )
            eid = ert["edge_id"]
            e_contact[eid].append(ok)
            e_force[eid].append(fn)
            e_pen[eid].append(pen)
            e_mode[eid].append(mode)

    out = {
        "t": np.asarray(t_rec, dtype=float),
        "pos": {n: np.asarray(pos[n], dtype=float) for n in body_names},
        "quat": {n: np.asarray(quat[n], dtype=float) for n in body_names},
        "edge_contact": {k: np.asarray(v, dtype=bool) for k, v in e_contact.items()},
        "edge_force": {k: np.asarray(v, dtype=float) for k, v in e_force.items()},
        "edge_pen": {k: np.asarray(v, dtype=float) for k, v in e_pen.items()},
        "edge_mode": e_mode,
    }
    return out


def generate_scene(
    name: str, seed: int = 0, hz: float = 100.0, noise_m: float = 5e-4
) -> MultiBodyScene:
    """Build, simulate, and label one multi-body SCENE (THEORY.md section 8).

    The scene-level analogue of :func:`generate`: instead of one body pair it produces a
    whole CONTACT GRAPH -- several bodies sharing a time base and a list of candidate
    ``ContactEdge`` s with a per-edge ``GroundTruth``. The graph layer runs the single-pair
    detector per edge (in each support's frame) and fuses the edges into a joint active-set
    posterior over the 2^E structures (THEORY.md s.8; exact enumeration is fine here since
    every scene keeps E <= 2 -- large E would need RJMCMC/particle methods).

    Parameters
    ----------
    name:
        One of :data:`SCENES`.
    seed:
        RNG seed for the additive mocap noise on the moving-body positions (reproducible).
    hz:
        Recording rate; the physics runs at the model timestep, subsampled to ~``1/hz``.
    noise_m:
        Std (m) of i.i.d. Gaussian position noise added to every body's recorded world
        positions, emulating optical mocap (THEORY.md s.4/s.9). Truth labels come from the
        CLEAN simulator state and are never derived from the noised poses.

    Returns
    -------
    MultiBodyScene
        ``bodies`` (name -> noised ``PoseTrajectory``; the world ground is implicit and
        re-created as an identity trajectory by the per-edge ``observe`` call when a
        support is "world"), ``edges`` (the candidate ``ContactEdge`` s), ``truth`` (edge_id
        -> per-edge ``GroundTruth``, frame-aligned and the same length as the time base),
        and ``meta``.

    Note
    ----
    Headless physics only -- no rendering (THEORY.md s.9). Each edge's ``support_body`` is a
    key into ``bodies`` *except* "world", which the per-edge geometry path treats as an
    identity pose (a static floor is the s.1 degenerate support of infinite mass).
    """
    if name not in _SCENE_BUILDERS:
        raise KeyError(f"unknown scene {name!r}; available: {SCENES}")

    model, build = _SCENE_BUILDERS[name]()
    # A scene builder may pin its own recording rate via build["record_hz"] (the
    # recording-cadence override, mirroring `generate`): chained-impact scenes contain brief
    # body-to-body strikes that are sub-frame at the default 100 Hz and only register as the
    # named IMPACT in the truth when sampled faster. Per-scene, and never narrows the caller's
    # request (we take the MAX).
    rec_hz = max(float(hz), float(build.get("record_hz") or hz))
    rec = _simulate_scene(model, build, rec_hz)

    rng = np.random.default_rng(seed)
    t = rec["t"]

    # --- emulate mocap: additive Gaussian noise on every body's recorded positions ---
    # (THEORY.md s.4: the detector only ever sees this noisy observable channel; the
    #  per-edge truth labels below come from the clean sim state, not the noised poses.)
    bodies: dict[str, PoseTrajectory] = {}
    for n, p in rec["pos"].items():
        noisy = p + rng.normal(0.0, noise_m, size=p.shape)
        bodies[n] = PoseTrajectory(t=t, position=noisy, quat=rec["quat"][n])

    # --- candidate edges + per-edge ground truth ---
    edges: list[ContactEdge] = []
    truth: dict[str, GroundTruth] = {}
    for e in build["edges"]:
        eid = e["edge_id"]
        edges.append(
            ContactEdge(
                edge_id=eid,
                moving_body=e["moving_body"],
                support_body=e["support_body"],
                surface=SupportSurface(
                    point=np.asarray(e["surface_point_local"], dtype=float),
                    normal=np.asarray(e["surface_normal_local"], dtype=float),
                ),
                contact_point_local=np.asarray(e["contact_point_local"], dtype=float),
                # Optional per-edge contact-geometry resolver (DESIGN.md III.1). Most edges
                # leave this absent -> ContactEdge.geometry defaults to None -> observe() wraps
                # surface + contact_point_local in a FlatRegion (today's bit-identical path).
                # An edge may attach a higher-fidelity resolver (e.g. SphereSphere on a
                # ball<->ball edge, DESIGN.md III.5 Phase 1) which observe() then uses instead.
                geometry=e.get("geometry"),
            )
        )
        truth[eid] = GroundTruth(
            t=t,
            in_contact=rec["edge_contact"][eid],
            mode=rec["edge_mode"][eid],
            normal_force=rec["edge_force"][eid],
            penetration=rec["edge_pen"][eid],
        )

    meta = {
        "scene": name,
        "seed": seed,
        "hz": rec_hz,
        "noise_m": noise_m,
        "timestep": float(model.opt.timestep),
        "bodies": list(build["bodies"]),
        "edge_ids": [e["edge_id"] for e in build["edges"]],
        "mode_thresholds": {
            "slip_eps": _SLIP_EPS,
            "spin_eps": _SPIN_EPS,
            "impact_vn": _IMPACT_VN,
            "roll_vtan": _ROLL_VTAN,
        },
        "note": (
            "MuJoCo truth is truth for MuJoCo's soft-constraint contact model "
            "(THEORY.md s.9). Per-edge truth labels come from the clean sim active set; "
            "only body positions are noised to emulate mocap. The 'world' support is an "
            "identity pose (the s.1 degenerate static floor)."
        ),
    }
    meta.update(build.get("meta", {}))

    return MultiBodyScene(name=name, bodies=bodies, edges=edges, truth=truth, meta=meta)


# ======================================================================================
# Demo-module registry merge (THEORY.md s.9: "the entire edge-case taxonomy on demand").
#
# The richer demos live in their own self-contained modules so this file stays focused on
# the core truth-factory machinery. Each demo module exposes module-level SCENARIO_BUILDERS
# and SCENE_BUILDERS dicts and imports NOTHING from `contact` (no `mujoco_gen`), so pulling
# them in HERE -- at the very end, after `_BUILDERS`, `_SCENE_BUILDERS`, `SCENARIOS`,
# `SCENES`, `generate`, and `generate_scene` are all defined -- is cycle-free: those modules
# only depend on `mujoco`/`numpy`, never back on us. We merge their builders into the
# registries so `generate` / `generate_scene` (and `SCENARIOS` / `SCENES`) see them too.
# ======================================================================================

from . import demos_motion, demos_impacts, demos_scenes_stack, demos_scenes_chain

for _m in (demos_motion, demos_impacts, demos_scenes_stack, demos_scenes_chain):
    _BUILDERS.update(getattr(_m, "SCENARIO_BUILDERS", {}))
    _SCENE_BUILDERS.update(getattr(_m, "SCENE_BUILDERS", {}))
SCENARIOS[:] = list(_BUILDERS)
SCENES[:] = list(_SCENE_BUILDERS)
