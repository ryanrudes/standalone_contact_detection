"""The core single-pair scenarios of THEORY.md §9 — the canonical edge-case taxonomy.

These seven builders are the scenarios the theory derives its demands from: the resting box
(`drop_rest`, `drop_rest_liftoff`), the friction transition (`push_to_slide`), the coupled
twist subspace (`rolling_ball`), the impact train (`bouncing_ball`), the relative-frame
payoff (`moving_support`), and the §7 observability rig (`indeterminate_rig`). Richer
regimes live in their sibling modules (`scenarios_motion`, `scenarios_impacts`).

Each builder compiles a MuJoCo model and returns ``(model, build)``, where ``build`` names
the entities the factory machinery extracts truth from (see `oracle.factory`). Builders
self-register by name via `oracle.registry.scenario`, so importing this module is what
makes these names available to `generate`.
"""

from __future__ import annotations

import numpy as np

import mujoco

from oracle._mjcf import obj_id as _id, options as _common_options
from oracle.registry import scenario
from oracle.specs import ScenarioSpec


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


@scenario("drop_rest")
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
    return ScenarioSpec(
        model=model,
        moving_body="box",
        moving_geom="boxg",
        support_body="world",
        support_geom="floor",
        surface_point_local=np.zeros(3),
        surface_normal_local=np.array([0.0, 0.0, 1.0]),
        contact_point_local=np.array([0.0, 0.0, -_BOX_HALF]),
        shape="box",
        duration=1.5,
        # Candidate contact corners for contact-implicit inverse dynamics (THEORY.md s.8).
        box_corners_local=_BOX_BOTTOM_CORNERS_LOCAL,
    )


@scenario("drop_rest_liftoff")
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

    return ScenarioSpec(
        model=model,
        moving_body="box",
        moving_geom="boxg",
        support_body="world",
        support_geom="floor",
        surface_point_local=np.zeros(3),
        surface_normal_local=np.array([0.0, 0.0, 1.0]),
        contact_point_local=np.array([0.0, 0.0, -_BOX_HALF]),
        shape="box",
        duration=1.6,
        forcing=forcing,
        box_corners_local=_BOX_BOTTOM_CORNERS_LOCAL,
    )


@scenario("push_to_slide")
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

    return ScenarioSpec(
        model=model,
        moving_body="box",
        moving_geom="boxg",
        support_body="world",
        support_geom="floor",
        surface_point_local=np.zeros(3),
        surface_normal_local=np.array([0.0, 0.0, 1.0]),
        contact_point_local=np.array([0.0, 0.0, -_BOX_HALF]),
        shape="box",
        # Keep the window short enough to stay in the clean static->sliding regime
        # before the box reaches a high enough speed to skip on the soft-constraint plane.
        duration=1.0,
        forcing=forcing,
        box_corners_local=_BOX_BOTTOM_CORNERS_LOCAL,
    )


@scenario("rolling_ball")
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

    return ScenarioSpec(
        model=model,
        moving_body="ball",
        moving_geom="ballg",
        support_body="world",
        support_geom="floor",
        # Sphere "material point" tracked is the center; the radius is the gap/offset
        # handled by the surface. We raise the (observation-side) plane by one radius so
        # the tracked CENTER's signed distance to it reads ~0 at contact, just like a
        # box's tracked bottom-face point. This is the surface absorbing the radius
        # exactly as documented for the sphere (THEORY.md s.3 note on tracking a single
        # material point). It changes only the OBSERVABLE gap; the MuJoCo truth labels
        # come from the geom-level contact and are unaffected.
        surface_point_local=np.array([0.0, 0.0, _BALL_R]),
        surface_normal_local=np.array([0.0, 0.0, 1.0]),
        contact_point_local=np.array([0.0, 0.0, 0.0]),
        shape="sphere",
        duration=1.0,
        init=init,
    )


@scenario("bouncing_ball")
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
    return ScenarioSpec(
        model=model,
        moving_body="ball",
        moving_geom="ballg",
        support_body="world",
        support_geom="floor",
        # Raise the observation-side plane by one radius so the tracked sphere CENTER's
        # signed distance reads ~0 at contact (see _build_rolling_ball for the rationale);
        # truth labels are geom-based and unaffected.
        surface_point_local=np.array([0.0, 0.0, _BALL_R]),
        surface_normal_local=np.array([0.0, 0.0, 1.0]),
        contact_point_local=np.array([0.0, 0.0, 0.0]),
        shape="sphere",
        duration=2.5,
    )


@scenario("moving_support")
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

    return ScenarioSpec(
        model=model,
        moving_body="box",
        moving_geom="boxg",
        support_body="cart",
        support_geom="cartg",
        # Surface in the CART's local frame: the deck top face.
        surface_point_local=np.array([0.0, 0.0, deck_top]),
        surface_normal_local=np.array([0.0, 0.0, 1.0]),
        contact_point_local=np.array([0.0, 0.0, -_BOX_HALF]),
        shape="box",
        duration=2.0,
        forcing=forcing,
        # The deck is the (moving) support; the box's four bottom corners are the
        # candidates. Gaps/forces are computed support-relative against the deck top.
        box_corners_local=_BOX_BOTTOM_CORNERS_LOCAL,
    )


# Local corner positions of the rig box's bottom face, in the box body frame. These are
# the K candidate point contacts whose individual loads s.7 says are unobservable from
# kinematics yet identifiable from per-corner penetration once compliance is known. They
# are exactly the shared box bottom-face corners (the same K=4 candidates the inverse-
# dynamics layer of s.8 reasons over), so we alias the one constant.
_RIG_CORNERS_LOCAL = _BOX_BOTTOM_CORNERS_LOCAL


@scenario("indeterminate_rig")
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
    return ScenarioSpec(
        model=model,
        moving_body="box",
        moving_geom="boxg",
        support_body="world",
        support_geom="floor",
        surface_point_local=np.zeros(3),
        surface_normal_local=np.array([0.0, 0.0, 1.0]),
        # Tracked center-of-bottom-face point, as for the other resting box scenarios.
        contact_point_local=np.array([0.0, 0.0, -_BOX_HALF]),
        shape="box",
        # Long enough to drop, dissipate the touchdown transient, and settle into a quiet
        # static multi-contact equilibrium before we read off the per-corner truth.
        duration=4.0,
        # Marks this builder so `generate` knows to harvest the K-corner contact_points.
        rig_corners_local=_RIG_CORNERS_LOCAL,
        # Also expose the unified inverse-dynamics candidate view (signed gap + per-corner
        # force + active flags) consistent with the new schema; these are the same K=4
        # corners, so meta["candidates"] aliases meta["contact_points"] downstream (s.8).
        box_corners_local=_RIG_CORNERS_LOCAL,
    )
