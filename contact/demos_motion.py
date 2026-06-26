"""Single-pair contact-detection demos showcasing varied contact MODES (THEORY.md s.9).

These are additional SCENARIO builders for the MuJoCo truth factory. Each one shapes the
*physics* so a particular twist regime (THEORY.md s.3) dominates and the truth labeler
(``mujoco_gen._classify_mode``) reports the named mode:

* ``incline_slide`` : a box on a TILTED plane slides downhill -> SLIDING, with a
  non-vertical support normal (the gap channel is tested against a tilted plane).
* ``skid_to_rest``  : a box launched horizontally on a high-friction floor decelerates
  -> SLIDING then STATIC (a friction-arrest transition).
* ``spinning_top``  : a cylinder spun FAST about the vertical (= contact-normal) axis,
  staying in place -> PIVOTING (spin about the normal).
* ``tumbling_box``  : a box thrown with linear + angular velocity bounces and tumbles
  across the floor -> alternating IMPACT / FREE / STATIC.

Self-contained by contract: this module imports ONLY ``mujoco``, ``numpy`` and a couple of
names from ``contact.types`` (the mode-string constants, for documentation only). It defines
its own tiny <option>/id helpers so it never has to import ``mujoco_gen`` (which would create
an import cycle, since ``mujoco_gen`` imports THIS file at its end). The generic
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
We deliberately do NOT set ``box_corners_local`` (that triggers the inverse-dynamics
metadata path reserved for the box-on-plane scenarios in ``mujoco_gen``).
"""

from __future__ import annotations

import numpy as np

import mujoco

# NOTE on self-containment (the import contract): this module imports ONLY ``mujoco`` and
# ``numpy``. The mode strings it produces (free/static/sliding/pivoting/impact) are defined in
# ``contact.types`` and emitted by ``mujoco_gen._classify_mode``; we never need them here, so
# we deliberately do NOT import any ``contact`` submodule -- ``mujoco_gen`` imports THIS file
# at its end, and importing it (or anything that imports it) back would create a cycle.


# --------------------------------------------------------------------------------------
# Tiny self-contained helpers (intentionally NOT imported from mujoco_gen -- see module
# docstring on the import cycle).
# --------------------------------------------------------------------------------------

def _common_options() -> str:
    """MuJoCo <option> block shared by these demos.

    Gravity -9.81 z, a small timestep for clean impacts, and a pyramidal friction cone
    (matching the rest of the truth factory). The small timestep matters most for
    ``tumbling_box``/``spinning_top``, where fast rotation + impacts otherwise smear.
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


# --------------------------------------------------------------------------------------
# Scenario builders
# --------------------------------------------------------------------------------------

def _build_incline_slide() -> tuple[mujoco.MjModel, dict]:
    """A box rests on a TILTED plane (~20 deg) and slides downhill under gravity.

    Physics (THEORY.md s.3, the sliding mode, and s.1, the support-relative gap with a
    NON-vertical normal). The plane is tilted by angle ``theta`` about the +y axis, so its
    outward normal is ``n = (sin theta, 0, cos theta)`` -- NOT +z. Gravity's component
    along the incline is ``g sin theta``; the static-friction ceiling is ``mu g cos theta``.
    We pick a low friction (mu = 0.3) and theta = 20 deg so that
    ``tan(20 deg) = 0.364 > mu = 0.3`` -> the box cannot stick and slides steadily downhill.

    The support normal is the TILTED plane normal (set in ``surface_normal_local``), and the
    tracked material point is the box face touching the incline. Because the box rides the
    incline, the COM accelerates DOWN the slope (constant ``g(sin theta - mu cos theta)``),
    so the material contact point slips tangentially the whole time -> a clean, sustained
    SLIDING label. We orient the box so one face lies flat on the incline (same tilt as the
    plane) and place it just touching, so it is in contact from t=0 with no big touchdown
    impact to pollute the early frames.
    """
    theta_deg = 18.0
    theta = np.deg2rad(theta_deg)
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    # mu just BELOW tan(18 deg)=0.325 -> kinetic net downslope accel g(sin-mu*cos) ~ 0.23 m/s^2,
    # a GENTLE steady slide (not a runaway), so the box stays well within the detector's
    # sliding-speed range instead of accelerating into the FREE-diffuse regime. We also give it
    # a small initial downhill velocity so it is cleanly SLIDING from frame 0 (no stick-slip
    # onset stutter), and stiffen+damp the contact so it does not chatter/skip on the soft plane.
    mu = 0.30
    v0 = 0.15  # initial downhill speed (m/s)

    # The ramp is a large BOX geom tilted by +theta about the world +y axis; its TOP face is
    # the incline. The ramp is fixed to the world (no joint), so the support is "world" and
    # the surface is given directly in WORLD coordinates. We derive the exact incline plane so
    # the observed support-relative gap reads ~0 at contact (a wrong surface point would leave
    # a spurious cm-scale standoff in the OBSERVED channel, like a mismeasured floor height).
    #
    # Geometry: the ramp box has half-z = 0.5 and origin at z = -0.5, so its top-face center
    # in the ramp-local frame is (0, 0, 0.5). Rotating by R(theta about +y) and translating
    # gives the top-face center and the outward normal in the world:
    R = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])  # rot about +y by +theta
    ramp_half_z = 0.5
    ramp_origin = np.array([0.0, 0.0, -ramp_half_z])
    top_world = ramp_origin + R @ np.array([0.0, 0.0, ramp_half_z])   # incline surface point
    n_world = R @ np.array([0.0, 0.0, 1.0])                           # incline outward normal
    # A FLAT SLAB (low height -> low CoM) rather than a cube: a sliding cube pitches forward
    # (friction at the base, weight up high) and lifts its trailing edge -> 2-corner contact
    # that micro-bounces and makes the truth contact flicker. A low slab has a tiny pitching
    # moment, so it slides flat in continuous contact. half-extents: wide footprint, thin.
    slab = np.array([0.16, 0.16, 0.025])
    # Seat the slab slightly INTO the incline (~0.5 mm), PRE-TILTED so its whole bottom face
    # lies flat on the incline -> a damped contact engaged from frame 0 with no touchdown
    # bounce, and the wide flat footprint keeps it seated as it slides.
    contact_world = top_world - n_world * 0.0005
    box_origin = contact_world + R @ np.array([0.0, 0.0, slab[2]])
    # Unit downhill tangent (gravity projected onto the incline surface): (cos, 0, -sin).
    downhill = np.array([ct, 0.0, -st])
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="ramp" type="box" size="3 3 {ramp_half_z}" pos="0 0 -{ramp_half_z}"
          euler="0 {theta_deg} 0" friction="{mu} 0.005 0.0001"
          solref="0.008 2" solimp="0.97 0.999 0.0001" margin="0.004" gap="0.004"/>
    <body name="box" pos="{box_origin[0]} {box_origin[1]} {box_origin[2]}"
          euler="0 {theta_deg} 0">
      <freejoint name="boxj"/>
      <geom name="boxg" type="box" size="{slab[0]} {slab[1]} {slab[2]}"
            density="1000" friction="{mu} 0.005 0.0001"
            solref="0.008 2" solimp="0.97 0.999 0.0001" margin="0.004" gap="0.004"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    box_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "box")

    def init(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Launch the box gently DOWN the incline so it is cleanly sliding from frame 0.
        adr = _free_dofadr(m, box_id)
        d.qvel[adr:adr + 3] = v0 * downhill

    build = {
        "init": init,
        "moving_body": "box",
        "moving_geom": "boxg",
        "support_body": "world",
        "support_geom": "ramp",
        # Surface point + normal in the support (world) frame: the actual incline top-face
        # point and its tilted outward normal n = (sin theta, 0, cos theta) -- NON-vertical,
        # the whole point of this demo (the support-relative gap with a tilted normal).
        "surface_point_local": top_world,
        "surface_normal_local": n_world,
        # Tracked material point on the box: the center of the face lying on the incline
        # (the slab-local -z face center; the slab is pre-tilted to match the incline).
        "contact_point_local": np.array([0.0, 0.0, -0.025]),
        "shape": "box",
        "duration": 1.2,
    }
    return model, build


def _build_skid_to_rest() -> tuple[mujoco.MjModel, dict]:
    """A box launched horizontally on a high-friction floor skids and decelerates to rest.

    Physics (THEORY.md s.3): an initial COM velocity along +x with the box already seated on
    the floor. Kinetic friction ``mu m g`` decelerates it at constant ``a = mu g`` until the
    material contact point's tangential slip drops below the slip threshold -> the mode
    transitions SLIDING -> STATIC. We give it enough speed (2.5 m/s) and a high-ish mu (0.6)
    that it slides visibly for a few tenths of a second, then comes to rest and stays put for
    the remainder -> both SLIDING and STATIC appear, with a clean arrest in between. No
    forcing: gravity seats it and friction does all the work after the initial kick.
    """
    mu = 0.6
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" pos="0 0 0"
          friction="{mu} 0.005 0.0001"/>
    <body name="box" pos="0 0 {_BOX_HALF}">
      <freejoint name="boxj"/>
      <geom name="boxg" type="box" size="{_BOX_HALF} {_BOX_HALF} {_BOX_HALF}"
            density="500" friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    box_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "box")

    def init(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Kick the box along +x. Stop distance ~ v^2 / (2 mu g) = 2.5^2/(2*0.6*9.81) ~ 0.53 m,
        # taking ~ v/(mu g) = 2.5/(0.6*9.81) ~ 0.42 s -> a clear SLIDING segment then a long
        # STATIC tail within the 1.5 s window.
        adr = _free_dofadr(m, box_id)
        d.qvel[adr + 0] = 2.5  # world +x linear velocity

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
        "init": init,
    }
    return model, build


def _build_spinning_top() -> tuple[mujoco.MjModel, dict]:
    """A top spun FAST about the vertical (= contact-normal) axis, staying in place.

    Physics (THEORY.md s.3, the pivoting mode): pivoting is spin about the contact NORMAL
    with ~no tangential slip and ~no normal closing. The discriminator is the *material
    contact point*: PIVOTING needs that point's tangential slip to stay small
    (``slip_tan < _SLIP_EPS``) while the relative spin about the normal is large
    (``spin_normal > _SPIN_EPS``).

    The naive "flat cylinder/puck on the floor" does NOT pivot in the truth labels: a flat
    face contacts the plane out at its RIM, and a rim point at radius ``r`` spinning at
    ``omega`` slips tangentially at ``omega*r`` (e.g. 60*0.08 ~ 4.8 m/s) -> the labeler reads
    SLIDING, not pivoting. A real top fixes this by touching at a POINT on the spin axis.

    So we model an actual top: a small spherical FOOT at the bottom center is the only
    colliding geom, with a heavy flywheel DISC mounted high on a thin stem (both
    non-colliding, contype/conaffinity=0 -- they only add mass/inertia). The contact point
    sits on the spin axis, so its tangential slip is ~0 while ``omega_z`` is large -> a clean
    PIVOTING label, and the high disc keeps the top upright and roughly in place (gyroscopic
    + low base). The tiny torsional friction (3rd ``friction`` term) lets the spin persist
    across the window, so PIVOTING dominates throughout.

    On the SPIN MAGNITUDE: we use omega_z = 2.0 rad/s. That is ~6.7x the truth pivot threshold
    (``_SPIN_EPS`` = 0.30 rad/s), so it is unambiguously PIVOTING in the labels, yet it sits at
    the detector's calibrated angular scale (``EmissionConfig.pivot_speed`` ~ 1.0 rad/s, with a
    FREE angular prior of only ~3 rad/s). A much larger spin (say 60-80 rad/s) is still
    correctly LABELLED pivoting, but lands tens of sigma outside every emission's angular scale,
    so the diffuse FREE state wins the likelihood and the detector mis-reads it as free. Keeping
    the spin near the model's physical scale lets the demo exhibit AND be detected as pivoting
    end to end -- the honest choice (the observable channel is what a mocap rig could resolve).
    """
    foot_r = 0.012   # spherical foot radius (m) -- the only colliding geom, on the axis
    disc_r = 0.08    # flywheel disc radius (m), high up (non-colliding, just inertia)
    disc_hh = 0.02   # disc half-height (m)
    stem = 0.05      # stem length from foot top to disc bottom (m)
    disc_z = foot_r + stem + disc_hh  # disc center height in body-local frame
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <!-- Low torsional friction (3rd term) so the spin about the normal is not braked away
         in the first frames; moderate sliding friction so the foot does not wander. -->
    <geom name="floor" type="plane" size="20 20 0.1" pos="0 0 0" friction="0.5 0.5 0.005"/>
    <body name="top" pos="0 0 {foot_r}">
      <freejoint name="topj"/>
      <!-- The ONLY colliding geom: a small sphere foot at the bottom center (on the axis). -->
      <geom name="footg" type="sphere" size="{foot_r}" pos="0 0 0" density="500"
            friction="0.5 0.5 0.005"/>
      <!-- Heavy flywheel disc mounted high (non-colliding): adds the inertia that keeps the
           top upright while it spins, without creating off-axis rim contacts. -->
      <geom name="discg" type="cylinder" size="{disc_r} {disc_hh}" pos="0 0 {disc_z}"
            density="3000" contype="0" conaffinity="0"/>
      <geom name="stemg" type="capsule" size="0.006 {stem / 2.0}" pos="0 0 {foot_r + stem / 2.0}"
            density="500" contype="0" conaffinity="0"/>
      <!-- A small off-axis "marker" nub on the disc rim (non-colliding, negligible mass) so the
           SPIN is visible in the render -- it orbits the axis as the top pivots. The visualizer
           paints any "*marker*" geom a contrasting accent colour. -->
      <geom name="markerg" type="box" size="0.012 0.012 0.012" pos="{0.72 * disc_r} 0 {disc_z}"
            mass="0" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    top_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "top")

    def init(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Spin about +z (the contact normal). The foot contacts on the spin axis, so its
        # tangential slip stays ~0 while omega_z is well above the pivot threshold -> PIVOTING.
        # 2.0 rad/s is ~6.7x the truth threshold yet within the detector's angular scale (see
        # the docstring on why a much larger spin would be mis-read as FREE).
        adr = _free_dofadr(m, top_id)
        d.qvel[adr + 5] = 2.0  # world +z angular velocity (rad/s)

    build = {
        "moving_body": "top",
        "moving_geom": "footg",
        "support_body": "world",
        "support_geom": "floor",
        "surface_point_local": np.zeros(3),
        "surface_normal_local": np.array([0.0, 0.0, 1.0]),
        # Tracked material point: the bottom of the foot (on the spin axis), so its tangential
        # slip is ~0 while the body spins -> the rigorous pivoting signature.
        "contact_point_local": np.array([0.0, 0.0, -foot_r]),
        "shape": "cylinder",
        "duration": 1.5,
        "init": init,
    }
    return model, build


def _build_tumbling_box() -> tuple[mujoco.MjModel, dict]:
    """A box thrown with linear + angular velocity tumbles across the floor then rests.

    Physics (THEORY.md s.6, impacts as force atoms; s.3 modes). The box starts in the air
    and is launched with a forward+upward COM velocity and a spin about +y (the lateral axis,
    so it tumbles end-over-end in the x-z plane). Each time a corner/edge strikes the floor
    the relative normal velocity is arrested almost discontinuously -> an IMPACT atom;
    between strikes the box is airborne -> FREE; after a few tumbles it loses energy to the
    impacts and friction and settles -> STATIC. This yields the named alternating
    IMPACT / FREE / STATIC sequence.

    Tuning for a CLEAN, labelable tumble (per the contract's guidance to prefer a few clean
    tumbles over chaos): a forward speed (2.0 m/s) plus a real upward toss (1.5 m/s) so the
    box genuinely arcs through the air between strikes, and a spin (9.2 rad/s) about +y so it
    rotates end-over-end as it flies; a moderately bouncy-but-damped floor (``solref``) so
    each touchdown is a sharp, distinct IMPACT rather than a mushy single thud or endless
    jitter; medium friction so it grips and tips (tumbles) rather than skating flat. The box
    starts ~0.3 m up so the first touchdown is a clear FREE->IMPACT transition. With these
    numbers it makes ~2 visible airborne tumbles (and ~6 strikes counting corner/edge sub-
    impacts) -- IMPACT atoms separated by FREE flight arcs -- losing energy each bounce until
    it settles into a sustained STATIC rest well inside the window.

    A deliberate refinement: the spin is tuned so the box lands back UPRIGHT (an integer
    number of quarter-turns, finishing flat on its ORIGINAL bottom face). This matters for the
    OBSERVABLE channel: the detector tracks a single body-fixed point (the bottom-face center,
    ``contact_point_local``), and only that point reads gap ~0 at rest if the box rests on
    that same face. Landing on a side face instead would leave the tracked point a box-half
    above the floor at rest (truth still correct -- it is geom-based -- but the observed gap
    would never close). Landing upright keeps the demo honest end to end.
    """
    start_z = 0.3
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="40 40 0.1" pos="0 0 0"
          friction="0.5 0.01 0.001" solref="0.004 0.5"/>
    <body name="box" pos="0 0 {start_z}">
      <freejoint name="boxj"/>
      <geom name="boxg" type="box" size="{_BOX_HALF} {_BOX_HALF} {_BOX_HALF}"
            density="500" friction="0.5 0.01 0.001" solref="0.004 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    box_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "box")

    def init(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Throw: forward (+x) and UP (+z) so it arcs through the air, plus spin about +y so
        # it tumbles end-over-end in the x-z plane (the forward direction of travel). Tuned
        # so it does ~2 visible tumbles, loses energy each bounce, and lands back UPRIGHT
        # (on its original bottom face) for a clean STATIC rest -- not skidding or jittering.
        adr = _free_dofadr(m, box_id)
        d.qvel[adr + 0] = 2.0    # +x linear (forward)
        d.qvel[adr + 2] = 1.5    # +z linear (real upward arc -> airborne FREE between strikes)
        d.qvel[adr + 4] = 9.2    # +y angular (tumble forward, end over end; lands upright)

    build = {
        "moving_body": "box",
        "moving_geom": "boxg",
        "support_body": "world",
        "support_geom": "floor",
        "surface_point_local": np.zeros(3),
        "surface_normal_local": np.array([0.0, 0.0, 1.0]),
        "contact_point_local": np.array([0.0, 0.0, -_BOX_HALF]),
        "shape": "box",
        "duration": 2.5,
        "init": init,
        # Attach a BoxPlane resolver (DESIGN.md PHASE 2): the box tumbles, so the contact is
        # whichever of its 8 CORNERS is currently lowest -- a MIGRATING contact. The legacy
        # fixed bottom-face point ([0,0,-_BOX_HALF]) sits ~225 mm in the air when a corner
        # strikes, so its gap never closes at a bounce and the per-corner IMPACT cannot fire.
        # The BoxPlane's nearest-corner gap closes at every bounce, so the impacts are seen.
        "geometry": {"kind": "box_plane", "half_extents": [_BOX_HALF, _BOX_HALF, _BOX_HALF]},
    }
    return model, build


# --------------------------------------------------------------------------------------
# Registries (the required module-level dicts). SCENE_BUILDERS is intentionally empty:
# every demo here is a single contact PAIR (a moving body vs one support).
# --------------------------------------------------------------------------------------

SCENARIO_BUILDERS: dict[str, callable] = {
    "incline_slide": _build_incline_slide,
    "skid_to_rest": _build_skid_to_rest,
    "spinning_top": _build_spinning_top,
    "tumbling_box": _build_tumbling_box,
}

SCENE_BUILDERS: dict[str, callable] = {}
