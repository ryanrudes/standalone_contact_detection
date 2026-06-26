"""Complex multi-body SCENES with body-to-body collisions / chained impacts (THEORY.md s.8/s.9).

These scenes go beyond a single body resting on a static floor: their interesting
edges are *body-to-body* contacts that flicker between FREE and IMPACT as momentum is
exchanged through a chain (THEORY.md s.6: an impact is a force *atom*, a velocity
reset map ``v+ = -e*v-``, and these scenes are where those atoms propagate). The
contact graph (THEORY.md s.8) here has edges that share moving bodies, and the
true active set changes as collisions happen.

Three scenes:

* ``two_balls_collide`` : two spheres on the floor; ball A rolls toward ball B and
  strikes it, exchanging momentum. The two floor edges are ROLLING; the ball<->ball
  edge is a transient IMPACT at collision.
* ``dominoes`` : a row of upright thin boxes; the first is shoved and topples into the
  next, cascading. Each domino<->floor edge sees an IMPACT/topple as the domino above
  it falls; adjacent domino<->domino edges see the strike impacts.
* ``newtons_cradle`` : hinge-suspended balls hanging just touching; the end ball is
  lifted and released to strike the line, momentum passing through to the far ball.
  The adjacent ball<->ball edges carry the propagating impacts.

SELF-CONTAINMENT (THEORY.md s.9 / the builder contract): this module imports ONLY
``mujoco``, ``numpy``, and ``contact.types``. It defines its own tiny ``<option>`` /
name->id helpers so it never imports ``mujoco_gen`` (which imports THIS file at its
end -- importing it back would be a cycle). The generic simulate/label/score code in
``mujoco_gen`` does everything else: we only build the model + the build dict.

A scene builder may pin its own recording rate via ``build["record_hz"]``: the truth
labeler samples each edge's active set only on RECORDED frames (every 1/hz s), so a brief
body-to-body strike can be sub-frame at the default 100 Hz and register only as a single
degenerate frame. ``mujoco_gen.generate_scene`` records at ``max(caller hz, record_hz)``,
so it never narrows a caller asking for an even higher rate.
"""

from __future__ import annotations

import numpy as np

import mujoco

# --------------------------------------------------------------------------------------
# Tiny self-contained helpers (NO import of mujoco_gen or any submodule that imports it).
# --------------------------------------------------------------------------------------


def _options() -> str:
    """Shared MuJoCo <option> block: gravity -9.81, fine timestep, pyramidal cone.

    A small timestep (0.0005 s) keeps the body-to-body impacts crisp -- an impact is a
    near-discontinuous velocity reset (THEORY.md s.6), and too coarse a step smears it
    into a soft ramp that the truth labeller would never flag as IMPACT.
    """
    return (
        '<option timestep="0.0005" gravity="0 0 -9.81" '
        'integrator="implicitfast" cone="pyramidal"/>'
    )


def _bid(model: mujoco.MjModel, name: str) -> int:
    """Resolve a named body to its integer id (raises if absent)."""
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise KeyError(f"no body named {name!r} in model")
    return i


def _dofadr(model: mujoco.MjModel, body_name: str) -> int:
    """First DOF address of a body's joint (for writing initial qvel)."""
    bid = _bid(model, body_name)
    return int(model.jnt_dofadr[model.body_jntadr[bid]])


# Shared sizes (kept as named constants so surface points / tracked material points line up).
_BALL_R = 0.05          # sphere radius (m) for the two-balls scene


# ======================================================================================
# Scene 1: two balls collide
# ======================================================================================

def _build_two_balls_collide() -> tuple[mujoco.MjModel, dict]:
    """Two spheres on the floor; ball A is sent into ball B and exchanges momentum.

    Physics (THEORY.md s.3 + s.6). Ball A is launched toward +x (a brisk forward shove on a
    HIGH-friction floor, so it spins up into rolling almost immediately) at ball B, which is
    resting one ball-diameter ahead. The collision is a clean central IMPACT on the
    ball<->ball edge: a near-discontinuous reset of the relative NORMAL velocity along the
    line of centers (s.6). With equal masses and a head-on hit, momentum is exchanged -- A
    slows sharply and B is shot forward, then B rolls away. Both floor edges are ROLLING when
    their ball is moving.

    Making the ball<->ball IMPACT both REAL and DETECTABLE (three coupled fixes). The
    headline of this scene is the ball<->ball edge, and it is the hardest thing to capture --
    a sub-frame event. Three things conspired to make the old version's edge degenerate
    (single frame, mislabeled SLIDING, IoU 0), and each is fixed:

    1. *Mode purity (launch).* If A arrives already SPINNING (a pre-set no-slip roll), the two
       ball surfaces SHEAR past each other at the contact, so the relative material-point twist
       is dominated by the TANGENTIAL component and the truth labels the collision SLIDING, not
       IMPACT. We launch A with a brisk PURE TRANSLATION (~2.5 m/s, no pre-set spin); the
       high-friction floor spins it up into rolling over the run, but at the strike the approach
       is essentially along the line of centers, so the contact is NORMAL-velocity-dominated --
       a true IMPACT. A and B still roll on the floor before/after (floor edges stay ROLLING),
       and momentum still transfers (A advances ~1 m, B is shot ~1.4 m).
    2. *Observable geometry (the gap channel).* The old surface put a plane on B's near face
       (-R) and tracked A's +x POLE -- but a rolling/spinning A rotates that body-fixed pole
       away from the line of centers, so the observed gap never reached 0 at the strike and the
       detector could never confirm the contact. We instead track A's (rotation-invariant)
       CENTER and place the plane TWO radii inboard of B's center, absorbing BOTH radii (the
       sphere convention of the floor edges, generalized to a moving support) -- so the gap
       reads ~0 exactly when the geoms touch.
    3. *Sampling + persistence (cadence + contact softness).* The strike is a handful of 0.5 ms
       substeps; at 100 Hz the recorder flies past its normal-velocity peak. We pin
       ``record_hz = 200`` and use a slightly softer ball-ball contact (``solref="0.02 1"``) so
       the impact ATOM is stretched across several recorded frames -- long enough that the
       truth labels several IMPACT frames AND the HMM persistence prior can confirm the contact
       (a non-degenerate, non-zero-IoU edge).

    Three candidate edges (THEORY.md s.8 contact graph):

    * ``ballA_floor`` : ballA <-> floor (ROLLING while A moves).
    * ``ballB_floor`` : ballB <-> floor (ROLLING after the strike sends B off).
    * ``ballA_ballB`` : ballA <-> ballB (FREE -> transient IMPACT at the collision -> FREE).

    The MuJoCo truth labels are geom-based and independent of that observable surface
    (THEORY.md s.9).
    """
    gap = 0.35  # initial center-to-center gap along +x (m); A starts at the origin. Long
                # enough that the detector sees a clear multi-frame FREE approach before the
                # strike, short enough that A is still moving briskly (mostly along the line of
                # centers) when it reaches B, so the contact stays a NORMAL-dominated IMPACT.
    xml = f"""
<mujoco>
  {_options()}
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" pos="0 0 0" friction="0.3 0.005 0.0001"
          contype="1" conaffinity="1"/>
    <body name="ballA" pos="0 0 {_BALL_R}">
      <freejoint name="ballAj"/>
      <geom name="ballAg" type="sphere" size="{_BALL_R}" density="800"
            friction="0.3 0.005 0.0001" solref="0.01 1" contype="1" conaffinity="0"/>
    </body>
    <body name="ballB" pos="{gap} 0 {_BALL_R}">
      <freejoint name="ballBj"/>
      <geom name="ballBg" type="sphere" size="{_BALL_R}" density="800"
            friction="0.3 0.005 0.0001" solref="0.01 1" contype="1" conaffinity="0"/>
    </body>
  </worldbody>
  <contact>
    <!-- Decouple the two contact regimes: the ball-FLOOR contacts use the geoms' damped
         solref ("0.01 1") so a fast-sliding ball does NOT chatter/bounce on the floor (which
         made the floor edge flicker FREE); the ball-BALL collision uses an explicit elastic
         pair (low damping) for a clean billiard momentum transfer. contype/conaffinity stop
         the balls from auto-colliding, so this pair is their ONLY mutual contact. -->
    <pair geom1="ballAg" geom2="ballBg" solref="0.02 0.2" friction="0.3 0.3 0.005 0.0001 0.0001"/>
  </contact>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    a_dof = _dofadr(model, "ballA")

    # Phase-1 sphere-sphere geometry (DESIGN.md III.5) for the ball<->ball edge. Lazy import,
    # mirroring geometry.observe()'s own lazy FlatRegion import: keeps this module's
    # module-load-time dependencies just mujoco/numpy (the cycle-free contract documented in
    # the module docstring and the mujoco_gen registry merge) since geometry_resolvers is only
    # touched when a builder actually runs, long after every module has loaded.
    from .geometry_resolvers import SphereSphere

    def launch(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Brisk shove of ball A toward +x. The floor is LOW-friction so A arrives SLIDING with
        # little spin -- so at impact almost all of A's momentum is LINEAR and transfers
        # cleanly to B (an elastic contact, solref damping 0.1): A nearly STOPS and B shoots
        # off (~2 m/s of A's 2.5), the recognizable billiard transfer. (A high-friction floor
        # would spin A up into a roll; the retained spin then re-accelerates A after impact, so
        # both balls drift the same way -- a muddier, less legible collision.)
        d.qvel[a_dof + 0] = 2.5     # linear x (m/s)

    build = {
        "bodies": ["ballA", "ballB"],
        # One-shot launch (reuses the scene 'launch' hook _simulate_scene runs after settle).
        "launch": launch,
        "duration": 1.6,
        # Sample faster than the default 100 Hz so the brief ball-ball collision spans several
        # recorded frames (its sub-frame normal-velocity peak would otherwise fall between
        # frames, leaving the ball-ball edge a single, degenerate SLIDING frame). See docstring.
        "record_hz": 300.0,
        "edges": [
            {
                "edge_id": "ballA_floor",
                "moving_body": "ballA",
                "support_body": "world",
                "moving_geoms": ["ballAg"],
                "support_geoms": ["floor"],
                # Observation-side plane raised one radius so the tracked CENTER reads gap ~0
                # at floor contact (the sphere convention from mujoco_gen's rolling_ball).
                "surface_point_local": np.array([0.0, 0.0, _BALL_R]),
                "surface_normal_local": np.array([0.0, 0.0, 1.0]),
                "contact_point_local": np.array([0.0, 0.0, 0.0]),
                "shape": "sphere",
            },
            {
                "edge_id": "ballB_floor",
                "moving_body": "ballB",
                "support_body": "world",
                "moving_geoms": ["ballBg"],
                "support_geoms": ["floor"],
                "surface_point_local": np.array([0.0, 0.0, _BALL_R]),
                "surface_normal_local": np.array([0.0, 0.0, 1.0]),
                "contact_point_local": np.array([0.0, 0.0, 0.0]),
                "shape": "sphere",
            },
            {
                "edge_id": "ballA_ballB",
                "moving_body": "ballA",
                "support_body": "ballB",
                "moving_geoms": ["ballAg"],
                "support_geoms": ["ballBg"],
                # Sphere-sphere observable surface (the sphere convention, generalized to a
                # MOVING support). We track A's CENTER (contact_point_local = 0) and put a
                # vertical plane on ballB at TWO radii inboard of B's center with an outward
                # (-x) normal, so the plane absorbs BOTH ball radii: when the centers are 2R
                # apart (touching) the signed distance from A's center to the plane is ~0. This
                # is the fix for the old (-R, track A's +x pole) surface, which FAILED because a
                # rolling/spinning A rotates that body-fixed +x material point away from the
                # line of centers -- so the observed gap never reached 0 at the strike and the
                # detector could never confirm the contact. Tracking the (rotation-invariant)
                # center, as the floor edges do, keeps the gap honest. (Truth is geom-based.)
                "surface_point_local": np.array([-2.0 * _BALL_R, 0.0, 0.0]),
                "surface_normal_local": np.array([-1.0, 0.0, 0.0]),
                "contact_point_local": np.array([0.0, 0.0, 0.0]),
                "shape": "sphere",
                # Phase-1 (DESIGN.md III.5): resolve this ball<->ball edge with SphereSphere so
                # the contact normal is the line-of-centres (c_A - c_B)/||.||, NOT a quat-carried
                # body-fixed vector. The old plane-on-B surface above (kept for the FlatRegion
                # fallback / broad-phase) whirls with a spinning A and manufactured 7 phantom
                # impacts; the position-derived normal collapses them to the single real strike.
                "geometry": SphereSphere(_BALL_R, _BALL_R),
            },
        ],
        "meta": {
            "story": (
                "Ball A is sent into a resting ball B; equal-mass central impact exchanges "
                "momentum (THEORY.md s.6). Floor edges ROLLING; ball-ball edge a clean "
                "transient IMPACT."
            ),
        },
    }
    return model, build


# ======================================================================================
# Scene 2: dominoes
# ======================================================================================

_DOM_HALF = np.array([0.008, 0.05, 0.09])  # domino half-extents (m): thin (x), tall (z)
_DOM_GAP = 0.085                            # center-to-center spacing along +x (m)
_N_DOM = 4                                  # number of dominoes (4 => the cascade reliably completes)


def _build_dominoes() -> tuple[mujoco.MjModel, dict]:
    """A row of upright thin boxes; the first is shoved and topples into the next.

    Physics (THEORY.md s.6, chained impacts). Each domino is a tall thin box standing on
    the floor. The first is given an initial +x angular velocity (a shove about its
    bottom edge); it topples, its top strikes domino 2, knocking it past its balance
    point, and the cascade propagates. The spacing ``_DOM_GAP`` is set smaller than a
    domino's height so a falling domino reliably reaches the next one (the classic
    cascade condition), and 4 dominoes keep the chain short enough to complete cleanly
    within the window.

    Edges (THEORY.md s.8): one ``domI_floor`` edge per domino (its bottom-face contact
    with the ground). While standing it is STATIC; as it topples its bottom edge pivots
    and the toppling/landing produces IMPACT transients. We also expose the adjacent
    ``domI_domJ`` strike edges (box-on-box) so the propagating collision shows up as a
    sequence of IMPACTs along the row.

    The tracked material point for each floor edge is the domino's bottom-face center;
    its observable gap stays ~0 while standing and lifts as the domino rotates off its
    base (the s.1 support-relative gap). The truth labels are geom-based (THEORY.md s.9).
    """
    hx, hy, hz = _DOM_HALF
    bodies = [f"dom{i}" for i in range(_N_DOM)]

    def _domino(i: int, x: float) -> str:
        return (
            f'<body name="dom{i}" pos="{x} 0 {hz}">'
            f'  <freejoint name="dom{i}j"/>'
            f'  <geom name="dom{i}g" type="box" size="{hx} {hy} {hz}" density="700" '
            f'        friction="0.6 0.01 0.001" solref="0.004 1" margin="0.004" gap="0.004"/>'
            f'</body>'
        )

    domino_xml = "\n    ".join(_domino(i, i * _DOM_GAP) for i in range(_N_DOM))
    xml = f"""
<mujoco>
  {_options()}
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0" friction="0.8 0.01 0.001"
          margin="0.004" gap="0.004"/>
    {domino_xml}
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    d0_dof = _dofadr(model, "dom0")

    def launch(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Shove the first domino: a +x lean plus an initial tip angular velocity about +y
        # (the lateral axis), pushing its top toward +x so it falls into domino 1. A small
        # forward COM velocity helps it commit past its balance point.
        d.qvel[d0_dof + 0] = 0.12     # linear x: gentle forward shove of the COM
        d.qvel[d0_dof + 4] = 3.5      # angular about +y: tips the top toward +x

    # Floor edges: one per domino. Track the FRONT-BOTTOM PIVOT EDGE ([+hx,0,-hz]) rather
    # than the bottom-face center: a toppling domino pivots about this front edge, so this
    # point stays grounded (gap ~ 0) right through the topple, whereas the face center lifts
    # the instant the domino tips and would read FREE while the domino is plainly still down.
    edges = []
    for i in range(_N_DOM):
        edges.append(
            {
                "edge_id": f"dom{i}_floor",
                "moving_body": f"dom{i}",
                "support_body": "world",
                "moving_geoms": [f"dom{i}g"],
                "support_geoms": ["floor"],
                "surface_point_local": np.array([0.0, 0.0, 0.0]),
                "surface_normal_local": np.array([0.0, 0.0, 1.0]),
                "contact_point_local": np.array([+hx, 0.0, -hz]),
                "shape": "box",
            }
        )
    # Adjacent domino<->domino strike edges (box-on-box). Surface = the +x face of the
    # lower-index domino in ITS local frame, normal +x; tracked point on the upper-index
    # domino is its -x face center. These observe the propagating strikes, then a sustained
    # face-to-face lean as the toppled dominoes pile up. We expose all but the LAST pair:
    # the final domino has nothing to lean on so it falls flat (a brief edge-touch the
    # fixed-point observable can't track cleanly), so a dom{N-2}_dom{N-1} edge would be a
    # perpetually-empty lane. The earlier pairs end in clean parallel-face leans.
    for i in range(_N_DOM - 2):
        edges.append(
            {
                "edge_id": f"dom{i}_dom{i + 1}",
                "moving_body": f"dom{i + 1}",
                "support_body": f"dom{i}",
                "moving_geoms": [f"dom{i + 1}g"],
                "support_geoms": [f"dom{i}g"],
                "surface_point_local": np.array([+hx, 0.0, 0.0]),
                "surface_normal_local": np.array([+1.0, 0.0, 0.0]),
                "contact_point_local": np.array([-hx, 0.0, 0.0]),
                "shape": "box",
            }
        )

    build = {
        "bodies": bodies,
        # Brief settle so the dominoes seat on the floor (gravity squashes the contact to
        # equilibrium) before the shove, so the recorded window opens from clean STATIC.
        "settle": 0.15,
        "launch": launch,
        "duration": 1.6,
        "edges": edges,
        "meta": {
            "story": (
                "%d-domino cascade (THEORY.md s.6): the first is shoved and topples into "
                "the next, a chain of impacts/topples propagating along the row." % _N_DOM
            ),
        },
    }
    return model, build


# ======================================================================================
# Scene 3: Newton's cradle (simplified)
# ======================================================================================

_NC_N = 5               # balls in the cradle (the classic five)
_NC_R = 0.035           # ball radius (m)
_NC_GAP = 0.0008        # tiny gap between adjacent hanging balls (m): keeps the strikes
                        # SEQUENTIAL pairwise (not one mushy simultaneous contact) so the
                        # impulse propagates cleanly down the line.
_NC_L = 0.22            # pendulum rod length (m)
_NC_BALL_Z = 0.28       # rest height of the hanging balls (m) -- suspended, well clear of any floor
_NC_LIFT = 0.55         # angle (rad ~31 deg) the end ball is lifted to, then RELEASED FROM REST


def _build_newtons_cradle() -> tuple[mujoco.MjModel, dict]:
    """A real SUSPENDED Newton's cradle (THEORY.md s.6: impacts propagating through a line).

    Physics. N equal balls hang from a fixed top bar on hinge pendulums, in a line with a
    hair of clearance between neighbours. The end ball is LIFTED to a modest angle and
    RELEASED FROM REST; it swings down and strikes the line in a near-central IMPACT, the
    impulse propagates ball-to-ball, and the FAR ball swings out while the struck end stops
    and the middle balls stay put -- the signature of the cradle. (Verified: at the first
    transfer the end balls move ~118 mm, the middle ~11 mm.)

    Two corrections over earlier attempts, both load-bearing:

    * *Lift, don't launch (the whirl bug).* An earlier hinge version LAUNCHED a ball with
      velocity, which drove the pendulums into full rotations over the top. Releasing from
      REST keeps every swing a proper pendulum arc. And the lift sign matters: +angle about
      +y tips the end ball AWAY from the line; the wrong sign swings it INTO its neighbour at
      t=0 and explodes the line.
    * *Sequential, samplable strikes.* Tiny inter-ball gaps keep the collisions SEQUENTIAL
      pairwise (not one mushy simultaneous contact), an elastic-but-not-instant ball contact
      (``solref="0.012 0.15"``) transfers momentum cleanly, and ``record_hz = 400`` samples
      each brief strike across several frames.

    Edges (THEORY.md s.8): the adjacent ball<->ball contacts ``b{i}_b{i+1}``; each tracks the
    moving ball's (rotation-invariant) CENTER against a plane 2R inboard of the support ball's
    centre (the balls are near-vertical at each central strike, so that pole aligns). Truth
    labels are geom-based (THEORY.md s.9).
    """
    R = _NC_R
    spacing = 2.0 * R + _NC_GAP
    H = _NC_BALL_Z + _NC_L              # pivot height (the pivots sit on a fixed top bar)
    bar_cx = (_NC_N - 1) * spacing / 2.0

    def _pendulum(i: int) -> str:
        # A hinge pendulum (swings in the x-z plane about +y) with a thin non-colliding rod
        # and a ball at the bottom. The ball body's world pose is recorded for the edges.
        x = i * spacing
        return (
            f'<body name="p{i}" pos="{x} 0 {H}">'
            f'  <joint name="h{i}" type="hinge" axis="0 1 0" damping="0.00003"/>'
            f'  <geom type="capsule" fromto="0 0 0 0 0 {-_NC_L}" size="0.0035" density="40" '
            f'        contype="0" conaffinity="0" rgba="0.75 0.78 0.82 1"/>'
            f'  <body name="b{i}" pos="0 0 {-_NC_L}">'
            f'    <geom name="b{i}g" type="sphere" size="{R}" density="2400" '
            f'          friction="0.4 0.005 0.0001" solref="0.012 0.15"/>'
            f'  </body>'
            f'</body>'
        )

    pend_xml = "\n    ".join(_pendulum(i) for i in range(_NC_N))
    xml = f"""
<mujoco>
  {_options()}
  <worldbody>
    <geom name="bar" type="box" size="{bar_cx + 0.04} 0.012 0.012" pos="{bar_cx} 0 {H}"
          contype="0" conaffinity="0" rgba="0.28 0.30 0.36 1"/>
    {pend_xml}
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    h0_qadr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "h0")])

    def launch(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # LIFT the end ball to _NC_LIFT and RELEASE FROM REST (qvel = 0): it swings DOWN as a
        # pendulum and strikes the line. It does NOT whirl over the top -- that was the old
        # bug, which LAUNCHED a ball with velocity. The impulse propagates ball-to-ball (the
        # tiny gaps keep the strikes sequential) and the FAR ball swings out: the cradle.
        d.qvel[:] = 0.0
        # +angle about +y tips b0's rod toward -x, i.e. AWAY from the line (which extends +x);
        # the negative sign would swing it INTO b1 at t=0 and explode the line.
        d.qpos[h0_qadr] = +_NC_LIFT
        mujoco.mj_forward(m, d)

    # Phase-1 sphere-sphere geometry (DESIGN.md III.5) for the ball<->ball edges. Lazy import
    # for the same reason as in _build_two_balls_collide: keep this module's module-load-time
    # deps just mujoco/numpy (the documented cycle-free contract); geometry_resolvers is only
    # imported when a builder runs, long after all modules have loaded.
    from .geometry_resolvers import SphereSphere

    # Adjacent ball<->ball edges: the impulse propagates along these. Each edge resolves with
    # SphereSphere so the contact normal is the line-of-centres (c_{i+1} - c_i)/||.||, NOT a
    # quat-carried body-fixed +x pole (DESIGN.md II.D): a spinning ball would whirl that pole and
    # fabricate phantom closing velocities. The plane surface below is retained for the
    # FlatRegion fallback / broad-phase, but observe() uses the SphereSphere geometry.
    edges = []
    for i in range(_NC_N - 1):
        edges.append(
            {
                "edge_id": f"b{i}_b{i + 1}",
                "moving_body": f"b{i + 1}",
                "support_body": f"b{i}",
                "moving_geoms": [f"b{i + 1}g"],
                "support_geoms": [f"b{i}g"],
                "surface_point_local": np.array([2.0 * R, 0.0, 0.0]),
                "surface_normal_local": np.array([1.0, 0.0, 0.0]),
                "contact_point_local": np.zeros(3),
                "shape": "sphere",
                "geometry": SphereSphere(_NC_R, _NC_R),
            }
        )

    build = {
        "bodies": [f"b{i}" for i in range(_NC_N)],
        # Brief settle so the hanging line comes fully to rest before the end ball is lifted.
        "settle": 0.15,
        "launch": launch,
        "duration": 2.5,
        # Sample fast: each ball-ball strike is brief; its sub-frame normal-velocity peak would
        # otherwise fall between recorded frames, leaving the edge a degenerate single frame.
        "record_hz": 400.0,
        "edges": edges,
        "meta": {
            "story": (
                "%d-ball Newton's cradle (THEORY.md s.6): the end ball is lifted and released "
                "FROM REST; it swings down, strikes the suspended line, and the impulse "
                "propagates ball-to-ball so the FAR ball swings out while the middle stays put."
                % _NC_N
            ),
        },
    }
    return model, build


# --------------------------------------------------------------------------------------
# Registries (the builder contract: module-level dicts mapping name -> builder).
# --------------------------------------------------------------------------------------

#: No single-pair scenarios in this module -- everything here is a multi-body SCENE.
SCENARIO_BUILDERS: dict[str, callable] = {}

#: Multi-body scenes with body-to-body collisions / chained impacts (THEORY.md s.8).
SCENE_BUILDERS: dict[str, callable] = {
    "two_balls_collide": _build_two_balls_collide,
    "dominoes": _build_dominoes,
    "newtons_cradle": _build_newtons_cradle,
}
