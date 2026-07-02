"""The founding multi-body scenes of THEORY.md §8 — the contact graph made concrete.

`person_on_skateboard` is the repo's headline: the person↔board edge must read STATIC while
both bodies scream across the world (§1's relative-frame payoff on a graph edge), and the
wheeled board↔ground edge is genuinely ROLLING. `box_on_two_blocks` is the minimal
active-set *change*: {L, R} → {L} when one support is lowered. Stacks/hand-offs and chained
impacts live in the sibling modules (`scenes_stacks`, `scenes_chains`).

Each builder compiles a MuJoCo model and returns ``(model, build)`` with the scene contract
(`bodies`, `edges`, optional settle/launch/forcing) consumed by
`oracle.factory.generate_scene`. Builders self-register via `oracle.registry.scene`.
"""

from __future__ import annotations

import numpy as np

import mujoco

from oracle._mjcf import obj_id as _id, options as _common_options
from oracle.registry import scene


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


@scene("person_on_skateboard")
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


@scene("box_on_two_blocks")
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
