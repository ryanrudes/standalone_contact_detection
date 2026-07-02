"""Multi-body SCENES exercising stacking and multi-surface hand-off (THEORY.md s.8).

These are additional multi-body scene demos focused on the
*contact graph* and, above all, on the **active set as the hidden structure** we infer
(THEORY.md s.8). A single body pair has one edge; the interesting graph signal is

* several edges active *simultaneously* (a stack: each box rests on the one below), and
* the active set *changing in time* (a box toppling off the top of a stack, or sliding
  off a table, going airborne, then landing on the floor -- a multi-surface hand-off).

All three scenes are built with ONLY ``mujoco`` + ``numpy`` (plus the scene contract
keys), registered via ``oracle.registry.scene`` (a leaf that imports nothing) without an
import cycle: it imports no ``contact`` submodule. The simulate/label machinery in
``oracle.factory`` (``_simulate_scene`` / ``generate_scene`` / ``_edge_frame_truth``) does all
the physics extraction; we only build the model + the build-dict.

The mode of every active edge here is STATIC (resting boxes do not slide/roll relative to
their support while seated) or, transiently, IMPACT (a falling box striking the floor): the
point of these scenes is the *structure* (which edges are active and when), not per-edge
mode richness -- exactly the structure-inference target of THEORY.md s.8.
"""

from __future__ import annotations

import numpy as np

import mujoco

from ._mjcf import body_id as _bid, options as _common_options

from oracle.registry import scene

# Imports only ``mujoco``/``numpy``, the leaf ``oracle._mjcf`` helpers, and the registry
# leaf the builders below self-register into — all cycle-free.


# Shared box half-extent for the stacked boxes (m). Bottom-face center material point of a
# box centered at its body origin is [0, 0, -_BOX_HALF]; top face is at +_BOX_HALF.
_BOX_HALF = 0.06


# --------------------------------------------------------------------------------------
# Scene 1: a stable three-box stack -> several simultaneous STATIC edges.
# --------------------------------------------------------------------------------------

@scene("stacked_boxes")
def _build_stacked_boxes() -> tuple[mujoco.MjModel, dict]:
    """Three equal boxes stacked on the floor, resting stably (THEORY.md s.8).

    Physics: box1 rests on the floor, box2 on box1, box3 on box2, all CoMs aligned over a
    wide common footprint with high friction. Gravity loads each interface; nothing slides
    or tips, so all three edges are sustained STATIC contacts with zero relative twist
    (THEORY.md s.3). The point of the scene is the *graph*: three candidate edges that are
    all active at once for the whole run (a small but genuine simultaneous active set).

    Stability comes from (a) perfectly aligned centers of mass (each box directly above
    the one below, so the gravity line stays inside every support polygon), (b) generous
    friction (mu = 1.0, so no tendency to slide), and (c) a short settle before t = 0 so
    the recorded window opens from a quiet equilibrium rather than a touchdown bounce.

    Edges (each box on the one below; box1 on the world floor):
      ``box1_floor`` : box1 <-> world floor.
      ``box2_box1``  : box2 <-> box1.
      ``box3_box2``  : box3 <-> box2.
    """
    h = _BOX_HALF
    # Stack the boxes face-to-face with a hair of overlap (-1 mm) so each interface is a
    # snug load-bearing contact from the first frame (gap ~ 0, real normal force) rather
    # than a few-mm standoff that would read as "not in contact".
    eps = 0.001
    z1 = h                      # box1 center: bottom face on the floor (z=0)
    z2 = z1 + 2.0 * h - eps     # box2 center: bottom face on box1 top
    z3 = z2 + 2.0 * h - eps     # box3 center: bottom face on box2 top
    fr = "1.0 0.02 0.001"       # high friction so the stack cannot creep/slide

    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0" friction="{fr}"/>
    <body name="box1" pos="0 0 {z1}">
      <freejoint name="box1j"/>
      <geom name="box1g" type="box" size="{h} {h} {h}" density="500" friction="{fr}"/>
    </body>
    <body name="box2" pos="0 0 {z2}">
      <freejoint name="box2j"/>
      <geom name="box2g" type="box" size="{h} {h} {h}" density="500" friction="{fr}"/>
    </body>
    <body name="box3" pos="0 0 {z3}">
      <freejoint name="box3j"/>
      <geom name="box3g" type="box" size="{h} {h} {h}" density="500" friction="{fr}"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)

    n_up = np.array([0.0, 0.0, 1.0])
    build = {
        "bodies": ["box1", "box2", "box3"],
        # Let the stack settle into a quiet equilibrium before recording (the clock is
        # reset to 0 after settle), so the recorded window is pure sustained STATIC.
        "settle": 0.5,
        "duration": 1.5,
        "edges": [
            {
                "edge_id": "box1_floor",
                "moving_body": "box1",
                "support_body": "world",
                "moving_geoms": ["box1g"],
                "support_geoms": ["floor"],
                "surface_point_local": np.array([0.0, 0.0, 0.0]),  # floor z=0 (world)
                "surface_normal_local": n_up,
                "contact_point_local": np.array([0.0, 0.0, -h]),   # box1 bottom-face center
                "shape": "box",
            },
            {
                "edge_id": "box2_box1",
                "moving_body": "box2",
                "support_body": "box1",
                "moving_geoms": ["box2g"],
                "support_geoms": ["box1g"],
                "surface_point_local": np.array([0.0, 0.0, h]),    # box1 TOP face (box1-local)
                "surface_normal_local": n_up,
                "contact_point_local": np.array([0.0, 0.0, -h]),   # box2 bottom-face center
                "shape": "box",
            },
            {
                "edge_id": "box3_box2",
                "moving_body": "box3",
                "support_body": "box2",
                "moving_geoms": ["box3g"],
                "support_geoms": ["box2g"],
                "surface_point_local": np.array([0.0, 0.0, h]),    # box2 TOP face (box2-local)
                "surface_normal_local": n_up,
                "contact_point_local": np.array([0.0, 0.0, -h]),   # box3 bottom-face center
                "shape": "box",
            },
        ],
        "meta": {
            "structure": (
                "All three edges active simultaneously for the whole run: a stable "
                "STATIC stack (THEORY.md s.8 -- a small fixed active set over the graph)."
            ),
        },
    }
    return model, build


# --------------------------------------------------------------------------------------
# Scene 2: a small stack whose TOP box is pushed off -> a CHANGING active set + impact.
# --------------------------------------------------------------------------------------

@scene("stack_topple")
def _build_stack_topple() -> tuple[mujoco.MjModel, dict]:
    """A 2-box stack whose TOP slab is shoved off, slides off the edge, and hits the floor.

    Physics: box1 (a cube) rests on the floor and box2 (a thin, wide SLAB) rests on box1,
    both STATIC. After a settle, a sustained horizontal world force (``xfrc_applied``) on
    the TOP slab overcomes friction and walks it off the edge of box1; once its CoM passes
    box1's edge it slides off, deactivating the ``box2_box1`` edge, free-falls (FREE), and
    strikes the floor -- an IMPACT on the new ``box2_floor`` edge (THEORY.md s.6) -- then
    settles flat (STATIC). box1 is heavy and stays put, so ``box1_floor`` is STATIC
    throughout.

    Why box2 is a thin wide SLAB (half-height 0.02, footprint 0.08 x 0.08) and not a cube,
    and why the push is gentle (0.8x weight) and brief (0.3 s): a cube shoved hard off an
    edge TUMBLES -- it lands rotated ~90 deg onto a side face, so the tracked body-fixed
    "bottom-face center" material point (the only thing the kinematic detector follows) is
    no longer the contacting point and the ``box2_floor`` edge reads as a ~half-extent gap
    forever (the detector cannot confirm the landing even though MuJoCo's geom truth sees
    it). A flat, tip-resistant slab pushed just hard enough to clear the edge slides off
    TRANSLATING and lands flat on its large bottom face, so the tracked point stays the
    contact point and the landing is cleanly observable (THEORY.md s.3: rolling/landing is
    a property of the tracked material point). This keeps the demo's named phenomenon -- a
    CHANGING active set with a landing impact -- both true AND detectable.

    This is the changing-active-set test of THEORY.md s.8: the true active structure goes
    ``{box1_floor, box2_box1}`` -> (box2 airborne) ``{box1_floor}`` -> (box2 landed)
    ``{box1_floor, box2_floor}``. box1's heavy mass + the floor-level box1 keep the lower
    interface a clean sustained STATIC contact while box2's edges flip.

    Edges:
      ``box1_floor`` : box1 <-> world floor -- active the whole run (STATIC).
      ``box2_box1``  : box2 <-> box1 -- active early, DEACTIVATES when box2 is pushed off.
      ``box2_floor`` : box2 <-> world floor -- INACTIVE early, ACTIVATES (with an impact)
                       after box2 falls.
    """
    h = _BOX_HALF                              # box1 cube half-extent
    slab = np.array([0.08, 0.08, 0.02])        # box2 half-extents: wide + thin (tip-resistant)
    sz = slab[2]                               # box2 half-height (its tracked bottom point z)
    eps = 0.001
    z1 = h                       # box1 bottom on floor
    z2 = 2.0 * h + sz - eps      # box2 slab bottom on box1 top
    fr = "0.6 0.02 0.001"        # moderate friction: a steady push can slide the slab off

    # box1 is dense/heavy so it does not budge under the push that moves the light slab.
    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0" friction="{fr}"/>
    <body name="box1" pos="0 0 {z1}">
      <freejoint name="box1j"/>
      <geom name="box1g" type="box" size="{h} {h} {h}" density="6000" friction="{fr}"/>
    </body>
    <body name="box2" pos="0 0 {z2}">
      <freejoint name="box2j"/>
      <geom name="box2g" type="box" size="{slab[0]} {slab[1]} {slab[2]}"
            density="500" friction="{fr}"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    box2_id = _bid(model, "box2")
    weight2 = float(model.body_mass[box2_id]) * 9.81

    def forcing(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Settle quietly for 0.4 s (the clock has been reset to 0 after the build's
        # pre-record settle, so this is measured from the recorded window start), then push
        # the TOP slab steadily in +x with ~0.8x its weight for a brief 0.3 s window -- just
        # enough (above the friction limit, mu=0.6) to slide it off box1's edge so gravity
        # then carries it flat to the floor. A gentle, brief push (vs. a hard sustained one)
        # is deliberate: it makes the slab slide off TRANSLATING and land flat (rather than
        # tumbling or being flung across the room), so the tracked bottom point stays the
        # contact point and the landing is observable. Applied to box2 only; box1 (heavy)
        # feels nothing and stays seated.
        d.xfrc_applied[box2_id, :] = 0.0
        if 0.4 < d.time < 0.7:
            d.xfrc_applied[box2_id, 0] = 0.8 * weight2

    n_up = np.array([0.0, 0.0, 1.0])
    build = {
        "bodies": ["box1", "box2"],
        "settle": 0.4,
        "duration": 2.0,
        "forcing": forcing,
        "edges": [
            {
                "edge_id": "box1_floor",
                "moving_body": "box1",
                "support_body": "world",
                "moving_geoms": ["box1g"],
                "support_geoms": ["floor"],
                "surface_point_local": np.array([0.0, 0.0, 0.0]),
                "surface_normal_local": n_up,
                "contact_point_local": np.array([0.0, 0.0, -h]),
                "shape": "box",
            },
            {
                "edge_id": "box2_box1",
                "moving_body": "box2",
                "support_body": "box1",
                "moving_geoms": ["box2g"],
                "support_geoms": ["box1g"],
                "surface_point_local": np.array([0.0, 0.0, h]),    # box1 TOP face (box1-local)
                "surface_normal_local": n_up,
                "contact_point_local": np.array([0.0, 0.0, -sz]),  # slab bottom-face center
                "shape": "box",
            },
            {
                "edge_id": "box2_floor",
                "moving_body": "box2",
                "support_body": "world",
                "moving_geoms": ["box2g"],
                "support_geoms": ["floor"],
                "surface_point_local": np.array([0.0, 0.0, 0.0]),
                "surface_normal_local": n_up,
                "contact_point_local": np.array([0.0, 0.0, -sz]),  # slab bottom-face center
                "shape": "box",
            },
        ],
        "meta": {
            "active_set_change": (
                "{box1_floor, box2_box1} -> {box1_floor} (box2 airborne) -> "
                "{box1_floor, box2_floor} (box2 lands, an impact). box1 stays put "
                "(THEORY.md s.8 changing active set + s.6 impact)."
            ),
        },
    }
    return model, build


# --------------------------------------------------------------------------------------
# Scene 3 (showcase): a box slides off a table, falls, and lands on the floor.
# --------------------------------------------------------------------------------------

@scene("box_off_table")
def _build_box_off_table() -> tuple[mujoco.MjModel, dict]:
    """A flat slab on a raised TABLE is pushed off the edge, falls, and lands FLAT on the FLOOR.

    Physics: a thin wide slab rests STATIC on a fixed raised block (the "table"). After a
    settle a steady horizontal world force overcomes friction and slides the slab toward the
    table edge; once its CoM clears the edge it slides off, free-falls (FREE), and lands on
    the floor below -- a sustained STATIC contact after the landing transient. This is the
    showcase MULTI-SURFACE hand-off of THEORY.md s.8: the SAME moving body's active edge
    migrates from one support to another, with an airborne gap in between, so the active
    set is ``{box_table} -> {} -> {box_floor}``.

    Why a thin wide SLAB (half 0.08 x 0.08 x 0.02) and a GENTLE, BRIEF push (0.8x weight over
    0.3 s) -- the same tip-resistance reasoning as ``stack_topple``, and the correction of the
    earlier version. A CUBE shoved off a table edge TUMBLES end-over-end and SKIDS/bounces
    across the floor (the earlier box_off_table travelled >3 m and flipped multiple times), so
    its tracked body-fixed "bottom-face center" material point is no longer the contacting
    point on landing: ``box_floor`` then chatters across many disjoint intervals and never
    settles to the documented sustained STATIC. A flat, tip-resistant slab pushed just hard
    enough to clear the edge slides off TRANSLATING and lands FLAT on its large bottom face, so
    the tracked point stays the contact point: ``box_floor`` activates with a short landing
    transient and then a single sustained STATIC interval (THEORY.md s.3: landing flat is a
    property of the tracked material point). This keeps the named phenomenon -- a clean
    multi-surface hand-off ending in a flat rest -- both true AND detectable.

    Geometry is tuned so the hand-off is clean and observable: the table is tall enough
    (0.30 m top) that the fall is long and unambiguous (a big z drop the detector cannot
    miss), the table is short in x so the slab reaches the edge quickly, and the floor is
    far below the table top so ``box_floor`` is decisively inactive while the slab is on the
    table (a real gap, not a sub-mm standoff).

    Edges:
      ``box_table`` : slab <-> table -- active first, DEACTIVATES when the slab slides off.
      ``box_floor`` : slab <-> world floor -- INACTIVE first, ACTIVATES after the slab lands.
    """
    # box2... the moving body is a thin wide SLAB (tip-resistant), mirroring stack_topple.
    slab = np.array([0.08, 0.08, 0.02])         # box half-extents: wide + thin (tip-resistant)
    sz = slab[2]                                 # slab half-height (its tracked bottom point z)
    # Table (a fixed raised block). Short in x so the slab reaches the edge quickly; the top
    # face sits at z = table_top. The slab starts toward the -x side of the table.
    table_half = np.array([0.12, 0.20, 0.15])  # half-extents (m); top at 2*0.15 = 0.30 m
    table_top = 2.0 * table_half[2]             # table top face world z (origin at +half_z)
    fr = "0.4 0.02 0.001"                        # low-ish friction so a modest push slides it off

    eps = 0.001
    box_z = table_top + sz - eps                 # slab bottom snug on the table top
    # Start the slab toward the -x side of the table so the +x push has room to accelerate it
    # before it reaches the +x edge (a cleaner slide-then-fall than starting at the edge).
    box_x0 = -table_half[0] + slab[0] + 0.01

    xml = f"""
<mujoco>
  {_common_options()}
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0" friction="{fr}"/>
    <body name="table" pos="0.30 0 {table_half[2]}">
      <geom name="tableg" type="box" size="{table_half[0]} {table_half[1]} {table_half[2]}"
            density="5000" friction="{fr}"/>
    </body>
    <body name="box" pos="{0.30 + box_x0} 0 {box_z}">
      <freejoint name="boxj"/>
      <geom name="boxg" type="box" size="{slab[0]} {slab[1]} {slab[2]}"
            density="400" friction="{fr}"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    box_id = _bid(model, "box")
    weight = float(model.body_mass[box_id]) * 9.81

    def forcing(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        # Settle quietly, then push the slab steadily in +x with ~0.8x its weight (above the
        # friction limit, mu=0.4) for a brief 0.3 s window -- just enough to slide it across
        # the short table and off the +x edge. A gentle, brief push (vs the old hard 1.2x
        # sustained one) is deliberate: it makes the slab slide off TRANSLATING and land flat,
        # rather than tumbling and skidding across the room. Stop pushing at 0.9 s -- by then
        # the slab is at/over the edge and gravity finishes the job, so the airborne phase and
        # landing are pure ballistics (no force smearing the FREE / landing-impact labels).
        d.xfrc_applied[box_id, :] = 0.0
        if 0.6 < d.time < 0.9:
            d.xfrc_applied[box_id, 0] = 0.8 * weight

    n_up = np.array([0.0, 0.0, 1.0])
    build = {
        "bodies": ["box", "table"],
        "settle": 0.3,
        "duration": 2.2,
        "forcing": forcing,
        "edges": [
            {
                "edge_id": "box_table",
                "moving_body": "box",
                "support_body": "table",
                "moving_geoms": ["boxg"],
                "support_geoms": ["tableg"],
                # Table TOP face, in the TABLE-local frame (origin at table center).
                "surface_point_local": np.array([0.0, 0.0, table_half[2]]),
                "surface_normal_local": n_up,
                "contact_point_local": np.array([0.0, 0.0, -sz]),  # slab bottom-face center
                "shape": "box",
            },
            {
                "edge_id": "box_floor",
                "moving_body": "box",
                "support_body": "world",
                "moving_geoms": ["boxg"],
                "support_geoms": ["floor"],
                "surface_point_local": np.array([0.0, 0.0, 0.0]),  # floor z=0 (world)
                "surface_normal_local": n_up,
                "contact_point_local": np.array([0.0, 0.0, -sz]),   # slab bottom-face center
                "shape": "box",
            },
        ],
        "meta": {
            "hand_off": (
                "{box_table} -> {} (airborne) -> {box_floor}: a multi-surface hand-off "
                "of the same body across an airborne gap (THEORY.md s.8). The table top "
                "is 0.30 m up so the fall is long and the active-set change is decisive."
            ),
        },
    }
    return model, build


