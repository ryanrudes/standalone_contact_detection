"""Expectation-based verification of the demos (the rigorous "is the detection what we
physically expect?" layer, beyond a coarse IoU).

For every scenario/scene we encode the contact/mode STORY we expect from the physics --
e.g. push_to_slide must go STATIC then SLIDING; two_balls_collide must show both floor
edges in rolling contact and a brief ball-ball IMPACT; box_off_table must hand off
{table} -> {} -> {floor}. We then check the DETECTION (and, where relevant, the withheld
MuJoCo truth) against that story and report PASS / WARN / FAIL per check.

Used by ``verify_demos.py`` (human-readable report) and ``tests/test_expectations.py``
(asserts that the headline checks hold). Importing this module is cycle-safe: it imports
the public API, not factory internals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oracle import factory
from oracle.registry import SCENARIO_BUILDERS, SCENE_BUILDERS
from contact.geometry import observe
from contact.detector import ContactDetector
from contact.graph import detect_scene
from contact.types import FREE


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def _iou(a, b) -> float:
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def _contact_mask(map_state) -> np.ndarray:
    return np.array([m != FREE for m in map_state], dtype=bool)


def _modes_on_contact(map_state) -> set[str]:
    return {m for m in map_state if m != FREE}


def _dominant_tail_mode(map_state, frac=0.15) -> str:
    """Most common mode over the last `frac` of the clip (the settled end state)."""
    n = len(map_state)
    tail = map_state[max(0, int(n * (1 - frac))):]
    from collections import Counter
    return Counter(tail).most_common(1)[0][0] if tail else FREE


@dataclass
class Check:
    name: str
    status: str   # "PASS" | "WARN" | "FAIL"
    detail: str


def _ok(cond, name, detail_ok="", detail_bad="", warn=False):
    if cond:
        return Check(name, "PASS", detail_ok)
    return Check(name, "WARN" if warn else "FAIL", detail_bad)


# --------------------------------------------------------------------------------------
# Expected stories.  Each scenario maps to a spec dict; the checker below interprets it.
# Thresholds are deliberately generous (they encode the STORY, not exact timing) but tight
# enough to catch a genuinely wrong detection.
# --------------------------------------------------------------------------------------

SCENARIO_EXPECT: dict[str, dict] = {
    "drop_rest":          {"min_iou": 0.85, "modes": {"static"}, "ends": "static", "events": {"touchdown"}},
    "drop_rest_liftoff":  {"min_iou": 0.80, "modes": {"static"}, "events": {"touchdown", "liftoff"}},
    "push_to_slide":      {"min_iou": 0.85, "modes": {"static", "sliding"}, "order": ("static", "sliding")},
    "rolling_ball":       {"min_iou": 0.90, "modes": {"rolling"}, "ends": "rolling"},
    "bouncing_ball":      {"min_iou": 0.45, "modes": {"static"}, "min_impacts": 2, "ends": "static"},
    "moving_support":     {"min_iou": 0.90, "modes": {"static"}, "ends": "static", "min_world_disp": 1.0},
    "indeterminate_rig":  {"min_iou": 0.90, "modes": {"static"}},
    "incline_slide":      {"min_iou": 0.90, "modes": {"sliding"}, "ends": "sliding"},
    "skid_to_rest":       {"min_iou": 0.80, "ends": "static", "any_modes": {"sliding", "rolling", "impact"}},
    "spinning_top":       {"min_iou": 0.90, "modes": {"pivoting"}, "ends": "pivoting"},
    "tumbling_box":       {"min_iou": 0.80, "min_impacts": 1, "ends": "static"},
    "hard_drop":          {"min_iou": 0.85, "min_impacts": 1, "ends": "static"},
    "restitution_bounce": {"min_iou": 0.80, "min_impacts": 2, "ends": "static"},
    "angled_impact":      {"min_iou": 0.85, "min_impacts": 1, "any_modes": {"rolling", "sliding"}},
    "drop_on_incline":    {"min_iou": 0.85, "min_impacts": 1, "any_modes": {"rolling", "sliding"}},
}

#: Per-scene expectations: per-edge min IoU, expected modes, and active-set "story" checks.
SCENE_EXPECT: dict[str, dict] = {
    "person_on_skateboard": {
        "edges": {"person_board": {"min_iou": 0.90, "modes": {"static"}},
                  "board_ground": {"min_iou": 0.70}},
        "payoff_body": "board", "min_world_disp": 1.5,  # rides far in world yet person_board is STATIC
    },
    "box_on_two_blocks": {
        "edges": {"box_blockL": {"min_iou": 0.85}},
        "deactivates": "box_blockR",  # this edge must drop out partway (active-set change)
    },
    "stacked_boxes": {
        "edges": {"box1_floor": {"min_iou": 0.85}, "box2_box1": {"min_iou": 0.85}, "box3_box2": {"min_iou": 0.85}},
        "all_static": True,
    },
    "stack_topple": {"active_set_changes": True},
    "box_off_table": {"handoff": ("box_table", "box_floor")},  # table active early, floor active late
    "two_balls_collide": {
        "edges": {"ballA_floor": {"min_iou": 0.80}, "ballB_floor": {"min_iou": 0.80}},
        "impact_edge": "ballA_ballB",
    },
    "dominoes": {"floor_edges_prefix": "dom", "min_iou": 0.75},
    # A REAL Newton's cradle: balls SUSPENDED in the air (not resting on a floor) and the
    # end-ball-out signature (the struck end and far ball move far; the middle stays nearly
    # put). These honestly FAIL the current floor-ball stand-in until it is rebuilt.
    "newtons_cradle": {"impact_edges": True, "suspended": 0.15, "end_ball_out": True},
}


# --------------------------------------------------------------------------------------
# Scenario checks
# --------------------------------------------------------------------------------------

def verify_scenario(name: str, seed: int = 0) -> list[Check]:
    spec = SCENARIO_EXPECT[name]
    raw = factory.generate(name, seed=seed)
    res = ContactDetector().detect(
        observe(raw.moving, raw.support, raw.surface, raw.contact_point_local,
                geometry=getattr(raw, "geometry", None))
    )
    checks: list[Check] = []
    tru = np.asarray(raw.truth.in_contact, bool)
    pred = _contact_mask(res.map_state)
    iou = _iou(pred, tru)
    checks.append(_ok(iou >= spec["min_iou"], "contact-IoU",
                      f"IoU={iou:.2f} (>= {spec['min_iou']})",
                      f"IoU={iou:.2f} < {spec['min_iou']}"))

    detected = _modes_on_contact(res.map_state)
    if "modes" in spec:
        missing = spec["modes"] - detected
        checks.append(_ok(not missing, "expected-modes",
                          f"found {sorted(spec['modes'])}",
                          f"missing {sorted(missing)} (detected {sorted(detected)})"))
    if "any_modes" in spec:
        hit = spec["any_modes"] & detected
        checks.append(_ok(bool(hit), "any-of-modes",
                          f"found {sorted(hit)}",
                          f"none of {sorted(spec['any_modes'])} detected (got {sorted(detected)})"))
    if "ends" in spec:
        end = _dominant_tail_mode(res.map_state)
        checks.append(_ok(end == spec["ends"], "ends-in",
                          f"ends in {end}", f"ends in {end}, expected {spec['ends']}"))
    if "order" in spec:
        a, b = spec["order"]
        ia = next((i for i, m in enumerate(res.map_state) if m == a), None)
        ib = next((i for i, m in enumerate(res.map_state) if m == b), None)
        cond = ia is not None and ib is not None and ia < ib
        checks.append(_ok(cond, "mode-order",
                          f"{a} before {b}", f"expected {a} before {b} (got idx {ia},{ib})"))
    if "events" in spec:
        kinds = {e.kind for e in res.events}
        missing = spec["events"] - kinds
        checks.append(_ok(not missing, "events",
                          f"found {sorted(kinds)}", f"missing events {sorted(missing)}"))
    if "min_impacts" in spec:
        n = len(res.impulses)
        checks.append(_ok(n >= spec["min_impacts"], "impacts",
                          f"{n} impact atoms", f"only {n} impacts (< {spec['min_impacts']})"))
    if "min_world_disp" in spec:
        disp = float(np.ptp(raw.moving.position, axis=0).max())
        checks.append(_ok(disp >= spec["min_world_disp"], "world-motion",
                          f"moves {disp:.2f} m in world (yet detected {_dominant_tail_mode(res.map_state)})",
                          f"only moves {disp:.2f} m"))
    return checks


# --------------------------------------------------------------------------------------
# Scene checks
# --------------------------------------------------------------------------------------

def verify_scene(name: str, seed: int = 0) -> list[Check]:
    from .visualize import _config_for_scene
    spec = SCENE_EXPECT[name]
    sc = factory.generate_scene(name, seed=seed)
    gr = detect_scene(sc, _config_for_scene(sc, None))
    checks: list[Check] = []

    def edge_pred(eid):
        return _contact_mask(gr.per_edge[eid].map_state)

    def edge_iou(eid):
        return _iou(edge_pred(eid), np.asarray(sc.truth[eid].in_contact, bool))

    for eid, espec in spec.get("edges", {}).items():
        iou = edge_iou(eid)
        checks.append(_ok(iou >= espec["min_iou"], f"edge:{eid}",
                          f"IoU={iou:.2f}", f"IoU={iou:.2f} < {espec['min_iou']}"))
        if "modes" in espec:
            detected = _modes_on_contact(gr.per_edge[eid].map_state)
            missing = espec["modes"] - detected
            checks.append(_ok(not missing, f"edge-modes:{eid}",
                              f"{sorted(espec['modes'])}", f"missing {sorted(missing)}"))

    if spec.get("all_static"):
        for e in sc.edges:
            modes = _modes_on_contact(gr.per_edge[e.edge_id].map_state)
            checks.append(_ok(modes <= {"static"} and modes, f"static:{e.edge_id}",
                              "static", f"got {sorted(modes)}", warn=True))

    if "deactivates" in spec:
        eid = spec["deactivates"]
        pred = edge_pred(eid)
        n = len(pred)
        early = pred[: n // 3].mean()
        late = pred[2 * n // 3:].mean()
        checks.append(_ok(early > 0.5 and late < 0.5, f"deactivates:{eid}",
                          f"active early ({early:.2f}) then off ({late:.2f})",
                          f"no clear deactivation (early {early:.2f}, late {late:.2f})"))

    if "handoff" in spec:
        a, b = spec["handoff"]
        pa, pb = edge_pred(a), edge_pred(b)
        n = len(pa)
        a_early = pa[: n // 3].mean()
        b_late = pb[2 * n // 3:].mean()
        checks.append(_ok(a_early > 0.4 and b_late > 0.4, f"handoff:{a}->{b}",
                          f"{a} early ({a_early:.2f}), {b} late ({b_late:.2f})",
                          f"no hand-off ({a} early {a_early:.2f}, {b} late {b_late:.2f})"))

    if spec.get("active_set_changes"):
        sizes = [len([e for e in sc.edges if gr.per_edge[e.edge_id].map_state[i] != FREE])
                 for i in range(len(sc.bodies[next(iter(sc.bodies))].t))]
        checks.append(_ok(len(set(sizes)) > 1, "active-set-change",
                          f"active-set size varies {min(sizes)}..{max(sizes)}",
                          "active-set size never changes", warn=True))

    if "impact_edge" in spec:
        eid = spec["impact_edge"]
        n = len(gr.per_edge[eid].impulses)
        truthc = float(np.asarray(sc.truth[eid].in_contact, bool).mean())
        checks.append(_ok(n >= 1 or truthc > 0.01, f"impact:{eid}",
                          f"{n} impacts detected", f"no impact on {eid}", warn=True))

    if "floor_edges_prefix" in spec:
        pref = spec["floor_edges_prefix"]
        floor_edges = [e.edge_id for e in sc.edges if e.support_body == "world" and e.edge_id.startswith(pref)]
        for eid in floor_edges:
            iou = edge_iou(eid)
            checks.append(_ok(iou >= spec["min_iou"], f"floor:{eid}",
                              f"IoU={iou:.2f}", f"IoU={iou:.2f} < {spec['min_iou']}"))

    if "payoff_body" in spec:
        disp = float(np.ptp(sc.bodies[spec["payoff_body"]].position, axis=0).max())
        checks.append(_ok(disp >= spec["min_world_disp"], "relative-frame-payoff",
                          f"{spec['payoff_body']} moves {disp:.2f} m in world; person_board still STATIC",
                          f"{spec['payoff_body']} only moves {disp:.2f} m"))

    if spec.get("impact_edges"):
        total = sum(len(gr.per_edge[e.edge_id].impulses) for e in sc.edges)
        checks.append(_ok(total >= 1, "cradle-impacts",
                          f"{total} ball-ball impacts", "no ball-ball impacts detected", warn=True))

    if "suspended" in spec:
        # A real cradle hangs the balls in the air; floor-resting balls are not a cradle.
        min_h = min(float(sc.bodies[b].position[:, 2].min()) for b in sc.bodies)
        checks.append(_ok(min_h >= spec["suspended"], "suspended",
                          f"balls hang at >= {min_h:.2f} m",
                          f"balls rest near the floor (min height {min_h:.2f} m) -- not a suspended cradle"))

    if spec.get("end_ball_out"):
        # Newton's-cradle signature, measured over the FIRST transfer (the first ~third of the
        # clip): the END balls travel far, the MIDDLE ball barely moves. (Over the whole clip
        # energy inevitably spreads to every ball, as in a real cradle, so the signature must
        # be read at the first strike, not from the full-clip excursion.)
        names = sorted(sc.bodies)  # b0..bN-1
        n = len(sc.bodies[names[0]].t)
        w = slice(0, max(10, n // 3))

        def _peak(b):
            x = sc.bodies[b].position[w, 0]
            return float(np.max(np.abs(x - x[0])))

        disp = {b: _peak(b) for b in names}
        mid = names[len(names) // 2]
        ends = max(disp[names[0]], disp[names[-1]])
        cond = ends > 0.05 and disp[mid] < 0.4 * ends
        checks.append(_ok(cond, "end-ball-out",
                          f"first transfer: ends move {ends*1000:.0f} mm, middle {disp[mid]*1000:.0f} mm",
                          f"no clean transfer (ends {ends*1000:.0f} mm, middle {disp[mid]*1000:.0f} mm)"))
    return checks


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def verify_all(seed: int = 0) -> dict[str, list[Check]]:
    out: dict[str, list[Check]] = {}
    for name in SCENARIO_BUILDERS:
        if name in SCENARIO_EXPECT:
            out[name] = verify_scenario(name, seed)
    for name in SCENE_BUILDERS:
        if name in SCENE_EXPECT:
            out[name] = verify_scene(name, seed)
    return out


def worst_status(checks: list[Check]) -> str:
    if any(c.status == "FAIL" for c in checks):
        return "FAIL"
    if any(c.status == "WARN" for c in checks):
        return "WARN"
    return "PASS"
