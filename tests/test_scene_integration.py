"""End-to-end validation of the multi-body contact-graph detector (THEORY.md s.8).

This is rung 5 of the pragmatic ladder (THEORY.md s.10) exercised as a whole. For each
MuJoCo *scene* we run the exact s.8/s.9 workflow ---

    mujoco_gen.generate_scene  ->  graph.detect_scene  ->  report.score (per edge)

--- handing the detector only the observable channel (noisy body poses) and scoring its
inferred *joint active-set structure* and per-edge contact decisions against the withheld
simulator truth. Each scene asserts the specific theoretical claim it was built to stress:

* ``person_on_skateboard`` -- the s.1/s.8 relative-frame + graph payoff: BOTH edges are in
  contact for the whole clip, and the person<->deck edge is STATIC *even though the person
  has a large WORLD velocity* (~1.1 m/s), because the contact is measured support-relative
  on a moving deck. The deck<->ground edge moves (rolling/sliding) consistent with the
  wheeled board rolling across the ground.

* ``box_on_two_blocks`` -- the s.8 changing-active-set test: the MAP active set drops from
  cardinality 2 ({box_blockL, box_blockR}) to 1 ({box_blockL}) when one support is removed
  in truth, recovered within a tolerance window of the true structural change.

Calibration notes (THEORY.md s.4/s.7 observability):

* The two scenes sit on opposite sides of the observability spectrum, so they take
  different *physically-interpretable* emission configs (real velocity / gap scales), not a
  single one-size default. The skateboard scene's board<->ground edge tracks the BOARD
  origin -- a translating, non-spinning material point -- so its observable kinematic
  signature is a fast STEADY tangential slip (~1.1 m/s); the default ``slide_speed`` (0.15)
  is tuned for a slow human slide and reads that fast steady slip as "free". We widen
  ``slide_speed`` / ``vel_sigma`` / ``free_vel_sigma`` to the board's actual speed regime so
  the genuinely-in-contact edge is recovered (it reads SLIDING, not the wheels' true
  ROLLING, precisely because the tracked board point does not itself spin -- a faithful,
  documented consequence of which material point the mocap rig tracks, THEORY.md s.3).

* The box_on_two_blocks deactivation is, by construction, a pure *loading* event: removing
  blockR redistributes weight with essentially NO motion (the wide blockL holds the box
  level; the box settles <0.1 mm and tilts <0.03 deg) and only a sub-millimetre (~0.4 mm)
  geometric separation. THEORY.md s.7's observability theorem says force/loading is
  unrecoverable from kinematics; the *only* kinematic trace of the support removal here is
  that tiny gap opening. To read it we therefore (a) use the clean (noise-free) observable
  channel -- the 0.4 mm signal is smaller than the default 0.5 mm mocap noise, so any noise
  would bury it (s.4), and we verified the assertion is impossible under default noise --
  and (b) tighten the gap emission tolerance (``gap_sigma_gap``) and disable the
  resting-bias EM so the 0.4 mm opening clears the still-body STATIC preference. This is a
  deliberately knife-edge demonstration that the structural change is *barely* observable
  from kinematics alone -- exactly the s.7 message -- not a claim that it is robust.

MuJoCo is required; if it is not importable the whole module is skipped (the graph/detector
core is tested elsewhere without the simulator).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable regardless of how pytest is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The simulator is the only hard external dependency of this suite; skip cleanly if absent.
mujoco = pytest.importorskip("mujoco")

from contact import graph, mujoco_gen
from oracle import report
from contact.config import DetectorConfig
from contact.types import FREE, ROLLING, SLIDING, STATIC

# A single fixed seed for every scene so the whole suite is reproducible (the seed only
# drives the additive mocap noise in generate_scene; the physics is deterministic).
SEED = 12345


# --------------------------------------------------------------------------------------
# Small helpers shared by the scene tests.
# --------------------------------------------------------------------------------------


def _dominant_mode(map_state: list[str]) -> str:
    """Most frequent non-FREE MAP label of an edge's per-frame path (its overall mode, s.3)."""
    counts = Counter(m for m in map_state if m != FREE)
    return counts.most_common(1)[0][0] if counts else FREE


def _map_cardinality(graph_result) -> np.ndarray:
    """Per-frame size of the MAP active set (number of simultaneously active edges, s.8)."""
    return np.array([len(s) for s in graph_result.map_active_set], dtype=int)


def _true_cardinality(scene, edges: list[str]) -> np.ndarray:
    """Per-frame size of the TRUE active set from each edge's withheld ``in_contact`` (s.9)."""
    masks = [np.asarray(scene.truth[eid].in_contact, dtype=int) for eid in edges]
    return np.sum(masks, axis=0)


def _first_drop_time(cardinality: np.ndarray, t: np.ndarray, hi: int, lo_max: int) -> float:
    """Time of the first frame where ``cardinality`` falls from ``hi`` to ``<= lo_max``.

    THEORY.md s.5/s.8: the active set persists, then changes at a discrete guard instant;
    this finds that downward structural transition. Returns ``nan`` if it never drops.
    """
    drops = np.flatnonzero((cardinality[:-1] == hi) & (cardinality[1:] <= lo_max))
    if drops.size == 0:
        return float("nan")
    # +1: the drop is realized at the first frame at the lower cardinality.
    return float(np.asarray(t, dtype=float)[drops[0] + 1])


# --------------------------------------------------------------------------------------
# person_on_skateboard: the relative-frame + graph payoff (THEORY.md s.1 / s.8).
# --------------------------------------------------------------------------------------


def test_person_on_skateboard_both_edges_in_contact_static_rider_on_moving_deck():
    """BOTH edges in contact; person<->deck STATIC despite large WORLD velocity (s.1/s.8).

    The whole point of measuring contact support-relative (THEORY.md s.1) is that a body
    riding a fast support reads ~0 *relative* twist: the person screams across the world at
    ~1.1 m/s yet is in solid STATIC contact with the deck. The graph layer (s.8) recovers
    both edges of the contact graph -- person<->deck and deck<->ground -- as simultaneously
    active for the whole clip.

    Config note: the board<->ground edge tracks the BOARD origin (a translating,
    non-spinning point), so its observable signature is a fast STEADY tangential slip; we
    widen the velocity emission scales to that regime (default ``slide_speed`` is tuned for
    a slow human slide and would call the fast steady slip "free"). It then reads SLIDING --
    rolling-vs-sliding here is set by which material point the rig tracks, THEORY.md s.3.
    """
    scene = mujoco_gen.generate_scene("person_on_skateboard", seed=SEED)

    cfg = DetectorConfig()
    # Tune the translational-velocity emission to the board's actual fast-steady-slip regime
    # (THEORY.md s.3/s.4: physically-interpretable speed scales, not simulator-specific
    # magic). ~1.1 m/s is the board's world translation speed, which is what the tracked
    # board origin's tangential velocity reads on the deck<->ground edge.
    cfg.emission.slide_speed = 1.1
    cfg.emission.vel_sigma = 0.3
    cfg.emission.free_vel_sigma = 3.0

    result = graph.detect_scene(scene, cfg)
    edges = list(result.edges)
    assert set(edges) == {"person_board", "board_ground"}

    # --- per-edge existence: both edges in contact for MOST of the clip (iou > 0.7) -------
    score_pb = report.score(result.per_edge["person_board"], scene.truth["person_board"])
    score_bg = report.score(result.per_edge["board_ground"], scene.truth["board_ground"])
    assert score_pb["contact_iou"] > 0.7, (
        f"person_board contact IoU too low: {score_pb['contact_iou']:.3f}"
    )
    assert score_bg["contact_iou"] > 0.7, (
        f"board_ground contact IoU too low: {score_bg['contact_iou']:.3f}"
    )

    # --- the joint structure: BOTH edges active simultaneously for most of the clip ------
    # active_posterior[t, e] = P(edge e active) after the joint structure inference (s.8).
    eidx = {e: i for i, e in enumerate(edges)}
    both_active = np.all(result.active_posterior > 0.5, axis=1)
    assert np.mean(both_active) > 0.9, (
        f"both edges should be jointly active for most frames; got "
        f"{np.mean(both_active):.2f}"
    )
    card = _map_cardinality(result)
    assert np.mean(card == 2) > 0.9, (
        f"MAP active set should be both-edges for most frames; got cardinality-2 fraction "
        f"{np.mean(card == 2):.2f}"
    )

    # --- person_board is dominated by STATIC ... -----------------------------------------
    pb_map = result.per_edge["person_board"].map_state
    assert _dominant_mode(pb_map) == STATIC, (
        f"person_board dominant mode should be STATIC; got {_dominant_mode(pb_map)!r}"
    )
    pb_static_frac = np.mean([m == STATIC for m in pb_map])
    assert pb_static_frac > 0.8, (
        f"person_board should be STATIC on most frames; got {pb_static_frac:.2f}"
    )

    # --- ... EVEN THOUGH the person has a large WORLD-frame velocity. ----------------------
    # This is the relative-frame + graph payoff (THEORY.md s.1/s.8): a world-frame speed
    # test would call this fast-moving contact "moving, therefore not in contact" and be
    # wrong; the support-relative frame on the moving deck reads ~0 relative twist -> STATIC.
    person = scene.bodies["person"]
    t = np.asarray(person.t, dtype=float)
    world_dx = person.position[-1, 0] - person.position[0, 0]
    world_speed = abs(world_dx) / (t[-1] - t[0])
    assert world_speed > 0.5, (
        f"sanity: the person should be moving fast in the world for this to be the s.1 "
        f"payoff; got {world_speed:.3f} m/s"
    )

    # --- board_ground mode is SLIDING, consistent across truth and observation. -----------
    # The edge tracks a BOARD-fixed point (the board origin), which translates over the
    # ground at the board's travel speed -> a steady tangential slip -> SLIDING. The scene's
    # truth mode is now classified from that same board-fixed point (truth_mode_body="board"
    # in mujoco_gen), so truth and observation agree on SLIDING. (The wheels' material point
    # truly rolls, but that point is not tracked -- THEORY.md s.3: rolling vs sliding depends
    # on which material point you follow.) We accept ROLLING too for robustness, but assert
    # the edge is genuinely MOVING, not STATIC/FREE.
    bg_map = result.per_edge["board_ground"].map_state
    bg_dom = _dominant_mode(bg_map)
    assert bg_dom in (ROLLING, SLIDING), (
        f"board_ground dominant mode should be ROLLING or SLIDING (a board moving over the "
        f"ground); got {bg_dom!r}"
    )
    # And the truth labels it a moving contact (rolling/sliding), confirming the scene.
    bg_truth_moving = Counter(
        m for m in scene.truth["board_ground"].mode if m in (ROLLING, SLIDING)
    )
    assert sum(bg_truth_moving.values()) > 0.8 * len(scene.truth["board_ground"].mode), (
        "scene sanity: board_ground truth should be a moving (rolling/sliding) contact"
    )


# --------------------------------------------------------------------------------------
# box_on_two_blocks: the changing active set (THEORY.md s.8).
# --------------------------------------------------------------------------------------


def test_box_on_two_blocks_active_set_cardinality_drops_when_support_removed():
    """The MAP active set drops from both edges to one when a support is removed (s.8).

    The s.8 changing-active-set test: a plank bridges two blocks ({box_blockL, box_blockR}),
    then one support (blockR) is removed mid-run and the true active set becomes
    {box_blockL}. We assert the MAP active-set cardinality DROPS (2 -> 1) at a time within a
    tolerance window of the true structural change.

    The removal is genuinely OBSERVABLE: blockR rides a vertical slide joint and is decoupled
    from the floor collision group (mujoco_gen._build_box_on_two_blocks), so when commanded
    down it descends the full ~0.30 m into the well below the deck -- opening a ~0.30 m gap on
    the box_blockR edge while the wide blockL holds the box level (the box stays put, the
    support leaves). That separation is three orders of magnitude above the mocap noise floor,
    so this runs at the DEFAULT noise level and the DEFAULT detector config -- no detuning.
    (An earlier version of the scene held blockR pinned against the floor so it could only
    sink sub-mm; that made the advertised structural change kinematically unobservable -- the
    s.7 trap. The scene now physically realizes the separation it labels.)
    """
    # Realistic mocap noise (the scene default ~0.5 mm) -- the ~0.30 m separation dwarfs it.
    scene = mujoco_gen.generate_scene("box_on_two_blocks", seed=SEED)

    # Default detector config: no gap-emission tightening, no EM disabling. The separation is
    # large and unambiguous, so the stock detector recovers the structural change on its own.
    cfg = DetectorConfig()

    result = graph.detect_scene(scene, cfg)
    edges = list(result.edges)
    assert set(edges) == {"box_blockL", "box_blockR"}

    t = np.asarray(result.t, dtype=float)
    card = _map_cardinality(result)
    true_card = _true_cardinality(scene, edges)

    # --- the true structural change: both -> one, at some drop time. ----------------------
    true_drop = _first_drop_time(true_card, t, hi=2, lo_max=1)
    assert np.isfinite(true_drop), "scene sanity: the true active set must drop 2 -> 1"
    # Both supports truly active early, only one truly active late (the scene contract).
    assert true_card[0] == 2
    assert true_card[-1] == 1

    # --- the inferred MAP active set drops in cardinality (2 -> 1). ------------------------
    early = t < (true_drop - 0.2)   # comfortably before the support is removed
    late = t > (true_drop + 0.2)    # comfortably after
    assert np.all(card[early] == 2), (
        f"MAP active set should be BOTH edges before the support is removed; got "
        f"cardinalities {sorted(set(card[early].tolist()))}"
    )
    assert np.all(card[late] == 1), (
        f"MAP active set should be a SINGLE edge after the support is removed; got "
        f"cardinalities {sorted(set(card[late].tolist()))}"
    )
    # The cardinality genuinely DROPS (the headline assertion of THEORY.md s.8).
    assert card[early].mean() > card[late].mean(), (
        "MAP active-set cardinality must drop across the support-removal event"
    )

    # --- the drop happens within a tolerance window of the true structural change. --------
    map_drop = _first_drop_time(card, t, hi=2, lo_max=1)
    assert np.isfinite(map_drop), "the MAP active set must drop 2 -> 1 somewhere"
    # 0.2 s tolerance: the smoothed HMM realizes the structural switch a few frames after
    # the guard (THEORY.md s.5: persistence delays the switch slightly), well within window.
    assert abs(map_drop - true_drop) < 0.2, (
        f"MAP active-set drop at t={map_drop:.3f}s should be within 0.2 s of the true "
        f"structural change at t={true_drop:.3f}s"
    )

    # --- the SURVIVING edge (box_blockL) stays active throughout; box_blockR deactivates. -
    # box_blockL is in contact for the whole clip (it never loses load); box_blockR is the
    # one whose existence flips off -- exactly the structure-inference signal of s.8.
    score_L = report.score(result.per_edge["box_blockL"], scene.truth["box_blockL"])
    score_R = report.score(result.per_edge["box_blockR"], scene.truth["box_blockR"])
    assert score_L["contact_iou"] > 0.9, (
        f"box_blockL (always loaded) should be a clean sustained contact; iou="
        f"{score_L['contact_iou']:.3f}"
    )
    assert score_R["contact_iou"] > 0.7, (
        f"box_blockR (deactivated mid-run) should still track the true active interval; "
        f"iou={score_R['contact_iou']:.3f}"
    )
    eidx = {e: i for i, e in enumerate(edges)}
    # The surviving single edge in the late MAP sets is box_blockL.
    late_sets = [
        set(result.map_active_set[i]) for i in np.flatnonzero(late)
    ]
    assert all(s == {"box_blockL"} for s in late_sets), (
        "after the support is removed, the single active edge should be box_blockL"
    )
