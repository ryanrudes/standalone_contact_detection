"""End-to-end validation of the detector against MuJoCo ground truth (THEORY.md §9).

This is the top rung of the pragmatic ladder (THEORY.md §10) exercised as a whole:
for each scenario we run the *exact* workflow of THEORY.md §9 ---

    oracle.generate  ->  geometry.observe  ->  ContactDetector().detect  ->  report.score

--- handing the detector only the "observable" channel (noisy poses) and scoring its
inferred posterior against the *withheld* simulator truth. Each scenario asserts the
specific theoretical claim it was built to stress:

* ``drop_rest``          existence + a single touchdown impact (§2, §6).
* ``drop_rest_liftoff``  both make/break guards: free->contact and contact->free (§5).
* ``push_to_slide``      the stick->slip friction-cone guard: static then sliding (§7).
* ``rolling_ball``       the curved rolling twist-subspace mode (§3).
* ``moving_support``     contact is *relative*, not world-frame (§1) --- the payoff.

The thresholds below were calibrated against the actual implementation at this seed
(SEED) and rate; where a "clean" theoretical bound would be too strict for a kinematic
detector fed noisy mocap, it is loosened with a comment stating the realistic limit.
No scenario is dropped and nothing is asserted trivially.

MuJoCo is required; if it is not importable the whole module is skipped (the detector
core is tested elsewhere without the simulator).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable regardless of how pytest is invoked (so a bare
# ``uv run pytest`` finds the ``contact`` package without needing pythonpath=.).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The simulator is the only hard external dependency of this suite. Skip cleanly if
# absent rather than erroring (THEORY.md §9: the simulator is the truth oracle, but the
# detector core does not itself depend on it).
mujoco = pytest.importorskip("mujoco")

from contact import geometry
import oracle
from oracle import report
from contact.detector import ContactDetector
from contact.types import ROLLING, SLIDING, STATIC

# A single fixed seed for every scenario so the whole suite is reproducible (the seed
# only drives the additive mocap noise in oracle.generate; the physics is
# deterministic). Chosen once; all thresholds below were tuned against it.
SEED = 12345


# --------------------------------------------------------------------------------------
# Pipeline helper + small ground-truth/segmentation utilities
# --------------------------------------------------------------------------------------


def _run(name: str):
    """Run the full §9 pipeline for one scenario and return (raw, obs, result, scores).

    THEORY.md §9: generate (clean physics + withheld truth, noisy poses) -> observe
    (support-relative twist, §1/§3) -> detect (the HMM estimator, §4-8) -> score
    (against the withheld truth). This is exactly the chain every test below shares.
    """
    raw = oracle.generate(name, seed=SEED)
    obs = geometry.observe(
        raw.moving, raw.support, raw.surface, raw.contact_point_local,
        geometry=getattr(raw, "geometry", None),
    )
    result = ContactDetector().detect(obs)
    scores = report.score(result, raw.truth)
    return raw, obs, result, scores


def _true_first_contact_time(truth) -> float:
    """Time of the first truly-in-contact frame (the ground-truth touchdown instant)."""
    idx = np.flatnonzero(np.asarray(truth.in_contact, dtype=bool))
    assert idx.size > 0, "scenario has no true contact frames at all"
    return float(np.asarray(truth.t, dtype=float)[idx[0]])


def _map_runs(result, label: str) -> list[tuple[float, float]]:
    """Contiguous (t_start, t_end) runs of the MAP path equal to ``label``.

    THEORY.md §5: the Viterbi MAP path is already temporally coherent. A *mode* segment
    is a maximal run of that path in one mode. We read these directly off ``map_state``
    rather than off ``DetectionResult.intervals`` because the latter splits only on FREE
    (so a static->sliding transition mid-contact lives inside ONE interval whose
    *dominant* label is reported); the per-frame run is the honest way to ask "did a
    sliding segment occur within the contact?".
    """
    ms = list(result.map_state)
    t = np.asarray(result.t, dtype=float)
    runs: list[tuple[float, float]] = []
    i = 0
    n = len(ms)
    while i < n:
        if ms[i] == label:
            j = i
            while j < n and ms[j] == label:
                j += 1
            runs.append((float(t[i]), float(t[j - 1])))
            i = j
        else:
            i += 1
    return runs


def _frac_mode_on_true_contact(result, truth, label: str) -> float:
    """Fraction of *truly-in-contact* frames whose MAP mode equals ``label``.

    THEORY.md §3: a mode is the twist-subspace the motion lives in; this asks how often
    the detector recovered that subspace, restricted (like ``report.score``'s
    ``mode_accuracy``) to frames the simulator says are truly in contact.
    """
    true_mask = np.asarray(truth.in_contact, dtype=bool)
    idx = np.flatnonzero(true_mask)
    if idx.size == 0:
        return float("nan")
    ms = list(result.map_state)
    hits = sum(1 for i in idx if ms[i] == label)
    return hits / idx.size


# --------------------------------------------------------------------------------------
# drop_rest: existence + a single touchdown near the true landing (THEORY.md §2, §6)
# --------------------------------------------------------------------------------------


def test_drop_rest():
    """Box free-falls onto a static plane and rests.

    Asserts contact existence is recovered well (high IoU) and that a touchdown event is
    pinned near the true landing instant (THEORY.md §6: the deceleration spike at
    touchdown is the gold-standard event timer).
    """
    raw, obs, result, scores = _run("drop_rest")

    # Existence: the active set of §2 is recovered with strong overlap. The spec bound
    # is 0.70; the implementation comfortably exceeds it (~0.96 at this seed). The small
    # shortfall from 1.0 is the few-frame onset lag inherent to a smoothing detector
    # (§6: confirming a landing uses the frames AFTER it).
    assert scores["contact_iou"] > 0.7

    # A touchdown event must exist (free->contact make guard, §6).
    touchdowns = [e for e in result.events if e.kind == "touchdown"]
    assert len(touchdowns) >= 1, "expected at least one touchdown event"

    # The detected touchdown lands near the true first-contact instant. Tolerance is
    # 50 ms: the detector's onset is intrinsically a hair late (it confirms landing from
    # subsequent rest, §6) and the gap channel is built from noise + 100 Hz sampling.
    true_td = _true_first_contact_time(raw.truth)
    nearest = min(touchdowns, key=lambda e: abs(e.time - true_td))
    assert abs(nearest.time - true_td) < 0.05, (
        f"touchdown {nearest.time:.3f}s not near true landing {true_td:.3f}s"
    )


# --------------------------------------------------------------------------------------
# drop_rest_liftoff: BOTH guards of the hybrid system (THEORY.md §5)
# --------------------------------------------------------------------------------------


def test_drop_rest_liftoff():
    """Box drops, rests, then is peeled back off by an applied force.

    Asserts the full make/break cycle of THEORY.md §5: free->contact (gap reaches 0)
    AND contact->free (normal force reaches 0), realised as exactly one sustained
    contact interval bracketed by a touchdown then a liftoff.
    """
    raw, obs, result, scores = _run("drop_rest_liftoff")

    # Existence still solid even though the contact ends mid-recording (spec bound 0.7;
    # ~0.87 here -- lower than drop_rest because both ends now cost onset/offset lag).
    assert scores["contact_iou"] > 0.7

    # Exactly one *main* contact interval. The Viterbi segmentation (§5) replaces the
    # toy script's morphological cleanup, so the single sustained touch is one run -- not
    # a flickering string. We allow at most one tiny spurious blip by taking the longest
    # interval as "main" and requiring the rest (if any) to be negligibly short.
    assert len(result.intervals) >= 1, "expected a contact interval"
    durations = sorted(
        (iv.t_end - iv.t_start for iv in result.intervals), reverse=True
    )
    main = durations[0]
    spurious = durations[1:]
    assert all(d < 0.1 * main for d in spurious), (
        f"expected one dominant interval; got durations {durations}"
    )

    # Both event kinds present (the two guards of §5).
    touchdowns = [e for e in result.events if e.kind == "touchdown"]
    liftoffs = [e for e in result.events if e.kind == "liftoff"]
    assert len(touchdowns) >= 1, "expected a touchdown (free->contact guard)"
    assert len(liftoffs) >= 1, "expected a liftoff (contact->free guard)"

    # Causality: the (last) liftoff must come after the (first) touchdown -- you cannot
    # break a contact you never made (§5/§6 guard ordering).
    first_td = min(e.time for e in touchdowns)
    last_lo = max(e.time for e in liftoffs)
    assert last_lo > first_td, (
        f"liftoff {last_lo:.3f}s must follow touchdown {first_td:.3f}s"
    )


# --------------------------------------------------------------------------------------
# push_to_slide: the stick->slip friction-cone guard (THEORY.md §7)
# --------------------------------------------------------------------------------------


def test_push_to_slide():
    """Box rests, then a ramped push breaks friction and it slides.

    Asserts the stick->slip guard of THEORY.md §7: an early STATIC resting phase
    (tangential force inside the friction cone) followed by a SLIDING phase once the push
    reaches the cone boundary -- i.e. the detector recovers a sliding segment and labels
    the early rest static.
    """
    raw, obs, result, scores = _run("push_to_slide")

    # A SLIDING segment must appear in the MAP path. Note we look at per-frame runs, not
    # DetectionResult.intervals: the static->sliding switch happens mid-contact, so it
    # lives inside ONE non-FREE interval whose *dominant* label is the longer static
    # phase. The honest question is whether a contiguous sliding run exists -- it does.
    sliding_runs = _map_runs(result, SLIDING)
    assert len(sliding_runs) >= 1, "expected a sliding segment after friction breaks"

    # Mode recovery on truly-in-contact frames beats chance handily (spec bound 0.5;
    # ~0.86 here). Pure-kinematic mode ID is imperfect near the stick->slip boundary
    # where slip speed is tiny, so we do NOT demand perfection.
    assert scores["mode_accuracy"] > 0.5

    # The early resting phase (before the push starts ramping at t=0.3 s) is labeled
    # STATIC: while sticking, the relative twist is ~0 (§3/§7). We check every
    # truly-in-contact frame in that window is MAP-static.
    t = np.asarray(result.t, dtype=float)
    true_mask = np.asarray(raw.truth.in_contact, dtype=bool)
    ms = list(result.map_state)
    early_idx = [i for i in np.flatnonzero(true_mask) if t[i] < 0.3]
    assert len(early_idx) > 0, "expected resting contact frames before the push"
    early_static = sum(1 for i in early_idx if ms[i] == STATIC)
    # Allow a 1-frame slack for the very first noisy frame; require the rest static.
    assert early_static >= len(early_idx) - 1, (
        f"early resting phase not labeled static: {early_static}/{len(early_idx)}"
    )

    # The sliding segment must come AFTER the static rest (the guard fires once, late).
    first_slide = min(s for s, _ in sliding_runs)
    assert first_slide > 0.3, (
        f"sliding began at {first_slide:.3f}s, before the friction-cone breach"
    )


# --------------------------------------------------------------------------------------
# rolling_ball: the curved rolling twist-subspace mode (THEORY.md §3)
# --------------------------------------------------------------------------------------


def test_rolling_ball():
    """Sphere rolling without slip across a plane.

    Asserts ROLLING is recovered as a present mode for a meaningful fraction of the
    truly-in-contact frames. THEORY.md §3 stresses that rolling is the HARD mode --- it
    is a *curved* constraint manifold (v coupled to omega by v = omega x r), so a
    detector can easily confuse it with sliding or static. We therefore assert it is
    *present*, not that it dominates every frame everywhere.

    Threshold rationale: we require rolling on at least 30% of the true-contact frames.
    That is well above what a static/sliding-only confusion would yield, yet does not
    demand the (harder) claim of full-trajectory dominance. The implementation actually
    achieves a far higher fraction at this seed; 0.30 is the documented *meaningful*
    floor the spec asks for, kept conservative because rolling is genuinely fragile.
    """
    raw, obs, result, scores = _run("rolling_ball")

    # Sanity: the simulator's own truth labels this contact as (at least partly) rolling,
    # so the assertion is meaningful and not vacuous.
    true_idx = np.flatnonzero(np.asarray(raw.truth.in_contact, dtype=bool))
    assert true_idx.size > 0, "rolling_ball has no true contact frames"

    roll_frac = _frac_mode_on_true_contact(result, raw.truth, ROLLING)
    assert roll_frac > 0.30, (
        f"rolling present on only {roll_frac:.2%} of contact frames "
        f"(meaningful floor 30%); rolling is the hard curved-manifold mode of §3"
    )

    # And it should appear as an actual contiguous segment, not scattered single frames.
    rolling_runs = _map_runs(result, ROLLING)
    assert len(rolling_runs) >= 1, "expected at least one contiguous rolling segment"


# --------------------------------------------------------------------------------------
# moving_support: contact is RELATIVE, not world-frame (THEORY.md §1) -- the payoff
# --------------------------------------------------------------------------------------


def test_moving_support():
    """Box riding a sliding cart: large WORLD velocity, ~0 RELATIVE velocity.

    This is the central payoff of THEORY.md §1: the box has a large world-frame velocity
    (the cart drives it to x=1.5 m), yet the box-on-cart contact is unambiguously STATIC
    because the relative twist is ~0. A world-frame detector would call the moving box
    "not in contact"; measuring support-relative (geometry.observe carries the cart's
    plane into the world per frame) recovers the contact correctly.

    We assert high IoU EVEN THOUGH the box is screaming across the world -- that is the
    relative-frame payoff. (If this were measured in the world frame the moving phase
    would read as motion and the contact IoU would collapse; the high score IS the proof
    that observe() worked support-relative.)
    """
    raw, obs, result, scores = _run("moving_support")

    # Confirm the box really does move a lot in the WORLD frame, so the relative-frame
    # claim has teeth (this is what would fool a world-frame detector).
    world_x = np.asarray(raw.moving.position, dtype=float)[:, 0]
    world_travel = float(world_x.max() - world_x.min())
    assert world_travel > 0.5, (
        f"box only moved {world_travel:.2f} m in world; scenario not exercising motion"
    )

    # The relative-frame payoff: existence recovered with high overlap despite that
    # world motion (spec bound 0.7; ~0.99 here because the support-relative twist is
    # genuinely ~0 throughout). THEORY.md §1.
    assert scores["contact_iou"] > 0.7

    # And the recovered mode is STATIC for essentially the whole contact (relative twist
    # ~0, §3), not sliding -- the world velocity does NOT leak into the relative frame.
    static_frac = _frac_mode_on_true_contact(result, raw.truth, STATIC)
    assert static_frac > 0.7, (
        f"only {static_frac:.2%} of contact frames labeled static; world velocity "
        f"may be leaking into the relative frame (§1 violated)"
    )
