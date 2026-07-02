"""The simulator-free end-to-end path: analytic truth → observe → detect → score.

`synthetic_drop_rest_liftoff` is the one scenario whose truth is *constructed* rather than
measured, so this is the cheapest full-pipeline check in the suite (no MuJoCo stepping) and
the reference expectation story: FREE → IMPACT → STATIC → liftoff → FREE.
"""

from __future__ import annotations

import numpy as np

from contact import ContactDetector, observe
from contact.types import FREE, STATIC
from oracle import synthetic_drop_rest_liftoff


def _detect(raw):
    obs = observe(raw.moving, raw.support, raw.surface, raw.contact_point_local)
    return obs, ContactDetector().detect(obs)


def test_recovers_the_canonical_story():
    raw = synthetic_drop_rest_liftoff()
    _, result = _detect(raw)

    from oracle.report import score

    sc = score(result, raw.truth)
    assert sc["contact_iou"] > 0.9, sc
    assert sc["mode_accuracy"] > 0.9, sc

    # The MAP path tells the story in order: free fall, a rest plateau, free flight again.
    map_state = list(result.map_state)
    t = np.asarray(result.t)
    t_impact = raw.meta["t_impact"]
    t_lift = raw.meta["t_lift"]
    assert map_state[0] == FREE and map_state[-1] == FREE
    rest = (t > t_impact + 0.1) & (t < t_lift - 0.1)
    assert all(s == STATIC for s in np.asarray(map_state, dtype=object)[rest])


def test_events_and_impact_atom_are_timed():
    raw = synthetic_drop_rest_liftoff()
    _, result = _detect(raw)
    t_impact = raw.meta["t_impact"]

    touchdowns = [e for e in result.events if e.kind == "touchdown"]
    liftoffs = [e for e in result.events if e.kind == "liftoff"]
    assert len(touchdowns) == 1 and len(liftoffs) == 1
    assert abs(touchdowns[0].time - t_impact) < 0.05
    assert liftoffs[0].time > raw.meta["t_lift"] - 0.05

    # The touchdown is a matched-filter impact atom at the analytic instant. Its measured
    # closing speed is the *smoothed* derivative at the arrest — attenuated well below the
    # analytic g*t_impact ≈ 2.6 m/s by the differentiation kernel — so assert timing and
    # sign/order, not the raw ballistic value.
    assert len(result.impulses) >= 1
    atom = min(result.impulses, key=lambda i: abs(i.time - t_impact))
    assert abs(atom.time - t_impact) < 0.05
    assert atom.closing_speed > 0.5


def test_em_recovers_the_modeled_clearance_bias():
    raw = synthetic_drop_rest_liftoff()
    _, result = _detect(raw)
    # The rest phase sits at a constructed +4 mm clearance; the EM gap-bias calibration
    # (THEORY.md §7) must find it from the noisy gap alone.
    assert abs(result.resting_bias - raw.meta["resting_bias"]) < 1.5e-3


def test_contact_posterior_is_calibrated_at_the_extremes():
    raw = synthetic_drop_rest_liftoff()
    _, result = _detect(raw)
    post = np.asarray(result.contact_posterior, dtype=float)
    t = np.asarray(result.t)
    # Mid-rest the posterior must be near-certain contact; in clean flight, near-certain free.
    mid_rest = (t > 1.0) & (t < 1.8)
    flight = (t < raw.meta["t_impact"] - 0.1) | (t > 2.6)
    assert post[mid_rest].min() > 0.95
    assert post[flight].max() < 0.05
