"""Ground-truth checks for the contact-detection pipeline.

The synthetic clip is fully known: the sole rests on the floor over [1.00s, 2.50s]
with a 4 mm modeled clearance offset injected. These tests assert the pipeline
recovers both, within tolerances loose enough to survive the smoothing/hysteresis
edge effects but tight enough to catch a real regression.
"""

import numpy as np

from main import (
    compute_features,
    contact_confidence,
    decide_contact,
    synthesize_trajectory,
    _runs,
)

# Ground truth baked into synthesize_trajectory().
TRUE_START, TRUE_END = 1.0, 2.5
TRUE_BIAS = 0.004


def run_pipeline(seed=0):
    t, points, yaw = synthesize_trajectory(seed=seed)
    features, bias = compute_features(t, points, yaw)
    confidence = contact_confidence(features)
    contact = decide_contact(t, confidence)
    return t, contact, bias


def test_recovers_single_contact_interval():
    t, contact, _ = run_pipeline()
    intervals = [(t[s], t[e - 1]) for s, e in _runs(contact, True)]
    assert len(intervals) == 1, f"expected one contact interval, got {intervals}"

    start, end = intervals[0]
    # Smoothing + hysteresis trim the edges inward; allow ~120 ms of slack and
    # require the detection to stay strictly inside the true rest window.
    assert TRUE_START <= start <= TRUE_START + 0.12
    assert TRUE_END - 0.12 <= end <= TRUE_END


def test_recovers_resting_bias():
    _, _, bias = run_pipeline()
    assert abs(bias - TRUE_BIAS) < 5e-4  # within 0.5 mm of the injected 4 mm


def test_confidence_in_unit_range():
    t, points, yaw = synthesize_trajectory()
    features, _ = compute_features(t, points, yaw)
    confidence = contact_confidence(features)
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))
    # High at rest, low in free flight.
    assert confidence[(t > 1.5) & (t < 2.0)].mean() > 0.9
    assert confidence[t > 3.5].mean() < 0.1


def test_stable_across_seeds():
    for seed in range(5):
        t, contact, bias = run_pipeline(seed=seed)
        intervals = [(t[s], t[e - 1]) for s, e in _runs(contact, True)]
        assert len(intervals) == 1, f"seed {seed}: got {intervals}"
        assert abs(bias - TRUE_BIAS) < 1e-3
