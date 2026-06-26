#!/usr/bin/env python3
"""
Contact detection from scratch — a self-contained, illustrative reimplementation.

This script depends on NOTHING but numpy + scipy. It does not import the `retarget`
package; it re-derives the whole contact-detection pipeline in one file so you can
read it top-to-bottom and see exactly how a boolean "is this patch touching the
surface?" signal is produced from a motion-capture trajectory.

The pipeline, in order:

  1. SIGNALS      synthesize a contact-point trajectory (here: a shoe sole that
                  drops onto the floor, rests, then lifts off again).
  2. FEATURES     for each frame, measure six physical channels that distinguish
                  "resting in contact" from "moving / in the air":
                    - clearance        : signed gap to the support plane (m)
                    - normal_speed     : speed perpendicular to the surface (m/s)
                    - tangential_speed : speed along the surface (m/s)
                    - quiet_activity   : windowed RMS speed (m/s)
                    - quiet_spread     : windowed position range (m)
                    - angular_speed    : spin rate (rad/s)
  3. NOISE        each channel has a noise scale sigma. A feature one sigma from
                  rest contributes ~1 to a chi-squared penalty. (These sigmas are
                  the hard-coded DEFAULT_NOISE used when no static calibration
                  recording is supplied — see the note at NOISE below.)
  4. CONFIDENCE   divide each feature by its sigma to get a standardized residual,
                  square-and-sum them into a Mahalanobis distance, and read off the
                  chi-squared survival probability: the chance residuals at least this
                  large would arise from sensor noise if the patch were truly in contact.
                  ~1 when resting, collapsing to 0 as any channel departs by many sigma.
  5. DECIDE       threshold the confidence with hysteresis (enter high, leave low),
                  then clean the boolean mask in time: bridge brief gaps, drop brief
                  blips, and require a minimum contact duration.

The real library generalizes each step (moving supports, polygon footprints,
support-relative motion, per-patch measured noise), but the skeleton is exactly this.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

# --------------------------------------------------------------------------------------
# Configuration — the few public knobs, with the rest pinned to the library's defaults.
# --------------------------------------------------------------------------------------

# Per-channel noise scale sigma, in physical units. A channel reading `sigma` away from
# its resting value contributes ~1 to the chi-squared penalty below. These specific
# numbers are the library's hard-coded DEFAULT_NOISE ("typical optical mocap, ~100-250 Hz,
# deliberately conservative"). They are NOT derived from a calibration recording — a real
# static-rig calibration would *measure* them and tighten the clearance sigma in particular.
NOISE = {
    "clearance": 0.0015,         # 1.5 mm  positional jitter of the contact point
    "normal_speed": 0.08,        # m/s     generous: differentiation amplifies position noise
    "tangential_speed": 0.08,    # m/s
    "quiet_activity": 0.08,      # m/s     windowed RMS speed
    "quiet_spread": 0.01,        # m       windowed position range
    "angular_speed": 0.3,        # rad/s   rolling/settling supports tilt
}

ENTER_CONFIDENCE = 0.95          # confidence required to ENTER contact
EXIT_CONFIDENCE = 0.95 * 0.85    # lower bar to STAY in contact (hysteresis)
PENETRATION_SOFTENING = 4.0      # penetration penalized this many x more gently than a gap
MAX_RESTING_BIAS = 0.01          # constant geometry offset absorbed at rest, clipped to +/-1 cm

MAX_GAP_TIME = 0.10              # bridge contact gaps up to this long (s)
MIN_CONTACT_TIME = 0.18          # discard contact intervals shorter than this (s)

# Smoothing / window lengths (seconds) for the motion + quiet channels.
VEL_SMOOTH_TIME = 0.05           # smooth position before differentiating (tames noise)
QUIET_WINDOW_TIME = 0.10         # window for RMS-speed "activity"
SPREAD_WINDOW_TIME = 0.12        # window for position-range "spread"


# --------------------------------------------------------------------------------------
# Small numeric helpers (time-aware smoothing / windowed statistics / differentiation).
# Pure numpy; written for clarity over speed (the trajectory is short).
# --------------------------------------------------------------------------------------

def gaussian_smooth(x: np.ndarray, t: np.ndarray, sigma_time: float) -> np.ndarray:
    """Smooth a (T,) or (T, D) signal with a Gaussian kernel measured in real time."""
    if sigma_time <= 0.0:
        return x
    dt = t[:, None] - t[None, :]                      # (T, T) pairwise time offsets
    w = np.exp(-0.5 * (dt / sigma_time) ** 2)         # Gaussian weights
    w /= w.sum(axis=1, keepdims=True)                 # row-normalize
    return w @ x                                       # weighted average over neighbours


def derivative(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central-difference d/dt of a (T,) or (T, D) signal (the library uses a local
    polynomial fit; finite difference is the same idea, less robust to noise)."""
    return np.gradient(x, t, axis=0)


def window_mask(t: np.ndarray, window_time: float) -> np.ndarray:
    """(T, T) boolean: row i selects the samples within +/- window/2 of time t[i]."""
    return np.abs(t[:, None] - t[None, :]) <= 0.5 * window_time


def windowed_rms(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-frame RMS of a (T,) signal over each row's time window."""
    sq = values ** 2
    return np.sqrt((mask @ sq) / mask.sum(axis=1))


def windowed_range(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-frame position spread: the norm of the per-axis (max - min) over each window."""
    spans = np.array([points[row].max(axis=0) - points[row].min(axis=0) for row in mask])
    return np.linalg.norm(spans, axis=1)


# --------------------------------------------------------------------------------------
# Step 1 — SIGNALS: synthesize a contact-point trajectory + its orientation.
# Story: the sole drops from ~0.5 m, lands and rests on the floor (with a 4 mm modeled
# offset, like a real calibrated contact origin), then lifts off and spins in the air.
# --------------------------------------------------------------------------------------

def synthesize_trajectory(seed: int = 0):
    """Return (timestamps (T,), points (T,3), yaw (T,)) for one contact point."""
    rng = np.random.default_rng(seed)
    hz = 100.0
    t = np.arange(0.0, 4.0, 1.0 / hz)
    n = len(t)

    REST_OFFSET = 0.004  # 4 mm: the modeled contact origin sits just above the fitted plane
    z = np.empty(n)
    for i, ti in enumerate(t):
        if ti < 1.0:                       # [0.0, 1.0)  airborne descent 0.5 m -> floor
            z[i] = REST_OFFSET + 0.5 * (1.0 - ti)
        elif ti < 2.5:                     # [1.0, 2.5)  resting on the floor
            z[i] = REST_OFFSET
        elif ti < 3.0:                     # [2.5, 3.0)  lift-off back up to ~0.4 m
            z[i] = REST_OFFSET + 0.4 * (ti - 2.5) / 0.5
        else:                              # [3.0, 4.0)  airborne, high
            z[i] = REST_OFFSET + 0.4

    points = np.column_stack([np.full(n, 0.2), np.full(n, -0.1), z])
    points += rng.normal(scale=0.0005, size=points.shape)  # 0.5 mm sensor noise

    # Orientation reduced to a yaw angle: still while resting, spinning once airborne.
    yaw = np.where(t < 3.0, 0.0, 2.0 * (t - 3.0))      # rad
    yaw += rng.normal(scale=0.002, size=n)
    return t, points, yaw


# --------------------------------------------------------------------------------------
# Step 2 — FEATURES: six per-frame channels measured against a STATIC ground plane.
# The plane is z = 0 with outward normal +z; clearance is signed (positive = a gap above
# the plane, negative = penetration). A moving support would instead carry a per-frame
# origin+normal, and motion would be measured relative to it — same six channels.
# --------------------------------------------------------------------------------------

def compute_features(t, points, yaw, plane_height=0.0):
    normal = np.array([0.0, 0.0, 1.0])

    # --- clearance: signed distance from the contact point to the support plane ---
    raw_clearance = points @ normal - plane_height

    # --- velocity (smooth the position first so finite differencing isn't dominated
    #     by sensor noise), split into normal vs tangential components ---
    smooth_pts = gaussian_smooth(points, t, VEL_SMOOTH_TIME)
    velocity = derivative(smooth_pts, t)
    normal_velocity = velocity @ normal
    tangential = velocity - normal_velocity[:, None] * normal
    normal_speed = np.abs(normal_velocity)
    tangential_speed = np.linalg.norm(tangential, axis=1)

    # --- quiet activity (windowed RMS speed) and spread (windowed position range) ---
    speed = np.linalg.norm(velocity, axis=1)
    activity = windowed_rms(speed, window_mask(t, QUIET_WINDOW_TIME))
    spread = windowed_range(smooth_pts, window_mask(t, SPREAD_WINDOW_TIME))

    # --- angular speed from the (smoothed) yaw path ---
    angular_speed = np.abs(derivative(gaussian_smooth(yaw, t, VEL_SMOOTH_TIME), t))

    # --- resting-bias correction: a patch at rest often shows a small *constant*
    #     clearance (the modeled contact origin sits a few mm off the fitted plane).
    #     Estimate it as the median of the quiet, near-contact frames and subtract it,
    #     so a true rest reads ~0. Clipped to +/-MAX_RESTING_BIAS so a real cm-scale gap
    #     can never be subtracted away. (Here it recovers the injected 4 mm offset.) ---
    quiet = quiet_mask(activity, spread)
    bias = resting_bias(raw_clearance, quiet)
    clearance = raw_clearance - bias

    return {
        "clearance": clearance,
        "normal_speed": normal_speed,
        "tangential_speed": tangential_speed,
        "quiet_activity": activity,
        "quiet_spread": spread,
        "angular_speed": angular_speed,
    }, bias


def quiet_mask(activity, spread):
    """Frames that are quiet in BOTH activity and spread, with hysteresis. Adaptive
    thresholds are read from each signal's own distribution (percentiles), so the
    notion of "quiet" scales to the clip."""
    a_on = max(0.003, np.percentile(activity, 45))
    a_off = max(0.010, np.percentile(activity, 65), 1.25 * a_on)
    s_on = max(np.percentile(spread, 45), 1e-12)
    s_off = max(np.percentile(spread, 65), 1.25 * s_on)
    mask = np.zeros(len(activity), dtype=bool)
    in_quiet = False
    for i in range(len(activity)):
        if in_quiet:
            if activity[i] >= a_off or spread[i] >= s_off:
                in_quiet = False
        elif activity[i] <= a_on and spread[i] <= s_on:
            in_quiet = True
        mask[i] = in_quiet
    return mask


def resting_bias(clearance, quiet):
    band = abs(MAX_RESTING_BIAS)
    candidates = clearance[quiet & np.isfinite(clearance) & (np.abs(clearance) <= band)]
    if candidates.size == 0:
        return 0.0
    return float(np.clip(np.median(candidates), -band, band))


# --------------------------------------------------------------------------------------
# Steps 3 & 4 — NOISE + CONFIDENCE: fuse the six channels into one probability in [0, 1].
# --------------------------------------------------------------------------------------

def contact_confidence(features):
    """Per-frame contact confidence in [0, 1] via a chi-squared survival probability."""
    # Standardized residual z = value / sigma for each channel.
    z = {ch: features[ch] / NOISE[ch] for ch in NOISE}

    # Clearance is ONE-SIDED: a gap (z > 0) is penalized fully, but penetration (z < 0,
    # e.g. soft-tissue squish or plane-fit error) is forgiven PENETRATION_SOFTENING x more.
    zc = z["clearance"]
    z["clearance"] = np.where(zc < 0.0, zc / PENETRATION_SOFTENING, zc)

    # Sum of squared residuals = squared Mahalanobis distance ~ chi-squared with one
    # degree of freedom per channel. The confidence is the survival function: the
    # probability that pure resting-contact noise would produce a penalty at least this
    # large. Small penalty (residuals near rest) -> survival ~1; large penalty -> ~0.
    penalty = sum(np.square(z[ch]) for ch in NOISE)
    k = len(NOISE)
    return np.clip(chi2.sf(penalty, df=k), 0.0, 1.0)


# --------------------------------------------------------------------------------------
# Step 5 — DECIDE: threshold the confidence with hysteresis, then clean in time.
# --------------------------------------------------------------------------------------

def hysteresis_mask(confidence, enter, exit_):
    """True while confidence stays high: latch on at `enter`, release only below `exit_`."""
    mask = np.zeros(len(confidence), dtype=bool)
    active = False
    for i, c in enumerate(confidence):
        if active:
            if c <= exit_:
                active = False
        elif c >= enter:
            active = True
        mask[i] = active
    return mask


def _runs(mask, value):
    """(start, end_exclusive) index pairs for contiguous runs of mask == value."""
    padded = np.r_[False, mask == value, False]
    change = np.diff(padded.astype(np.int8))
    return list(zip(np.where(change == 1)[0], np.where(change == -1)[0]))


def bridge_short_gaps(t, mask, max_gap_time):
    """Fill brief interior False gaps (momentary tracking dropouts), but never runs that
    touch a series edge — their true extent beyond the recording is unknown."""
    out = mask.copy()
    for start, end in _runs(mask, False):
        touches_edge = start == 0 or end == len(mask)
        if not touches_edge and t[end - 1] - t[start] <= max_gap_time:
            out[start:end] = True
    return out


def drop_short_runs(t, mask, min_time):
    """Remove True runs shorter than `min_time` seconds (spurious blips)."""
    out = mask.copy()
    for start, end in _runs(mask, True):
        if t[end - 1] - t[start] < min_time:
            out[start:end] = False
    return out


def decide_contact(t, confidence):
    mask = hysteresis_mask(confidence, ENTER_CONFIDENCE, EXIT_CONFIDENCE)
    mask = bridge_short_gaps(t, mask, MAX_GAP_TIME)
    mask = drop_short_runs(t, mask, MIN_CONTACT_TIME)
    return mask


# --------------------------------------------------------------------------------------
# Run the whole pipeline and report.
# --------------------------------------------------------------------------------------

def main():
    t, points, yaw = synthesize_trajectory()
    features, bias = compute_features(t, points, yaw)
    confidence = contact_confidence(features)
    contact = decide_contact(t, confidence)

    print(f"Recovered resting-clearance bias: {bias * 1000:.2f} mm (injected 4.00 mm)\n")

    # Boolean intervals where the sole is in contact.
    intervals = [(t[s], t[e - 1]) for s, e in _runs(contact, True)]
    print("Detected contact intervals:")
    for a, b in intervals:
        print(f"  [{a:5.2f}s, {b:5.2f}s]   ({b - a:.2f}s)")
    print("  (ground truth: the sole rests on the floor over [1.00s, 2.50s])\n")

    # A compact text timeline: '#' = in contact, '.' = airborne (subsampled).
    step = max(1, len(t) // 80)
    strip = "".join("#" if c else "." for c in contact[::step])
    print("Timeline (left=0s, right=4s):")
    print("  " + strip)
    print("  " + "".join("^" if i % 25 == 0 else " " for i in range(len(strip))))

    # Optional: a labelled plot if matplotlib is installed.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib to also see a plotted timeline)")
        return
    import os

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
    # Height (cm) and confidence (0-1) live on separate y-axes so both are readable.
    ax1.plot(t, points[:, 2] * 100, color="tab:blue", label="height (cm)")
    ax1.set_ylabel("height (cm)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    axc = ax1.twinx()
    axc.plot(t, confidence, color="tab:orange", label="contact confidence")
    axc.axhline(ENTER_CONFIDENCE, ls="--", c="gray", lw=0.8, label="enter threshold")
    axc.set_ylabel("contact confidence", color="tab:orange")
    axc.tick_params(axis="y", labelcolor="tab:orange")
    axc.set_ylim(-0.05, 1.05)
    axc.legend(loc="center right")
    ax2.fill_between(t, contact.astype(float), step="mid", alpha=0.4)
    ax2.set_ylabel("in contact")
    ax2.set_xlabel("time (s)")
    ax2.set_yticks([0, 1])
    fig.suptitle("Contact detection from scratch — sole vs. floor")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contact_timeline.png")
    fig.savefig(out, dpi=120)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
