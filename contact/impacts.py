"""Impact detection and characterization (THEORY.md s.6).

The sustained-contact modes of s.3-s.5 live on a *smooth* timescale: within a mode
the relative twist follows a continuous flow. The make/break instants are different
in kind. At touchdown the relative normal velocity is arrested almost
discontinuously -- in the hybrid-system language of s.5 this is a **reset map**,
``v+ = -e v-`` with ``e`` the coefficient of restitution. The clean object that
holds both sustained force and these jumps at once is the contact force as a
**measure** (s.6):

    dnu = lambda(t) dt  +  sum_i p_i delta(t - t_i)

-- a smooth part (ordinary sustained force) *plus atoms* (impulses ``p_i`` at impact
instants ``t_i``), with the velocity allowed to jump at those atoms. This module
finds those atoms and characterizes each one.

Three facts from s.6 dictate *how* we detect them, and they shape every choice below:

* **Impacts are the precise event timers.** The arrest of the normal velocity pins
  the touchdown *time* far more sharply than the gradual onset of the "contact"
  state, so we localize on the velocity step itself, sub-frame.
* **They momentarily reveal force and material.** The impulse equals the change in
  momentum, ``integral lambda dt = m * delta v`` (a force reading with no force
  plate, when mass is known); the velocity ratio across the event estimates the
  restitution ``e``.
* **Smoothing here must be local.** ``v_normal`` is a function of bounded variation
  -- smooth with genuine jumps. A single wide global smoother forbids those jumps and
  is therefore *wrong at exactly the moments we most care about*. So we smooth only
  lightly (``ImpactParams.detect_smooth_time``) and detect the event by fitting a
  parametric arrest **template** (a matched filter) rather than by blindly
  differentiating noisy positions.

Detection method (matched filter)
---------------------------------
A clean impact is a step in the relative normal velocity: the body closes at some
speed (``v_normal < 0`` -- by the s.1 convention +ve is *separating*, so closing is
negative) and is abruptly arrested to ~0 (a stick/plastic landing) or reversed to
``-e v_before`` (a bounce). Differentiating a step gives a *spike*; equivalently, the
matched filter for a step is a step-edge / derivative-of-step kernel. We therefore
correlate ``v_normal`` against an antisymmetric **arrest template**

    K(tau) = -1   for  -W <= tau < 0      (the "before": closing, more negative)
    K(tau) = +1   for   0 <  tau <= W     (the "after": arrested/rebounded, less negative)

(zero-mean, unit-norm) of half-width ``W = template_halfwidth_time`` seconds. The
local correlation ``s[i] = sum_tau K(tau) v_normal[i+tau]`` is large and positive
exactly where ``v_normal`` *rises* over the window -- i.e. where a closing (negative)
velocity is arrested toward / past zero. This is the discrete matched filter for the
arrest edge; its peaks are impact candidates. (A pure separation event -- liftoff --
also rises through zero but starts from ~0, not from a closing speed, so the
``min_closing_speed`` gate below rejects it: an impact is an *arrest of approach*, not
the *onset of departure*.)

A candidate at the correlation peak is accepted as an impact iff:

  1. it is a local maximum of ``s`` (the template is best aligned there), and
  2. the closing speed just before exceeds ``min_closing_speed`` -- a genuine
     approach was arrested, not sensor jitter near rest.

Restitution estimator
----------------------
Across an accepted impact we bracket the event by the template half-width and read

    v_before = (representative v_normal in the W-window *before* the arrest)
    v_after  = (representative v_normal in the W-window *after*  the arrest)

using a robust median over each side (the velocity is piecewise-smooth on each side,
so a median rejects the single transition sample and any spikes). Then, per s.6,

    closing_speed  = |v_before|                     (the approach speed, always >= 0)
    restitution e  = max(0, -v_after / v_before)     (ratio of separating to closing)

The sign logic: closing means ``v_before < 0`` (approaching). A bounce reverses the
sign, ``v_after > 0`` (separating), so ``-v_after / v_before = -(+)/(-) = +`` -- a
positive ``e`` in ``(0, 1]``. A plastic/sticking landing has ``v_after ~ 0`` so
``e ~ 0``. We clip negatives to 0 (a body cannot leave faster *into* the surface),
and report ``NaN`` when ``e`` is not measurable: when ``v_before`` is ~0 (no real
approach) or either side is non-finite. Note we deliberately do NOT substitute a
prior here (``ImpactParams.restitution_default`` is the HMM's prior for *unmeasured*
events elsewhere); a detected impact reports what its own kinematics support, and
``NaN`` honestly flags "bounce not resolved" rather than fabricating a number.

Normal-impulse atom
-------------------
With the moving body's ``mass`` known, the atom magnitude is the momentum jump
(s.6, ``integral lambda dt = m * delta v``):

    normal_impulse = mass * (v_after - v_before)        (N*s)

For a closing-then-arrested event ``v_after - v_before >= 0``, so the impulse is the
(positive) normal momentum delivered by the surface. With ``mass=None`` the atom's
magnitude is unobservable from kinematics alone (s.7), so we report ``NaN``.

This module imports only ``contact.types``, ``contact.config``, ``contact.signals``
and numpy, per the package dependency rules. It is a pure, offline (smoothing-window)
estimator: like the s.6 latency note, using frames *after* the arrest is what makes
the bounce measurable at all.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .config import ImpactParams
from .signals import gaussian_smooth
from .types import ContactImpulse, ContactObservations

__all__ = ["detect_impacts"]


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------

def _typical_dt(t: np.ndarray) -> float:
    """Representative sample spacing (seconds) of a possibly non-uniform clock.

    The matched-filter template half-width is specified in *time*; to build it as an
    integer number of samples we need a sample period. We use the median frame gap so
    that a few dropped frames or clock jitter do not distort the kernel size.
    """
    if t.shape[0] < 2:
        return 0.0
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0.0)]
    if d.size == 0:
        return 0.0
    return float(np.median(d))


def _arrest_template(half_samples: int) -> np.ndarray:
    """Zero-mean, unit-L2-norm antisymmetric arrest (velocity-step) kernel.

    The kernel is ``-1`` over the ``half_samples`` taps *before* the centre and ``+1``
    over the ``half_samples`` taps *after* it (the centre tap is 0). Correlating
    ``v_normal`` with this kernel measures how much the normal velocity *rises* across
    the window -- the signature of an arrest of approach (THEORY.md s.6). It is the
    discrete matched filter for a velocity step: the step's "derivative" is the edge,
    and an edge detector is exactly this antisymmetric difference of two boxcars.

    Zero-mean makes the response insensitive to any constant offset in ``v_normal``
    (only the *change* matters); unit-norm makes peak heights comparable across
    records so a single absolute threshold logic is meaningful.
    """
    n = 2 * half_samples + 1
    k = np.zeros(n, dtype=float)
    k[:half_samples] = -1.0
    k[half_samples + 1:] = +1.0
    norm = np.linalg.norm(k)
    if norm > 0.0:
        k /= norm
    return k


def _correlate_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Sliding correlation of ``x`` with ``kernel``, edge-replicated, length-preserving.

    ``s[i] = sum_j kernel[j] * x[i + j - half]`` with ``x`` extended at the boundaries
    by edge replication (not zero-padding: zero-padding would invent a spurious step at
    the record ends and hallucinate an impact there). Returns an array the same length
    as ``x``.
    """
    half = (kernel.shape[0] - 1) // 2
    xp = np.pad(x, (half, half), mode="edge")
    # np.correlate(..., "valid") slides the kernel without flipping it (correlation,
    # not convolution), which is what "matched filter" means here.
    return np.correlate(xp, kernel, mode="valid")


def _side_velocity(
    v: np.ndarray, center: int, half_samples: int, *, after: bool, guard: int = 0
) -> float:
    """Robust representative normal velocity on one side of an arrest.

    ``v`` is piecewise-smooth on each side of the event (s.6), so a *median* over a
    window on that side cleanly rejects the transition samples and any isolated noise
    spike, giving a stable ``v_before`` / ``v_after``.

    The light pre-smoothing (and the discrete step itself) blurs the velocity across a
    few samples *straddling* the arrest, so reading right up against the centre would
    bias ``v_before``/``v_after`` toward each other (a measured restitution dragged
    toward 0.5 of its true value). We therefore skip a ``guard`` band of samples
    immediately adjacent to the centre and read the *asymptotic plateau* on each side:

      * ``after=False`` reads the closing side ``[center-guard-W, center-guard-1]``;
      * ``after=True``  reads the rebound/arrest side ``[center+guard+1, center+guard+W]``.

    Windows are clamped to the array; if a side collapses at a boundary we fall back to
    the nearest single available sample so the estimate degrades gracefully rather than
    returning NaN at the record edges.
    """
    n = v.shape[0]
    if after:
        lo, hi = center + guard + 1, center + guard + half_samples
    else:
        lo, hi = center - guard - half_samples, center - guard - 1
    lo = max(0, lo)
    hi = min(n - 1, hi)
    if hi < lo:
        # Window collapsed at a boundary: fall back to the single nearest sample on
        # this side, ignoring the guard so we never return NaN purely from clamping.
        idx = center + (1 if after else -1)
        idx = min(n - 1, max(0, idx))
        return float(v[idx])
    seg = v[lo:hi + 1]
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        return float("nan")
    return float(np.median(seg))


def _refine_center_subframe(s: np.ndarray, peak: int) -> float:
    """Sub-frame location of the matched-filter peak by parabolic interpolation.

    Fitting a parabola through the correlation values ``s[peak-1], s[peak], s[peak+1]``
    and taking its vertex gives a fractional-index peak finer than the frame grid --
    the same spirit as the zero-crossing interpolation in ``events.py``. The vertex
    offset is ``0.5 (s[-] - s[+]) / (s[-] - 2 s[0] + s[+])`` and is clamped to
    ``[-0.5, 0.5]`` so it never leaves the sampled bracket.
    """
    n = s.shape[0]
    if peak <= 0 or peak >= n - 1:
        return float(peak)
    a = s[peak - 1]
    b = s[peak]
    c = s[peak + 1]
    denom = a - 2.0 * b + c
    if not np.isfinite(denom) or denom == 0.0:
        return float(peak)
    offset = 0.5 * (a - c) / denom
    if not np.isfinite(offset):
        return float(peak)
    offset = float(np.clip(offset, -0.5, 0.5))
    return peak + offset


def _interp_time(t: np.ndarray, frac_index: float) -> float:
    """Time at a (possibly fractional) frame index by linear interpolation of ``t``.

    Matches ``events.py`` so impact times and make/break event times share one
    sub-frame convention on a non-uniform clock.
    """
    n = t.shape[0]
    if n == 0:
        return float("nan")
    if frac_index <= 0.0:
        return float(t[0])
    if frac_index >= n - 1:
        return float(t[-1])
    i = int(np.floor(frac_index))
    alpha = frac_index - i
    return float(t[i] + alpha * (t[i + 1] - t[i]))


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def detect_impacts(
    obs: ContactObservations,
    params: ImpactParams,
    mass: float | None = None,
) -> list[ContactImpulse]:
    """Detect and characterize impact atoms in the force measure (THEORY.md s.6).

    Pipeline (each step traced to s.6):

      1. **Light local smoothing.** Smooth ``obs.v_normal`` with a Gaussian of
         ``params.detect_smooth_time`` seconds. This tames differentiation/sensor
         noise without forbidding the velocity jump -- over-smoothing destroys the
         very timing we are after, so the time constant is kept small.
      2. **Matched filter.** Correlate the lightly-smoothed ``v_normal`` against the
         antisymmetric arrest (velocity-step) template of half-width
         ``params.template_halfwidth_time``. The response peaks where a *closing*
         (negative) normal velocity is rapidly *arrested* toward / past zero.
      3. **Peak gating.** Keep response peaks (local maxima of the correlation) whose
         pre-event closing speed ``|v_before|`` exceeds ``params.min_closing_speed``.
         The closing-speed gate is what separates a true touchdown (arrest of
         approach) from a liftoff (onset of departure, which also rises through zero
         but from rest).
      4. **Characterize.** For each accepted impact, measure ``closing_speed``,
         ``restitution = max(0, -v_after / v_before)`` (NaN if unmeasurable / not a
         bounce), and ``normal_impulse = mass * (v_after - v_before)`` (NaN if ``mass``
         is None). Time is sub-frame (parabolic peak interpolation); ``index`` is the
         nearest frame.

    Parameters
    ----------
    obs:
        Per-frame support-relative observations; only ``obs.t`` and ``obs.v_normal``
        are consumed (s.1: support-relative; s.6: the normal channel carries the
        arrest).
    params:
        ``ImpactParams`` (template half-width, min closing speed, smoothing time).
    mass:
        Moving body's mass (kg) for the impulse atom; ``None`` => impulse unobservable
        from kinematics (s.7), reported as ``NaN``.

    Returns
    -------
    list[ContactImpulse]
        One per detected impact, sorted by ``time``.
    """
    t = np.asarray(obs.t, dtype=float).ravel()
    v = np.asarray(obs.v_normal, dtype=float).ravel()
    n = t.shape[0]
    if n != v.shape[0]:
        raise ValueError(
            f"obs.t and obs.v_normal must have matching length; "
            f"got {n} and {v.shape[0]}"
        )
    if n < 3:
        # Too few frames to define a velocity step and its bracketing windows.
        return []

    # --- step 1: light, LOCAL smoothing (s.6: never wide -- it would erase the jump) ---
    v_s = gaussian_smooth(v, t, sigma_time=max(0.0, params.detect_smooth_time))

    # --- build the arrest template at the resolution of this clock ---
    dt = _typical_dt(t)
    if dt <= 0.0:
        return []
    half = int(round(params.template_halfwidth_time / dt))
    half = max(1, half)                       # need at least one tap per side
    half = min(half, (n - 1) // 2)            # and the kernel must fit in the record
    if half < 1:
        return []
    kernel = _arrest_template(half)

    # Guard band (samples) to skip immediately either side of an arrest when reading the
    # plateau velocities: the Gaussian pre-smooth (sigma = detect_smooth_time) blurs the
    # step over roughly +-2 sigma, and the discrete step itself spans one sample. Skipping
    # this band reads the asymptotic v_before / v_after instead of the blurred ramp, so the
    # restitution estimate is not dragged toward (v_before+v_after)/2. We cap the guard so
    # that guard + a 1-sample read still fits inside the template half-width.
    guard = int(np.ceil(2.0 * max(0.0, params.detect_smooth_time) / dt))
    guard = max(1, guard)
    guard = min(guard, max(0, half - 1))

    # --- step 2: matched filter (correlation), edge-replicated to preserve length ---
    response = _correlate_same(v_s, kernel)   # (n,)

    # --- step 3: find strong, well-separated response peaks = arrest candidates ---
    # The response of a genuine arrest is a clean triangular bump whose height scales
    # with the velocity step it matched. We turn that into discrete candidates with a
    # significance threshold (rejects flat/zero stretches and float wobble) followed by
    # non-maximum suppression over the template half-width (one atom per arrest -- s.6).
    #
    # The threshold is grounded in the physical closing-speed gate, NOT in the strongest
    # peak: a multi-bounce sequence has *decaying* peaks, so a "fraction of the max" floor
    # would silently drop the later (weaker but real) bounces. A unit-norm step kernel
    # maps a velocity step of magnitude dv to a peak response of dv*||boxcar|| = dv*sqrt(half/2),
    # so the smallest step worth detecting -- a `min_closing_speed` arrest to rest --
    # produces a response of `min_closing_speed * sqrt(half/2)`. We threshold at half of
    # that: low enough to keep every physically real arrest, high enough that the exactly-
    # zero free-flight/rest response (and its +-0.0 float wobble) never qualifies. The
    # definitive physical filter is then the closing-speed gate applied per candidate
    # in step 4.
    min_step_response = params.min_closing_speed * np.sqrt(half / 2.0)
    threshold = max(0.5 * min_step_response, 1e-9)
    # Strong, well-separated positive maxima. find_peaks applies the significance floor
    # (height) and the greedy descending non-maximum suppression (distance) in one call;
    # distance=half+1 reproduces the old strict ``|i - j| > half`` separation bit-for-bit
    # (one atom per arrest, s.6). The caller sorts the final impulses by time.
    candidates, _ = find_peaks(response, height=threshold, distance=half + 1)

    # --- step 4: gate by closing speed, then characterize each accepted impact ---
    impulses: list[ContactImpulse] = []
    for c in candidates:
        v_before = _side_velocity(v_s, c, half, after=False, guard=guard)
        v_after = _side_velocity(v_s, c, half, after=True, guard=guard)

        # Closing speed = magnitude of the (approaching, i.e. negative) pre-event
        # velocity. The gate rejects liftoffs (v_before ~ 0) and noise near rest.
        if not np.isfinite(v_before):
            continue
        closing_speed = -v_before if v_before < 0.0 else 0.0
        if closing_speed < params.min_closing_speed:
            continue

        # An impact is an *arrest of approach*: the velocity must actually rise across
        # the event (v_after > v_before). A body that keeps closing at the same speed
        # (v_after ~ v_before) has not hit anything -- that pattern is uniform approach
        # plus noise, not an atom. We require the rise to clear the same physical scale
        # as the entry gate, which rejects the residual noise-driven false positives that
        # the matched filter alone passes (a downward-then-upward noise blip on a steady
        # closing velocity) while keeping every genuine arrest, partial or full.
        if not np.isfinite(v_after):
            continue
        if (v_after - v_before) < params.min_closing_speed:
            continue

        # Restitution e = max(0, -v_after / v_before). NaN if not measurable.
        if np.isfinite(v_after) and v_before != 0.0:
            e = -v_after / v_before
            restitution = float(max(0.0, e))
        else:
            restitution = float("nan")

        # Impulse atom = m * delta-v_normal (s.6). NaN if mass unknown (s.7).
        if mass is not None and np.isfinite(v_after) and np.isfinite(v_before):
            normal_impulse = float(mass * (v_after - v_before))
        else:
            normal_impulse = float("nan")

        # Sub-frame time from the matched-filter peak; nearest integer frame index.
        frac = _refine_center_subframe(response, c)
        frac = float(np.clip(frac, 0.0, n - 1))
        time = _interp_time(t, frac)
        index = int(np.clip(round(frac), 0, n - 1))

        impulses.append(
            ContactImpulse(
                time=time,
                index=index,
                closing_speed=float(closing_speed),
                restitution=restitution,
                normal_impulse=normal_impulse,
            )
        )

    impulses.sort(key=lambda imp: imp.time)
    return impulses
