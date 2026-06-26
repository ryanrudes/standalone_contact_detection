"""Make/break contact event detection (THEORY.md section 6).

The sustained-contact posterior (sections 4-5) tells us *that* a contact existed
over some span of frames, but it pins the *instant* of touchdown / liftoff only to
within the resolution of the discrete state estimate -- which is exactly the part
that matters least for "am I touching?" and most for timing. THEORY.md section 6
argues that the make/break instants are singular: they are the guards/resets of the
hybrid system, where the relative normal velocity is arrested (touchdown) or first
allowed to grow (liftoff), and where the gap crosses zero. The *kinematics near the
boundary* therefore localize the event far more sharply than the onset of the
"contact" label does.

This module takes the (already temporally-coherent) boolean ``in_contact`` mask and,
for each free->contact and contact->free transition it contains, refines the event
*time* by sub-frame linear interpolation of the relevant zero-crossing in the
kinematics straddling the boundary:

* **touchdown** (free -> contact): the gap reaches zero, i.e. the bodies meet. We
  interpolate the zero-crossing of ``gap`` across the transition. If the gap never
  actually crosses zero in the local window (noisy / already-resting data) we fall
  back to the zero-crossing of the relative normal velocity ``v_normal`` -- the
  moment the closing motion is arrested (``v_normal`` rising through 0 from the
  approaching, negative side), which section 6 calls out as the precise event timer.
* **liftoff** (contact -> free): the contact starts separating. The gap reopens
  (``gap`` crosses zero going positive) and/or ``v_normal`` becomes persistently
  positive. We interpolate the gap zero-crossing, falling back to the ``v_normal``
  sign change into separation.

Sub-frame interpolation gives a fractional-index time strictly more precise than the
frame grid; ``index`` is reported as the nearest frame for callers that need an
integer anchor. Events are returned sorted by time.

This module imports only ``contact.types`` (and, when available, the leaf
``contact.signals`` smoothing helpers), per the package's dependency rules.
"""

from __future__ import annotations

import numpy as np

from .types import ContactEvent, ContactObservations

# ``contact.signals`` is a leaf helper module (THEORY.md section 6: events should be
# detected on a *lightly*-filtered signal, with smoothing kept local so it does not
# erase the very jumps we are timing). We use it opportunistically for a gentle
# pre-smooth of v_normal when present, but never hard-depend on it: the refinement
# math below is exact zero-crossing interpolation and stands on its own.
try:  # pragma: no cover - availability depends on build order
    from . import signals as _signals  # type: ignore
except Exception:  # pragma: no cover
    _signals = None  # type: ignore[assignment]


__all__ = ["detect_events"]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _as_obs(obs: ContactObservations) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull the (t, gap, v_normal) channels we need as float arrays.

    THEORY.md section 6: the gap zero-crossing locates the geometric meeting; the
    relative-normal-velocity sign change locates the arrest/separation. Those two
    channels are all the kinematics the event refinement consumes.
    """
    t = np.asarray(obs.t, dtype=float).ravel()
    gap = np.asarray(obs.gap, dtype=float).ravel()
    v_normal = np.asarray(obs.v_normal, dtype=float).ravel()
    return t, gap, v_normal


def _interp_time_at_index(t: np.ndarray, frac_index: float) -> float:
    """Map a (possibly fractional) frame index to a time by linear interpolation.

    A fractional index ``i + alpha`` (0 <= alpha <= 1) corresponds to the time
    ``t[i] + alpha * (t[i+1] - t[i])`` -- this is what makes the refined event time
    finer than the frame grid even when ``t`` is non-uniform.
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


def _zero_crossing_frac_index(
    y: np.ndarray,
    lo: int,
    hi: int,
    *,
    rising: bool | None = None,
) -> float | None:
    """Fractional index of the zero-crossing of ``y`` in the closed span [lo, hi].

    We scan consecutive samples and return the first bracket ``[k, k+1]`` whose
    endpoints straddle zero (with the requested sign of crossing, if any), then place
    the crossing by linear interpolation:

        frac = k + y[k] / (y[k] - y[k+1])

    ``rising`` selects the crossing direction:
      * ``True``  -> from negative to positive (e.g. v_normal arrested into separation,
        or gap reopening at liftoff);
      * ``False`` -> from positive to negative (e.g. gap closing at touchdown);
      * ``None``  -> either direction (first sign change of any kind).

    Returns ``None`` if there is no qualifying crossing in the span -- the caller then
    falls back to the other channel. (THEORY.md section 6: the gap crossing is the
    primary timer; v_normal's arrest/onset is the principled fallback.)
    """
    lo = max(0, lo)
    hi = min(y.shape[0] - 1, hi)
    if hi <= lo:
        return None
    for k in range(lo, hi):
        a = y[k]
        b = y[k + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        # Exact landing on a sample.
        if a == 0.0:
            if rising is None or (rising and b >= 0.0) or ((not rising) and b <= 0.0):
                return float(k)
        # Strict straddle of zero.
        if a < 0.0 < b:  # rising crossing
            if rising is None or rising:
                return float(k) + a / (a - b)
        elif a > 0.0 > b:  # falling crossing
            if rising is None or (not rising):
                return float(k) + a / (a - b)
    # Endpoint exactly zero.
    if y[hi] == 0.0:
        return float(hi)
    return None


def _smooth_v_normal(t: np.ndarray, v_normal: np.ndarray) -> np.ndarray:
    """Lightly pre-smooth v_normal for the sign-change fallback, if signals allows.

    THEORY.md section 6 insists smoothing here be *local* so it never erases the jump
    we are timing. If ``contact.signals`` exposes a smoothing helper we use it with a
    short time constant; otherwise we return the raw channel unchanged (the gap-based
    primary path is unaffected either way).
    """
    if _signals is None:
        return v_normal
    fn = (
        getattr(_signals, "smooth", None)
        or getattr(_signals, "gaussian_smooth", None)
        or getattr(_signals, "smooth_signal", None)
    )
    if fn is None:
        return v_normal
    try:  # pragma: no cover - exact signature unknown until signals lands
        out = np.asarray(fn(t, v_normal), dtype=float).ravel()
        if out.shape == v_normal.shape:
            return out
    except Exception:
        pass
    return v_normal


def _refine_transition(
    kind: str,
    boundary: int,
    t: np.ndarray,
    gap: np.ndarray,
    v_normal: np.ndarray,
) -> tuple[float, int]:
    """Refine one transition into a (time, nearest_index) pair.

    ``boundary`` is the index of the FIRST frame of the new state, so the transition
    physically occurs somewhere in the bracket ``[boundary - 1, boundary]``. We widen
    the search by one frame on each side to tolerate sampling jitter, then:

      * touchdown: locate the gap *falling* through zero (bodies meeting). Fallback:
        v_normal *rising* through zero (the closing motion arrested) -- section 6.
      * liftoff: locate the gap *rising* through zero (gap reopening). Fallback:
        v_normal *rising* through zero (motion turns to separation).

    Returns the interpolated event time and the nearest integer frame index.
    """
    lo = boundary - 2
    hi = boundary + 1

    if kind == "touchdown":
        frac = _zero_crossing_frac_index(gap, lo, hi, rising=False)
        if frac is None:
            frac = _zero_crossing_frac_index(v_normal, lo, hi, rising=True)
    else:  # liftoff
        frac = _zero_crossing_frac_index(gap, lo, hi, rising=True)
        if frac is None:
            frac = _zero_crossing_frac_index(v_normal, lo, hi, rising=True)

    if frac is None:
        # No usable kinematic crossing in the window: fall back to the geometric
        # midpoint of the bracket the mask itself reported.
        frac = boundary - 0.5

    # Clamp to the valid index range so a noisy widened window can't escape the array.
    frac = float(np.clip(frac, 0.0, t.shape[0] - 1))
    time = _interp_time_at_index(t, frac)
    index = int(np.clip(round(frac), 0, t.shape[0] - 1))
    return time, index


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def detect_events(
    obs: ContactObservations,
    in_contact: np.ndarray,
    t: np.ndarray | None = None,
) -> list[ContactEvent]:
    """Detect and sub-frame-refine touchdown/liftoff events (THEORY.md section 6).

    Find every free->contact (touchdown) and contact->free (liftoff) transition in the
    boolean ``in_contact`` mask, then refine each event's *time* using the kinematics
    straddling the boundary: a touchdown is placed at the gap zero-crossing (or, as a
    fallback, the arrest of the relative normal closing velocity); a liftoff at the gap
    reopening (or the onset of separating normal velocity). Times are obtained by
    linear interpolation of the zero-crossing, yielding a fractional-index instant more
    precise than the frame grid; ``index`` is the nearest frame.

    The mask is allowed to *start or end mid-contact*: an initial ``True`` run is not a
    touchdown (we never saw it land) and a trailing ``True`` run is not a liftoff (we
    never saw it leave), so no spurious event is emitted at ``t[0]`` or ``t[-1]`` for an
    already-active contact.

    Parameters
    ----------
    obs:
        Per-frame support-relative observations; only ``obs.t``, ``obs.gap`` and
        ``obs.v_normal`` are consumed (THEORY.md sections 1 & 6).
    in_contact:
        ``(T,)`` boolean contact mask (typically ``DetectionResult.in_contact`` from
        the Viterbi segmentation of section 5).
    t:
        Optional ``(T,)`` time base overriding ``obs.t`` (e.g. a resampled grid). When
        ``None`` (default) the timestamps in ``obs`` are used.

    Returns
    -------
    list[ContactEvent]
        ``ContactEvent(kind, time, index)`` for each transition, sorted by ``time``.
        ``kind`` is ``"touchdown"`` or ``"liftoff"``.
    """
    obs_t, gap, v_normal = _as_obs(obs)
    time_base = obs_t if t is None else np.asarray(t, dtype=float).ravel()

    mask = np.asarray(in_contact).ravel().astype(bool)
    n = mask.shape[0]

    # Degenerate inputs: nothing to transition between.
    if n < 2 or time_base.shape[0] < 2:
        return []

    # Light local smoothing of the fallback channel only (section 6: keep it local).
    v_for_fallback = _smooth_v_normal(time_base, v_normal)

    # A transition happens at frame i (1 <= i <= n-1) when mask[i] != mask[i-1]; frame i
    # is the first frame of the new state. mask[i] True => free->contact (touchdown);
    # mask[i] False => contact->free (liftoff). Boundaries are interior by construction,
    # so an initial or trailing mid-contact run produces no event (handled for free).
    events: list[ContactEvent] = []
    changes = np.nonzero(mask[1:] != mask[:-1])[0] + 1  # indices i with a switch at i
    for i in changes:
        boundary = int(i)
        kind = "touchdown" if mask[boundary] else "liftoff"
        chan_v = v_for_fallback
        time, index = _refine_transition(kind, boundary, time_base, gap, chan_v)
        events.append(ContactEvent(kind=kind, time=time, index=index))

    events.sort(key=lambda e: e.time)
    return events
