"""Time-aware numeric helpers: smoothing and differentiation on non-uniform clocks.

This is a *leaf* module (numpy/scipy only, no other `contact` submodules). Everything
here is a pure function of its inputs.

Why "time-aware"? Real mocap is rarely perfectly uniform: frames drop, clocks jitter,
and streams from different sensors get resampled. A helper that secretly assumes a
fixed `dt` will silently mis-scale velocities the moment the sampling wobbles. So
every routine below takes the explicit timestamp array `t` and measures its windows
and kernels in *seconds*, not in samples.

The governing caution comes from THEORY.md section 6: the contact velocity is a
function of *bounded variation* — smooth within a mode but with genuine jumps at
impacts (the reset map `v+ = -e v-`). A single aggressive global smoother forbids
those jumps and therefore destroys the very make/break timing that section 6 says
carries the most information. Hence smoothing here is always *optional and local*:
small `sigma_time` / `window_time` measured in real seconds, never a wide filter.
"""

from __future__ import annotations

import numpy as np


def _as_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return `x` viewed as (T, D) plus a flag recording whether it was originally 1-D.

    Internal helper so every public function can handle both a scalar channel (T,) and
    a vector channel (T, D) with one code path, then restore the caller's shape.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None], True
    if x.ndim == 2:
        return x, False
    raise ValueError(f"signal must be (T,) or (T, D); got shape {x.shape}")


def gaussian_smooth(x: np.ndarray, t: np.ndarray, sigma_time: float) -> np.ndarray:
    """Gaussian-smooth a (T,) or (T, D) signal with a kernel measured in real time.

    The kernel width `sigma_time` is in *seconds*, so the effective number of samples
    averaged adapts automatically to the (possibly non-uniform) sampling in `t`. Weights
    are row-normalized, so a constant signal is returned unchanged (DC is preserved) and
    boundaries are handled without bias. `sigma_time <= 0` is a no-op (returns `x`).

    Keep `sigma_time` small: THEORY.md section 6 warns that wide smoothing forbids the
    velocity jumps at impacts and so destroys make/break timing.

    Parameters
    ----------
    x : (T,) or (T, D) signal samples.
    t : (T,) strictly-increasing timestamps in seconds.
    sigma_time : Gaussian standard deviation in seconds. <= 0 => no smoothing.

    Returns
    -------
    (T,) or (T, D) smoothed signal, same shape as `x`.
    """
    t = np.asarray(t, dtype=float)
    if sigma_time <= 0.0:
        return np.asarray(x, dtype=float)

    xx, was_1d = _as_2d(x)
    T = xx.shape[0]
    if t.shape != (T,):
        raise ValueError(f"t must be ({T},) to match the signal; got {t.shape}")

    # Full dense (T, T) kernel weighted by the *time* gap between sample i and sample j.
    # T is small for these trajectories, so the O(T^2) build is fine and is far simpler
    # (and exactly correct for non-uniform t) than an FFT/box approximation.
    dt = t[:, None] - t[None, :]                     # (T, T) signed time differences
    # Work in log-space then exponentiate once: avoids forming/normalizing tiny products
    # and is numerically stable (we subtract the per-row max before exp).
    log_w = -0.5 * (dt / sigma_time) ** 2            # (T, T) log un-normalized weights
    log_w -= log_w.max(axis=1, keepdims=True)        # stabilize before exp (per row)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)                # row-normalize => preserves DC

    out = w @ xx                                     # (T, T) @ (T, D) -> (T, D)
    return out[:, 0] if was_1d else out


def savgol_derivative(
    x: np.ndarray, t: np.ndarray, window_time: float, polyorder: int = 2
) -> np.ndarray:
    """Local least-squares (Savitzky-Golay-style) time derivative of a (T,) / (T, D) signal.

    At each sample we fit a degree-`polyorder` polynomial in time to the neighbours
    inside a window of `window_time` seconds (centred where possible, truncated at the
    ends) and read off the analytic slope at that sample. Because the fit is in real
    time, this handles non-uniform sampling correctly, and because it is *local* it does
    not smear the velocity jump at an impact across the whole record (THEORY.md s.6).

    Noise/robustness tradeoff (inline): a wider `window_time` averages out more
    differentiation noise (the killer of bare finite differences) but a window that
    spans an impact will fit a smooth polynomial straight through the discontinuity and
    blur its timing. So `window_time` is the single knob trading variance for impact
    fidelity — keep it of order the impact duration, not the whole contact.

    Parameters
    ----------
    x : (T,) or (T, D) signal samples.
    t : (T,) strictly-increasing timestamps in seconds.
    window_time : full window width in seconds for the local fit (must be > 0).
    polyorder : polynomial degree of the local fit (default 2 => locally quadratic).

    Returns
    -------
    (T,) or (T, D) time derivative dx/dt, same shape as `x`.
    """
    if window_time <= 0.0:
        raise ValueError("window_time must be > 0 for a local-polynomial derivative")

    t = np.asarray(t, dtype=float)
    xx, was_1d = _as_2d(x)
    T, D = xx.shape
    if t.shape != (T,):
        raise ValueError(f"t must be ({T},) to match the signal; got {t.shape}")
    if T < 2:
        # Not enough data to define a slope; derivative of a single sample is 0.
        out = np.zeros_like(xx)
        return out[:, 0] if was_1d else out

    half = 0.5 * window_time
    out = np.empty_like(xx)

    for i in range(T):
        # Neighbours within +/- half a window in TIME (not in sample count) of sample i.
        lo = np.searchsorted(t, t[i] - half, side="left")
        hi = np.searchsorted(t, t[i] + half, side="right")
        idx = np.arange(lo, hi)

        # The local fit needs at least (polyorder + 1) distinct points; if the window is
        # too sparse near the ends, shrink the polynomial degree to what the data support
        # (degree 1 => a plain local linear slope, the most robust fallback).
        npts = idx.size
        deg = min(polyorder, npts - 1)
        if deg < 1:
            # Degenerate single-point window: fall back to a one-sided finite difference.
            j = i + 1 if i == 0 else i - 1
            out[i] = (xx[i] - xx[j]) / (t[i] - t[j])
            continue

        # Centre time at t[i] so the fitted coefficient for the linear term IS the
        # derivative at t[i] (Vandermonde column 1), with no extra evaluation needed.
        tau = t[idx] - t[i]                          # (npts,) local time coords
        V = np.vander(tau, N=deg + 1, increasing=True)  # [1, tau, tau^2, ...]
        # Least-squares fit of every channel at once: coeffs is (deg+1, D).
        coeffs, *_ = np.linalg.lstsq(V, xx[idx], rcond=None)
        out[i] = coeffs[1]                           # d/dt at tau=0 == linear coefficient

    return out[:, 0] if was_1d else out


def derivative(
    x: np.ndarray, t: np.ndarray, smooth_time: float = 0.0
) -> np.ndarray:
    """Robust time derivative d/dt of a (T,) or (T, D) signal on a non-uniform clock.

    Two regimes, chosen by `smooth_time`:
      * `smooth_time > 0` : a *local* least-squares fit over a window of `smooth_time`
        seconds (`savgol_derivative`). This is markedly more robust to noise than raw
        finite differences, which amplify it (THEORY.md s.4), while staying local so it
        does not smear impact timing (THEORY.md s.6).
      * `smooth_time <= 0`: plain `np.gradient`, which already uses correct non-uniform
        second-order central differences from `t`. Use this when you want the rawest,
        lowest-latency estimate and will handle noise elsewhere.

    Parameters
    ----------
    x : (T,) or (T, D) signal samples.
    t : (T,) strictly-increasing timestamps in seconds.
    smooth_time : local-fit window in seconds. <= 0 => finite-difference fallback.

    Returns
    -------
    (T,) or (T, D) time derivative, same shape as `x`.
    """
    if smooth_time > 0.0:
        return savgol_derivative(x, t, window_time=smooth_time, polyorder=2)

    t = np.asarray(t, dtype=float)
    xx, was_1d = _as_2d(x)
    T = xx.shape[0]
    if t.shape != (T,):
        raise ValueError(f"t must be ({T},) to match the signal; got {t.shape}")
    if T < 2:
        out = np.zeros_like(xx)
        return out[:, 0] if was_1d else out

    # np.gradient differentiates along axis 0 against the explicit (non-uniform) t.
    out = np.gradient(xx, t, axis=0)
    return out[:, 0] if was_1d else out
