#!/usr/bin/env python3
r"""Contact detection from first principles — the whole method, in one file.

This single, self-contained module is the *complete story* of what the
``contact/`` package does, written so it can be read top to bottom instead of
chased across thirty files. It imports nothing from that package; it depends only
on numpy + a handful of scipy primitives, and it reproduces the package's
detection results bit-for-bit (see ``_selftest`` / the companion verification at
the bottom).

THE PROBLEM. We are handed the noisy recorded *motion* of two rigid bodies and we
must invert physics: recover, per frame and with calibrated uncertainty, whether
the two are in contact and what *kind* of contact (static / sliding / pivoting /
rolling / impact), plus the make/break instants. Physics is a forward map (a world
with contacts produces motion); we only see the motion, so this is an inverse
problem and its difficulty is set by what is *recoverable* from the data.

THE STORY (each idea is forced by a concrete failure of the one before it; the
``§`` tags point at the sections of the project's THEORY.md):

  §1  Contact is *relative & geometric*. There is no privileged floor — a foot on a
      moving skateboard is in solid contact though it screams across the world. So
      every quantity is measured in the *support's* frame: the signed **gap** and
      the relative **twist** (3 linear + 3 angular velocity) at the contact.
  §2  Contact obeys a complementarity law (Signorini: ``g≥0, λ≥0, g·λ=0``). "In
      contact?" is exactly "which branch are we on?" — detecting the active set.
  §3  A contact constrains *motion*; the pattern of constraint is the contact
      *type*. Each **mode** is a subspace of the 6-D twist (rolling = tangential
      velocity *coupled* to spin), distinguished by the correlations between
      channels, not any channel alone.
  §4  We never see truth, so we reason probabilistically: each mode is a *proper
      generative density* over (gap, twist); the decision is a calibrated
      **likelihood ratio**, free vs. contact.
  §5  Contact persists in time — a hybrid dynamical system. Its discrete shadow is
      an **HMM**: forward–backward gives the smoothed P(contact); Viterbi gives the
      clean segmentation; a gap-gated transition and an explicit-duration
      (semi-Markov) dwell replace every ad-hoc cleanup heuristic.
  §6  Make/break instants are singular — **impacts**. A velocity step (a matched
      filter) times touchdown sub-frame and, with mass, reads the impulse.
  §7  What is *knowable*? Force magnitude is unobservable from kinematics alone;
      **compliance** (``λ = k·δ``) is the regularizer that restores it. The friction
      cone predicts stick vs. slip.
  §8  Assemble the whole object: a Bayesian posterior over *active-constraint
      structures* on a multi-body **contact graph** — the single-pair estimator run
      per edge and fused into a joint active-set posterior.

The public surface mirrors the package:

    obs    = observe(moving, support, surface, contact_point_local[, geometry])
    result = ContactDetector().detect(obs)            # §1–§7, one body pair
    graph  = detect_scene(scene)                       # §8, a whole contact graph
    scores = score(result, truth)                      # validation vs. an oracle

Run ``python contact_detection_standalone.py`` for a self-contained synthetic demo
(no simulator needed): it builds a drop→rest→liftoff trajectory analytically, runs
the full pipeline, and prints the recovered story.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Protocol

import numpy as np
from scipy import integrate
from scipy.signal import find_peaks
from scipy.special import erf, expit, logsumexp
from scipy.stats import nbinom

# ======================================================================================
# §0  Mode vocabulary and data contracts
#
# These dataclasses are the interfaces between every stage below. The only inputs the
# detector ever sees are pose streams (PoseTrajectory) + a support surface; everything
# else it derives. The truth labels (GroundTruth) come from a simulator and are used
# only to *score*, never by the detector.
# ======================================================================================

FREE = "free"
STATIC = "static"
SLIDING = "sliding"
PIVOTING = "pivoting"
ROLLING = "rolling"
IMPACT = "impact"

#: Every contact mode (all states except FREE).
CONTACT_MODES: list[str] = [STATIC, SLIDING, PIVOTING, ROLLING, IMPACT]
#: Canonical state ordering — index 0 is always FREE. Posterior/emission columns follow this.
ALL_STATES: list[str] = [FREE] + CONTACT_MODES


@dataclass
class PoseTrajectory:
    """Time-stamped pose of one rigid body in the world frame.

    t (T,) seconds; position (T,3) world origin (m); quat (T,4) scalar-first unit
    quaternions (w, x, y, z) that rotate body-local vectors into the world.
    """

    t: np.ndarray
    position: np.ndarray
    quat: np.ndarray


@dataclass
class SupportSurface:
    """A planar support, expressed in the support body's *local* frame.

    point (3,) a point on the plane; normal (3,) the outward unit normal. For a static
    floor the support's pose is identity, so local == world.
    """

    point: np.ndarray
    normal: np.ndarray


@dataclass
class ContactObservations:
    """Per-frame, support-relative observations for ONE candidate body pair (§1, §3).

    Everything lives in the support's instantaneous contact frame (z = outward normal):
    gap (T,) signed surface distance (+ separation, − penetration); v_normal (T,) relative
    normal velocity (+ separating); v_tangent (T,2) tangential velocity; omega_normal (T,)
    spin about the normal; omega_tangent (T,2) the rolling axis. ``normal_force`` is the
    optional measured-force channel (§7 / force axis); ``None`` ⇒ kinematics only.
    """

    t: np.ndarray
    gap: np.ndarray
    v_normal: np.ndarray
    v_tangent: np.ndarray
    omega_normal: np.ndarray
    omega_tangent: np.ndarray
    meas_cov: np.ndarray | None = None
    normal_force: np.ndarray | None = None


@dataclass
class GroundTruth:
    """Per-frame oracle labels from a simulator — used only for scoring, never by detect."""

    t: np.ndarray
    in_contact: np.ndarray
    mode: list[str]
    normal_force: np.ndarray
    penetration: np.ndarray


@dataclass
class ContactEvent:
    """A make/break event (§6): kind ∈ {touchdown, liftoff}, a sub-frame time, nearest index."""

    kind: str
    time: float
    index: int


@dataclass
class ContactImpulse:
    """An impulsive contact event — an atom of the force measure (§6).

    closing_speed (m/s, ≥0); restitution e = −v_after/v_before (NaN if unresolved);
    normal_impulse = m·Δv (NaN when mass is unknown — unobservable from kinematics, §7).
    """

    time: float
    index: int
    closing_speed: float
    restitution: float = float("nan")
    normal_impulse: float = float("nan")


@dataclass
class ContactInterval:
    """A contiguous detected contact interval with its dominant mode."""

    t_start: float
    t_end: float
    mode: str


@dataclass
class DetectionResult:
    """Everything the single-pair detector returns.

    contact_posterior (T,) = 1 − P(free); state_posterior (T,S); map_state length-T MAP
    labels; in_contact derived mask; intervals; events; resting_bias (EM gap offset);
    normal_force (T,) if stiffness known else None; states ordering; impulses; slip_state.
    """

    t: np.ndarray
    contact_posterior: np.ndarray
    state_posterior: np.ndarray
    map_state: list[str]
    in_contact: np.ndarray
    intervals: list[ContactInterval]
    events: list[ContactEvent]
    resting_bias: float
    normal_force: np.ndarray | None = None
    states: list[str] = field(default_factory=lambda: list(ALL_STATES))
    impulses: list[ContactImpulse] = field(default_factory=list)
    slip_state: list[str] | None = None


# --- The narrow waist (§1/§3): a per-frame, world-frame contact description ----------
# A `ContactGeometry` resolver turns the two bodies' poses into one `ContactFrame` per
# recorded frame; `observe()` asks it for (point, normal, gap) then runs an identical
# twist decomposition. Swapping the resolver (flat → sphere → box) leaves every stage
# downstream untouched (§8 / DESIGN axis 1: geometry fidelity).

@dataclass
class ContactPoint:
    """One world-frame contact: point (3,), outward unit normal (3,), signed gap, σ provenance."""

    point: np.ndarray
    normal: np.ndarray
    gap: float
    normal_sigma: float = 0.0
    gap_sigma: float = 0.0


ContactFrame = list[ContactPoint]  # >1 point ⇒ an area/face contact


class ContactGeometry(Protocol):
    def resolve(self, moving: PoseTrajectory, support: PoseTrajectory) -> list[ContactFrame]:
        """One ContactFrame per recorded frame (length T)."""
        ...


@dataclass
class ContactEdge:
    """One candidate contact in a multi-body scene (§8): a moving body vs. a support body."""

    edge_id: str
    moving_body: str
    support_body: str
    surface: SupportSurface
    contact_point_local: np.ndarray
    geometry: ContactGeometry | None = None


@dataclass
class MultiBodyScene:
    """A scene: bodies (name→PoseTrajectory, shared clock), candidate edges, per-edge truth, meta."""

    name: str
    bodies: dict[str, PoseTrajectory]
    edges: list[ContactEdge]
    truth: dict[str, GroundTruth]
    meta: dict = field(default_factory=dict)


@dataclass
class GraphDetectionResult:
    """Joint contact-state estimate over a scene (§8).

    edges = column order of active_posterior; per_edge = each edge's DetectionResult;
    active_posterior (T,E) = marginal P(edge active); map_active_set = per-frame MAP edge sets.
    """

    t: np.ndarray
    edges: list[str]
    per_edge: dict[str, DetectionResult]
    active_posterior: np.ndarray
    map_active_set: list[list[str]]
    meta: dict = field(default_factory=dict)


# ======================================================================================
# §0  Configuration — physically interpretable parameters (not tuned to any simulator).
#
# Defaults are conservative values for optical mocap (~100–250 Hz). They are the same
# numbers the package ships, because the detection results must match exactly.
# ======================================================================================


@dataclass
class EmissionParams:
    """Per-state emission scales (§3, §4). Contact modes are sharp peaks; FREE is diffuse."""

    gap_sigma_gap: float = 0.0015      # tight tolerance ABOVE the surface (a real gap ⇒ free)
    gap_sigma_pen: float = 0.0060      # looser BELOW (squish / plane-fit error)
    gap_free_range: float = 1.0        # diffuse FREE clearance prior width (m)
    vel_sigma: float = 0.05            # contact: relative-velocity noise at rest
    slide_speed: float = 0.15          # sliding: characteristic tangential speed
    slide_width_frac: float = 0.7      # sliding ring WIDTH as a fraction of slide_speed
    free_vel_sigma: float = 0.50       # FREE: broad velocity prior
    omega_sigma: float = 0.30          # contact: angular-rate noise at rest
    slide_omega_broad_weight: float = 0.25  # broad weight of sliding's tight+broad spin mixture
    pivot_speed: float = 1.00          # pivoting: characteristic spin rate
    free_omega_sigma: float = 3.00     # FREE: broad angular prior
    roll_radius: float = 0.05          # rolling: |v_t| ≈ roll_radius·|ω_t| (m)
    roll_sigma: float = 0.03           # tolerance on the rolling-constraint residual (m/s)
    impact_speed: float = 0.30         # impact: characteristic normal closing speed


@dataclass
class ForceEmissionParams:
    """Per-state NORMAL-force emission scales (the measured-force channel; gated, off by default)."""

    sigma_free: float = 0.15           # half-normal width for FREE (normalized units)
    s_load: float = 1.0                # Rayleigh scale for LOADED contact (normalized load ≈ 1)
    s_impact: float = 4.0              # Rayleigh scale for IMPACT (a spike, ≈ 4× median load)
    w_unloaded: float = 0.5            # weight of the unloaded (free-like) contact-force component


@dataclass
class TransitionParams:
    """Temporal prior for the HMM/HSMM (§5)."""

    mean_dwell_time: float = 0.20      # s, baseline expected dwell
    impact_dwell_time: float = 0.04    # s, IMPACT is a short transient
    gap_gate: float = 0.008            # m, gap within which free→contact entry is enabled
    gap_gate_softness: float = 0.004   # m, logistic softness of the gap gate
    use_semi_markov: bool = True       # explicit-duration (HSMM) decoding vs plain Markov
    dwell_concentration: float = 4.0   # duration sharpness (higher ⇒ tighter dwell)


@dataclass
class ImpactParams:
    """Impact detection (§6) — a matched filter on a lightly smoothed normal velocity."""

    template_halfwidth_time: float = 0.03  # s, half-width of the velocity-step template
    min_closing_speed: float = 0.06        # m/s, minimum closing speed to call an impact
    detect_smooth_time: float = 0.01       # s, light smoothing (preserve sharpness)
    restitution_default: float = 0.0       # prior restitution when unmeasured


@dataclass
class MaterialParams:
    """Contact material (§7). With known stiffness, penetration becomes a force gauge λ = k·δ."""

    stiffness: float | None = None     # N/m; None ⇒ purely kinematic (force not estimated)
    damping: float = 0.0               # N/(m/s)
    friction: float = 0.6              # Coulomb coefficient μ
    slip_speed_threshold: float = 0.02  # m/s, tangential speed above which a contact is sliding


@dataclass
class CalibrationParams:
    """EM self-calibration of the resting-gap bias (§7, §8)."""

    max_resting_bias: float = 0.01     # clip the estimated gap offset to ± this (m)
    em_iters: int = 8


@dataclass
class GraphParams:
    """Multi-body contact-graph / active-set inference (§8)."""

    proximity_gap: float = 0.05            # m, broad-phase: propose an edge only within this gap
    active_set_dwell_time: float = 0.20    # s, temporal prior on the active-set sequence
    use_energy_prior: bool = True          # soft global energy/dissipation consistency factor
    use_balance_prior: bool = False        # soft CoM-over-support-polygon factor (needs masses)


@dataclass
class InferenceParams:
    """Structure-inference knobs (§8). Exact 2^E enumeration up to enumerate_max_edges."""

    enumerate_max_edges: int = 4
    n_particles: int = 256
    use_uncertainty: bool = False


@dataclass
class DetectorConfig:
    """Top-level configuration bundle (defaults reproduce the package exactly)."""

    emission: EmissionParams = field(default_factory=EmissionParams)
    force: ForceEmissionParams = field(default_factory=ForceEmissionParams)
    transition: TransitionParams = field(default_factory=TransitionParams)
    material: MaterialParams = field(default_factory=MaterialParams)
    calibration: CalibrationParams = field(default_factory=CalibrationParams)
    impact: ImpactParams = field(default_factory=ImpactParams)
    graph: GraphParams = field(default_factory=GraphParams)
    inference: InferenceParams = field(default_factory=InferenceParams)
    vel_smooth_time: float = 0.05      # Gaussian smoothing time before differentiation (s)


# ======================================================================================
# §1 (numerics)  Time-aware smoothing and differentiation.
#
# Real mocap is non-uniform: frames drop, clocks jitter. Every routine takes the explicit
# timestamps and measures its kernels in *seconds*, not samples. And — §6 — the contact
# velocity is of bounded variation (smooth with genuine jumps at impacts), so smoothing is
# always small and LOCAL: a wide global filter would erase the very make/break timing we want.
# ======================================================================================


def gaussian_smooth(x: np.ndarray, t: np.ndarray, sigma_time: float) -> np.ndarray:
    """Gaussian-smooth a (T,) or (T,D) signal with a kernel measured in real time.

    A dense (T,T) kernel weighted by the *time* gap between samples, row-normalized so DC
    is preserved and the boundaries are unbiased. ``sigma_time ≤ 0`` is a no-op. T is small
    for these clips, so the O(T²) build is fine and is exactly right for non-uniform t.
    """
    t = np.asarray(t, dtype=float)
    if sigma_time <= 0.0:
        return np.asarray(x, dtype=float)
    xx = np.asarray(x, dtype=float)
    was_1d = xx.ndim == 1
    if was_1d:
        xx = xx[:, None]
    dt = t[:, None] - t[None, :]                       # (T,T) signed time differences
    log_w = -0.5 * (dt / sigma_time) ** 2              # log un-normalized weights
    log_w -= log_w.max(axis=1, keepdims=True)          # stabilize before exp (per row)
    w = np.exp(log_w)
    w /= w.sum(axis=1, keepdims=True)                  # row-normalize ⇒ preserves DC
    out = w @ xx
    return out[:, 0] if was_1d else out


def savgol_derivative(x: np.ndarray, t: np.ndarray, window_time: float, polyorder: int = 2) -> np.ndarray:
    """Local least-squares (Savitzky–Golay) time derivative on a non-uniform clock.

    At each sample fit a degree-``polyorder`` polynomial in time to the neighbours inside a
    ``window_time``-second window and read off the analytic slope. Local ⇒ it does not smear
    the velocity jump at an impact across the whole record (§6).
    """
    if window_time <= 0.0:
        raise ValueError("window_time must be > 0 for a local-polynomial derivative")
    t = np.asarray(t, dtype=float)
    xx = np.asarray(x, dtype=float)
    was_1d = xx.ndim == 1
    if was_1d:
        xx = xx[:, None]
    T = xx.shape[0]
    if T < 2:
        out = np.zeros_like(xx)
        return out[:, 0] if was_1d else out
    half = 0.5 * window_time
    out = np.empty_like(xx)
    for i in range(T):
        lo = np.searchsorted(t, t[i] - half, side="left")
        hi = np.searchsorted(t, t[i] + half, side="right")
        idx = np.arange(lo, hi)
        npts = idx.size
        deg = min(polyorder, npts - 1)
        if deg < 1:
            j = i + 1 if i == 0 else i - 1
            out[i] = (xx[i] - xx[j]) / (t[i] - t[j])
            continue
        tau = t[idx] - t[i]                            # local time coords; slope at τ=0 is the deriv
        V = np.vander(tau, N=deg + 1, increasing=True)  # [1, τ, τ², …]
        coeffs, *_ = np.linalg.lstsq(V, xx[idx], rcond=None)
        out[i] = coeffs[1]                             # d/dt at τ=0 == linear coefficient
    return out[:, 0] if was_1d else out


def derivative(x: np.ndarray, t: np.ndarray, smooth_time: float = 0.0) -> np.ndarray:
    """Robust d/dt: a local least-squares fit when ``smooth_time>0``, else non-uniform gradient."""
    if smooth_time > 0.0:
        return savgol_derivative(x, t, window_time=smooth_time, polyorder=2)
    t = np.asarray(t, dtype=float)
    xx = np.asarray(x, dtype=float)
    was_1d = xx.ndim == 1
    if was_1d:
        xx = xx[:, None]
    if xx.shape[0] < 2:
        out = np.zeros_like(xx)
        return out[:, 0] if was_1d else out
    out = np.gradient(xx, t, axis=0)                   # correct non-uniform central differences
    return out[:, 0] if was_1d else out


# ======================================================================================
# §1 & §3  The relative-frame core: poses → support-relative ContactObservations.
#
# Quaternions are scalar-first unit (w,x,y,z), rotating body-local → world. observe() does,
# per frame: place the tracked point in the world; carry the support plane into the world
# (it may translate/rotate); measure the support-relative gap; build a continuous contact
# frame (z = normal, x/y a no-flip tangent basis); compute the RELATIVE twist of the
# coincident material points and split it into normal/tangent. Measuring relative to the
# support is the whole point (§1): a foot on a moving deck reads ~0 relative motion.
# ======================================================================================


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quaternions) of scalar-first q."""
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a·b of scalar-first quaternions (rotation composition)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix/matrices R(q) with v_world = R(q) @ v_local. Input normalized defensively."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[..., 0, 1] = 2.0 * (x * y - z * w)
    R[..., 0, 2] = 2.0 * (x * z + y * w)
    R[..., 1, 0] = 2.0 * (x * y + z * w)
    R[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[..., 1, 2] = 2.0 * (y * z - x * w)
    R[..., 2, 0] = 2.0 * (x * z - y * w)
    R[..., 2, 1] = 2.0 * (y * z + x * w)
    R[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate body-local vector(s) v into the world by q: R(q) @ v (broadcasts over time)."""
    R = quat_to_matrix(q)
    v = np.asarray(v, dtype=float)
    if R.ndim == 2 and v.ndim == 1:
        return R @ v
    if R.ndim == 2 and v.ndim == 2:
        return v @ R.T
    if R.ndim == 3 and v.ndim == 1:
        return np.einsum("tij,j->ti", R, v)
    return np.einsum("tij,tj->ti", R, v)


def _angular_velocity_world(quat: np.ndarray, t: np.ndarray, sigma_time: float) -> np.ndarray:
    """World angular velocity ω(t) = vector part of 2·(dq/dt)·conj(q).

    Smooth the quaternion stream first (differentiating raw orientation is hopeless, §4), and
    resolve the antipodal double cover (q and −q are the same rotation) so dq/dt is meaningful.
    """
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = q.copy()
    flip = np.cumprod(np.sign(np.sum(q[1:] * q[:-1], axis=1) + 1e-300))
    q[1:] *= np.where(flip[:, None] < 0.0, -1.0, 1.0)
    q_smooth = gaussian_smooth(q, t, sigma_time)
    q_smooth = q_smooth / np.linalg.norm(q_smooth, axis=-1, keepdims=True)
    dq = derivative(q_smooth, t)
    omega_quat = 2.0 * quat_mul(dq, quat_conjugate(q_smooth))
    return omega_quat[..., 1:]


def plane_gap(points_world: np.ndarray, plane_point_world: np.ndarray, plane_normal_world: np.ndarray) -> np.ndarray:
    """Signed distance of points to a (possibly moving) plane; + on the outward-normal side (§1)."""
    p = np.atleast_2d(np.asarray(points_world, dtype=float))
    p0 = np.atleast_2d(np.asarray(plane_point_world, dtype=float))
    n = np.atleast_2d(np.asarray(plane_normal_world, dtype=float))
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    return np.sum((p - p0) * n, axis=-1)


def _tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continuous orthonormal tangent basis (x̂, ŷ) for a normal stream (§3).

    The tangent plane carries sliding/rolling, so its axes must not flip frame-to-frame. Pick
    one tangent at frame 0, then parallel-transport (project the previous x̂ into each new
    tangent plane and re-normalize). ŷ = ẑ × x̂ completes a right-handed frame.
    """
    z = np.asarray(normals, dtype=float)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    T = z.shape[0]
    x_hat = np.empty((T, 3), dtype=float)
    y_hat = np.empty((T, 3), dtype=float)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, z[0])) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    x0 = seed - np.dot(seed, z[0]) * z[0]
    x0 /= np.linalg.norm(x0)
    x_hat[0] = x0
    y_hat[0] = np.cross(z[0], x_hat[0])
    for i in range(1, T):
        x = x_hat[i - 1] - np.dot(x_hat[i - 1], z[i]) * z[i]
        nrm = np.linalg.norm(x)
        if nrm < 1e-12:                                # normal flipped ~180°: reseed from y_hat
            x = y_hat[i - 1] - np.dot(y_hat[i - 1], z[i]) * z[i]
            nrm = np.linalg.norm(x)
            if nrm < 1e-12:
                s = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(s, z[i])) > 0.9:
                    s = np.array([0.0, 1.0, 0.0])
                x = s - np.dot(s, z[i]) * z[i]
                nrm = np.linalg.norm(x)
        x_hat[i] = x / nrm
        y_hat[i] = np.cross(z[i], x_hat[i])
    return x_hat, y_hat


# --- The fidelity ladder of resolvers (DESIGN axis 1). All return the same ContactFrame, so
# observe() is identical downstream. FlatRegion is the validated default; SphereSphere fixes
# the spinning-normal artifact on ball↔ball (position-derived normal); BoxPlane gives the
# migrating nearest-corner contact for a tumbling box. ----------------------------------


class FlatRegion:
    """Default resolver: a flat plane on the support + a fixed tracked point on the moving body."""

    migrating = False

    def __init__(self, surface: SupportSurface, contact_point_local: np.ndarray = np.zeros(3)) -> None:
        self.surface = surface
        self.contact_point_local = contact_point_local

    def resolve(self, moving: PoseTrajectory, support: PoseTrajectory) -> list[ContactFrame]:
        mov_pos = np.asarray(moving.position, dtype=float)
        mov_quat = np.asarray(moving.quat, dtype=float)
        sup_pos = np.asarray(support.position, dtype=float)
        sup_quat = np.asarray(support.quat, dtype=float)
        cpl = np.asarray(self.contact_point_local, dtype=float)
        p = mov_pos + quat_rotate(mov_quat, cpl)                          # world contact point
        plane_pt_w = sup_pos + quat_rotate(sup_quat, self.surface.point)  # plane point (rotate+translate)
        normal_w = quat_rotate(sup_quat, self.surface.normal)            # plane normal (rotate only)
        normal_w = normal_w / np.linalg.norm(normal_w, axis=-1, keepdims=True)
        gap = plane_gap(p, plane_pt_w, normal_w)
        return [[ContactPoint(point=p[i], normal=normal_w[i], gap=float(gap[i]))] for i in range(p.shape[0])]


class SpherePlane:
    """A sphere (radius r_moving) on a planar support: gap = centre-distance − r, point = the foot."""

    migrating = False

    def __init__(self, r_moving: float, surface: SupportSurface, contact_point_local: np.ndarray = np.zeros(3)) -> None:
        self.r_moving = float(r_moving)
        self.surface = surface
        self.contact_point_local = contact_point_local

    def resolve(self, moving: PoseTrajectory, support: PoseTrajectory) -> list[ContactFrame]:
        mov_pos = np.asarray(moving.position, dtype=float)
        sup_pos = np.asarray(support.position, dtype=float)
        sup_quat = np.asarray(support.quat, dtype=float)
        normal_w = quat_rotate(sup_quat, self.surface.normal)
        normal_w = normal_w / np.linalg.norm(normal_w, axis=-1, keepdims=True)
        plane_pt_w = sup_pos + quat_rotate(sup_quat, self.surface.point)
        c = mov_pos
        gap = plane_gap(c, plane_pt_w, normal_w) - self.r_moving
        point = c - self.r_moving * normal_w
        return [[ContactPoint(point=point[i], normal=normal_w[i], gap=float(gap[i]))] for i in range(c.shape[0])]


class SphereSphere:
    """Two spheres — the position-derived-normal resolver (fixes ball↔ball phantom impacts).

    The normal comes from the *line of centres* d = c₁ − c₂, NEVER a body-fixed vector rotated
    by a (spinning) quaternion; gap = ‖d‖ − r₁ − r₂; the reported point is the MOVING sphere's
    surface point c₁ − r₁·n̂ (so observe() recovers the moving body's closing velocity).
    """

    migrating = False

    def __init__(self, r_moving: float, r_support: float) -> None:
        self.r_moving = float(r_moving)
        self.r_support = float(r_support)

    def resolve(self, moving: PoseTrajectory, support: PoseTrajectory) -> list[ContactFrame]:
        c1 = np.asarray(moving.position, dtype=float)
        c2 = np.asarray(support.position, dtype=float)
        d = c1 - c2
        dist = np.linalg.norm(d, axis=-1)
        normal = np.tile(np.array([0.0, 0.0, 1.0]), (d.shape[0], 1))
        safe = dist >= 1e-12
        normal[safe] = d[safe] / dist[safe, None]
        gap = dist - self.r_moving - self.r_support
        point = c1 - self.r_moving * normal
        return [[ContactPoint(point=point[i], normal=normal[i], gap=float(gap[i]))] for i in range(c1.shape[0])]


class BoxPlane:
    """A box (8 corners) vs. a plane — the *migrating-contact* resolver (fixes the tumbling box).

    The contact point is whichever corner is currently lowest, and it JUMPS between bounces, so
    it is not a fixed material point: ``migrating=True`` tells observe() to take the moving
    point's velocity analytically from the body twist (v = v_com + ω×r) rather than by
    differentiating the teleporting world point. A frame is multi-point on a flat face/edge.
    """

    migrating = True
    eps = 1e-3                                          # group corners within eps of the lowest

    def __init__(self, half_extents: np.ndarray, surface: SupportSurface, contact_point_local: np.ndarray = np.zeros(3)) -> None:
        self.half_extents = np.asarray(half_extents, dtype=float)
        self.surface = surface
        self.contact_point_local = contact_point_local

    def resolve(self, moving: PoseTrajectory, support: PoseTrajectory) -> list[ContactFrame]:
        mov_pos = np.asarray(moving.position, dtype=float)
        mov_quat = np.asarray(moving.quat, dtype=float)
        sup_pos = np.asarray(support.position, dtype=float)
        sup_quat = np.asarray(support.quat, dtype=float)
        he = np.asarray(self.half_extents, dtype=float)
        signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)], dtype=float)
        corners_local = signs * he
        normal_w = quat_rotate(sup_quat, self.surface.normal)
        normal_w = normal_w / np.linalg.norm(normal_w, axis=-1, keepdims=True)
        plane_pt_w = sup_pos + quat_rotate(sup_quat, self.surface.point)
        corners_world = np.stack([mov_pos + quat_rotate(mov_quat, corners_local[j]) for j in range(8)], axis=1)
        d = np.einsum("tjk,tk->tj", corners_world - plane_pt_w[:, None, :], normal_w)
        eps = float(self.eps)
        frames: list[ContactFrame] = []
        for i in range(corners_world.shape[0]):
            gap_i = float(d[i].min())
            sel = np.nonzero(d[i] <= gap_i + eps)[0]
            frames.append([ContactPoint(point=corners_world[i, j], normal=normal_w[i], gap=float(d[i, j])) for j in sel])
        return frames


def observe(
    moving: PoseTrajectory,
    support: PoseTrajectory,
    surface: SupportSurface,
    contact_point_local: np.ndarray = np.zeros(3),
    vel_smooth_time: float = 0.05,
    geometry: ContactGeometry | None = None,
) -> ContactObservations:
    """Turn a moving body + (possibly moving) support into support-relative observations (§1, §3).

    The per-frame world (point, normal, gap) come from a ``ContactGeometry`` resolver (default:
    ``FlatRegion``); the twist decomposition that follows is identical for every resolver.
    """
    if geometry is None:
        geometry = FlatRegion(surface, contact_point_local)

    t = np.asarray(moving.t, dtype=float)
    mov_pos = np.asarray(moving.position, dtype=float)
    sup_pos = np.asarray(support.position, dtype=float)
    sup_quat = np.asarray(support.quat, dtype=float)
    mov_quat = np.asarray(moving.quat, dtype=float)

    frames = geometry.resolve(moving, support)
    # A face/edge contact is several points but ONE kinematic mode: reduce each frame to its
    # min-gap (closest) representative for the twist decomposition (a single-point frame is itself).
    reps = [min(frame, key=lambda cp: cp.gap) for frame in frames]
    p = np.stack([rep.point for rep in reps])                # (T,3) world contact point
    normal_w = np.stack([rep.normal for rep in reps])        # (T,3) world unit normal
    gap = np.array([rep.gap for rep in reps], dtype=float)   # (T,) signed plane distance

    z_hat = normal_w
    x_hat, y_hat = _tangent_basis(z_hat)

    omega_moving = _angular_velocity_world(mov_quat, t, vel_smooth_time)  # (T,3) world

    # (a) velocity of the moving material point.
    if getattr(geometry, "migrating", False):
        # The contact point teleports between corners, so take the velocity of the body's
        # material point currently at the contact analytically: v = v_com + ω × (p − com).
        v_com = derivative(gaussian_smooth(mov_pos, t, vel_smooth_time), t)
        v_moving_point = v_com + np.cross(omega_moving, p - mov_pos)
    else:
        # Fixed material point: differentiate its smooth world trajectory.
        v_moving_point = derivative(gaussian_smooth(p, t, vel_smooth_time), t)

    # (b) velocity of the support point coincident with p: v_origin + ω_support × r.
    v_sup_origin = derivative(gaussian_smooth(sup_pos, t, vel_smooth_time), t)
    omega_support = _angular_velocity_world(sup_quat, t, vel_smooth_time)
    r = p - sup_pos
    v_support_point = v_sup_origin + np.cross(omega_support, r)

    # (c) relative velocity, decomposed into the contact frame.
    v_rel = v_moving_point - v_support_point
    v_normal = np.sum(v_rel * z_hat, axis=-1)                # + = separating
    v_tangent = np.stack([np.sum(v_rel * x_hat, axis=-1), np.sum(v_rel * y_hat, axis=-1)], axis=-1)

    omega_rel = omega_moving - omega_support
    omega_normal = np.sum(omega_rel * z_hat, axis=-1)        # spin about the normal
    omega_tangent = np.stack([np.sum(omega_rel * x_hat, axis=-1), np.sum(omega_rel * y_hat, axis=-1)], axis=-1)

    return ContactObservations(
        t=t, gap=gap, v_normal=v_normal, v_tangent=v_tangent,
        omega_normal=omega_normal, omega_tangent=omega_tangent,
    )


# ======================================================================================
# §3 & §4  Per-state emission log-likelihoods — the contact modes as generative models.
#
# Each mode is a PROPER, normalized density over the WHOLE observation (gap, v_normal∈ℝ,
# v_tangent∈ℝ², ω_normal∈ℝ, ω_tangent∈ℝ²). We keep every normalization constant (e.g. the
# −log(σ√2π) of each Gaussian) so cross-state log-RATIOS stay calibrated (§4) — dropping
# them would silently bias the decision toward the sharp states. FREE is diffuse; a contact
# mode pins the gap near the resting bias and concentrates the twist on the subspace it
# allows (rolling: a *cross-channel* coupling no per-channel model could represent).
# ======================================================================================

_LOG_2PI = float(np.log(2.0 * np.pi))
_LOG_SQRT_2PI = 0.5 * _LOG_2PI
_LOG_SQRT_PI_OVER_2 = 0.5 * float(np.log(np.pi / 2.0))
_LOG_2_OVER_PI = float(np.log(2.0 / np.pi))
_FORCE_EPS = 1e-12


def _log_normal_1d(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    z = (np.asarray(x, dtype=float) - mean) / sigma
    return -np.log(sigma) - _LOG_SQRT_2PI - 0.5 * z * z


def _log_normal_2d_iso(x: np.ndarray, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sq = np.sum(x * x, axis=-1)
    return -2.0 * np.log(sigma) - _LOG_2PI - 0.5 * sq / (sigma * sigma)


def _log_uniform(width: float) -> float:
    return -float(np.log(width))


def _log_split_normal_gap(gap: np.ndarray, mean: float, sigma_hi: float, sigma_lo: float) -> np.ndarray:
    """Two-piece Gaussian gap density: σ_hi above the mean (a real gap ⇒ free, so tight), σ_lo
    below (penetration tolerated more). Gross penetration falls in the lower tail ⇒ ~0
    contact likelihood — the bounded behaviour §2 demands, without an ad-hoc clip.
    """
    gap = np.asarray(gap, dtype=float)
    sigma = np.where(gap >= mean, sigma_hi, sigma_lo)
    log_z = _LOG_SQRT_PI_OVER_2 + np.log(sigma_hi + sigma_lo)  # Z = √(π/2)·(σ_hi+σ_lo)
    z = (gap - mean) / sigma
    return -log_z - 0.5 * z * z


def _log_mix_zero_1d(x: np.ndarray, sigma_tight: float, sigma_broad: float, w_broad: float) -> np.ndarray:
    """Zero-mean 1-D Gaussian mixture (1−w)·N(0,σ_t²)+w·N(0,σ_b²): peaked at 0 yet heavy-tailed."""
    lp_t = np.log1p(-w_broad) + _log_normal_1d(x, 0.0, sigma_tight)
    lp_b = np.log(w_broad) + _log_normal_1d(x, 0.0, sigma_broad)
    return np.logaddexp(lp_t, lp_b)


def _log_mix_zero_2d(x: np.ndarray, sigma_tight: float, sigma_broad: float, w_broad: float) -> np.ndarray:
    lp_t = np.log1p(-w_broad) + _log_normal_2d_iso(x, sigma_tight)
    lp_b = np.log(w_broad) + _log_normal_2d_iso(x, sigma_broad)
    return np.logaddexp(lp_t, lp_b)


def _log_offset_magnitude_1d(x: np.ndarray, speed: float, sigma: float) -> np.ndarray:
    """Proper density on ℝ peaked at ±speed: ½N(+speed,σ²)+½N(−speed,σ²) (sign uninformative)."""
    x = np.asarray(x, dtype=float)
    log_half = -float(np.log(2.0))
    lp_plus = log_half + _log_normal_1d(x, +speed, sigma)
    lp_minus = log_half + _log_normal_1d(x, -speed, sigma)
    return np.logaddexp(lp_plus, lp_minus)


def _log_offset_magnitude_2d(x: np.ndarray, speed: float, sigma: float) -> np.ndarray:
    """Proper ℝ² density on the ring ‖x‖ = speed: p(x) ∝ exp(−½(‖x‖−speed)²/σ²).

    The polar normalizer (Jacobian r·dr) is Z = 2πσ[σ·exp(−speed²/2σ²) + speed·√(π/2)(1+erf(...))],
    collapsing to the isotropic Gaussian as speed → 0.
    """
    x = np.asarray(x, dtype=float)
    r = np.sqrt(np.sum(x * x, axis=-1))
    s2 = sigma * sigma
    term_gauss = sigma * np.exp(-(speed * speed) / (2.0 * s2))
    term_ring = speed * np.sqrt(np.pi / 2.0) * (1.0 + erf(speed / (np.sqrt(2.0) * sigma)))
    z = 2.0 * np.pi * sigma * (term_gauss + term_ring)
    return -float(np.log(z)) - 0.5 * (r - speed) ** 2 / s2


@lru_cache(maxsize=None)
def _log_rolling_residual_normalizer(free_vel_sigma: float, free_omega_sigma: float, roll_radius: float, roll_sigma: float) -> float:
    """log Z_res restoring properness to the ROLLING column.

    Rolling multiplies two broad isotropic tangential priors by the coupling-residual factor
    N(|v_t|−r|ω_t|; 0, roll_sigma²); since the residual depends on BOTH tangential vectors the
    product integrates to Z_res (~0.66 at defaults), not 1. Every other column integrates to 1,
    so we subtract log Z_res. Z_res is a 2-D quadrature over the Rayleigh magnitudes (cached).
    """
    sv, sw, rr, rs = float(free_vel_sigma), float(free_omega_sigma), float(roll_radius), float(roll_sigma)

    def f_rayleigh(x: float, scale: float) -> float:
        return x / (scale * scale) * np.exp(-x * x / (2.0 * scale * scale))

    def n1(x: float) -> float:
        return 1.0 / (rs * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x / rs) ** 2)

    def inner(b: float) -> float:
        val, _ = integrate.quad(lambda a: f_rayleigh(a, sv) * n1(a - rr * b), 0.0, 8.0 * sv, limit=200)
        return val

    z_res, _ = integrate.quad(lambda b: f_rayleigh(b, sw) * inner(b), 0.0, 8.0 * sw, limit=200)
    return float(np.log(z_res))


# --- Force-channel densities on [0,∞) (the optional measured-force factor; off by default) ---
def _log_half_normal(f: np.ndarray, sigma: float) -> np.ndarray:
    f = np.maximum(np.asarray(f, dtype=float), 0.0)
    z = f / sigma
    return 0.5 * _LOG_2_OVER_PI - np.log(sigma) - 0.5 * z * z


def _log_rayleigh(f: np.ndarray, scale: float) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    return np.log(np.maximum(f, _FORCE_EPS)) - 2.0 * np.log(scale) - (f * f) / (2.0 * scale * scale)


def _force_log_density(obs: ContactObservations, state: str, force: ForceEmissionParams) -> np.ndarray:
    """Per-state force log-density over the robustly normalized normal force.

    FREE: half-normal at 0 (no load). Sustained contact: a MIXTURE w·HN + (1−w)·Rayleigh — a
    touch may be unloaded (f≈0, the gap decides) OR loaded. IMPACT: a larger-scale Rayleigh spike.
    """
    f = np.asarray(obs.normal_force, dtype=float).ravel()
    pos = f[f > 0.0]
    s = float(np.median(pos)) if pos.size > 0 else 1.0
    if not np.isfinite(s) or s <= 0.0:
        s = 1.0
    fn = f / s
    if state == FREE:
        return _log_half_normal(fn, force.sigma_free)
    if state == IMPACT:
        return _log_rayleigh(fn, force.s_impact)
    w = float(force.w_unloaded)
    return np.logaddexp(np.log(w) + _log_half_normal(fn, force.sigma_free), np.log1p(-w) + _log_rayleigh(fn, force.s_load))


# ======================================================================================
# §3/§4 (encapsulated)  The channel densities as first-class objects.
#
# Each contact mode is a PRODUCT of independent per-channel densities (a SUM of these
# log-densities); the modes differ only in WHICH density sits on each channel. Naming each
# density as a small immutable object lets a mode read as its generative signature, and makes
# every normalization constant a property that can be tested in isolation (see _density_selftest:
# each density is proper — unit mass — and the documented limit laws are checked). Every .logpdf
# is a thin, proper wrapper over the primitives above, so a composed mode is bit-for-bit identical
# to the inline accumulation it replaces.
# ======================================================================================


@dataclass(frozen=True)
class Normal1D:
    """1-D Gaussian N(mean, σ²) on ℝ — a proper density (normalizer included)."""

    mean: float
    sigma: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return _log_normal_1d(x, self.mean, self.sigma)


@dataclass(frozen=True)
class IsoNormal2D:
    """Isotropic 2-D Gaussian N(0, σ²·I) on ℝ²."""

    sigma: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return _log_normal_2d_iso(x, self.sigma)


@dataclass(frozen=True)
class SplitNormalGap:
    """Two-piece Gaussian gap density: σ_hi above the mean, σ_lo below (§2). σ_hi=σ_lo ⇒ N(mean, σ²)."""

    mean: float
    sigma_hi: float
    sigma_lo: float

    def logpdf(self, gap: np.ndarray) -> np.ndarray:
        return _log_split_normal_gap(gap, self.mean, self.sigma_hi, self.sigma_lo)


@dataclass(frozen=True)
class OffsetMagnitude1D:
    """Proper density on ℝ peaked at ±speed (sign uninformative). speed→0 ⇒ N(0, σ²)."""

    speed: float
    sigma: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return _log_offset_magnitude_1d(x, self.speed, self.sigma)


@dataclass(frozen=True)
class OffsetMagnitude2D:
    """Proper ℝ² density on the ring ‖x‖=speed. speed→0 ⇒ the isotropic Gaussian IsoNormal2D(σ)."""

    speed: float
    sigma: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return _log_offset_magnitude_2d(x, self.speed, self.sigma)


@dataclass(frozen=True)
class MixZero1D:
    """Zero-mean 1-D Gaussian mixture (1−w)·N(0,σ_t²)+w·N(0,σ_b²). w→0 ⇒ N(0, σ_t²)."""

    sigma_tight: float
    sigma_broad: float
    w_broad: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return _log_mix_zero_1d(x, self.sigma_tight, self.sigma_broad, self.w_broad)


@dataclass(frozen=True)
class MixZero2D:
    """Zero-mean isotropic 2-D Gaussian mixture (1−w)·N(0,σ_t²I)+w·N(0,σ_b²I)."""

    sigma_tight: float
    sigma_broad: float
    w_broad: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return _log_mix_zero_2d(x, self.sigma_tight, self.sigma_broad, self.w_broad)


@dataclass(frozen=True)
class UniformClearance:
    """Diffuse uniform gap prior of width ``width`` — the FREE clearance (no surface pins the gap)."""

    width: float

    def logpdf(self, gap: np.ndarray) -> np.ndarray:
        return _log_uniform(self.width) * np.ones_like(np.asarray(gap, dtype=float))


def _compose(terms: tuple) -> np.ndarray:
    """Sum a sequence of ``(channel_value, Density)`` into one per-frame log-density.

    Reduced strictly left-to-right so the result is bit-for-bit identical to the equivalent
    ``lp = d0.logpdf(x0); lp = lp + d1.logpdf(x1); …`` accumulation (float addition is not
    associative, so the order is load-bearing for the standalone-equivalence gate).
    """
    (x0, d0), rest = terms[0], terms[1:]
    lp = d0.logpdf(x0)
    for x, d in rest:
        lp = lp + d.logpdf(x)
    return lp


class ContactMode:
    """A latent mode as a generative model: a kinematic signature + the optional force channel."""

    name: str = ""

    def kinematic_log_density(self, obs: ContactObservations, params: EmissionParams, gap_bias: float) -> np.ndarray:
        raise NotImplementedError

    def log_density(self, obs, params, gap_bias, material=None, force=None):
        lp = self.kinematic_log_density(obs, params, gap_bias)
        if obs.normal_force is not None and force is not None:
            lp = lp + _force_log_density(obs, self.name, force)
        return lp


class Free(ContactMode):
    """FREE: nothing pinned — diffuse on every channel (gap_bias unused)."""

    name = FREE

    def kinematic_log_density(self, obs, params, gap_bias):
        return _compose((
            (obs.gap,           UniformClearance(params.gap_free_range)),
            (obs.v_normal,      Normal1D(0.0, params.free_vel_sigma)),
            (obs.v_tangent,     IsoNormal2D(params.free_vel_sigma)),
            (obs.omega_normal,  Normal1D(0.0, params.free_omega_sigma)),
            (obs.omega_tangent, IsoNormal2D(params.free_omega_sigma)),
        ))


class Static(ContactMode):
    """STATIC: the whole twist pinned to ~0 — a contact at rest."""

    name = STATIC

    def kinematic_log_density(self, obs, params, gap_bias):
        return _compose((
            (obs.gap,           SplitNormalGap(gap_bias, params.gap_sigma_gap, params.gap_sigma_pen)),
            (obs.v_normal,      Normal1D(0.0, params.vel_sigma)),
            (obs.v_tangent,     IsoNormal2D(params.vel_sigma)),
            (obs.omega_normal,  Normal1D(0.0, params.omega_sigma)),
            (obs.omega_tangent, IsoNormal2D(params.omega_sigma)),
        ))


class Sliding(ContactMode):
    """SLIDING: tangential-linear motion only; spin is off-subspace (a heavy-tailed mixture)."""

    name = SLIDING

    def kinematic_log_density(self, obs, params, gap_bias):
        slide_width = max(params.vel_sigma, params.slide_width_frac * params.slide_speed)
        wb = params.slide_omega_broad_weight
        return _compose((
            (obs.gap,           SplitNormalGap(gap_bias, params.gap_sigma_gap, params.gap_sigma_pen)),
            (obs.v_normal,      Normal1D(0.0, params.vel_sigma)),
            (obs.v_tangent,     OffsetMagnitude2D(params.slide_speed, slide_width)),
            (obs.omega_normal,  MixZero1D(params.omega_sigma, params.free_omega_sigma, wb)),
            (obs.omega_tangent, MixZero2D(params.omega_sigma, params.free_omega_sigma, wb)),
        ))


class Pivoting(ContactMode):
    """PIVOTING: normal-angular motion only (spin about the normal)."""

    name = PIVOTING

    def kinematic_log_density(self, obs, params, gap_bias):
        return _compose((
            (obs.gap,           SplitNormalGap(gap_bias, params.gap_sigma_gap, params.gap_sigma_pen)),
            (obs.v_normal,      Normal1D(0.0, params.vel_sigma)),
            (obs.v_tangent,     IsoNormal2D(params.vel_sigma)),
            (obs.omega_normal,  OffsetMagnitude1D(params.pivot_speed, params.omega_sigma)),
            (obs.omega_tangent, IsoNormal2D(params.omega_sigma)),
        ))


class Rolling(ContactMode):
    """ROLLING: tangential-linear COUPLED to tangential-angular by the residual |v_t|−r|ω_t| ≈ 0.

    The ONE non-product mode: v_tangent and ω_tangent are not independent, so the tangential block is
    a Gaussian on the coupling residual (each magnitude left broad) renormalized by Z_res. In the
    composition this is just a *derived channel* (the residual) plus a trailing block normalizer —
    the _compose interface needs no special case; the coupling is a value, not new machinery.
    """

    name = ROLLING

    def kinematic_log_density(self, obs, params, gap_bias):
        v_t = np.asarray(obs.v_tangent, dtype=float)
        w_t = np.asarray(obs.omega_tangent, dtype=float)
        speed_t = np.sqrt(np.sum(v_t * v_t, axis=-1))
        omega_t = np.sqrt(np.sum(w_t * w_t, axis=-1))
        residual = speed_t - params.roll_radius * omega_t           # the rolling coupling |v_t|−r|ω_t|
        log_z_res = _log_rolling_residual_normalizer(
            params.free_vel_sigma, params.free_omega_sigma, params.roll_radius, params.roll_sigma)
        return _compose((
            (obs.gap,           SplitNormalGap(gap_bias, params.gap_sigma_gap, params.gap_sigma_pen)),
            (obs.v_normal,      Normal1D(0.0, params.vel_sigma)),
            (v_t,               IsoNormal2D(params.free_vel_sigma)),
            (w_t,               IsoNormal2D(params.free_omega_sigma)),
            (residual,          Normal1D(0.0, params.roll_sigma)),
            (obs.omega_normal,  Normal1D(0.0, params.omega_sigma)),
        )) - log_z_res


class Impact(ContactMode):
    """IMPACT: a short-lived transient — a large closing normal velocity at a (wider) gap ~0."""

    name = IMPACT

    def kinematic_log_density(self, obs, params, gap_bias):
        return _compose((
            (obs.gap,           SplitNormalGap(gap_bias, 2.0 * params.gap_sigma_gap, 2.0 * params.gap_sigma_pen)),
            (obs.v_normal,      OffsetMagnitude1D(params.impact_speed, params.vel_sigma)),
            (obs.v_tangent,     IsoNormal2D(params.free_vel_sigma)),
            (obs.omega_normal,  Normal1D(0.0, params.free_omega_sigma)),
            (obs.omega_tangent, IsoNormal2D(params.free_omega_sigma)),
        ))


MODES: dict[str, ContactMode] = {m.name: m for m in (Free(), Static(), Sliding(), Pivoting(), Rolling(), Impact())}


def log_emissions(obs, params, gap_bias, states, material=None, force=None) -> np.ndarray:
    """Assemble the (T, len(states)) emission log-likelihood matrix; column j = log p(obs | states[j])."""
    T = int(np.asarray(obs.gap, dtype=float).shape[0])
    out = np.empty((T, len(states)), dtype=float)
    for j, name in enumerate(states):
        out[:, j] = MODES[name].log_density(obs, params, gap_bias, material, force)
    return out


# ======================================================================================
# §5  Temporal inference — the HMM (the discrete shadow of the hybrid dynamical system).
#
# Contacts persist; deciding each frame in isolation throws that away. The transition prior
# is a continuous-time Markov jump discretized per frame, P(stay over dt) = exp(−dt/dwell),
# with the off-diagonal mass split along the hybrid system's GUARDS (FREE is the gateway;
# IMPACT bridges free↔contact). A gap-GATE makes free→contact entry rise as the gap nears 0.
# forward–backward gives the smoothed posterior; Viterbi the clean segmentation; a semi-Markov
# duration model makes 1-frame blips intrinsically improbable. This replaces every cleanup pass.
# ======================================================================================

_FLOOR = 0.02  # a small floor on every transition propensity ⇒ log is finite everywhere (§4)


def base_transition_matrix(states: list[str], dt: float, params: TransitionParams) -> np.ndarray:
    """Time-homogeneous (S,S) row-stochastic matrix with the guard-structured off-diagonal."""
    states = list(states)
    S = len(states)
    idx = {name: i for i, name in enumerate(states)}
    dt = max(float(dt), 0.0)
    tau = max(float(params.mean_dwell_time), 1e-6)
    tau_impact = max(float(params.impact_dwell_time), 1e-6)
    dwell = {name: tau for name in states}
    if IMPACT in idx:
        dwell[IMPACT] = tau_impact
    sustained = [m for m in CONTACT_MODES if m != IMPACT]

    def jump_weights(src: str) -> np.ndarray:
        w = np.full(S, _FLOOR, dtype=float)
        if src == FREE:
            if IMPACT in idx:
                w[idx[IMPACT]] = 1.0           # FREE re-enters mainly via the IMPACT transient
            for m in sustained:
                if m in idx:
                    w[idx[m]] = 0.15
        elif src == IMPACT:
            for m in sustained:
                if m in idx:
                    w[idx[m]] = 1.0            # IMPACT establishes a sustained contact
            if FREE in idx:
                w[idx[FREE]] = 0.6
        else:
            if FREE in idx:
                w[idx[FREE]] = 1.0            # a sustained mode mostly breaks back to FREE
            for m in sustained:
                if m in idx and m != src:
                    w[idx[m]] = 0.25
            if IMPACT in idx:
                w[idx[IMPACT]] = 0.20
        w[idx[src]] = 0.0
        return w

    P = np.zeros((S, S), dtype=float)
    for src in states:
        i = idx[src]
        stay = float(np.exp(-dt / dwell[src]))
        P[i, i] = stay
        w = jump_weights(src)
        total = float(w.sum())
        if total <= 0.0:
            if S > 1:
                P[i] = (1.0 - stay) / (S - 1)
                P[i, i] = stay
            else:
                P[i, i] = 1.0
        else:
            P[i] += (1.0 - stay) * w / total
    return P


def gated_transition_tensor(obs, states: list[str], dt: float, params: TransitionParams) -> np.ndarray:
    """Per-frame (T,S,S): the FREE→contact entry mass is gated by gap proximity (the §5 make guard).

    g(t) = sigmoid((gap_gate − gap)/softness): ~0 far above the surface, ~1 once within reach. Of
    the base FREE-row entry mass m0, only fraction g(t) is offered to the contact states (the rest
    returns to the FREE diagonal); the relative split across modes stays the base one. A floor
    keeps a sliver open even when shut, so a surprising touchdown is never impossible.
    """
    states = list(states)
    S = len(states)
    idx = {name: i for i, name in enumerate(states)}
    gap = np.asarray(obs.gap, dtype=float).ravel()
    T = gap.shape[0]
    base = base_transition_matrix(states, dt, params)
    if FREE not in idx:
        return np.broadcast_to(base, (T, S, S)).copy()
    free_i = idx[FREE]
    contact_cols = np.array([j for j, s in enumerate(states) if s != FREE], dtype=np.intp)
    if contact_cols.size == 0:
        return np.broadcast_to(base, (T, S, S)).copy()
    softness = max(float(params.gap_gate_softness), 1e-9)
    z = (float(params.gap_gate) - gap) / softness
    gate = expit(z)
    base_free_row = base[free_i].copy()
    base_free_diag = float(base_free_row[free_i])
    base_contact = base_free_row[contact_cols].copy()
    m0 = float(base_contact.sum())
    if m0 > 0.0:
        contact_shape = base_contact / m0
    else:
        contact_shape = np.full(contact_cols.shape[0], 1.0 / contact_cols.shape[0])
    floor_frac = _FLOOR
    tensor = np.broadcast_to(base, (T, S, S)).copy()
    for t in range(T):
        g = float(gate[t])
        g_eff = floor_frac + (1.0 - floor_frac) * g
        offered = m0 * g_eff
        returned = m0 - offered
        row = np.empty(S, dtype=float)
        row[free_i] = base_free_diag + returned
        row[contact_cols] = offered * contact_shape
        row /= row.sum()
        tensor[t, free_i] = row
    return tensor


def _broadcast_trans(log_trans: np.ndarray, T: int, S: int) -> np.ndarray:
    """Return transitions as (T−1, S, S): tile a homogeneous (S,S), or drop the trailing (T,S,S) slice."""
    log_trans = np.asarray(log_trans, dtype=float)
    if log_trans.shape == (S, S):
        return np.broadcast_to(log_trans, (max(T - 1, 0), S, S))
    if log_trans.shape == (T, S, S):
        return log_trans[: T - 1]
    raise ValueError(f"log_trans must be (S,S) or (T,S,S); got {log_trans.shape}")


def forward_backward(log_emission: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray) -> tuple[np.ndarray, float]:
    """Smoothed state posterior γ and total log-likelihood (log-space, §5).

    α[t,s] = emit[t,s] + logΣ_j(α[t−1,j] + A[t−1,j,s]); β symmetric; γ ∝ exp(α+β) per row.
    Conditioning each frame on the whole record is why an offline detector beats a causal one (§6).
    """
    log_emission = np.asarray(log_emission, dtype=float)
    T, S = log_emission.shape
    log_init = np.asarray(log_init, dtype=float)
    log_A = _broadcast_trans(log_trans, T, S)
    log_alpha = np.empty((T, S), dtype=float)
    log_alpha[0] = log_init + log_emission[0]
    for t in range(1, T):
        prev = log_alpha[t - 1][:, None] + log_A[t - 1]
        log_alpha[t] = log_emission[t] + logsumexp(prev, axis=0)
    total_loglik = float(logsumexp(log_alpha[T - 1]))
    log_beta = np.empty((T, S), dtype=float)
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        nxt = log_A[t] + (log_emission[t + 1] + log_beta[t + 1])[None, :]
        log_beta[t] = logsumexp(nxt, axis=1)
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, total_loglik


def viterbi(log_emission: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray) -> np.ndarray:
    """MAP state path (the clean contiguous segmentation, §5) via the max-product recursion."""
    log_emission = np.asarray(log_emission, dtype=float)
    T, S = log_emission.shape
    log_init = np.asarray(log_init, dtype=float)
    log_A = _broadcast_trans(log_trans, T, S)
    log_delta = np.empty((T, S), dtype=float)
    psi = np.zeros((T, S), dtype=np.intp)
    log_delta[0] = log_init + log_emission[0]
    for t in range(1, T):
        scores = log_delta[t - 1][:, None] + log_A[t - 1]
        psi[t] = np.argmax(scores, axis=0)
        log_delta[t] = log_emission[t] + np.max(scores, axis=0)
    path = np.empty(T, dtype=np.intp)
    path[T - 1] = int(np.argmax(log_delta[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path.astype(int)


# --- Semi-Markov (explicit-duration) Viterbi (§5). A plain HMM's dwell is geometric ⇒
# memoryless ⇒ it still admits 1-frame blips. A shifted negative-binomial dwell concentrates
# mass away from d=1 (concentration=1 recovers the geometric), so short spurious segments are
# intrinsically expensive — the principled replacement for "drop short runs". The package's
# detect() takes its MAP path from THIS decoder (use_semi_markov=True) and its posterior from
# the plain forward–backward above, so we only need the segmental Viterbi here. ----------

_LOG_ZERO = -1e30


def _nb_params(mean_dwell_frames: float, concentration: float) -> tuple[float, float]:
    r = float(max(concentration, 1e-6))
    mean_k = float(max(mean_dwell_frames, 1.0)) - 1.0
    p = r / (r + mean_k) if mean_k > 0.0 else 1.0
    return r, min(max(p, 1e-12), 1.0 - 1e-15)


def duration_logpmf(d, mean_dwell_frames: float, concentration: float):
    """log-pmf of a dwell of d≥1 frames: a shifted NB(r, p) with k=d−1, mean = mean_dwell_frames."""
    d_arr = np.asarray(d, dtype=float)
    scalar_in = d_arr.ndim == 0
    r, p = _nb_params(mean_dwell_frames, concentration)
    valid = (d_arr >= 1.0) & (np.abs(d_arr - np.round(d_arr)) < 1e-9)
    logpmf = np.where(valid, nbinom.logpmf(np.where(valid, d_arr - 1.0, 0.0), r, p), _LOG_ZERO)
    return float(logpmf) if scalar_in else logpmf


def _duration_logsf(d: int, mean_dwell_frames: float, concentration: float) -> float:
    """log P(duration ≥ d) — the right-censored mass for a segment that hits the cap."""
    r, p = _nb_params(mean_dwell_frames, concentration)
    return float(nbinom.logsf(d - 2, r, p))


def _duration_table(mean_dwell_frames: np.ndarray, concentration: float, max_dur: int) -> np.ndarray:
    """(S, max_dur) table: pmf for d=1..max_dur−1, survival (censored) for d=max_dur."""
    S = mean_dwell_frames.shape[0]
    table = np.empty((S, max_dur), dtype=float)
    if max_dur >= 2:
        durations = np.arange(1, max_dur, dtype=float)
        for s in range(S):
            table[s, : max_dur - 1] = duration_logpmf(durations, float(mean_dwell_frames[s]), concentration)
    for s in range(S):
        table[s, max_dur - 1] = _duration_logsf(max_dur, float(mean_dwell_frames[s]), concentration)
    return table


def _default_max_dur(mean_dwell_frames: np.ndarray, max_dur: int | None) -> int:
    if max_dur is not None:
        return int(max(1, max_dur))
    return int(max(1, np.ceil(5.0 * float(np.max(mean_dwell_frames)))))


def _interseg_logtrans(log_trans: np.ndarray) -> np.ndarray:
    """The transition restricted to between-segment (state-changing) jumps: zero the diagonal, renorm."""
    A = np.array(log_trans, dtype=float, copy=True)
    np.fill_diagonal(A, _LOG_ZERO)
    row_norm = logsumexp(A, axis=1)
    safe = np.where(np.isfinite(row_norm) & (row_norm > _LOG_ZERO / 2), row_norm, 0.0)
    return A - safe[:, None]


def hsmm_viterbi(log_emission, log_trans, log_init, mean_dwell_frames, concentration, max_dur=None) -> np.ndarray:
    """MAP state path under an explicit-duration semi-Markov model (segmental Viterbi with
    duration censoring). A segment of state s over d frames scores Σemit + duration_logpmf(d|s)
    + inter-segment transition. A segment that hits the cap D is right-censored (scores the
    survival mass) and may continue as the same state at zero cost, so a bout longer than D is
    exact, not truncated. We track natural-end vs. censored flavours to backtrace whole segments.
    """
    emit = np.asarray(log_emission, dtype=float)
    T, S = emit.shape
    log_trans = np.asarray(log_trans, dtype=float)
    log_init = np.asarray(log_init, dtype=float)
    mean = np.asarray(mean_dwell_frames, dtype=float)
    D = _default_max_dur(mean, max_dur)
    log_dur = _duration_table(mean, concentration, D)
    A = _interseg_logtrans(log_trans)

    E = np.zeros((T + 1, S), dtype=float)
    np.cumsum(emit, axis=0, out=E[1:])
    NEG = _LOG_ZERO
    V_end = np.full((T + 1, S), NEG, dtype=float)
    V_cens = np.full((T + 1, S), NEG, dtype=float)
    bd_end = np.zeros((T + 1, S), dtype=np.intp)
    bp_end = np.full((T + 1, S), -1, dtype=np.intp)
    bf_end = np.zeros((T + 1, S), dtype=np.intp)
    bd_cens = np.zeros((T + 1, S), dtype=np.intp)
    bp_cens = np.full((T + 1, S), -1, dtype=np.intp)
    bf_cens = np.zeros((T + 1, S), dtype=np.intp)
    s_range = np.arange(S)

    for t in range(1, T + 1):
        d_max = min(t, D)
        seg_emit = E[t][None, :] - E[t - d_max: t][::-1, :]       # (d_max, S)
        dur_term = log_dur[:, :d_max].T                          # (d_max, S)
        tot = np.maximum(V_end, V_cens)
        tot_prev = tot[t - d_max: t][::-1, :]                    # (d_max, S')
        cens_prev = V_cens[t - d_max: t][::-1, :]                # same-state continuation
        trans_score = tot_prev[:, :, None] + A[None, :, :]        # (d_max, S', S)
        switch_in = np.max(trans_score, axis=1)
        switch_idx = np.argmax(trans_score, axis=1)
        cont_in = cens_prev
        entry = np.maximum(switch_in, cont_in)
        if d_max == t:
            entry = entry.copy()
            start_vals = np.maximum(log_init, entry[t - 1])
            entry[t - 1] = start_vals
        cand = entry + seg_emit + dur_term
        use_cont = cont_in >= switch_in
        src = switch_idx
        idx_d = np.arange(d_max)[:, None]
        t_pred = (t - (idx_d + 1))
        ve_src = V_end[t_pred, src]
        vc_src = V_cens[t_pred, src]
        switch_pflav = (vc_src > ve_src).astype(np.intp)
        pstate = np.where(use_cont, s_range[None, :], src)
        pflav = np.where(use_cont, 1, switch_pflav)
        if d_max == t:
            is_start = start_vals >= np.maximum(switch_in[t - 1], cont_in[t - 1])
            pstate[t - 1] = np.where(is_start, -1, pstate[t - 1])
            pflav[t - 1] = np.where(is_start, 0, pflav[t - 1])
        is_cap = d_max == D
        if is_cap:
            nat = cand[: D - 1] if D >= 2 else np.full((0, S), NEG)
            cens = cand[D - 1]
        else:
            nat = cand
            cens = None
        if nat.shape[0] > 0:
            best_nat = np.argmax(nat, axis=0)
            V_end[t] = nat[best_nat, s_range]
            bd_end[t] = best_nat + 1
            bp_end[t] = pstate[best_nat, s_range]
            bf_end[t] = pflav[best_nat, s_range]
        if is_cap and cens is not None:
            V_cens[t] = cens
            bd_cens[t] = D
            bp_cens[t] = pstate[D - 1]
            bf_cens[t] = pflav[D - 1]

    term_end = V_end[T]
    term_cens = V_cens[T]
    s = int(np.argmax(np.maximum(term_end, term_cens)))
    flav = 1 if term_cens[s] > term_end[s] else 0
    path = np.empty(T, dtype=np.intp)
    t = T
    while t > 0:
        bd, bp, bf = (bd_cens, bp_cens, bf_cens) if flav == 1 else (bd_end, bp_end, bf_end)
        d = int(bd[t, s])
        prev_s = int(bp[t, s])
        prev_f = int(bf[t, s])
        path[t - d: t] = s
        t -= d
        if prev_s < 0:
            break
        s = prev_s
        flav = prev_f
    return path.astype(int)


class HMM:
    """Bundles a temporal prior so emissions are the only per-call input (forward–backward + Viterbi)."""

    def __init__(self, log_trans, log_init):
        self.log_trans = log_trans
        self.log_init = log_init

    def posterior(self, log_emission):
        return forward_backward(log_emission, self.log_trans, self.log_init)

    def map_path(self, log_emission):
        return viterbi(log_emission, self.log_trans, self.log_init)


class SemiMarkovHMM:
    """Explicit-duration HMM — same interface as HMM; map_path is the segmental Viterbi above."""

    def __init__(self, log_trans, log_init, mean_dwell_frames, concentration, max_dur=None):
        self.log_trans = log_trans
        self.log_init = log_init
        self.mean_dwell_frames = mean_dwell_frames
        self.concentration = concentration
        self.max_dur = max_dur

    def map_path(self, log_emission):
        return hsmm_viterbi(log_emission, self.log_trans, self.log_init, self.mean_dwell_frames, self.concentration, self.max_dur)


# ======================================================================================
# §6  Impacts and make/break events — the singular instants.
#
# At touchdown the relative normal velocity is arrested almost discontinuously (a reset map
# v⁺ = −e·v⁻). The force is a MEASURE: a smooth part plus atoms (impulses) at impact instants.
# Those atoms carry the sharpest timing and briefly reveal force/material. We find them with a
# matched filter — correlate v_normal against an antisymmetric arrest (velocity-step) template
# on a *lightly* smoothed signal (over-smoothing erases the jump), gate by closing speed, and
# read restitution from robust medians on each side. Events (touchdown/liftoff) are then placed
# sub-frame at the gap (or v_normal) zero-crossing straddling each contact-mask transition.
# ======================================================================================


def _typical_dt(t: np.ndarray) -> float:
    if t.shape[0] < 2:
        return 0.0
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0.0)]
    return float(np.median(d)) if d.size else 0.0


def _arrest_template(half_samples: int) -> np.ndarray:
    """Zero-mean, unit-L2 antisymmetric kernel: −1 before the centre, +1 after — the step edge."""
    n = 2 * half_samples + 1
    k = np.zeros(n, dtype=float)
    k[:half_samples] = -1.0
    k[half_samples + 1:] = +1.0
    norm = np.linalg.norm(k)
    if norm > 0.0:
        k /= norm
    return k


def _correlate_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Length-preserving sliding correlation, edge-replicated (zero-padding would invent a step)."""
    half = (kernel.shape[0] - 1) // 2
    xp = np.pad(x, (half, half), mode="edge")
    return np.correlate(xp, kernel, mode="valid")


def _side_velocity(v: np.ndarray, center: int, half_samples: int, *, after: bool, guard: int = 0) -> float:
    """Robust (median) representative velocity on one side of the arrest, skipping a guard band
    so the blurred ramp does not drag v_before/v_after toward each other."""
    n = v.shape[0]
    if after:
        lo, hi = center + guard + 1, center + guard + half_samples
    else:
        lo, hi = center - guard - half_samples, center - guard - 1
    lo = max(0, lo)
    hi = min(n - 1, hi)
    if hi < lo:
        idx = center + (1 if after else -1)
        idx = min(n - 1, max(0, idx))
        return float(v[idx])
    seg = v[lo:hi + 1]
    seg = seg[np.isfinite(seg)]
    return float(np.median(seg)) if seg.size else float("nan")


def _refine_center_subframe(s: np.ndarray, peak: int) -> float:
    """Sub-frame peak by parabolic interpolation of s[peak−1:peak+2]."""
    n = s.shape[0]
    if peak <= 0 or peak >= n - 1:
        return float(peak)
    a, b, c = s[peak - 1], s[peak], s[peak + 1]
    denom = a - 2.0 * b + c
    if not np.isfinite(denom) or denom == 0.0:
        return float(peak)
    offset = 0.5 * (a - c) / denom
    if not np.isfinite(offset):
        return float(peak)
    return peak + float(np.clip(offset, -0.5, 0.5))


def _interp_time(t: np.ndarray, frac_index: float) -> float:
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


def detect_impacts(obs: ContactObservations, params: ImpactParams, mass: float | None = None) -> list[ContactImpulse]:
    """Detect and characterize impact atoms (§6): matched filter on a lightly-smoothed v_normal,
    gated by closing speed; report closing speed, restitution e = max(0, −v_after/v_before), and
    the impulse m·Δv (NaN if mass unknown)."""
    t = np.asarray(obs.t, dtype=float).ravel()
    v = np.asarray(obs.v_normal, dtype=float).ravel()
    n = t.shape[0]
    if n < 3:
        return []
    v_s = gaussian_smooth(v, t, sigma_time=max(0.0, params.detect_smooth_time))
    dt = _typical_dt(t)
    if dt <= 0.0:
        return []
    half = int(round(params.template_halfwidth_time / dt))
    half = max(1, half)
    half = min(half, (n - 1) // 2)
    if half < 1:
        return []
    kernel = _arrest_template(half)
    guard = int(np.ceil(2.0 * max(0.0, params.detect_smooth_time) / dt))
    guard = max(1, guard)
    guard = min(guard, max(0, half - 1))
    response = _correlate_same(v_s, kernel)
    # A unit-norm step kernel maps a velocity step dv to a peak dv·√(half/2); threshold at half the
    # smallest physically-real arrest so the exactly-zero free-flight response never qualifies.
    min_step_response = params.min_closing_speed * np.sqrt(half / 2.0)
    threshold = max(0.5 * min_step_response, 1e-9)
    candidates, _ = find_peaks(response, height=threshold, distance=half + 1)
    impulses: list[ContactImpulse] = []
    for c in candidates:
        v_before = _side_velocity(v_s, c, half, after=False, guard=guard)
        v_after = _side_velocity(v_s, c, half, after=True, guard=guard)
        if not np.isfinite(v_before):
            continue
        closing_speed = -v_before if v_before < 0.0 else 0.0
        if closing_speed < params.min_closing_speed:
            continue
        if not np.isfinite(v_after):
            continue
        if (v_after - v_before) < params.min_closing_speed:   # require a genuine rise (arrest)
            continue
        if np.isfinite(v_after) and v_before != 0.0:
            restitution = float(max(0.0, -v_after / v_before))
        else:
            restitution = float("nan")
        if mass is not None and np.isfinite(v_after) and np.isfinite(v_before):
            normal_impulse = float(mass * (v_after - v_before))
        else:
            normal_impulse = float("nan")
        frac = float(np.clip(_refine_center_subframe(response, c), 0.0, n - 1))
        impulses.append(ContactImpulse(
            time=_interp_time(t, frac), index=int(np.clip(round(frac), 0, n - 1)),
            closing_speed=float(closing_speed), restitution=restitution, normal_impulse=normal_impulse,
        ))
    impulses.sort(key=lambda imp: imp.time)
    return impulses


def _interp_time_at_index(t: np.ndarray, frac_index: float) -> float:
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


def _zero_crossing_frac_index(y: np.ndarray, lo: int, hi: int, *, rising: bool | None = None) -> float | None:
    """Fractional index of the first zero-crossing of y in [lo, hi] (of the requested direction)."""
    lo = max(0, lo)
    hi = min(y.shape[0] - 1, hi)
    if hi <= lo:
        return None
    for k in range(lo, hi):
        a = y[k]
        b = y[k + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0:
            if rising is None or (rising and b >= 0.0) or ((not rising) and b <= 0.0):
                return float(k)
        if a < 0.0 < b:
            if rising is None or rising:
                return float(k) + a / (a - b)
        elif a > 0.0 > b:
            if rising is None or (not rising):
                return float(k) + a / (a - b)
    if y[hi] == 0.0:
        return float(hi)
    return None


def _refine_transition(kind: str, boundary: int, t: np.ndarray, gap: np.ndarray, v_normal: np.ndarray) -> tuple[float, int]:
    """Refine one mask transition to a (time, nearest_index): touchdown ⇒ gap falling through 0
    (fallback v_normal rising); liftoff ⇒ gap rising through 0 (fallback v_normal rising)."""
    lo = boundary - 2
    hi = boundary + 1
    if kind == "touchdown":
        frac = _zero_crossing_frac_index(gap, lo, hi, rising=False)
        if frac is None:
            frac = _zero_crossing_frac_index(v_normal, lo, hi, rising=True)
    else:
        frac = _zero_crossing_frac_index(gap, lo, hi, rising=True)
        if frac is None:
            frac = _zero_crossing_frac_index(v_normal, lo, hi, rising=True)
    if frac is None:
        frac = boundary - 0.5
    frac = float(np.clip(frac, 0.0, t.shape[0] - 1))
    return _interp_time_at_index(t, frac), int(np.clip(round(frac), 0, t.shape[0] - 1))


def detect_events(obs: ContactObservations, in_contact: np.ndarray, t: np.ndarray | None = None) -> list[ContactEvent]:
    """Find every free→contact (touchdown) and contact→free (liftoff) transition in the mask and
    refine each event time sub-frame from the kinematics straddling the boundary (§6). A mask that
    starts/ends mid-contact emits no spurious event at the record ends."""
    obs_t = np.asarray(obs.t, dtype=float).ravel()
    gap = np.asarray(obs.gap, dtype=float).ravel()
    v_normal = np.asarray(obs.v_normal, dtype=float).ravel()
    time_base = obs_t if t is None else np.asarray(t, dtype=float).ravel()
    mask = np.asarray(in_contact).ravel().astype(bool)
    n = mask.shape[0]
    if n < 2 or time_base.shape[0] < 2:
        return []
    events: list[ContactEvent] = []
    changes = np.nonzero(mask[1:] != mask[:-1])[0] + 1
    for i in changes:
        boundary = int(i)
        kind = "touchdown" if mask[boundary] else "liftoff"
        time, index = _refine_transition(kind, boundary, time_base, gap, v_normal)
        events.append(ContactEvent(kind=kind, time=time, index=index))
    events.sort(key=lambda e: e.time)
    return events


# ======================================================================================
# §7  Dynamics & material — what is knowable.
#
# Force magnitude is a Lagrange multiplier set by the dynamics, unobservable from kinematics
# alone. The instant we grant a known compliance k, penetration becomes a calibrated force gauge
# λ = k·δ (measured below the EM-calibrated resting datum). Friction is set-valued: while
# sticking ‖λ_t‖ ≤ μλ_n; gross sliding begins at the cone boundary. With force in hand the cone
# cross-checks the kinematic stick/slip label — but we never let the reconstructed force override
# the observed motion (the motion is the hard evidence).
# ======================================================================================


def normal_force_from_penetration(gap, gap_bias, in_contact, material: MaterialParams) -> np.ndarray:
    """λ = k·δ with δ = max(0, −(gap − gap_bias)), zeroed off contact frames (Signorini, §2).
    All-NaN when stiffness is unknown (force unobservable)."""
    gap = np.asarray(gap, dtype=float)
    in_contact = np.asarray(in_contact, dtype=bool)
    if material.stiffness is None:
        return np.full(gap.shape, np.nan, dtype=float)
    k = float(material.stiffness)
    penetration = np.maximum(0.0, -(gap - gap_bias))
    force = np.maximum(0.0, k * penetration)
    return np.where(in_contact, force, 0.0)


def friction_stick_slip(obs: ContactObservations, normal_force, material: MaterialParams) -> list[str]:
    """Per-frame stick/slip label (§7). Kinematic decision ‖v_t‖ vs slip_speed_threshold is always
    available; when the force is known the Coulomb cone refines a borderline stick. '' off contact."""
    v_tan = np.asarray(obs.v_tangent, dtype=float)
    speed = np.linalg.norm(v_tan, axis=-1)
    T = speed.shape[0]
    nf = np.asarray(normal_force, dtype=float)
    force_known = nf.shape == speed.shape and not np.all(np.isnan(nf))
    mu = float(material.friction)
    v_thresh = float(material.slip_speed_threshold)
    if force_known:
        peak = float(np.nanmax(nf)) if np.any(np.isfinite(nf)) else 0.0
        force_floor = 1e-3 * peak if peak > 0.0 else 0.0
        loaded = np.isfinite(nf) & (nf > force_floor)
    else:
        loaded = np.ones(T, dtype=bool)
    labels: list[str] = []
    for i in range(T):
        if not loaded[i]:
            labels.append("")
            continue
        kinematic_slip = speed[i] > v_thresh
        if not force_known:
            labels.append("slip" if kinematic_slip else "stick")
            continue
        cone_capacity = mu * float(nf[i])
        cap_floor = 1e-3 * mu * (float(np.nanmax(nf)) if np.any(np.isfinite(nf)) else 0.0)
        cone_can_stick = cone_capacity > cap_floor
        if kinematic_slip:
            labels.append("slip")
        else:
            labels.append("stick" if cone_can_stick else "slip")
    return labels


# ======================================================================================
# §4–§8  The assembled single-pair detector — the generative HMM wired end to end.
#
# detect(obs) runs the pragmatic ladder: (1) the six states FREE + five modes; (2) the gap-gated
# transition prior; (3) EM self-calibration of the resting-gap bias (the contact-responsibility-
# weighted mean gap — the principled replacement for a quiet-frame median); (4) the smoothed
# posterior from forward–backward, the MAP segmentation from the semi-Markov decoder, the events
# and impact atoms, and — if a stiffness is known — the force gauge and stick/slip labels.
# ======================================================================================


def _median_dt(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float).ravel()
    if t.shape[0] < 2:
        return 1.0
    dts = np.diff(t)
    dts = dts[dts > 0.0]
    return float(np.median(dts)) if dts.size else 1.0


def _intervals_from_map(t: np.ndarray, map_labels: list[str]) -> list[ContactInterval]:
    """Contiguous non-FREE runs of the MAP path, each tagged with its dominant (most frequent) mode."""
    t = np.asarray(t, dtype=float).ravel()
    n = len(map_labels)
    intervals: list[ContactInterval] = []
    i = 0
    while i < n:
        if map_labels[i] == FREE:
            i += 1
            continue
        j = i
        while j < n and map_labels[j] != FREE:
            j += 1
        run = map_labels[i:j]
        counts: dict[str, int] = {}
        for lbl in run:
            counts[lbl] = counts.get(lbl, 0) + 1
        dominant = max(counts, key=lambda k: counts[k])
        intervals.append(ContactInterval(t_start=float(t[i]), t_end=float(t[j - 1]), mode=dominant))
        i = j
    return intervals


def _emission_scaled_to_motion(cfg: DetectorConfig, obs) -> DetectorConfig:
    """Size the sliding scale to THIS pair's own motion (90th-percentile tangential speed), never
    narrowing below the defaults — so a fast slider (struck ball, skidding box) is not read FREE."""
    vt = np.linalg.norm(np.asarray(obs.v_tangent, dtype=float), axis=1)
    if vt.size == 0:
        return cfg
    scale = float(np.percentile(vt, 90))
    if scale <= cfg.emission.slide_speed:
        return cfg
    c = copy.deepcopy(cfg)
    c.emission.slide_speed = scale
    c.emission.free_vel_sigma = max(c.emission.free_vel_sigma, 2.0 * scale)
    return c


class ContactDetector:
    """Infer the per-frame contact state from support-relative observations (§4–§8)."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config if config is not None else DetectorConfig()

    def detect(self, obs: ContactObservations) -> DetectionResult:
        cfg = _emission_scaled_to_motion(self.config, obs)
        states = list(ALL_STATES)
        contact_state_idx = [i for i, s in enumerate(states) if s != FREE]
        t = np.asarray(obs.t, dtype=float).ravel()
        gap = np.asarray(obs.gap, dtype=float).ravel()
        S = len(states)
        free_idx = states.index(FREE)

        dt = _median_dt(t)
        gated = gated_transition_tensor(obs, states, dt, cfg.transition)
        log_trans_gated = np.log(gated)
        base = base_transition_matrix(states, dt, cfg.transition)
        log_trans_base = np.log(base)

        init = np.full(S, (1.0 - 0.5) / (S - 1), dtype=float)
        init[free_idx] = 0.5
        log_init = np.log(init)
        smoother = HMM(log_trans_gated, log_init)

        # Per-frame measurement-uncertainty tempering is OFF by default (no meas_cov) — a no-op.
        temper_w = None

        # (c) EM self-calibration of the resting-gap bias.
        gap_bias = self._calibrate_gap_bias(obs, cfg, states, gap, smoother, temper_w, contact_state_idx)

        # (d) final smoothed inference with the calibrated bias.
        log_em = log_emissions(obs, cfg.emission, gap_bias, states, cfg.material, force=cfg.force)
        gamma, _loglik = smoother.posterior(log_em)
        contact_posterior = 1.0 - gamma[:, free_idx]

        if cfg.transition.use_semi_markov:
            dt_safe = max(dt, 1e-9)
            mean_dwell_frames = np.full(S, float(cfg.transition.mean_dwell_time) / dt_safe, dtype=float)
            if IMPACT in states:
                mean_dwell_frames[states.index(IMPACT)] = float(cfg.transition.impact_dwell_time) / dt_safe
            decoder = SemiMarkovHMM(log_trans_base, log_init, mean_dwell_frames, concentration=float(cfg.transition.dwell_concentration))
            path = decoder.map_path(log_em)
        else:
            path = smoother.map_path(log_em)
        map_state = [states[int(s)] for s in path]
        in_contact = np.array([s != FREE for s in map_state], dtype=bool)
        intervals = _intervals_from_map(t, map_state)

        # (e) make/break events + impact atoms (mass unknown ⇒ NaN impulse magnitudes, §7).
        ev = detect_events(obs, in_contact, t=t)
        impulses = detect_impacts(obs, cfg.impact, mass=None)

        # (f) dynamics & material.
        if cfg.material.stiffness is not None:
            normal_force = np.asarray(normal_force_from_penetration(gap, gap_bias, in_contact, cfg.material), dtype=float)
            slip = friction_stick_slip(obs, normal_force, cfg.material)
        else:
            normal_force = None
            kin = friction_stick_slip(obs, np.full(t.shape[0], np.nan, dtype=float), cfg.material)
            slip = [kin[i] if in_contact[i] else "" for i in range(len(kin))]

        return DetectionResult(
            t=t, contact_posterior=np.asarray(contact_posterior, dtype=float),
            state_posterior=np.asarray(gamma, dtype=float), map_state=map_state,
            in_contact=in_contact, intervals=intervals, events=ev, resting_bias=float(gap_bias),
            normal_force=normal_force, states=states, impulses=impulses, slip_state=slip,
        )

    @staticmethod
    def _calibrate_gap_bias(obs, cfg, states, gap, smoother, temper_w, contact_state_idx):
        """EM: each step re-estimates the bias as the contact-responsibility-weighted mean gap (§7/§8)."""
        max_bias = abs(float(cfg.calibration.max_resting_bias))
        gap_bias = 0.0
        for _ in range(max(0, int(cfg.calibration.em_iters))):
            log_em = log_emissions(obs, cfg.emission, gap_bias, states, cfg.material, force=cfg.force)
            gamma, _ = smoother.posterior(log_em)
            w = gamma[:, contact_state_idx].sum(axis=1)
            wsum = float(w.sum())
            if wsum > 1e-12:
                gap_bias = float(np.clip(np.dot(w, gap) / wsum, -max_bias, max_bias))
        return gap_bias


# ======================================================================================
# §8  The multi-body contact graph — a posterior over active-constraint *structures*.
#
# Lift the single-pair estimator from one pair to a whole graph whose edges are candidate
# body-pair contacts. The hidden thing is no longer a bit per edge but WHICH SET of edges is
# active, over time. We run the per-edge detector, then over the 2^E subsets (exact for the
# small graphs here) put a joint HMM: the subset emission is the per-edge sum of active/inactive
# log-evidence (edges observe disjoint pairs ⇒ conditionally independent given the set); a soft
# global energy/dissipation factor couples them; a Markov dwell keeps the active set coherent.
# forward–backward marginalizes to P(edge active); Viterbi gives the MAP active set.
# ======================================================================================

_WORLD = "world"
_PROB_EPS = 1e-6
_ENERGY_GAIN = 0.5   # max per-frame nudge (nats) of the energy factor — tips ties, never overrides


def _resolve_support(scene: MultiBodyScene, support_body: str, like: PoseTrajectory | None) -> PoseTrajectory | None:
    """Return the support's trajectory, synthesizing a static identity 'world' floor on demand (§1)."""
    body = scene.bodies.get(support_body)
    if body is not None:
        return body
    if support_body != _WORLD or like is None:
        return None
    t = np.asarray(like.t, dtype=float).ravel()
    T = int(t.shape[0])
    quat = np.zeros((T, 4), dtype=float)
    quat[:, 0] = 1.0
    return PoseTrajectory(t=t, position=np.zeros((T, 3), dtype=float), quat=quat)


def build_candidate_edges(scene: MultiBodyScene, params: GraphParams | None = None) -> list[ContactEdge]:
    """Broad-phase: keep an edge only if its bodies come within proximity_gap at some frame (§8).
    The shipped scenes already carry only plausible edges, so this is the identity on them."""
    if params is None:
        params = DetectorConfig().graph
    gap_thresh = float(getattr(params, "proximity_gap", 0.05))
    kept: list[ContactEdge] = []
    for edge in scene.edges:
        moving = scene.bodies.get(edge.moving_body)
        support = _resolve_support(scene, edge.support_body, moving)
        if moving is None or support is None:
            continue
        try:
            obs = observe(moving, support, edge.surface, edge.contact_point_local)
            min_gap = float(np.nanmin(np.asarray(obs.gap, dtype=float)))
        except Exception:
            kept.append(edge)
            continue
        if min_gap <= gap_thresh:
            kept.append(edge)
    return kept


# --- The soft global energy/dissipation factor (§8). Mechanical energy may only decrease
# through a genuinely dissipative active contact (sliding/impact); an energy drop with no active
# dissipative edge is penalized, a spontaneous gain is mildly down-weighted. It is a fraction-of-
# a-nat nudge that breaks ties the per-edge kinematics leave open, never a veto. --------------


def _edge_ids(edges) -> list[str]:
    # Duck-typed (accept any edge-like object exposing .edge_id, or a bare id string).
    return [e.edge_id if hasattr(e, "edge_id") else str(e) for e in edges]


def _body_mass(masses, name: str) -> float:
    if masses is None:
        return 1.0
    try:
        m = float(masses[name])
    except (KeyError, TypeError, ValueError):
        return 1.0
    return m if np.isfinite(m) and m > 0.0 else 1.0


def _energy_budget(scene: MultiBodyScene, masses=None) -> dict:
    """Per-body KE+PE over time and the scene total mechanical energy E_mech(t); dE its per-frame
    change (translational KE only; unit masses ⇒ 'relative' energy — the sign of dE is unaffected)."""
    bodies = getattr(scene, "bodies", {}) or {}
    t_ref = None
    for traj in bodies.values():
        t_ref = np.asarray(traj.t, dtype=float).ravel()
        break
    if t_ref is None or t_ref.size == 0:
        return {"t": np.zeros(0), "bodies": {}, "E_mech": np.zeros(0), "dE": np.zeros(0)}
    T = t_ref.shape[0]
    if T >= 2:
        dts = np.diff(t_ref)
        dts = dts[dts > 0.0]
        med_dt = float(np.median(dts)) if dts.size else 0.0
    else:
        med_dt = 0.0
    sigma_time = max(0.05, 3.0 * med_dt) if med_dt > 0.0 else 0.05
    per_body: dict[str, dict] = {}
    E_mech = np.zeros(T, dtype=float)
    for name, traj in bodies.items():
        pos = np.asarray(traj.position, dtype=float)
        if pos.ndim != 2 or pos.shape[0] != T or pos.shape[1] < 3:
            continue
        m = _body_mass(masses, name)
        pos_s = gaussian_smooth(pos, t_ref, sigma_time)
        vel = derivative(pos_s, t_ref)
        speed2 = np.sum(vel * vel, axis=1)
        ke = 0.5 * m * speed2
        pe = m * 9.81 * pos[:, 2]
        per_body[name] = {"speed": np.sqrt(speed2)}
        E_mech += pe + ke
    dE = np.zeros(T, dtype=float)
    if T >= 2:
        dE[1:] = np.diff(E_mech)
    return {"t": t_ref, "bodies": per_body, "E_mech": E_mech, "dE": dE}


def _normalize_subset_index(subset_index_per_state, valid_ids: set[str], ids_order: list[str]) -> list[set[str]]:
    """Coerce the subset alphabet (tuples of edge indices, e.g. (), (0,), (0,1)) to sets of edge ids."""
    out: list[set[str]] = []
    for entry in subset_index_per_state:
        ids = set()
        for x in entry:
            i = int(x)
            ids.add(ids_order[i] if 0 <= i < len(ids_order) else str(i))
        out.append({i for i in ids if i in valid_ids})
    return out


def energy_log_factor(scene: MultiBodyScene, edges, subset_index_per_state, masses=None) -> np.ndarray:
    """(T, n_states) soft log-factor enforcing the energy/dissipation budget (§8); all-zeros (no-op)
    when it cannot be computed; mean-centred per frame (a pure relative preference)."""
    ids = _edge_ids(edges)
    states = _normalize_subset_index(subset_index_per_state, set(ids), ids)
    n_states = len(states)
    budget = _energy_budget(scene, masses=masses)
    t = np.asarray(budget.get("t", np.zeros(0)), dtype=float)
    T = t.shape[0]
    dE = np.asarray(budget.get("dE", np.zeros(0)), dtype=float)
    if T == 0 or n_states == 0:
        return np.zeros((max(T, 0), max(n_states, 0)), dtype=float)
    per_body = budget.get("bodies", {})
    all_speed = [np.asarray(b.get("speed", np.zeros(T)), dtype=float) for b in per_body.values()]
    if all_speed:
        sp = np.concatenate(all_speed)
        sp = sp[sp > 0.0]
        speed_scale = float(np.median(sp)) if sp.size else 0.05
    else:
        speed_scale = 0.05
    move_thresh = max(0.02, 0.25 * speed_scale)
    edge_to_moving = {e.edge_id: e.moving_body for e in edges if hasattr(e, "moving_body")}
    edge_moving: dict[str, np.ndarray] = {}
    for eid in ids:
        mb = edge_to_moving.get(eid)
        spd = np.asarray(per_body[mb]["speed"], dtype=float) if (mb in per_body) else None
        if spd is None or spd.shape[0] != T:
            edge_moving[eid] = np.ones(T, dtype=bool)
        else:
            edge_moving[eid] = spd > move_thresh
    state_dissipative = np.zeros((n_states, T), dtype=bool)
    for s_idx, active in enumerate(states):
        if not active:
            continue
        avail = np.zeros(T, dtype=bool)
        for eid in active:
            avail |= edge_moving.get(eid, np.ones(T, dtype=bool))
        state_dissipative[s_idx] = avail
    abs_dE = np.abs(dE)
    e_scale = float(np.median(abs_dE[abs_dE > 0.0])) if np.any(abs_dE > 0.0) else 0.0
    out = np.zeros((T, n_states), dtype=float)
    if e_scale <= 0.0:
        return out
    drop = np.tanh(np.maximum(0.0, -dE) / e_scale)
    gain = np.tanh(np.maximum(0.0, dE) / e_scale)
    for s_idx in range(n_states):
        diss = state_dissipative[s_idx].astype(float)
        out[:, s_idx] = _ENERGY_GAIN * drop * (2.0 * diss - 1.0) - _ENERGY_GAIN * gain
    out -= out.mean(axis=1, keepdims=True)
    return out


def _enumerate_subsets(num_edges: int) -> list[tuple[int, ...]]:
    """All 2^E active sets as edge-index tuples ordered by the bitmask integer (index 0 = ∅)."""
    return [tuple(e for e in range(num_edges) if (k >> e) & 1) for k in range(1 << num_edges)]


def _subset_active_mask(num_edges: int) -> np.ndarray:
    n_subsets = 1 << num_edges
    mask = np.zeros((n_subsets, num_edges), dtype=bool)
    for k in range(n_subsets):
        for e in range(num_edges):
            if (k >> e) & 1:
                mask[k, e] = True
    return mask


def _subset_log_transition(n_subsets: int, dt: float, dwell_time: float) -> np.ndarray:
    """Markov chain on the subsets: P(stay)=exp(−dt/dwell), the rest split uniformly over others."""
    if n_subsets <= 1:
        return np.zeros((1, 1), dtype=float)
    dt = max(float(dt), 1e-9)
    dwell = max(float(dwell_time), 1e-9)
    p_stay = float(np.exp(-dt / dwell))
    p_switch = max((1.0 - p_stay) / (n_subsets - 1), _PROB_EPS / n_subsets)
    A = np.full((n_subsets, n_subsets), p_switch, dtype=float)
    np.fill_diagonal(A, p_stay)
    A /= A.sum(axis=1, keepdims=True)
    return np.log(A)


def detect_scene(scene: MultiBodyScene, config: DetectorConfig | None = None) -> GraphDetectionResult:
    """Infer the joint active-set posterior over a multi-body contact graph (§8).

    Per edge: observe + ContactDetector.detect (each edge self-scales to its own motion). Joint:
    enumerate the 2^E active sets, build the per-edge-sum subset emission, add the optional energy
    factor, run forward–backward + Viterbi over the subset alphabet, marginalize to per-edge actives.
    """
    cfg = config if config is not None else DetectorConfig()
    edges = list(scene.edges)
    edge_ids = [e.edge_id for e in edges]
    E = len(edges)
    any_body = next(iter(scene.bodies.values())) if scene.bodies else None
    t = np.asarray(any_body.t, dtype=float).ravel() if any_body is not None else np.zeros(1, dtype=float)
    T = int(t.shape[0])

    # (1) per-edge single-pair detection.
    per_edge: dict[str, DetectionResult] = {}
    per_edge_posterior = np.zeros((T, max(E, 0)), dtype=float)
    for j, edge in enumerate(edges):
        moving = scene.bodies[edge.moving_body]
        support = _resolve_support(scene, edge.support_body, moving)
        if support is None:
            raise KeyError(f"edge {edge.edge_id!r}: support {edge.support_body!r} not in scene and not 'world'")
        obs = observe(moving, support, edge.surface, edge.contact_point_local, cfg.vel_smooth_time, geometry=edge.geometry)
        res = ContactDetector(cfg).detect(obs)
        per_edge[edge.edge_id] = res
        per_edge_posterior[:, j] = np.asarray(res.contact_posterior, dtype=float).ravel()

    if E == 0:
        return GraphDetectionResult(
            t=t, edges=[], per_edge={}, active_posterior=np.zeros((T, 0), dtype=float),
            map_active_set=[[] for _ in range(T)],
            meta={"num_edges": 0, "num_subsets": 1, "energy_prior_active": False, "balance_prior_active": False},
        )

    # (2) joint active-set inference over the 2^E subsets (exact enumeration; E ≤ 4 here).
    p = np.clip(per_edge_posterior, _PROB_EPS, 1.0 - _PROB_EPS)
    log_active, log_inactive = np.log(p), np.log1p(-p)
    subsets = _enumerate_subsets(E)
    active_mask = _subset_active_mask(E)
    n_subsets = len(subsets)
    # subset emission = Σ_e (active if e∈subset else inactive) log-evidence.
    log_emission = np.where(active_mask[None, :, :], log_active[:, None, :], log_inactive[:, None, :]).sum(axis=2)

    energy_active = False
    if getattr(cfg.graph, "use_energy_prior", False):
        masses = scene.meta.get("masses") if isinstance(scene.meta, dict) else None
        energy = energy_log_factor(scene, edges, subsets, masses)
        energy = np.asarray(energy, dtype=float)
        if energy.shape == (T, n_subsets) and np.any(energy):
            log_emission = log_emission + energy
            energy_active = True

    dt = _median_dt(t)
    log_trans = _subset_log_transition(n_subsets, dt, cfg.graph.active_set_dwell_time)
    init = np.full(n_subsets, 0.5 / (n_subsets - 1) if n_subsets > 1 else 1.0, dtype=float)
    init[0] = 0.5 if n_subsets > 1 else 1.0
    init /= init.sum()
    log_init = np.log(init)

    gamma, total_loglik = forward_backward(log_emission, log_trans, log_init)
    active_posterior = np.clip(gamma @ active_mask.astype(float), 0.0, 1.0)
    map_path = viterbi(log_emission, log_trans, log_init)
    map_active_set = [[edge_ids[e] for e in subsets[int(k)]] for k in map_path]

    meta = {
        "num_edges": E, "num_subsets": n_subsets, "inference": "exact",
        "joint_loglik": float(total_loglik), "active_set_dwell_time": float(cfg.graph.active_set_dwell_time),
        "energy_prior_active": energy_active, "balance_prior_active": False,
    }
    return GraphDetectionResult(t=t, edges=edge_ids, per_edge=per_edge, active_posterior=active_posterior, map_active_set=map_active_set, meta=meta)


# ======================================================================================
# Validation — score the inferred posterior against a withheld oracle (§9).
# ======================================================================================


def score(result: DetectionResult, truth: GroundTruth) -> dict:
    """Frame-for-frame scoring: contact IoU/F1 over the masks, mode accuracy on truly-in-contact frames."""
    pred = np.asarray(result.in_contact, dtype=bool)
    true = np.asarray(truth.in_contact, dtype=bool)
    intersection = int(np.count_nonzero(pred & true))
    union = int(np.count_nonzero(pred | true))
    contact_iou = 1.0 if union == 0 else intersection / union
    tp = intersection
    fp = int(np.count_nonzero(pred & ~true))
    fn = int(np.count_nonzero(~pred & true))
    denom = 2 * tp + fp + fn
    contact_f1 = 1.0 if denom == 0 else (2.0 * tp) / denom
    map_state = list(result.map_state)
    true_mode = list(truth.mode)
    n_true_contact = int(np.count_nonzero(true))
    if n_true_contact == 0:
        mode_accuracy = float("nan")
    else:
        idx = np.flatnonzero(true)
        matches = sum(1 for i in idx if map_state[i] == true_mode[i])
        mode_accuracy = matches / n_true_contact
    return {
        "contact_iou": float(contact_iou), "contact_f1": float(contact_f1),
        "mode_accuracy": float(mode_accuracy), "contact_frames_true": n_true_contact,
        "contact_frames_pred": int(np.count_nonzero(pred)),
    }


# ======================================================================================
# Executable derivation — the channel densities verify themselves (§3/§4).
#
# Because each Density is an isolated, encapsulated object, its two load-bearing properties can be
# checked WITHOUT running the detector: (a) it is a PROPER density (integrates to 1 — the
# calibration THEORY §4 relies on), and (b) the limit laws stated in prose above hold exactly
# (e.g. the sliding ring collapses to the isotropic Gaussian as speed→0). This turns those
# comments into machine-checked claims. Run:  python contact_detection_standalone.py --check-densities
# ======================================================================================


def _density_selftest(verbose: bool = True) -> None:
    """Assert every channel Density is proper (unit mass) and that the documented limits hold."""

    def mass_1d(d, lo: float, hi: float) -> float:
        val, _ = integrate.quad(lambda x: float(np.exp(d.logpdf(np.array([x]))[0])), lo, hi, limit=400)
        return val

    def mass_2d(d, hi: float) -> float:  # isotropic ⇒ logpdf depends only on r=‖x‖; ∫ p(r)·2πr dr
        val, _ = integrate.quad(
            lambda r: float(np.exp(d.logpdf(np.array([[r, 0.0]]))[0])) * 2.0 * np.pi * r, 0.0, hi, limit=400)
        return val

    masses = [
        ("Normal1D mass", mass_1d(Normal1D(0.0, 0.7), -10.0, 10.0)),
        ("SplitNormalGap mass", mass_1d(SplitNormalGap(0.002, 0.0015, 0.006), -0.1, 0.1)),
        ("OffsetMagnitude1D mass", mass_1d(OffsetMagnitude1D(1.0, 0.3), -5.0, 5.0)),
        ("MixZero1D mass", mass_1d(MixZero1D(0.3, 3.0, 0.25), -40.0, 40.0)),
        ("IsoNormal2D mass", mass_2d(IsoNormal2D(0.5), 8.0)),
        ("OffsetMagnitude2D mass", mass_2d(OffsetMagnitude2D(0.15, 0.1), 3.0)),
        ("MixZero2D mass", mass_2d(MixZero2D(0.3, 3.0, 0.25), 40.0)),
        ("UniformClearance mass", mass_1d(UniformClearance(2.0), 0.0, 2.0)),
    ]

    x1 = np.array([-0.3, 0.0, 0.4, 1.1])
    x2 = np.array([[0.05, -0.2], [0.3, 0.1], [-0.4, 0.25]])
    with np.errstate(divide="ignore"):  # the w→0 / speed→0 degenerate probes hit log(0) by design
        limits = [
            ("ring → isotropic as speed→0",
             float(np.max(np.abs(OffsetMagnitude2D(0.0, 0.5).logpdf(x2) - IsoNormal2D(0.5).logpdf(x2))))),
            ("offset1D → Normal1D as speed→0",
             float(np.max(np.abs(OffsetMagnitude1D(0.0, 0.7).logpdf(x1) - Normal1D(0.0, 0.7).logpdf(x1))))),
            ("split-normal → Normal1D when σ_hi=σ_lo",
             float(np.max(np.abs(SplitNormalGap(0.0, 0.5, 0.5).logpdf(x1) - Normal1D(0.0, 0.5).logpdf(x1))))),
            ("mixture → tight component as w→0",
             float(np.max(np.abs(MixZero1D(0.4, 3.0, 0.0).logpdf(x1) - Normal1D(0.0, 0.4).logpdf(x1))))),
        ]

    bad: list[str] = []
    for name, val in masses:
        ok = abs(val - 1.0) < 3e-3
        if verbose:
            print(f"  [{'ok' if ok else 'FAIL'}] {name:26s} = {val:.6f}   (target 1.000)")
        if not ok:
            bad.append(name)
    for name, err in limits:
        ok = err < 1e-9
        if verbose:
            print(f"  [{'ok' if ok else 'FAIL'}] {name:34s} max|Δ| = {err:.2e}")
        if not ok:
            bad.append(name)
    if bad:
        raise AssertionError("density self-test FAILED: " + ", ".join(bad))
    if verbose:
        print("density self-test: all densities proper and all limit laws hold.")


def _invariant_selftest(verbose: bool = True) -> None:
    """Executable derivation of two structural invariants the whole method rests on.

      (§1) SUPPORT-RELATIVITY. observe() measures motion in the *support's* frame, so a body
           rigidly co-moving with a translating support reads STATIC (relative twist ≈ 0) even as
           it sweeps metres through the world — THE foundational claim (a foot on a moving deck is
           in solid contact). We assert the relative twist is ~0 while the world speed is O(1 m/s);
           a non-support-relative observe() would report the world motion and fail by ~1e9×.

      (§4) ROLLING PROPERNESS. Every mode column must be a proper density for the likelihood ratio
           to stay calibrated. Rolling is the one non-product mode: its coupled (v_t, ω_t) block is
           renormalized by Z_res. We recompute that normalizer by an INDEPENDENT 2-D quadrature (a
           dense grid, vs. the code's nested adaptive quad) and assert they agree — i.e. the coupled
           block integrates to 1.
    """
    bad: list[str] = []

    # (§1) a body co-moving with a translating support reads static.
    hz = 100.0
    t = np.arange(0.0, 2.0, 1.0 / hz)
    T = t.shape[0]
    sup_pos = np.zeros((T, 3))
    sup_pos[:, 0] = 1.0 * t                       # ~2 m of world travel along x
    sup_pos[:, 1] = 0.3 * np.sin(2.0 * t)         # a bob, so the world velocity is non-trivial
    sup_pos[:, 2] = 0.5
    ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))
    support = PoseTrajectory(t=t, position=sup_pos, quat=ident)
    body = PoseTrajectory(t=t, position=sup_pos + np.array([0.1, 0.0, 0.05]), quat=ident.copy())
    surface = SupportSurface(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))
    obs = observe(body, support, surface, np.array([0.0, 0.0, -0.05]))
    world_speed = float(np.median(np.linalg.norm(np.diff(sup_pos, axis=0) * hz, axis=1)))
    rel = max(
        float(np.max(np.abs(obs.v_normal))),
        float(np.max(np.linalg.norm(obs.v_tangent, axis=1))),
        float(np.max(np.abs(obs.omega_normal))),
        float(np.max(np.linalg.norm(obs.omega_tangent, axis=1))),
    )
    ok1 = (rel < 1e-9) and (world_speed > 0.9)
    if verbose:
        print(f"  [{'ok' if ok1 else 'FAIL'}] support-relativity: |relative twist| = {rel:.2e}"
              f"   (world speed ≈ {world_speed:.2f} m/s)")
    if not ok1:
        bad.append("support-relativity")

    # (§4) rolling's coupled block is proper: independent recompute of Z_res.
    sv, sw, rr, rs = 0.50, 3.00, 0.05, 0.03       # EmissionParams defaults for the rolling block
    code_z = float(np.exp(_log_rolling_residual_normalizer(sv, sw, rr, rs)))
    n = 1200
    a = np.linspace(0.0, 8.0 * sv, n)
    b = np.linspace(0.0, 8.0 * sw, n)
    ray_a = a / (sv * sv) * np.exp(-a * a / (2.0 * sv * sv))
    ray_b = b / (sw * sw) * np.exp(-b * b / (2.0 * sw * sw))
    resid = 1.0 / (rs * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((a[:, None] - rr * b[None, :]) / rs) ** 2)
    grid_z = float(np.sum(ray_a[:, None] * ray_b[None, :] * resid) * (a[1] - a[0]) * (b[1] - b[0]))
    err = abs(code_z - grid_z) / max(code_z, 1e-12)
    ok2 = err < 5e-3
    if verbose:
        print(f"  [{'ok' if ok2 else 'FAIL'}] rolling Z_res proper: code={code_z:.6f} indep-grid={grid_z:.6f}"
              f"   rel-err={err:.2e}")
    if not ok2:
        bad.append("rolling-properness")

    if bad:
        raise AssertionError("invariant self-test FAILED: " + ", ".join(bad))
    if verbose:
        print("invariant self-test: support-relativity holds and rolling Z_res is proper.")


# ======================================================================================
# What this file deliberately leaves to prose (the rest of the §8 story).
#
# Everything above is the *validated default pipeline* and reproduces the contact/ package's
# detection bit-for-bit (single pair: 15 scenarios; contact graph: 8 scenes). The package also
# carries, OFF the default path, the frontier of THEORY.md §7–§8 — sketched here so the story is
# complete, but not reproduced (each is a self-contained extra layer, not part of "is this in
# contact, and what kind"):
#
#   • Contact-implicit INVERSE DYNAMICS (§8, the "north star"). The dual of this kinematic
#     detector: recover the per-contact forces + active set that explain the observed motion
#     under Newton–Euler with Signorini complementarity (§2) and the Coulomb cone (§7), as a
#     per-frame convex solve. It is what makes force-mediated contacts (a Newton's-cradle clack:
#     ~0 relative velocity, a sharp force pulse) observable — which kinematics, by theorem,
#     cannot see (§7). Output: a virtual force sensor feeding the optional ``normal_force``
#     channel handled above.
#   • Unsupervised MODE DISCOVERY (§8): a sticky HDP-HMM learns the contact-mode vocabulary from
#     data instead of presupposing the canonical five.
#   • A geometry-fidelity ladder beyond the three resolvers here (convex MESH vs. plane / mesh
#     vs. mesh via GJK/EPA), and a CAPABILITY REGISTRY + value-of-information that picks the
#     richest estimator from whatever the user can declare (shape, force, material…).
#   • A large-graph STRUCTURE filter: above ~4 edges the package swaps the exact 2^E enumeration
#     in ``detect_scene`` for a Rao–Blackwellized particle smoother (an approximation of the
#     exact computation done here; this file always enumerates, which is the reference).
#   • A measurement-uncertainty tempering factor and soft energy/balance graph priors (the
#     energy prior IS reproduced above; balance is off by default).
#
# The thread that ties it all together (THEORY.md §8): one Bayesian estimator with a fixed
# target — the per-frame contact state — fed by whatever evidence is available, each capability
# an optional factor that sharpens the posterior and is a no-op when absent.


# ======================================================================================
# A self-contained demonstration (no physics simulator required).
#
# The package validates against MuJoCo ground truth; "method only," we instead synthesize the
# canonical story analytically — a box free-falls, impacts a floor, rests, and is lifted off —
# feed only the noisy poses through observe()+detect(), and recover the contact story. This is
# the smallest end-to-end run that exercises FREE → IMPACT → STATIC → liftoff → FREE.
# ======================================================================================


def synthetic_drop_rest_liftoff(noise_m: float = 3e-4, seed: int = 0) -> tuple[PoseTrajectory, PoseTrajectory, SupportSurface, np.ndarray, GroundTruth]:
    """Build an analytic drop→rest→liftoff clip: returns (moving, support, surface, cpl, truth).

    A box (half-height h) on a floor at z=0. It free-falls from rest, its bottom face arrests at
    the floor (a velocity step — an impact), rests with a small modeled clearance bias, then is
    lifted off. Only the (noised) box pose is "observable"; the truth labels are withheld for
    scoring. ``contact_point_local`` is the bottom-face centre, ``surface`` the floor plane.
    """
    g, h = 9.81, 0.10
    z0, bias = 0.45, 0.004                 # drop height (m); modeled resting clearance (m)
    t_lift, a_lift = 2.0, 5.0              # liftoff time (s); upward acceleration (m/s²)
    hz = 100.0
    t = np.arange(0.0, 3.0, 1.0 / hz)
    t_impact = float(np.sqrt(2.0 * (z0 - h) / g))   # bottom face (z−h) reaches 0
    z = np.empty_like(t)
    in_contact = np.zeros(t.shape, dtype=bool)
    mode = [FREE] * t.shape[0]
    for i, ti in enumerate(t):
        if ti < t_impact:                  # free fall: centre z = z0 − ½ g t²
            z[i] = z0 - 0.5 * g * ti * ti
        elif ti < t_lift:                  # rest on the floor with the clearance bias
            z[i] = h + bias
            in_contact[i] = True
            mode[i] = STATIC
        else:                              # lift off: rise under a_lift until clear
            z[i] = h + bias + 0.5 * a_lift * (ti - t_lift) ** 2
            if z[i] - h < 0.02:            # still essentially touching for the first instants
                in_contact[i] = True
                mode[i] = STATIC
    # The touchdown frame is an impact (the normal velocity is arrested across it).
    k_impact = int(np.searchsorted(t, t_impact))
    if 0 <= k_impact < len(mode):
        mode[k_impact] = IMPACT

    rng = np.random.default_rng(seed)
    position = np.zeros((t.shape[0], 3))
    position[:, 2] = z
    position = position + rng.normal(0.0, noise_m, size=position.shape)   # emulate mocap noise
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (t.shape[0], 1))       # no rotation
    moving = PoseTrajectory(t=t, position=position, quat=quat)

    ident = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (t.shape[0], 1))
    support = PoseTrajectory(t=t, position=np.zeros((t.shape[0], 3)), quat=ident)  # static world floor
    surface = SupportSurface(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))
    contact_point_local = np.array([0.0, 0.0, -h])                        # bottom-face centre
    truth = GroundTruth(t=t, in_contact=in_contact, mode=mode,
                        normal_force=np.where(in_contact, 9.81, 0.0), penetration=np.zeros(t.shape[0]))
    return moving, support, surface, contact_point_local, truth


def _ascii_timeline(mask: np.ndarray, width: int = 64) -> str:
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    if n == 0:
        return ""
    w = min(width, n)
    cell = (np.arange(n) * w) // n
    cells = np.zeros(w, dtype=bool)
    np.logical_or.at(cells, cell, mask)
    return "".join("#" if c else "." for c in cells)


def print_story(name: str, obs: ContactObservations, result: DetectionResult, truth: GroundTruth | None = None) -> None:
    """Render one detection as a readable terminal summary (intervals, events, impacts, scores)."""
    line = "=" * 72
    print(line)
    print(f"SCENARIO: {name}")
    print(line)
    print("Detected contact intervals (start, end, mode):")
    for iv in result.intervals:
        print(f"  [{iv.t_start:7.3f}, {iv.t_end:7.3f}] s   {iv.mode}")
    if not result.intervals:
        print("  (none)")
    print("Make/break events:")
    for ev in result.events:
        print(f"  {ev.kind:10s} t={ev.time:7.3f} s  (frame {ev.index})")
    if not result.events:
        print("  (none)")
    print(f"Impact atoms ({len(result.impulses)}):")
    for imp in result.impulses:
        e = "  e=  n/a" if np.isnan(imp.restitution) else f"  e={imp.restitution:5.3f}"
        print(f"  t={imp.time:7.3f} s  closing={imp.closing_speed:5.3f} m/s{e}")
    print(f"Recovered resting bias: {result.resting_bias * 1e3:+.2f} mm")
    if truth is not None:
        sc = score(result, truth)
        print(f"Scores vs truth: contact IoU={sc['contact_iou']:.3f}  F1={sc['contact_f1']:.3f}  "
              f"mode-acc={sc['mode_accuracy']:.3f}")
        print(f"Timeline ('#'=contact):  pred {_ascii_timeline(result.in_contact)}")
        print(f"                         true {_ascii_timeline(np.asarray(truth.in_contact, dtype=bool))}")
    line2 = "-" * 72
    print(line2)
    # A few posterior samples so the calibrated P(contact) is visible, not just the hard decision.
    post = np.asarray(result.contact_posterior, dtype=float)
    t = np.asarray(result.t, dtype=float)
    for frac in (0.05, 0.20, 0.40, 0.60, 0.90):
        i = int(frac * (len(t) - 1))
        print(f"  t={t[i]:6.3f} s   P(contact)={post[i]:.3f}   gap={float(obs.gap[i])*1e3:+7.2f} mm   MAP={result.map_state[i]}")
    print(line)


def _demo() -> None:
    """Run the analytic drop→rest→liftoff clip end to end and print the recovered story."""
    moving, support, surface, cpl, truth = synthetic_drop_rest_liftoff()
    obs = observe(moving, support, surface, cpl)
    result = ContactDetector().detect(obs)
    print_story("synthetic drop → rest → liftoff", obs, result, truth)
    print("\nThe detector saw only noisy box positions. From them it recovered, support-relative,")
    print("the gap and twist (§1/§3), scored each frame as a calibrated likelihood ratio (§4),")
    print("smoothed it through a gap-gated semi-Markov HMM (§5), timed the touchdown impact (§6),")
    print("and self-calibrated the resting clearance by EM (§7) — the whole method, in one file.")
    # (Output-equivalence to the contact/ package is checked by the separate verify_standalone.py;
    # this file imports nothing from that package, so it stands entirely on its own.)


if __name__ == "__main__":
    import sys

    if "--check-densities" in sys.argv:
        _density_selftest()
    elif "--check-invariants" in sys.argv:
        _invariant_selftest()
    elif "--check" in sys.argv:
        _density_selftest()
        _invariant_selftest()
    else:
        _demo()

