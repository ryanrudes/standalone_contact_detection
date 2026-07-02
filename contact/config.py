"""Physical and statistical parameters for the detector.

These are deliberately *physically interpretable* (real noise scales, real material
stiffness) rather than tuned to any simulator — see THEORY.md §9 on keeping
emission/material models transferable. Defaults are conservative values for typical
optical mocap (~100-250 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmissionParams:
    """Parameters of the per-state emission likelihoods (THEORY.md §4 & §3).

    The contact states are sharp peaks on a twist subspace; FREE is diffuse. Each
    scale is the standard deviation of a (log-)Gaussian unless noted.
    """

    # --- gap channel (m): two-piece Gaussian about the contact mean (the resting bias) ---
    gap_sigma_gap: float = 0.0015   # tight tolerance ABOVE the surface (a real gap => free)
    gap_sigma_pen: float = 0.0060   # looser tolerance BELOW (squish / plane-fit error)
    gap_free_range: float = 1.0     # diffuse FREE clearance prior: ~uniform over this range (m)

    # --- translational velocity (m/s) ---
    vel_sigma: float = 0.05         # contact: relative velocity noise at rest
    slide_speed: float = 0.15       # sliding: characteristic tangential speed scale
    slide_width_frac: float = 0.7   # sliding ring WIDTH as a fraction of slide_speed: the ring
                                    # must be broad (a sliding body sweeps a range of speeds),
                                    # not a razor peak at one speed -- else fast/variable
                                    # sliding wrongly reads FREE (width = frac * slide_speed)
    free_vel_sigma: float = 0.50    # FREE: broad velocity prior

    # --- angular velocity (rad/s) ---
    omega_sigma: float = 0.30       # contact: angular rate noise at rest
    slide_omega_broad_weight: float = 0.25  # sliding's rotation is a tight+broad MIXTURE; this
                                    # is the broad component's weight -- it bounds the penalty
                                    # when a sliding body is also spinning (a ball spinning up),
                                    # while the tight component still rewards a non-spinning slide
    pivot_speed: float = 1.00       # pivoting: characteristic spin rate about the normal
    free_omega_sigma: float = 3.00  # FREE: broad angular prior

    # --- rolling coupling (THEORY.md §3): |v_tangent| ~ roll_radius * |omega_tangent| ---
    roll_radius: float = 0.05       # effective rolling radius (m)
    roll_sigma: float = 0.03        # tolerance on the rolling constraint residual (m/s)

    # --- impact transient (THEORY.md §6) ---
    impact_speed: float = 0.30      # characteristic relative normal closing speed at impact


@dataclass
class ForceEmissionParams:
    """Per-state NORMAL-force emission scales (DESIGN.md PART II.A; PHASE 4a).

    The MEASURED-force channel (`ContactObservations.normal_force`, gated) contributes one
    proper density on `[0, inf)` per state, added as a single `lp = lp + ...` term inside each
    emission builder so cross-state log-ratios stay calibrated (the module invariant in
    `emissions.py`):

      * FREE                                -> half-normal `HN(sigma_free)` (mode at 0: a
                                               separated body carries no load),
      * STATIC / SLIDING / PIVOTING / ROLLING -> a MIXTURE
                                               `w_unloaded*HN(sigma_free) + (1-w_unloaded)*R(s_load)`
                                               -- a body in contact may be UNLOADED (a resting
                                               touch carrying ~0 force) OR loaded, so the density
                                               must allow BOTH. A pure Rayleigh (zero at f=0)
                                               wrongly pulls a touching-but-unloaded body to FREE
                                               (it sank the cradle's resting contact 452->16); the
                                               unloaded component makes `f~0` nearly neutral
                                               (consistent with free OR resting contact -> the GAP
                                               decides), while appreciable force still pulls to
                                               contact via the loaded Rayleigh component,
      * IMPACT                              -> Rayleigh `R(s_impact)`, `s_impact >> s_load` (a
                                               brief, large force spike; an impact is never
                                               unloaded, so zero-at-0 is correct here).

    The observed force is normalized ONCE (in `emissions._force_log_density`) by its own robust
    positive scale `s = median(force[force > 0])` (fallback 1.0), so `fn = force / s` is
    dimensionless and these scales live in NORMALIZED units (DESIGN.md PART II.A calibration
    rung 2): the normalized median load is ~1 by construction (`s_load = 1`) and an impact spike
    is a few times larger (`s_impact = 4`). The whole term is gated on `obs.normal_force is not
    None`, so with no force channel there is no factor and behaviour is unchanged.
    """

    sigma_free: float = 0.15   # half-normal width for FREE (normalized units; ~no load at f=0)
    s_load: float = 1.0        # Rayleigh scale (mode) for LOADED contact; normalized load ~1
    s_impact: float = 4.0      # Rayleigh scale for IMPACT; cradle clacks are ~4x the median load
    w_unloaded: float = 0.5    # weight of the UNLOADED (free-like) component of the contact-force
                               # mixture: makes f~0 ~neutral (a resting touch carries no load), so
                               # force never overrides the gap for an unloaded contact


@dataclass
class TransitionParams:
    """Temporal prior for the HMM/HSMM (THEORY.md §5).

    Baseline is a continuous-time Markov jump discretized per frame:
    P(stay over dt) = exp(-dt/dwell). The full model upgrades this two ways:
      * STATE-DEPENDENT transitions: free->contact entry is gated by gap proximity
        (the guard of the hybrid system), via a logistic on `gap_gate`/`gap_gate_softness`.
      * SEMI-MARKOV (explicit-duration) decoding: dwell times are not memoryless but
        drawn from a duration distribution with hazard sharpness `dwell_concentration`.
    """

    mean_dwell_time: float = 0.20    # s, baseline expected dwell before switching
    impact_dwell_time: float = 0.04  # s, IMPACT is a short transient bridging free<->contact
    gap_gate: float = 0.008          # m, gap within which free->contact entry is enabled (the §5 guard)
    gap_gate_softness: float = 0.004 # m, logistic softness of the gap gate
    use_semi_markov: bool = True     # explicit-duration (HSMM) decoding vs plain Markov
    dwell_concentration: float = 4.0 # duration-distribution shape; higher => sharper/more deterministic dwell


@dataclass
class ImpactParams:
    """Impact detection and characterization (THEORY.md §6).

    Impacts are singular events (velocity steps / force atoms). They are detected by a
    matched filter on a LIGHTLY-smoothed normal velocity (over-smoothing destroys their
    timing), characterized by their impulse (m * delta-v) and restitution (-v_after/v_before).
    """

    template_halfwidth_time: float = 0.03  # s, half-width of the velocity-step matched-filter template
    min_closing_speed: float = 0.06        # m/s, minimum relative normal closing speed to call an impact
                                           # (low enough to catch the small late bounces of a
                                           # nearly-settled object; still rejects rolling/slide noise)
    detect_smooth_time: float = 0.01       # s, light smoothing for impact detection (preserve sharpness)
    restitution_default: float = 0.0       # prior restitution when it cannot be measured


@dataclass
class MaterialParams:
    """Contact material properties (THEORY.md §7).

    When `stiffness` is known, penetration depth becomes a calibrated force gauge
    (lambda = stiffness * penetration), which is what makes contact force observable.
    Leave `stiffness=None` to run purely kinematically (force not estimated).
    """

    stiffness: float | None = None    # N/m
    damping: float = 0.0              # N/(m/s)
    friction: float = 0.6             # Coulomb coefficient
    slip_speed_threshold: float = 0.02  # m/s, tangential speed above which a contact is deemed sliding (stick/slip, §7)


@dataclass
class CalibrationParams:
    """EM self-calibration of the resting-gap bias (THEORY.md §7 & §8)."""

    max_resting_bias: float = 0.01  # clip the estimated gap offset to +/- this (m)
    em_iters: int = 8


@dataclass
class GraphParams:
    """Multi-body contact-graph / active-set inference (THEORY.md §8)."""

    proximity_gap: float = 0.05        # m, broad-phase: propose an edge only within this gap
    active_set_dwell_time: float = 0.20  # s, temporal prior on the active-set sequence
    use_energy_prior: bool = True      # soft global energy/dissipation consistency factor
    use_balance_prior: bool = False    # soft CoM-over-support-polygon factor (needs masses/geometry)


@dataclass
class InferenceParams:
    """Research-frontier inference knobs (THEORY.md §8 & §10).

    * Structure posterior: exact 2^E enumeration up to `enumerate_max_edges`, a
      Rao-Blackwellized particle filter over the active-set sequence beyond that.
    * Mode discovery: a sticky HDP-HMM learns the contact-mode vocabulary from data
      instead of presupposing it (truncated at `max_modes`).
    * Uncertainty: propagate per-frame measurement covariance into the emissions so
      noisy/occluded frames contribute less.
    """

    enumerate_max_edges: int = 4     # exact subset enumeration up to this E; particle filter beyond
    n_particles: int = 256           # particles for the structure filter on large graphs
    max_modes: int = 8               # HDP-HMM truncation level
    hdp_concentration: float = 1.0   # DP concentration (how readily new modes appear)
    hdp_stickiness: float = 10.0     # sticky self-transition bias (mode persistence)
    use_uncertainty: bool = False    # propagate ContactObservations.meas_cov into emissions


@dataclass
class InverseDynamicsParams:
    """Contact-implicit inverse dynamics — the THEORY.md §8 "north star".

    Explain the OBSERVED motion of a body of known mass/inertia with physically valid
    contact forces, subject to the Signorini complementarity (force only where the gap
    is closed, force >= 0) and the Coulomb friction cone. The active set and forces fall
    out as the unique-up-to-indeterminacy explanation of the kinematics under Newton-Euler.
    """

    gravity: float = 9.81             # m/s^2, along -z of the world
    accel_smooth_time: float = 0.04   # s, smoothing before the double-differentiation to accel
    active_force_threshold: float = 1.0  # N, minimum normal force to declare a candidate active
    force_regularization: float = 1e-6   # Tikhonov term to pick a minimum-norm force among indeterminate solutions
    complementarity_gap: float = 0.005   # m, a candidate may carry force only when |gap| < this (Signorini)


@dataclass
class DetectorConfig:
    """Top-level configuration bundle."""

    emission: EmissionParams = field(default_factory=EmissionParams)
    force: ForceEmissionParams = field(default_factory=ForceEmissionParams)
    transition: TransitionParams = field(default_factory=TransitionParams)
    material: MaterialParams = field(default_factory=MaterialParams)
    calibration: CalibrationParams = field(default_factory=CalibrationParams)
    impact: ImpactParams = field(default_factory=ImpactParams)
    graph: GraphParams = field(default_factory=GraphParams)
    inference: InferenceParams = field(default_factory=InferenceParams)
    inverse_dynamics: InverseDynamicsParams = field(default_factory=InverseDynamicsParams)
    vel_smooth_time: float = 0.05   # Gaussian smoothing time before differentiation (s)
