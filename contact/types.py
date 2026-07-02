"""Core data contracts shared across the whole package.

These dataclasses are the *interfaces between modules*. Every module imports its
types from here and nothing else cross-module (except the leaf helpers in
`signals` and `geometry`). The field names and shapes below are authoritative —
implementers must conform to them exactly.

Frame conventions
-----------------
* World quantities are in a fixed inertial frame.
* A *contact frame* is attached to the support surface: its z-axis is the surface
  outward normal, and x/y span the tangent plane. All `ContactObservations` live
  in this (possibly moving, possibly non-inertial) contact frame — see THEORY.md
  §1 & §3 for why contact must be measured support-relative.
* Quaternions are scalar-first (w, x, y, z), unit norm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

# --------------------------------------------------------------------------------------
# Mode vocabulary (THEORY.md §3: modes are twist-subspaces of the relative motion)
# --------------------------------------------------------------------------------------

FREE = "free"
STATIC = "static"
SLIDING = "sliding"
PIVOTING = "pivoting"
ROLLING = "rolling"
IMPACT = "impact"

#: Every contact mode (i.e. all states except FREE).
CONTACT_MODES: list[str] = [STATIC, SLIDING, PIVOTING, ROLLING, IMPACT]

#: Canonical ordering of all latent states. Index 0 is always FREE. Posteriors and
#: emission matrices are columned in this order unless a subset is explicitly passed.
ALL_STATES: list[str] = [FREE] + CONTACT_MODES


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------

@dataclass
class PoseTrajectory:
    """Time-stamped pose of one rigid body in the world frame.

    t:        (T,) seconds, strictly increasing.
    position: (T, 3) world position of the body origin (m).
    quat:     (T, 4) world orientation, scalar-first unit quaternions (w, x, y, z).
    """

    t: np.ndarray
    position: np.ndarray
    quat: np.ndarray


@dataclass
class SupportSurface:
    """A planar support surface, expressed in the support body's *local* frame.

    For a static floor the support body's pose is identity for all t, so local and
    world coincide. For a moving support (e.g. a skateboard deck) the plane travels
    and rotates with the body.

    point:  (3,) a point on the plane, in the support body's local frame.
    normal: (3,) outward unit normal, in the support body's local frame.
    """

    point: np.ndarray
    normal: np.ndarray


@dataclass
class ContactObservations:
    """Per-frame, support-relative observations for ONE candidate body-pair contact.

    Everything is expressed in the support's instantaneous contact frame (z = outward
    normal). These are the only quantities the detector consumes.

    t:             (T,) seconds.
    gap:           (T,) signed distance to the surface (m); >0 separation, <0 penetration.
    v_normal:      (T,) relative normal velocity (m/s); +ve = separating.
    v_tangent:     (T, 2) relative tangential velocity in the tangent plane (m/s).
    omega_normal:  (T,) relative angular velocity about the normal (rad/s) — spin/pivot.
    omega_tangent: (T, 2) relative angular velocity in the tangent plane (rad/s) — rolling axis.
    """

    t: np.ndarray
    gap: np.ndarray
    v_normal: np.ndarray
    v_tangent: np.ndarray
    omega_normal: np.ndarray
    omega_tangent: np.ndarray
    meas_cov: np.ndarray | None = None  # optional per-frame measurement variance of the contact
    #: point (T,) or (T,3,3); when present and enabled, scales emission noise so noisy/occluded
    #: frames contribute less (THEORY.md §8). None => homogeneous noise.
    normal_force: np.ndarray | None = None  # optional (T,) measured normal contact force (N);
    #: DESIGN.md PART II.A / PHASE 4a. The MEASURED-force observation channel, mirroring the
    #: optional `meas_cov`. When present, a per-state force emission term (FREE: half-normal at 0;
    #: sustained contact: Rayleigh; IMPACT: a larger-scale Rayleigh spike) multiplies into each
    #: emission builder, gated on this field. None => the force channel is absent (no factor; the
    #: byte-identical kinematics-only behaviour). Inferred force (from dynamics) is a later phase.


# --------------------------------------------------------------------------------------
# The narrow waist (DESIGN.md PART III, III.1/III.2): a per-frame, world-frame contact
# description that `geometry.observe` consumes. A `ContactGeometry` resolver turns the two
# bodies' pose streams into one `ContactFrame` per recorded frame; `observe` asks for
# `(point, normal, gap)` and then runs the twist decomposition unchanged. Swapping the
# resolver (flat plane -> sphere -> mesh) leaves everything downstream untouched. The
# default resolver, `contact.geometry_resolvers.FlatRegion`, wraps the legacy
# `(surface, contact_point_local)` spec and is bit-identical to the pre-refactor pipeline.
# --------------------------------------------------------------------------------------

@dataclass
class ContactPoint:
    """One world-frame contact between a moving body and its support (DESIGN.md III.1).

    point:        (3,) world contact point (m).
    normal:       (3,) world outward UNIT normal (support -> moving).
    gap:          signed distance along `normal` (m); >0 separation, <0 penetration.
    normal_sigma: provenance/fidelity uncertainty of the normal (rad); 0.0 => exact. A crude
                  resolver declares it large, an exact one small (DESIGN.md III.5); it feeds
                  the measurement-tempering path. 0.0 (the Phase-0 value) is a no-op.
    gap_sigma:    provenance/fidelity uncertainty of the gap (m); 0.0 => exact (no-op).
    """

    point: np.ndarray
    normal: np.ndarray
    gap: float
    normal_sigma: float = 0.0
    gap_sigma: float = 0.0


#: One `ContactFrame` per recorded frame; a list of >1 points represents an area/face
#: contact (DESIGN.md III.1/III.4). `FlatRegion` always produces a single point per frame.
ContactFrame = list[ContactPoint]


class ContactGeometry(Protocol):
    """The narrow waist: a per-frame, world-frame contact-geometry resolver (DESIGN.md III.1).

    A resolver maps the pose streams of a moving body and its (possibly moving) support to
    one `ContactFrame` per recorded frame, in the world frame. `geometry.observe` consumes
    only this, so any resolver on the fidelity ladder (flat plane -> primitive -> mesh) plugs
    in without touching emissions, the HMM, the active-set, or inverse dynamics.
    """

    def resolve(
        self, moving: PoseTrajectory, support: PoseTrajectory
    ) -> list[ContactFrame]:
        """Return one `ContactFrame` per recorded frame (length T)."""
        ...


# --------------------------------------------------------------------------------------
# Ground truth (from the simulator; used only for validation, never by the detector)
# --------------------------------------------------------------------------------------

@dataclass
class GroundTruth:
    """Per-frame ground-truth labels extracted from the physics simulator.

    t:            (T,) seconds.
    in_contact:   (T,) bool.
    mode:         length-T list of mode strings (FREE on frames not in contact).
    normal_force: (T,) normal contact force magnitude (N).
    penetration:  (T,) penetration depth (m, >= 0).
    """

    t: np.ndarray
    in_contact: np.ndarray
    mode: list[str]
    normal_force: np.ndarray
    penetration: np.ndarray


@dataclass
class RawScenario:
    """A complete labeled scenario as produced by the MuJoCo harness.

    The detector never sees this directly; `geometry.observe(...)` turns the raw
    poses + surface into `ContactObservations`, exercising the same support-relative
    path that real captured data would.

    name:               human-readable scenario id.
    moving:             PoseTrajectory of the body whose contact we test (foot/box/ball).
    support:            PoseTrajectory of the support body (identity poses for a static floor).
    surface:            SupportSurface in the support's local frame.
    contact_point_local:(3,) the tracked material point on the moving body, in its local frame.
    truth:              GroundTruth labels.
    meta:               free-form metadata (material params, notes).
    geometry:           optional ContactGeometry resolver (DESIGN.md III.1). When None (the
                        default), `observe` falls back to a FlatRegion wrapping `surface` +
                        `contact_point_local`, i.e. today's bit-identical behaviour. A scenario
                        may attach a higher-fidelity resolver (e.g. BoxPlane on the tumbling box,
                        DESIGN.md PHASE 2) which `observe` then uses instead.
    """

    name: str
    moving: PoseTrajectory
    support: PoseTrajectory
    surface: SupportSurface
    contact_point_local: np.ndarray
    truth: GroundTruth
    meta: dict = field(default_factory=dict)
    geometry: ContactGeometry | None = None


# --------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------

@dataclass
class ContactEvent:
    """A make/break event (THEORY.md §6)."""

    kind: str  # "touchdown" | "liftoff"
    time: float
    index: int


@dataclass
class ContactImpulse:
    """An impulsive contact event — an atom in the force measure (THEORY.md §6).

    time:           event time (s).
    index:          nearest frame index.
    closing_speed:  relative normal closing speed just before impact (m/s).
    restitution:    measured e = -v_after / v_before (NaN if unmeasured).
    normal_impulse: integral of normal force over the impact = m * delta-v_normal (N*s);
                    NaN when the moving body's mass is unknown.
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
class ContactEdge:
    """One candidate contact in a multi-body scene (THEORY.md §8).

    A contact is between a pair of bodies; the surface is carried by the support body
    and the tracked material point by the moving body. The single-pair detector runs
    per edge (in the support's frame), and the graph layer fuses the edges into a joint
    active-set posterior.

    edge_id:             unique label.
    moving_body:         key into MultiBodyScene.bodies (the body whose contact we test).
    support_body:        key into MultiBodyScene.bodies (carries the surface; may itself move).
    surface:             SupportSurface in the support body's local frame.
    contact_point_local: (3,) tracked material point on the moving body, local frame.
    geometry:            optional ContactGeometry resolver (DESIGN.md III.1). When None
                         (the default), `observe` falls back to a FlatRegion wrapping
                         `surface` + `contact_point_local`, i.e. today's behaviour.
    """

    edge_id: str
    moving_body: str
    support_body: str
    surface: SupportSurface
    contact_point_local: np.ndarray
    geometry: ContactGeometry | None = None


@dataclass
class MultiBodyScene:
    """A scene of several bodies and the candidate contacts among them (THEORY.md §8).

    name:   scene id.
    bodies: dict body-name -> PoseTrajectory (all sharing the same time base).
    edges:  candidate ContactEdges (after broad-phase, or all plausible pairs).
    truth:  dict edge_id -> GroundTruth (per-edge labels from the simulator).
    meta:   free-form (masses, support-polygon corners, notes).
    """

    name: str
    bodies: dict[str, PoseTrajectory]
    edges: list[ContactEdge]
    truth: dict[str, GroundTruth]
    meta: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Everything the detector returns.

    t:                 (T,) seconds.
    contact_posterior: (T,) P(in contact) = 1 - P(free).
    state_posterior:   (T, S) posterior over `states`.
    map_state:         length-T list of MAP mode labels (Viterbi).
    in_contact:        (T,) bool, derived from map_state != FREE.
    intervals:         list[ContactInterval].
    events:            list[ContactEvent].
    resting_bias:      estimated constant gap offset (m), from EM calibration.
    normal_force:      (T,) estimated normal force (N) if material stiffness known, else None.
    states:            the state ordering used for `state_posterior` columns.
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
    impulses: list[ContactImpulse] = field(default_factory=list)  # force-as-measure atoms (§6)
    slip_state: list[str] | None = None  # per-frame "stick"/"slip"/"" (§7), or None if not computed


@dataclass
class DiscoveredModeResult:
    """Output of unsupervised contact-mode discovery (THEORY.md §8: HDP-HMM).

    labels:          (T,) integer id of the discovered mode active at each frame.
    n_modes:         number of distinct modes the model actually used.
    signatures:      dict mode_id -> mean twist signature (a small feature vector:
                     [gap, |v_normal|, |v_tangent|, |omega_normal|, |omega_tangent|]).
    alignment:       dict mode_id -> best-matching canonical mode name (FREE/STATIC/...),
                     for validation only; the discovery itself is label-free.
    """

    labels: np.ndarray
    n_modes: int
    signatures: dict[int, np.ndarray]
    alignment: dict[int, str]


@dataclass
class InverseDynamicsResult:
    """Output of contact-implicit inverse dynamics (THEORY.md §8, the north star).

    The contact forces that explain the observed motion under Newton-Euler dynamics with
    complementarity + friction, plus the active set they imply.

    t:                   (T,) seconds.
    contact_normal_force:(T, K) recovered normal force per candidate contact point (N, >= 0).
    contact_tangent_force:(T, K, 2) recovered friction force per candidate, in the tangent plane (N).
    active_set:          length-T list of the active candidate indices (normal force above threshold).
    total_normal_force:  (T,) summed normal force (at rest this should equal m*g).
    wrench_residual:     (T,) norm of the unexplained net wrench (how well the forces explain the motion).
    candidate_points:    (K, 3) the candidate contact points used, in body-local coordinates.
    """

    t: np.ndarray
    contact_normal_force: np.ndarray
    contact_tangent_force: np.ndarray
    active_set: list[list[int]]
    total_normal_force: np.ndarray
    wrench_residual: np.ndarray
    candidate_points: np.ndarray


@dataclass
class GraphDetectionResult:
    """Joint contact-state estimate over a multi-body scene (THEORY.md §8).

    t:               (T,) seconds.
    edges:           ordered list of edge ids (columns of active_posterior).
    per_edge:        dict edge_id -> the single-pair DetectionResult for that edge.
    active_posterior:(T, E) marginal P(edge active) after the joint structure inference.
    map_active_set:  length-T list of the MAP set of active edge ids per frame.
    meta:            diagnostics (energy-budget residual, balance-prior margins, etc.).
    """

    t: np.ndarray
    edges: list[str]
    per_edge: dict[str, "DetectionResult"]
    active_posterior: np.ndarray
    map_active_set: list[list[str]]
    meta: dict = field(default_factory=dict)
