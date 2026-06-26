"""Contact detection from first principles.

A probabilistic, support-relative contact-state estimator. See THEORY.md for the
full derivation. This top-level module re-exports the stable data contracts; the
detector and generators are imported from their submodules.
"""

from __future__ import annotations

from .config import (
    CalibrationParams,
    DetectorConfig,
    EmissionParams,
    GraphParams,
    ImpactParams,
    InferenceParams,
    InverseDynamicsParams,
    MaterialParams,
    TransitionParams,
)
from .types import (
    ALL_STATES,
    CONTACT_MODES,
    FREE,
    IMPACT,
    PIVOTING,
    ROLLING,
    SLIDING,
    STATIC,
    ContactEdge,
    ContactEvent,
    ContactImpulse,
    ContactInterval,
    ContactObservations,
    DetectionResult,
    DiscoveredModeResult,
    GraphDetectionResult,
    GroundTruth,
    InverseDynamicsResult,
    MultiBodyScene,
    PoseTrajectory,
    RawScenario,
    SupportSurface,
)

# The assembled detector and the relative-frame core (pure numpy/scipy).
from .geometry import observe
from .model import ContactDetector

# The s.5-s.7 leaf entrypoints, re-exported so users can call the individual rungs of
# the ladder (THEORY.md s.10) without reaching into submodules:
#   * detect_impacts                  -- the matched-filter impact-atom detector (s.6).
#   * friction_stick_slip             -- the Coulomb-cone stick/slip labeller (s.7).
#   * normal_force_from_penetration   -- penetration as a calibrated force gauge (s.7).
#   * observability_demo              -- the indeterminate-rig observability theorem (s.7).
#   * base_transition_matrix /        -- the temporal-prior builders (s.5): homogeneous
#     gated_transition_tensor            base matrix and the gap-gated per-frame tensor.
from .dynamics import (
    friction_stick_slip,
    normal_force_from_penetration,
    observability_demo,
)

# Contact-implicit inverse dynamics (THEORY.md s.8, the north star; the final rung beyond
# s.10's ladder). The dual of the kinematic detector: jointly recover contact existence,
# the active set, and the per-candidate force as the physically-valid Newton-Euler
# explanation of the observed motion under Signorini complementarity (s.2) and the Coulomb
# friction cone (s.7). All OFF the default detection path -- a separate analysis layer.
#   * body_accelerations       -- CoM linear accel + angular accel/velocity from a pose.
#   * required_wrench          -- the net external wrench Newton-Euler demands (s.8).
#   * contact_wrench_map       -- the per-candidate force -> net-wrench grasp map G(t).
#   * solve_contact_implicit   -- the per-frame Signorini+cone constrained force solve.
#   * contact_implicit_from_raw-- the end-to-end pipeline from a labeled RawScenario.
from .dynamics_id import (
    body_accelerations,
    contact_implicit_from_raw,
    contact_wrench_map,
    required_wrench,
    solve_contact_implicit,
)
from .impacts import detect_impacts
from .transitions import base_transition_matrix, gated_transition_tensor

# The multi-body contact-graph detector (THEORY.md s.8, rung 5): the single-pair detector
# lifted to a whole contact graph, inferring the joint active-set posterior over the 2^E
# structures. `build_candidate_edges` is the proximity broad-phase; `detect_scene` is the
# full pipeline.
from .graph import build_candidate_edges, detect_scene

# Research-frontier entrypoints (THEORY.md s.8 & s.10), all OFF the default detection path:
#   * discover_modes                 -- unsupervised contact-mode discovery via a sticky
#     HDP-HMM (learn the mode vocabulary from data instead of presupposing it).
#   * exact_active_sets /            -- the active-set structure posterior: exact 2^E
#     particle_filter_active_sets       enumeration (reference) and a Rao-Blackwellized
#                                       particle smoother that scales past enumeration.
#   * emission_tempering             -- per-frame measurement-uncertainty tempering of the
#                                       emissions (noisy/occluded frames contribute less).
from .mode_discovery import discover_modes
from .structure_inference import exact_active_sets, particle_filter_active_sets
from .uncertainty import emission_tempering

# Synced real-time side-by-side animations (scene + signals + detections).
from .visualize import animate_scene, animate_scenario

# The MuJoCo truth factory (THEORY.md s.9). Imported lazily-friendly but eager here:
# `mujoco` is a declared dependency, and the top-level convenience API exposes the
# generator alongside the detector. `generate`/`SCENARIOS` are the single-pair scenarios;
# `generate_scene`/`SCENES` are the multi-body contact-graph scenes (s.8).
from .mujoco_gen import SCENARIOS, SCENES, generate, generate_scene

__all__ = [
    "ALL_STATES",
    "CONTACT_MODES",
    "FREE",
    "STATIC",
    "SLIDING",
    "PIVOTING",
    "ROLLING",
    "IMPACT",
    "ContactEvent",
    "ContactImpulse",
    "ContactInterval",
    "ContactObservations",
    "DetectionResult",
    "GroundTruth",
    "PoseTrajectory",
    "RawScenario",
    "SupportSurface",
    "DetectorConfig",
    "EmissionParams",
    "TransitionParams",
    "MaterialParams",
    "CalibrationParams",
    "ImpactParams",
    "GraphParams",
    "InferenceParams",
    "InverseDynamicsParams",
    "ContactEdge",
    "MultiBodyScene",
    "GraphDetectionResult",
    "DiscoveredModeResult",
    "InverseDynamicsResult",
    "ContactDetector",
    "observe",
    "generate",
    "SCENARIOS",
    # multi-body contact-graph layer (THEORY.md s.8, rung 5)
    "build_candidate_edges",
    "detect_scene",
    "generate_scene",
    "SCENES",
    "animate_scenario",
    "animate_scene",
    # s.5-s.7 leaf entrypoints
    "detect_impacts",
    "friction_stick_slip",
    "normal_force_from_penetration",
    "observability_demo",
    "base_transition_matrix",
    "gated_transition_tensor",
    # contact-implicit inverse dynamics (THEORY.md s.8, the north star), off by default
    "body_accelerations",
    "required_wrench",
    "contact_wrench_map",
    "solve_contact_implicit",
    "contact_implicit_from_raw",
    # research-frontier layer (THEORY.md s.8 & s.10), off by default
    "discover_modes",
    "exact_active_sets",
    "particle_filter_active_sets",
    "emission_tempering",
]
