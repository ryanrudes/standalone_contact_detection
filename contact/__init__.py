"""Contact detection from first principles — the estimator.

A probabilistic, support-relative contact-state estimator. See THEORY.md for the full
derivation. This package is the method only: it consumes noisy poses and never sees
ground truth. Truth generation, scoring, and visualization live in the sibling
`oracle` package (`oracle` imports `contact`; never the reverse — the import law that
makes THEORY §9's "truth is withheld from the detector" structural).
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
from .detector import ContactDetector

# The §5-§7 leaf entrypoints, re-exported so users can call the individual rungs of
# the ladder (THEORY.md §10) without reaching into submodules:
#   * detect_impacts                  -- the matched-filter impact-atom detector (§6).
#   * friction_stick_slip             -- the Coulomb-cone stick/slip labeller (§7).
#   * normal_force_from_penetration   -- penetration as a calibrated force gauge (§7).
#   * observability_demo              -- the indeterminate-rig observability theorem (§7).
#   * base_transition_matrix /        -- the temporal-prior builders (§5): homogeneous
#     gated_transition_tensor            base matrix and the gap-gated per-frame tensor.
from .dynamics import (
    friction_stick_slip,
    normal_force_from_penetration,
    observability_demo,
)

# Contact-implicit inverse dynamics (THEORY.md §8, the north star; the final rung beyond
# §10's ladder). The dual of the kinematic detector: jointly recover contact existence,
# the active set, and the per-candidate force as the physically-valid Newton-Euler
# explanation of the observed motion under Signorini complementarity (§2) and the Coulomb
# friction cone (§7). All OFF the default detection path -- a separate analysis layer.
#   * body_accelerations       -- CoM linear accel + angular accel/velocity from a pose.
#   * required_wrench          -- the net external wrench Newton-Euler demands (§8).
#   * contact_wrench_map       -- the per-candidate force -> net-wrench grasp map G(t).
#   * solve_contact_implicit   -- the per-frame Signorini+cone constrained force solve.
#   * contact_implicit_from_raw-- the end-to-end pipeline from a labeled RawScenario.
from .inverse_dynamics import (
    body_accelerations,
    contact_implicit_from_raw,
    contact_wrench_map,
    required_wrench,
    solve_contact_implicit,
)
from .impacts import detect_impacts
from .transitions import base_transition_matrix, gated_transition_tensor

# The method's two halves, made explicit. The generic, contact-free inference ENGINES
# (THEORY.md §5): an ``HMM`` / ``SemiMarkovHMM`` is a ``TemporalSmoother`` that turns a
# per-frame log-emission matrix into a smoothed posterior + a MAP path. The contact SCIENCE
# they consume (§3/§4): ``MODES`` is the bank of per-mode generative models, each a
# ``ContactMode`` proper density over the (gap, twist) observation; ``log_emissions`` stacks
# them into the matrix the engine smooths.
from .emissions import MODES, ContactMode, log_emissions
from .hmm import HMM, TemporalSmoother
from .hsmm import SemiMarkovHMM

# The capability-driven front door (DESIGN.md): declare what you have — shape, force,
# material — and `detect_pair` assembles the richest estimator those capabilities admit
# (an empty declaration reproduces the kinematic flat-floor detector exactly);
# `value_of_information` ranks what to provide next.
from .capabilities import Capabilities, detect_pair, value_of_information

# The multi-body contact-graph detector (THEORY.md §8, rung 5): the single-pair detector
# lifted to a whole contact graph, inferring the joint active-set posterior over the 2^E
# structures. `build_candidate_edges` is the proximity broad-phase; `detect_scene` is the
# full pipeline.
from .graph import build_candidate_edges, detect_scene

# Research-frontier entrypoints (THEORY.md §8 & §10), all OFF the default detection path:
#   * discover_modes                 -- unsupervised contact-mode discovery via a sticky
#     HDP-HMM (learn the mode vocabulary from data instead of presupposing it).
#   * exact_active_sets /            -- the active-set structure posterior: exact 2^E
#     particle_filter_active_sets       enumeration (reference) and a Rao-Blackwellized
#                                       particle smoother that scales past enumeration.
#   * emission_tempering             -- per-frame measurement-uncertainty tempering of the
#                                       emissions (noisy/occluded frames contribute less).
from .mode_discovery import discover_modes
from .structure_inference import (
    StructurePosterior,
    exact_active_sets,
    particle_filter_active_sets,
)
from .uncertainty import emission_tempering

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
    # the capability-driven front door (DESIGN.md)
    "Capabilities",
    "detect_pair",
    "value_of_information",
    # multi-body contact-graph layer (THEORY.md §8, rung 5)
    "build_candidate_edges",
    "detect_scene",
    # §5-§7 leaf entrypoints
    "detect_impacts",
    "friction_stick_slip",
    "normal_force_from_penetration",
    "observability_demo",
    "base_transition_matrix",
    "gated_transition_tensor",
    # the inference engines (§5) + the per-mode generative models (§3/§4)
    "HMM",
    "SemiMarkovHMM",
    "TemporalSmoother",
    "ContactMode",
    "MODES",
    "log_emissions",
    # contact-implicit inverse dynamics (THEORY.md §8, the north star), off by default
    "body_accelerations",
    "required_wrench",
    "contact_wrench_map",
    "solve_contact_implicit",
    "contact_implicit_from_raw",
    # research-frontier layer (THEORY.md §8 & §10), off by default
    "discover_modes",
    "exact_active_sets",
    "particle_filter_active_sets",
    "StructurePosterior",
    "emission_tempering",
]
