"""DESIGN.md PHASE 5 (the capstone): the CAPABILITY REGISTRY + VALUE-OF-INFORMATION.

This module is the thin, declarative orchestration layer the whole DESIGN points at
(PART I sections 4 "the capability registry", 5 "evidence factors", 8 "value of
information"; PART III III.1 the data contracts). It unifies the optional, capability-gated
evidence -- per-frame contact *geometry* (the fidelity ladder of
:mod:`contact.geometry_resolvers`) and the *force* channel
(:attr:`contact.types.ContactObservations.normal_force`) -- behind ONE small data class,
:class:`Capabilities`, and a single entrypoint, :func:`detect_pair`, that selects the
richest resolver the declaration supports and wires the declared evidence into the *existing*
detector.

It is **purely additive and strictly on top of the validated floor**. It does NOT touch the
detector, the emissions, the geometry core, or any existing detection path: every call here
funnels through the already-shipped, already-gated seams --
:func:`contact.geometry.observe(..., geometry=<resolved>)` and
:meth:`contact.model.ContactDetector.detect` -- so with an empty :class:`Capabilities` the
result is byte-identical to today's kinematic/flat-floor pipeline (DESIGN.md PART I sections
2 & 7: "the validated flat-floor/kinematic path is the guaranteed floor"). Declaring a shape
swaps in a higher-fidelity resolver; declaring a (measured) force adds the gated force factor;
neither rewrites a single line of the inference.

On top of the registry, :func:`value_of_information` answers DESIGN.md PART I section 8 --
*what should the user provide next?* -- by counterfactually re-running detection with each
hypothesized capability and ranking them by how much the answer actually moves. The robust
signal here (DESIGN.md PART II premise note) is the **MAP-mode change fraction**, not the
posterior entropy: the HMM posterior is over-confident, so entropy barely budges even where a
capability flips the decision, whereas the MAP path *does* flip -- so the fraction of frames
whose MAP mode changes is the honest measure of a capability's informativeness.

Imports are deliberately limited to numpy, :mod:`dataclasses`, and the ``contact`` leaves
this layer composes (geometry / geometry_resolvers / model / config / types), keeping it a
thin, side-effect-free registry over the existing factors.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .config import DetectorConfig, MaterialParams
from .geometry import observe
from .geometry_resolvers import BoxPlane, MeshConvex, MeshPlane, SpherePlane, SphereSphere
from .model import ContactDetector
from .types import (
    IMPACT,
    ContactGeometry,
    ContactObservations,
    DetectionResult,
    PoseTrajectory,
    SupportSurface,
)

__all__ = ["Capabilities", "detect_pair", "value_of_information", "_resolve_geometry"]


# --------------------------------------------------------------------------------------
# The capability declaration (DESIGN.md PART I section 4 / PART III III.1).
#
# One small, introspectable record of "what the user can give us" for a single body-pair
# (edge). Every field defaults to the ABSENCE of that capability, so the default
# `Capabilities()` selects exactly the validated floor (FlatRegion geometry, no force
# factor, no material, no tempering) -- DESIGN.md PART I section 7 invariant 1.
# --------------------------------------------------------------------------------------


@dataclass
class Capabilities:
    """Declarative "what evidence is available" for one body-pair edge (DESIGN.md §4).

    A selector (:func:`_resolve_geometry` + :func:`detect_pair`) maps this record to a
    concrete :class:`~contact.types.ContactGeometry` resolver plus the set of enabled
    evidence factors, then runs the *existing* detector. Everything defaults to ABSENT, so
    a bare :class:`Capabilities` reproduces today's kinematic/flat-floor estimate
    bit-for-bit (the no-op-when-absent contract, DESIGN.md PART I sections 5 & 7).

    Fields
    ------
    shape:
        ``None`` (default) -> the floor: :func:`contact.geometry.observe` falls back to its
        :class:`~contact.geometry_resolvers.FlatRegion`. Otherwise one of
        ``"sphere_plane"`` / ``"sphere_sphere"`` / ``"box_plane"`` -- the primitive whose
        resolver position-derives the per-frame contact normal/point (DESIGN.md PART II.D),
        killing the spinning-normal artifacts a flat-region approximation manufactures on a
        curved/rotating support.
    params:
        Shape parameters consumed by :func:`_resolve_geometry`, e.g.
        ``{"r_moving": .., "r_support": ..}`` for the sphere resolvers or
        ``{"half_extents": [..]}`` for the box. Empty by default.
    force:
        The force channel (DESIGN.md PART I section 6). ``"none"`` (default) -> no force
        factor (the kinematics-only floor). ``"measured"`` -> the caller supplies a sensor
        stream (the ``truth_force`` argument of :func:`detect_pair`) that populates
        :attr:`~contact.types.ContactObservations.normal_force`. ``"inferred"`` -> the
        virtual-force-sensor path, which is a WHOLE-BODY/scenario-level quantity and so is
        not available to the bare-pair API (see :func:`detect_pair`).
    material:
        Optional :class:`~contact.config.MaterialParams` (mu / stiffness / restitution).
        When set, detection runs against a config copy carrying it (e.g. a known stiffness
        turns penetration into a calibrated force gauge, THEORY.md s.7). ``None`` -> the
        config's own material is used unchanged.
    meas_cov:
        Declared availability of a per-frame measurement covariance (DESIGN.md §3.5). When
        ``True`` AND the observations actually carry a ``meas_cov`` stream, detection enables
        the measurement-tempering factor (``use_uncertainty``) so low-fidelity/occluded
        frames are down-weighted. The Phase-0/1 resolvers declare *exact* provenance
        (``normal_sigma = gap_sigma = 0``) and :func:`observe` does not yet synthesize a
        ``meas_cov`` from it, so on the bare-pair path this flag is currently a safe no-op --
        it is wired here for forward-compatibility and never changes the floor.
    """

    shape: str | None = None
    params: dict = field(default_factory=dict)
    force: str = "none"
    material: MaterialParams | None = None
    meas_cov: bool = False

    def merge(self, candidate: "Capabilities") -> "Capabilities":
        """Return a copy of ``self`` (the BASE) with ``candidate``'s NON-DEFAULT fields applied.

        The asymmetric merge used by :func:`value_of_information`: a counterfactual
        *candidate* declaration overrides the base only where it actually says something
        (a field different from that field's default). Fields the candidate leaves at their
        default are taken from the base. Hence a candidate that declares nothing
        (``Capabilities()``) merges to a copy equal to the base -- the "adds nothing"
        case that VoI scores as zero gain.
        """
        overrides = {
            f.name: getattr(candidate, f.name)
            for f in dataclasses.fields(self)
            if not _is_field_default(f, getattr(candidate, f.name))
        }
        return dataclasses.replace(self, **overrides)


def _is_field_default(f: "dataclasses.Field", value: object) -> bool:
    """True iff ``value`` equals dataclass field ``f``'s declared default (factory-aware).

    Compares against ``f.default`` or, for a ``default_factory`` field (``params``), a
    freshly-built default (``{}``). Equality is taken defensively: a value whose ``==`` is
    not a plain bool (e.g. a stray numpy array tucked into ``params``) is treated as
    non-default (overrides), so the merge never raises on an exotic declaration.
    """
    if f.default is not dataclasses.MISSING:
        default = f.default
    elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        default = f.default_factory()  # type: ignore[misc]
    else:
        return False
    try:
        return bool(value == default)
    except Exception:
        return value is default


# --------------------------------------------------------------------------------------
# Geometry selection (DESIGN.md PART I section 4: "pick the richest resolver the
# (shapeA, shapeB) pair supports, else FlatRegion"). The dispatch is on the declared
# `shape` string; an absent or unrecognized shape returns None, which `observe` reads as
# "use the default FlatRegion" -- never worse than today (DESIGN.md s.11).
# --------------------------------------------------------------------------------------


#: shape id -> a resolver factory ``(caps, surface, contact_point_local) -> ContactGeometry``. Adding
#: a shape is one entry here -- the same registry pattern as ``emissions.MODES``. A shape absent from
#: the registry (or ``shape=None``) degrades to ``None`` => ``observe`` uses the byte-identical FlatRegion.
SHAPE_RESOLVERS: dict[str, Callable[[Capabilities, SupportSurface, np.ndarray], ContactGeometry]] = {
    "sphere_plane": lambda caps, surface, cpl: SpherePlane(caps.params["r_moving"], surface, cpl),
    "sphere_sphere": lambda caps, surface, cpl: SphereSphere(caps.params["r_moving"], caps.params["r_support"]),
    "box_plane": lambda caps, surface, cpl: BoxPlane(np.asarray(caps.params["half_extents"]), surface),
    "mesh_plane": lambda caps, surface, cpl: MeshPlane(np.asarray(caps.params["vertices"]), surface, cpl),
    "mesh_mesh": lambda caps, surface, cpl: MeshConvex(
        np.asarray(caps.params["vertices_moving"]), np.asarray(caps.params["vertices_support"])
    ),
}


def _resolve_geometry(
    caps: Capabilities,
    surface: SupportSurface,
    contact_point_local: np.ndarray,
) -> ContactGeometry | None:
    """Map a :class:`Capabilities` declaration to a concrete resolver (or ``None``).

    Dispatch on ``caps.shape`` (DESIGN.md PART II.D / III.1):

    * ``"sphere_plane"``  -> :class:`~contact.geometry_resolvers.SpherePlane` (moving sphere
      of radius ``params["r_moving"]`` on the support plane; the plane is the legacy
      ``surface`` carried by the support, the tracked point its ``contact_point_local``).
    * ``"sphere_sphere"`` -> :class:`~contact.geometry_resolvers.SphereSphere` with radii
      ``params["r_moving"]`` / ``params["r_support"]`` -- the position-derived normal that
      turns the ball-ball "7 phantom impacts" into the single real collision.
    * ``"box_plane"``     -> :class:`~contact.geometry_resolvers.BoxPlane` with
      ``params["half_extents"]`` -- the migrating nearest-corner resolver for the plane.
    * ``"mesh_plane"``    -> :class:`~contact.geometry_resolvers.MeshPlane` with
      ``params["vertices"]`` -- a convex vertex cloud against the support plane (the Phase-3
      generalization of ``box_plane`` to an arbitrary cloud; a box mesh reproduces ``BoxPlane``).
    * ``"mesh_mesh"``     -> :class:`~contact.geometry_resolvers.MeshConvex` with
      ``params["vertices_moving"]`` / ``params["vertices_support"]`` -- two convex clouds via
      GJK/EPA, the position-derived-normal generalization of ``sphere_sphere`` (DESIGN.md III.5).
    * ``None``            -> ``None``: :func:`observe` then defaults to ``FlatRegion``
      (the validated, byte-identical floor).
    * anything else       -> ``None`` (FlatRegion fallback). NOTE: an UNRECOGNIZED shape is
      intentionally degraded to the floor rather than raised, so a forward/typo'd shape can
      never make the estimate *worse* than today's flat-plane baseline (DESIGN.md PART I
      section 7 / s.11 "unsupported pairs fall back to FlatRegion").
    """
    make = SHAPE_RESOLVERS.get(caps.shape)  # None for shape=None or an UNRECOGNIZED shape
    if make is None:
        # Degrade to the FlatRegion floor (documented no-worse-than-today fallback):
        # `observe(geometry=None)` reproduces the bit-identical baseline.
        return None
    return make(caps, surface, contact_point_local)


# --------------------------------------------------------------------------------------
# The single-edge entrypoint (DESIGN.md PART I sections 4-6): declaration -> DetectionResult.
# A thin selector ON TOP of the existing `observe` + `ContactDetector.detect`; with
# `Capabilities()` it is byte-identical to today's pipeline.
# --------------------------------------------------------------------------------------


def _force_none(obs, truth_force):
    return obs  # no factor -> the byte-identical kinematics-only behaviour.


def _force_measured(obs, truth_force):
    if truth_force is None:
        raise ValueError(
            "Capabilities(force='measured') needs a caller-supplied `truth_force` sensor stream "
            "(T,), but got None. Pass the measured normal force, or declare force='none' to run "
            "kinematics-only."
        )
    return dataclasses.replace(obs, normal_force=np.asarray(truth_force, dtype=float))


def _force_inferred(obs, truth_force):
    raise NotImplementedError(
        "Capabilities(force='inferred') is unsupported by the bare-pair detect_pair API. Inferred "
        "force is a WHOLE-BODY quantity: it is recovered at the scene/body level from the body's "
        "mass/inertia and the scenario-level raw candidate points/normals/gaps via "
        "`contact.dynamics_id.infer_normal_force(raw, config)`, which requires a RawScenario "
        "(inertials + candidates) that the bare (moving, support, surface, contact_point_local) pair "
        "does not carry. Either use force='measured' with a supplied sensor stream, or run the "
        "inferred-force virtual sensor at the scene/body level (DESIGN.md PART II.B / III.4) and pass "
        "its (T,) output in here as the measured stream."
    )


#: force mode -> a preparer returning the (possibly force-augmented) observations, or raising.
#: The same registry pattern as SHAPE_RESOLVERS / emissions.MODES; a new force mode is one entry.
FORCE_PREPARERS: dict[str, Callable[[ContactObservations, np.ndarray | None], ContactObservations]] = {
    "none": _force_none,
    "measured": _force_measured,
    "inferred": _force_inferred,
}


def detect_pair(
    moving: PoseTrajectory,
    support: PoseTrajectory,
    surface: SupportSurface,
    contact_point_local: np.ndarray,
    caps: Capabilities,
    config: DetectorConfig | None = None,
    truth_force: np.ndarray | None = None,
) -> DetectionResult:
    """Detect contact for one body-pair under a :class:`Capabilities` declaration.

    The capability registry's main verb (DESIGN.md PART I sections 4-6). It (1) selects the
    geometry resolver the declaration supports, (2) runs the *existing*
    :func:`contact.geometry.observe` through that resolver, (3) wires the declared force /
    material / measurement-covariance evidence into the *existing*
    :meth:`contact.model.ContactDetector.detect`, and (4) returns its
    :class:`~contact.types.DetectionResult` unchanged. No detection logic lives here -- this
    is orchestration only, so an empty :class:`Capabilities` reproduces today's result
    bit-for-bit (DESIGN.md PART I section 7 invariant 1).

    Parameters
    ----------
    moving, support:
        Pose streams of the moving body and its (possibly moving) support.
    surface, contact_point_local:
        The legacy flat-plane spec; it configures the default ``FlatRegion`` and is passed
        to the sphere/plane resolvers that still need a plane + tracked point.
    caps:
        The :class:`Capabilities` declaration selecting the resolver and the evidence factors.
    config:
        A :class:`~contact.config.DetectorConfig`; ``None`` -> defaults. Never mutated:
        material / tempering overrides are applied to a :func:`dataclasses.replace` copy.
    truth_force:
        The caller-supplied normal-force sensor stream ``(T,)`` used ONLY when
        ``caps.force == "measured"`` (it then populates
        :attr:`~contact.types.ContactObservations.normal_force`). Ignored otherwise.

    Force channel (DESIGN.md PART I section 6)
    ------------------------------------------
    * ``"none"``     -> the observations are left untouched (no force factor; the floor).
    * ``"measured"`` -> ``obs = dataclasses.replace(obs, normal_force=truth_force)`` -- the
      gated per-state force emission then multiplies in (a cradle clack, invisible
      kinematically, becomes a decisive force pulse). Requires ``truth_force``.
    * ``"inferred"`` -> raises :class:`NotImplementedError`: inferred force is a WHOLE-BODY
      quantity recovered at the scene/body level from mass/inertia and the candidate
      points/normals/gaps via :func:`contact.dynamics_id.infer_normal_force`, which needs a
      :class:`~contact.types.RawScenario` -- inputs the bare ``(moving, support, surface,
      contact_point_local)`` pair does not carry. Run the virtual sensor at the scene level
      (DESIGN.md PART II.B / III.4) and feed its output back in as a ``"measured"`` stream.
    """
    cfg = config if config is not None else DetectorConfig()

    # (1) Geometry: pick the resolver the declaration supports (None => FlatRegion floor).
    geom = _resolve_geometry(caps, surface, contact_point_local)

    # (2) Observe through the EXISTING narrow waist. With geom=None this is byte-identical to
    # today's pipeline (DESIGN.md III.2); with a resolver, `observe` consumes its per-frame
    # (point, normal, gap) and runs the same twist decomposition unchanged.
    obs = observe(
        moving,
        support,
        surface,
        contact_point_local,
        cfg.vel_smooth_time,
        geometry=geom,
    )

    # (3a) Force channel: the gated optional observation (DESIGN.md PART I section 6 / II.A).
    prepare = FORCE_PREPARERS.get(caps.force)
    if prepare is None:
        raise ValueError(
            f"unknown force mode {caps.force!r}; expected 'none', 'measured', or 'inferred'."
        )
    obs = prepare(obs, truth_force)

    # (3b) Material / measurement-covariance: apply to a CONFIG COPY (never mutate the
    # caller's config). `dataclasses.replace` shares the untouched sub-configs by reference;
    # `ContactDetector.detect` deep-copies before it ever mutates emission scales, so nothing
    # leaks back. Each override is gated on the capability being present (no-op otherwise).
    det_cfg = cfg
    if caps.material is not None:
        det_cfg = dataclasses.replace(det_cfg, material=caps.material)
    if caps.meas_cov and obs.meas_cov is not None:
        # Enable the measurement-tempering factor only when a per-frame covariance is
        # actually present (DESIGN.md §3.5). Toggling a nested field needs a nested copy.
        det_cfg = dataclasses.replace(
            det_cfg,
            inference=dataclasses.replace(det_cfg.inference, use_uncertainty=True),
        )

    # (4) Run the EXISTING detector and return its result unchanged.
    return ContactDetector(det_cfg).detect(obs)


# --------------------------------------------------------------------------------------
# Value of information (DESIGN.md PART I section 8): "what should the user provide next?"
# --------------------------------------------------------------------------------------


class _VoIRanking(list):
    """The ranking returned by :func:`value_of_information` (a ``list`` with one extra).

    IS a ``list[tuple[str, float]]`` -- ``(capability name, MAP-change gain)`` sorted by gain
    DESC, the required core -- so it indexes / iterates / ``len`` / compares exactly like a
    plain list. The single ADDITIVE extra is :attr:`guidance`: a list of short canned-guidance
    strings (DESIGN.md PART I section 8), e.g. a force recommendation when the kinematic
    detector fires impulse atoms at frames whose MAP never reaches IMPACT (the force-transfer,
    "unobservable from kinematics" signature). The guidance NEVER affects the ordering -- it
    is a diagnostic bonus closing the "best with what you can give us" loop.
    """

    def __init__(self, items=(), guidance: list[str] | None = None) -> None:
        super().__init__(items)
        self.guidance: list[str] = list(guidance) if guidance is not None else []


def _force_transfer_guidance(base: DetectionResult) -> list[str]:
    """Canned VoI guidance from the BASE detection (DESIGN.md PART I section 8).

    The "unobservable from kinematics" signature: the matched-filter impact detector fires
    velocity-step atoms (it *sees* a closing velocity), yet the MAP path at those frames
    never enters IMPACT -- the contact's force pulse is invisible to kinematics (the cradle
    clack, THEORY.md s.6-s.8). When that happens we recommend declaring a force channel.
    """
    impulses = list(base.impulses)
    n = len(base.map_state)
    unseen = [
        a
        for a in impulses
        if 0 <= int(a.index) < n and base.map_state[int(a.index)] != IMPACT
    ]
    if not unseen:
        return []
    return [
        f"force-transfer / 'unobservable from kinematics' signature: {len(unseen)} of "
        f"{len(impulses)} impulse atom(s) fire at frames whose MAP mode never reaches "
        f"'{IMPACT}' -- the closing velocity is seen but the contact's force pulse is not. "
        f"Declare a force channel (force='measured' with a sensor stream, or the "
        f"scene/body-level inferred virtual sensor) to resolve these clacks "
        f"(DESIGN.md PART I sections 6 & 8)."
    ]


def value_of_information(
    moving: PoseTrajectory,
    support: PoseTrajectory,
    surface: SupportSurface,
    contact_point_local: np.ndarray,
    base: Capabilities,
    candidates: dict[str, Capabilities],
    config: DetectorConfig | None = None,
    truth_force: np.ndarray | None = None,
) -> _VoIRanking:
    """Rank hypothesized capabilities by how much they change the answer (DESIGN.md §8).

    The diagnostic that closes the loop on "best with what you can give us": given the
    capabilities already in hand (``base``) and a menu of ones the user *could* add
    (``candidates``), tell them which is worth providing next. We detect once with ``base``
    to get the reference MAP path, then for each named candidate merge ``base`` + candidate
    (the candidate's NON-DEFAULT fields override -- :meth:`Capabilities.merge`), detect, and
    score the candidate by the **fraction of frames whose MAP mode changes** versus the base.

    Why MAP-change, not entropy (DESIGN.md PART II premise note): the HMM/HSMM posterior is
    over-confident, so its entropy barely moves even where an added factor flips the decision;
    the MAP path, by contrast, *does* flip exactly there. The MAP-change fraction is therefore
    the robust, honest measure of a capability's informativeness for this estimator.

    A candidate that declares nothing new (its merge with ``base`` equals ``base``) is scored
    ``0.0`` without a redundant re-detection. The returned :class:`_VoIRanking` is the
    ``(name, gain)`` list sorted by gain DESC, and additionally carries ``.guidance`` (a short
    canned force-recommendation when the base shows the force-transfer signature, §8).

    Parameters mirror :func:`detect_pair`; ``truth_force`` is forwarded so a
    ``force='measured'`` candidate (or base) sees the supplied sensor stream.
    """
    cfg = config if config is not None else DetectorConfig()

    # Reference: detect once with the capabilities already in hand. `truth_force` is forwarded
    # but only consumed if `base` itself declares a measured force (else it is ignored).
    base_result = detect_pair(
        moving, support, surface, contact_point_local, base, cfg, truth_force
    )
    base_map = list(base_result.map_state)
    denom = float(len(base_map)) if base_map else 1.0

    gains: list[tuple[str, float]] = []
    for name, candidate in candidates.items():
        merged = base.merge(candidate)
        if _caps_equal(merged, base):
            # Adds nothing (no NON-DEFAULT field beyond the base) -> exactly zero gain; the
            # re-detection would reproduce base_map frame-for-frame, so we skip it.
            gains.append((name, 0.0))
            continue
        cand_result = detect_pair(
            moving, support, surface, contact_point_local, merged, cfg, truth_force
        )
        cand_map = cand_result.map_state
        m = min(len(base_map), len(cand_map))
        changed = sum(1 for i in range(m) if base_map[i] != cand_map[i])
        gains.append((name, float(changed) / denom))

    ranking = sorted(gains, key=lambda kv: kv[1], reverse=True)
    return _VoIRanking(ranking, guidance=_force_transfer_guidance(base_result))


def _caps_equal(a: Capabilities, b: Capabilities) -> bool:
    """Whether two :class:`Capabilities` are equal, defensive against exotic ``params``.

    Dataclass equality is exact field-by-field; a ``params`` dict holding only plain scalars
    (the supported case) compares cleanly. Should an exotic value (e.g. a numpy array) make
    ``==`` raise, we report "not equal" so VoI falls through to the real re-detection rather
    than mis-scoring the candidate as a no-op.
    """
    try:
        return bool(a == b)
    except Exception:
        return False
