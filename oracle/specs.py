"""The typed build contracts a scenario/scene builder returns (THEORY.md §9).

A builder's job is to *compile a world and name what matters in it*: which body is the
moving one, which geom pair realizes the candidate contact, where the support surface and
tracked material point sit, and how long to simulate. These frozen dataclasses ARE that
contract — the factory machinery (`oracle.factory`) reads them by attribute, so a typo is
an `AttributeError` at construction, not a silent `KeyError` mid-simulation, and the full
recipe of every scenario is readable from its constructor call.

`ScenarioSpec` is one candidate contact between one moving body and one support
(`factory.generate`); `SceneSpec` is a multi-body contact graph — several bodies sharing a
time base plus a list of candidate `EdgeSpec`s (`factory.generate_scene`, THEORY.md §8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:  # annotations only — this leaf stays import-light
    import mujoco

    from contact.geometry import ContactGeometry
    from contact.types import SupportSurface


@dataclass(frozen=True)
class ScenarioSpec:
    """One single-pair scenario: a compiled model + the named-entity recipe.

    model:                 the compiled MuJoCo model.
    moving_body/geom:      the body whose contact we test, and the geom realizing it.
    support_body/geom:     the support side; ``support_body`` may be ``"world"`` (the
                           static floor — an identity pose of infinite mass, THEORY §1).
    surface_point_local /
    surface_normal_local:  the support plane, in the support body's local frame.
    contact_point_local:   the tracked material point on the moving body (its frame).
    shape:                 ``"box" | "sphere" | "cylinder"`` — steers the truth labeler's
                           rolling test (a low-slip translating sphere is ROLLING, §3).
    duration:              seconds of physics to record.
    init:                  optional one-time ``f(model, data)`` after the first forward
                           pass (e.g. the rolling ball's matched v/ω initial state).
    forcing:               optional per-substep ``f(model, data)`` (e.g. the push ramp).
    record_hz:             optional recording-cadence floor: energetic sub-frame impacts
                           only register in the truth when sampled faster; ``generate``
                           records at ``max(caller hz, record_hz)``.
    resolver:              optional ``f(surface) -> ContactGeometry`` attaching a
                           higher-fidelity geometry resolver to the emitted RawScenario
                           (DESIGN axis 1); None -> the default FlatRegion path.
    rig_corners_local:     (K, 3) corner points of the §7 statically-indeterminate rig;
                           when set, per-corner penetration/force truth is harvested.
    box_corners_local:     (K, 3) candidate contact points for contact-implicit inverse
                           dynamics (§8): per-corner signed gap / true force / Signorini
                           active flags are harvested and emitted in ``meta``.
    """

    model: "mujoco.MjModel"
    moving_body: str
    moving_geom: str
    support_body: str
    support_geom: str
    surface_point_local: np.ndarray
    surface_normal_local: np.ndarray
    contact_point_local: np.ndarray
    shape: str
    duration: float
    init: Callable | None = None
    forcing: Callable | None = None
    record_hz: float | None = None
    resolver: "Callable[[SupportSurface], ContactGeometry] | None" = None
    rig_corners_local: np.ndarray | None = None
    box_corners_local: np.ndarray | None = None


@dataclass(frozen=True)
class EdgeSpec:
    """One candidate contact-graph edge of a scene (THEORY.md §8).

    The per-edge analogue of `ScenarioSpec`'s naming: which body pair, which geom SETS
    realize the contact (a set so e.g. board↔ground aggregates all four wheel geoms),
    the support plane and tracked point, and the twist shape for the truth labeler.

    truth_mode_body:  classify the truth MODE from this body's material point instead of
                      the contacting sub-body's — for edges whose *observation* tracks a
                      parent body (the board origin genuinely SLIDES while its wheels
                      roll; truth and observation must describe the same body, §3).
    geometry:         optional per-edge ContactGeometry resolver carried onto the emitted
                      ContactEdge (e.g. SphereSphere on a ball↔ball edge, DESIGN axis 1).
    """

    edge_id: str
    moving_body: str
    support_body: str
    moving_geoms: tuple[str, ...]
    support_geoms: tuple[str, ...]
    surface_point_local: np.ndarray
    surface_normal_local: np.ndarray
    contact_point_local: np.ndarray
    shape: str
    truth_mode_body: str | None = None
    geometry: "ContactGeometry | None" = None


@dataclass(frozen=True)
class SceneSpec:
    """One multi-body scene: a compiled model, its named bodies, and candidate edges.

    bodies:   the body names whose (noised) poses the scene exposes as observables.
    edges:    the candidate `EdgeSpec`s — the contact graph the detector infers over.
    settle:   optional un-recorded seating phase (s) before the clock starts.
    launch:   optional one-shot ``f(model, data)`` fired after settling (an unactuated
              initial velocity, e.g. shoving the skateboard).
    forcing:  optional per-substep ``f(model, data)`` (e.g. lowering a support block).
    record_hz: recording-cadence floor, as on `ScenarioSpec`.
    meta:     free-form notes merged into the emitted MultiBodyScene.meta.
    """

    model: "mujoco.MjModel"
    bodies: tuple[str, ...]
    edges: tuple[EdgeSpec, ...]
    duration: float
    settle: float = 0.0
    launch: Callable | None = None
    forcing: Callable | None = None
    record_hz: float | None = None
    meta: dict = field(default_factory=dict)
