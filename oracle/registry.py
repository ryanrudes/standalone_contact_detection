"""The scenario / scene registry — how demos join the truth factory (THEORY.md §9).

THEORY §9 wants "the entire edge-case taxonomy on demand": every scenario and scene is a
*builder* — a zero-argument callable compiling a MuJoCo model plus the build recipe the
factory machinery consumes — registered here under its public name. Builder modules
self-register at import:

    from oracle.registry import scenario

    @scenario("incline_slide")
    def _build_incline_slide(): ...

`oracle/__init__` imports every builder module, so importing `oracle` yields the complete
registry; `oracle.SCENARIOS` / `oracle.SCENES` are the resulting name lists, and
`factory.generate` / `factory.generate_scene` look names up here at call time. This module
imports nothing, so builder modules can depend on it without any cycle.
"""

from __future__ import annotations

from typing import Callable

#: name -> zero-argument builder for a single body-pair scenario (`factory.generate`).
SCENARIO_BUILDERS: dict[str, Callable] = {}

#: name -> zero-argument builder for a multi-body scene (`factory.generate_scene`).
SCENE_BUILDERS: dict[str, Callable] = {}


def scenario(name: str) -> Callable:
    """Register a single-pair scenario builder under ``name`` (decorator)."""

    def register(builder: Callable) -> Callable:
        if name in SCENARIO_BUILDERS:
            raise ValueError(f"duplicate scenario name {name!r}")
        SCENARIO_BUILDERS[name] = builder
        return builder

    return register


def scene(name: str) -> Callable:
    """Register a multi-body scene builder under ``name`` (decorator)."""

    def register(builder: Callable) -> Callable:
        if name in SCENE_BUILDERS:
            raise ValueError(f"duplicate scene name {name!r}")
        SCENE_BUILDERS[name] = builder
        return builder

    return register
