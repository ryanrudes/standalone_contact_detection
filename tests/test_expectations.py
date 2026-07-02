"""Expectation-based asserts: every demo's DETECTION must match the physically-expected
contact/mode story (contact/verification.py), not merely score a passing IoU.

This is the test-suite half of the verification the report (`verify_demos.py`) prints.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.verification import (  # noqa: E402
    SCENARIO_EXPECT,
    SCENE_EXPECT,
    verify_scenario,
    verify_scene,
)

#: Demos known not to meet their full expected story yet (xfail, strict so a fix flips green).
_KNOWN_PENDING: dict[str, str] = {}


def _scene_params():
    out = []
    for name in SCENE_EXPECT:
        if name in _KNOWN_PENDING:
            out.append(pytest.param(name, marks=pytest.mark.xfail(
                reason=_KNOWN_PENDING[name], strict=True)))
        else:
            out.append(name)
    return out


@pytest.mark.parametrize("name", list(SCENARIO_EXPECT))
def test_scenario_matches_expectation(name):
    fails = [c for c in verify_scenario(name) if c.status == "FAIL"]
    assert not fails, "; ".join(f"{c.name}: {c.detail}" for c in fails)


@pytest.mark.parametrize("name", _scene_params())
def test_scene_matches_expectation(name):
    fails = [c for c in verify_scene(name) if c.status == "FAIL"]
    assert not fails, "; ".join(f"{c.name}: {c.detail}" for c in fails)
