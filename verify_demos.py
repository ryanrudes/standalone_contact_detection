#!/usr/bin/env python3
"""Print the expectation-based verification report for every demo.

    uv run python verify_demos.py            # all demos
    uv run python verify_demos.py dominoes   # one demo

For each demo it checks the DETECTION against the physically-expected contact/mode story
(see contact/verification.py) and prints PASS / WARN / FAIL per check.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contact import mujoco_gen
from oracle.verification import (
    SCENARIO_EXPECT,
    SCENE_EXPECT,
    verify_scenario,
    verify_scene,
    worst_status,
)

_GLYPH = {"PASS": "[OK]  ", "WARN": "[warn]", "FAIL": "[FAIL]"}


def main() -> None:
    names = sys.argv[1:]
    if not names:
        names = list(mujoco_gen.SCENARIOS) + list(mujoco_gen.SCENES)

    n_pass = n_warn = n_fail = 0
    for name in names:
        if name in SCENARIO_EXPECT:
            checks = verify_scenario(name)
        elif name in SCENE_EXPECT:
            checks = verify_scene(name)
        else:
            print(f"  (no expectation encoded for {name})")
            continue
        status = worst_status(checks)
        n_pass += status == "PASS"
        n_warn += status == "WARN"
        n_fail += status == "FAIL"
        print(f"\n{_GLYPH[status]} {name}")
        for c in checks:
            print(f"    {_GLYPH[c.status]} {c.name:24s} {c.detail}")

    print(f"\n{'=' * 60}")
    print(f"  {n_pass} PASS   {n_warn} WARN   {n_fail} FAIL   ({len(names)} demos)")


if __name__ == "__main__":
    main()
