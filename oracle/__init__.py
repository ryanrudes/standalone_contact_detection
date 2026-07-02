"""The ground-truth oracle and everything that consumes its withheld labels (THEORY.md §9).

The repo's epistemology is a package boundary: `contact/` is the estimator and sees only
noisy poses; `oracle/` is the experimenter's side — it *makes* truth (MuJoCo scenario/scene
factories, plus one analytic synthesizer), withholds it from the detector, and then spends it
on scoring, expectation checks, and synced visualizations.

The import law, machine-checked in `tests/test_import_law.py`: **`oracle` imports `contact`;
`contact` never imports `oracle`** (nor mujoco/matplotlib — the method stays numpy/scipy).

    from oracle import generate                      # truth factory (§9)
    from contact import observe, ContactDetector     # the estimator (§1–§8)

    raw = generate("push_to_slide")
    obs = observe(raw.moving, raw.support, raw.surface, raw.contact_point_local)
    result = ContactDetector().detect(obs)
    oracle.score(result, raw.truth)                  # the labels come back out only here
"""

from __future__ import annotations

from oracle.report import (
    plot_graph,
    plot_result,
    print_graph_report,
    print_inverse_dynamics,
    print_report,
    score,
)
from oracle.verification import verify_scenario, verify_scene, worst_status
from oracle.visualize import animate_scene, animate_scenario

__all__ = [
    "score",
    "print_report",
    "print_inverse_dynamics",
    "plot_result",
    "print_graph_report",
    "plot_graph",
    "verify_scenario",
    "verify_scene",
    "worst_status",
    "animate_scenario",
    "animate_scene",
]
