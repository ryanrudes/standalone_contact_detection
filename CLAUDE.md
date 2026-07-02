# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A probabilistic, **support-relative contact-state estimator** for motion data: given two bodies'
noisy pose trajectories it infers, per frame and with calibrated uncertainty, whether they are in
contact and what *kind* (static / sliding / pivoting / rolling / impact), plus make/break event
times and (with known stiffness) contact force. The method is a generative hybrid dynamical system
decoded as a Bayesian posterior — modes are subspaces of the relative twist, scored as a calibrated
likelihood ratio and smoothed by an HMM/HSMM; multi-body scenes lift to an active-set posterior over
a contact graph. Ground truth comes from MuJoCo and is **withheld from the detector, used only to
score**.

The repo is theory-first: read [`THEORY.md`](THEORY.md) (§0–§10, the derivation the code cites by
section number throughout), then [`DESIGN.md`](DESIGN.md) (the capability-driven generalization), then
[`README.md`](README.md) (the module table + demos). Don't restate their content — defer to them.

## Environment & commands

`uv`-managed, Python 3.12 (`.python-version`). `uv run …` auto-syncs from `uv.lock`; use `uv sync` to
materialize `.venv` explicitly. Heavyweight deps: `mujoco` (truth oracle), `coal` (convex collision /
GJK+EPA), `cvxpy`+`clarabel` (inverse-dynamics cone/QP solves), `numpy`/`scipy` (core math).

```bash
# Single-pair: generate (MuJoCo) → observe → detect → score → plot
uv run python detect.py --scenario rolling_ball
uv run python detect.py --scenario drop_rest_liftoff --stiffness 50000 --no-plot
uv run python detect.py --scenario push_to_slide --inverse-dynamics   # opt-in §8 north star

# Multi-body contact-graph scene
uv run python detect_scene.py --scene person_on_skateboard

# Synced side-by-side videos (writes under git-ignored media/)
uv run python viz.py person_on_skateboard --pairs        # add --force for force-mediated contacts

# Tests (math + integration + expectation asserts; pytest is scoped via pyproject testpaths)
uv run pytest
uv run pytest tests/test_units.py::test_name             # single test by node id
uv run pytest -k "rolling and not mujoco"                # single test by keyword
uv run pytest test_main.py                               # the rung-0 toy's tests (root, imports main.py)

# The non-pytest validation harness
uv run python verify_demos.py                            # PASS/WARN/FAIL per demo vs its physical story
```

Scenario/scene names live in `contact.mujoco_gen.SCENARIOS` / `.SCENES` (also listed in the README).

No linter/type-checker is wired into the project (no dev-dep, no config). The local `.ruff_cache/` and
`.mypy_cache/` (3.12, over `contact/`) show the author runs them ad hoc — `uvx ruff check` /
`uvx mypy contact` with default settings — not as a gate.

## Architecture

The **`contact/` package** is the single implementation of the method, validated against MuJoCo
physics; the module-to-THEORY-§ map is the table in the README. (A historical single-file literate
retelling, `contact_detection_standalone.py`, plus its bit-for-bit equivalence gate, lived here
through the `standalone-final` tag — retired when the goal shifted from one readable file to a
readable repository.) Do not edit tests to make an expectation check pass — fix the code.

### The pipeline (the narrow waist)

The implementation is one data flow, and most modules slot onto one stage of it:

```
generate(scenario)              MuJoCo truth factory; exposes only noisy poses, withholds contacts
  → observe(moving, support,    poses → support-relative ContactObservations (signed gap + 6-D twist).
      surface, cpl[, geometry]) THE core invariant — everything measured in the support's frame, so a
                                foot on a moving skateboard reads static. Geometry resolver is pluggable.
  → ContactDetector().detect()  the generative-HMM estimator (§4–§8): EM-calibrated, emissions→HMM/HSMM
  → report.score / print_report score the posterior against the withheld truth (§9)
detect_scene(scene)             the single-pair detector lifted to a contact graph + active-set posterior (§8)
```

### Engine / science split

Inference **engines** are generic and contact-free: `HMM` / `SemiMarkovHMM` are `TemporalSmoother`s
that turn a per-frame log-emission matrix into a smoothed posterior + MAP path. The contact **science**
they consume is built in three encapsulated layers:
`Density` primitives (frozen dataclasses — `Normal1D`, `SplitNormalGap`, `OffsetMagnitude1D/2D`,
`MixZero1D/2D`, `UniformClearance`, … — each a proper *unit-mass* log-density wrapping the `_log_*`
math); the six `MODES`, each now PURELY a kinematic *composition* of `Density`s over its channels
(`_compose`, strict left-to-right; `Rolling` is the one non-product mode — a coupling residual + `Z_res`
normalizer); and the emission itself as a **sum of factors** — `log_emissions` = `KinematicFactor` +
optional gated `ForceFactor` (the `EmissionFactor` family, `_sum_emissions`), with the scene grid the
analogous `SubsetFactor` sum (evidence + optional energy/balance). Keep that boundary — new contact
physics goes in the densities/modes/factors; new temporal machinery goes in the engine.

### DESIGN.md generalization (capability-driven, regression-locked)

`DESIGN.md` extends the validated flat-floor detector *without disturbing it*, as **optional gated
factors** fed by declared capabilities (`contact/capabilities.py`, `detect_pair`):
- **Geometry fidelity** — a `ContactGeometry` resolver chosen per declared shape, behind the single
  `observe(..., geometry=…)` waist: `FlatRegion` (default) / `SpherePlane` / `SphereSphere` / `BoxPlane`
  / `MeshPlane` / `MeshConvex` (`contact/geometry_resolvers.py`, `mesh_collision.py`).
- **Force** — an optional `normal_force` channel + per-state force emission, fed by a measured sensor
  or the inferred virtual sensor (`dynamics_id.infer_normal_force`); recovers contacts kinematics can't
  see (e.g. Newton's-cradle clacks).

**Invariant:** an empty capability declaration must reproduce the kinematic flat-floor detector
byte-for-byte — now **structural**, not merely tested: the emission is a sum of factors and an absent
capability contributes the `ZERO` identity (see the engine/science split). Capability→resolver and
→force-mode dispatch is a registry (`capabilities.py`: `SHAPE_RESOLVERS` / `FORCE_PREPARERS`),
mirroring `MODES`. Frontier features (particle-filter active sets, HDP-HMM mode discovery, emission
tempering, contact-implicit inverse dynamics) are **off by default and non-regressing**.

### Rungs

`main.py` is "rung 0" — the original single-file chi-squared toy that THEORY.md critiques and
supersedes, kept for contrast with its own `test_main.py`. It is *not* the current method; the real
pipeline is everything above.

## Validation philosophy

Tests are four layers: unit (`tests/test_*` — HMM recursions, emission densities, relative-frame
geometry, differentiation), integration, **expectation checks** (`contact/verification.py`) that
encode each demo's physically-expected contact/mode *story* (e.g. `push_to_slide` must go
static→sliding; `moving_support` must read static despite ~1.4 m of world motion) and assert the
detection matches it — not merely a passing IoU — and **executable-derivation checks**
(`tests/test_density.py`, `tests/test_invariants.py`) that turn THEORY's prose claims into
assertions: each `Density` is proper (∫=1), the documented limit laws hold, `observe` is
support-relative, and rolling's `Z_res` is correct. `tests/test_synthetic.py` is the simulator-free
end-to-end run (analytic truth → observe → detect → score). When you change behavior, the
expectation checks are what catch regressions; run `uv run pytest` and `verify_demos.py`.
