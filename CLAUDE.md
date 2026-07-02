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
section number throughout; `docs/theory-long.md` is the gentle long form), then
[`DESIGN.md`](DESIGN.md) (the capability-driven generalization; its build record is
`docs/design-history.md`), then [`README.md`](README.md) (the two-package module map + demos).
Don't restate their content — defer to them.

## Environment & commands

`uv`-managed, Python 3.12 (`.python-version`), installed editable (hatchling), so `import contact` /
`import oracle` work from anywhere. `uv run …` auto-syncs from `uv.lock`. Heavyweight deps —
`mujoco` (truth oracle), `coal` (convex collision), `cvxpy`+`clarabel` (inverse-dynamics solves),
`matplotlib` — load only on the code paths that use them; `import contact` is numpy/scipy(+markovlib)
only, and a test enforces that.

```bash
# Single-pair: generate (MuJoCo) → observe → detect → score → plot
uv run python detect.py --scenario rolling_ball
uv run python detect.py --scenario drop_rest_liftoff --stiffness 50000 --no-plot
uv run python detect.py --scenario push_to_slide --inverse-dynamics   # opt-in §8 north star

# Multi-body contact-graph scene
uv run python detect_scene.py --scene person_on_skateboard

# Synced side-by-side videos (writes under git-ignored media/)
uv run python viz.py person_on_skateboard --pairs        # add --force for force-mediated contacts

# Tests (unit + integration + expectation + invariant asserts; scoped via pyproject testpaths)
uv run pytest
uv run pytest tests/test_units.py::test_name             # single test by node id
uv run pytest -k "rolling and not mujoco"                # single test by keyword
uv run pytest rung0                                      # the rung-0 toy's tests only

# The non-pytest validation harness
uv run python verify_demos.py                            # PASS/WARN/FAIL per demo vs its physical story
```

Scenario/scene names live in the oracle registry (`oracle.SCENARIOS` / `oracle.SCENES`; also listed
in the README). The vendored `markovlib/` submodule has its own env and suite — run
`uv run pytest markovlib/tests` explicitly when bumping the pin.

No linter/type-checker is wired in as a gate; the author runs `uvx ruff check` / `uvx mypy` ad hoc.

## Architecture: two packages, one import law

**`contact/` is the estimator; `oracle/` is the experimenter.** `oracle` imports `contact`;
`contact` never imports `oracle` (nor mujoco/matplotlib at module scope) — THEORY §9's
"truth is withheld from the detector" as a package boundary rather than a discipline. The law is
machine-checked (`tests/test_import_law.py`), and the README's module tables are locked to the tree
(`tests/test_readme_table.py`). Do not edit tests to make an expectation or invariant check pass —
fix the code.

### The pipeline (the narrow waist)

```
oracle.generate(scenario)       MuJoCo truth factory; exposes only noisy poses, withholds contacts
  → contact.observe(moving,     poses → support-relative ContactObservations (signed gap + 6-D twist).
      support, surface, cpl     THE core invariant — everything measured in the support's frame, so a
      [, geometry])             foot on a moving skateboard reads static. Geometry resolver is pluggable.
  → ContactDetector().detect()  the generative-HMM estimator (§4–§8): EM-calibrated, emissions→HMM/HSMM
  → oracle.score / print_report score the posterior against the withheld truth (§9)
contact.detect_scene(scene)     the single-pair detector lifted to a contact graph + active-set posterior (§8)
```

`oracle.synthetic_drop_rest_liftoff()` is the simulator-free analytic truth factory — the smallest
end-to-end run (exercised by `tests/test_synthetic.py`).

### The truth factory (oracle)

`oracle/factory.py` is machinery only (simulate → label → extract; `generate` / `generate_scene`).
Every scenario/scene is a builder in a themed module (`scenarios_{core,motion,impacts}`,
`scenes_{graph,stacks,chains}`) returning a typed **`ScenarioSpec` / `SceneSpec`** (`oracle/specs.py`
— the named-entity contract as frozen dataclasses) and self-registering by name via
`oracle/registry.py`'s `@scenario` / `@scene` decorators; importing `oracle` assembles the registry.
New demo → new decorated builder; the machinery, `viz.py`, and the CLIs pick it up by name.

### Engine / science split (contact)

Inference **engines** are generic and contact-free: `HMM` / `SemiMarkovHMM` are `TemporalSmoother`
adapters over the vendored **`markovlib/`** submodule (PyPI `marmo`; developed in its own repo) —
their docstrings keep the recursions on the page as the *specification* of what markovlib computes.
The contact **science** they consume is three encapsulated layers: `Density` primitives (frozen
dataclasses, each a proper *unit-mass* log-density); the six `MODES`, each purely a kinematic
*composition* of `Density`s over its channels (`Rolling` is the one non-product mode — a coupling
residual + `Z_res` normalizer); and the emission as a **sum of factors** — `log_emissions` =
`KinematicFactor` + optional gated `ForceFactor`, with the scene grid the analogous `SubsetFactor`
sum. Keep that boundary — new contact physics goes in the densities/modes/factors; new temporal
machinery goes in markovlib.

### DESIGN.md generalization (capability-driven, structurally regression-safe)

Optional gated factors fed by declared capabilities (`contact/capabilities.py`: `Capabilities`,
`detect_pair`, `value_of_information` — all exported from `contact`):
- **Geometry fidelity** — a `ContactGeometry` resolver per declared shape behind the single
  `observe(..., geometry=…)` waist: `FlatRegion` (default) / `SpherePlane` / `SphereSphere` /
  `BoxPlane` / `MeshPlane` / `MeshConvex`.
- **Force** — an optional `normal_force` channel + per-state force emission, measured or inferred
  (`inverse_dynamics.infer_normal_force`); recovers contacts kinematics can't see (cradle clacks).

**Invariant:** an empty capability declaration reproduces the kinematic flat-floor detector — a
structural fact (absent capability = the `ZERO` factor identity), not merely a tested one.
Dispatch is registries (`SHAPE_RESOLVERS` / `FORCE_PREPARERS`), mirroring `MODES`. Frontier
features (particle-filter active sets, HDP-HMM mode discovery, emission tempering, contact-implicit
inverse dynamics) are **off by default and non-regressing**.

### Rungs

`rung0/` holds the original single-file chi-squared toy THEORY.md §0 critiques, with its own tests —
kept for contrast; it is *not* the current method. (A retired intermediate rung — the single-file
literate telling of the whole method plus its bit-for-bit equivalence gate — is preserved at the
`standalone-final` tag.)

## Validation philosophy

Tests are five layers: **unit** (HMM recursions, emission densities, relative-frame geometry,
differentiation), **integration**, **expectation checks** (`oracle/verification.py`) that encode
each demo's physically-expected contact/mode *story* (e.g. `push_to_slide` must go static→sliding;
`moving_support` must read static despite ~1.4 m of world motion) and assert the detection matches
it — not merely a passing IoU — **executable-derivation checks** (`tests/test_density.py`,
`tests/test_invariants.py`) that turn THEORY's prose claims into assertions (every `Density` is
proper (∫=1), the limit laws hold, `observe` is support-relative, rolling's `Z_res` is correct), and
**organization invariants** (`tests/test_import_law.py`, `tests/test_readme_table.py`) that keep the
package boundary and the README map honest. `tests/test_synthetic.py` is the simulator-free
end-to-end story. When you change behavior, run `uv run pytest` and `uv run python verify_demos.py`;
those two are the gate.
