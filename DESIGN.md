# DESIGN — A Capability-Driven Contact Estimator

> Forward-looking architecture for generalizing the detector beyond the validated
> "known point on a body, resting/sliding/rolling on a flat, non-rotating floor" regime.
> Companion to `THEORY.md` (the first-principles model) — this document is the *engineering
> path* from today's single-point/flat-plane/kinematics-only observer to an estimator that
> uses **whatever the user can provide** and degrades gracefully when they can't.

## Build status

Both architecture axes are now implemented end-to-end, each **gated + additive** with the
validated kinematic/flat-floor floor **bit-identical** throughout (regression-locked by a
per-edge observation oracle):

- **Phase 0** — `ContactGeometry` narrow waist + `FlatRegion` (bit-identical refactor). ✅
- **Phase 1** — `SpherePlane` / `SphereSphere` (position-derived normal). ✅ ball-ball 7 phantom
  impacts → 1 (vₙ 24→1.04 m/s); cradle closing velocity 0.04→0.21 m/s.
- **Phase 2** — `BoxPlane` nearest-corner (multi-point) + analytic migrating-contact velocity.
  ✅ tumbling gap 225→~0 mm at bounces, impact frames 5 = truth, contact-IoU 0.96.
- **Phase 4a** — measured-force channel + per-state force emission (free half-normal /
  **contact mixture** / impact spike). ✅ cradle clacks 0→2 with contact recall preserved.
- **Phase 4b** — inferred-force **virtual sensor** (`dynamics_id.infer_normal_force`). ✅
  drop_rest corr 0.94 / 5% err, graceful `None` when unsupported. *Caveats:* noisier than a
  measured sensor (acceleration double-diff); single rigid body only (articulated cradle out
  of scope).
- **Phase 5** — capability registry + value-of-information (`contact/capabilities.py`:
  `Capabilities`, `detect_pair`, `value_of_information`). ✅ purely additive (no detection-path
  module imports it → floor bit-identical by construction). Registry: declaring the sphere
  shape selects `SphereSphere` → ball-ball impacts 7→1. VoI ranks by **MAP-change** (entropy is
  over-confident, ~0 even when wrong) → cradle ranks `force` top + emits the "unobservable from
  kinematics → force" guidance. *Caveat:* VoI is a heuristic (counterfactual MAP-change), and
  the bare-pair API can't do `force='inferred'` (whole-body; use the scene-level virtual sensor).
- **Phase 3** — convex **mesh** geometry (`contact/mesh_collision.py`: GJK distance + EPA
  penetration; `MeshPlane` / `MeshConvex` resolvers; `mesh_plane`/`mesh_mesh` in the registry).
  ✅ purely additive. `MeshPlane(box corners) == BoxPlane` **exactly** (0.00e+00); `MeshConvex`
  (icospheres) `≈ SphereSphere` **1.22 mm** separated, penetration **−0.0195 vs −0.020 analytic**.
  Penetration is **watertight for all convex polytopes** via a Separating-Axis close-out
  (`_penetration_sat`) on the touching/overlapping case: **coplanar box-vs-box penetration is now
  exact** (depth `-0.3000` for a 0.3 overlap; flush boxes read 0, not a spurious deep value),
  curved-mesh penetration unchanged. EPA remains only the fallback for a degenerate non-3-D cloud.

**The architecture is complete — no open frontier.** The full geometry fidelity ladder (flat →
sphere/box primitives → arbitrary convex mesh, with watertight penetration) and both observability
axes (geometry + force, measured *or* inferred) are implemented, unified behind the capability
registry with value-of-information, every piece gated/additive with the validated
kinematic/flat-floor path bit-identical throughout, and **permanently regression-locked** (160
tests). Everything in the plan is built and validated.

---

## 0. Thesis

> The user might have access to various things; we want to give them the **best and most
> robust** estimator using **what they can give us**.

Made precise:

- **One estimator, a fixed target, pluggable evidence.** The thing inferred — the per-frame
  contact state — never changes. Everything the user supplies enters as an **optional factor**
  (a likelihood or a prior) in the *same* Bayesian inference. Present → multiplies in and
  sharpens. Absent → not a factor; the posterior stays exactly as broad as the remaining
  evidence honestly warrants.
- **Best** = efficient fusion: use *all* available evidence, optimally combined; monotonic
  (more information can only sharpen, never degrade).
- **Robust** = graceful degradation + calibrated honesty: with the bare minimum (noisy poses)
  it still returns a *correct, if vaguer* answer; never hinges on one channel; each input is
  down-weighted by its declared noise so a bad sensor can't hijack it; and it *reports* the
  residual uncertainty instead of guessing.

**Observability is a function of the priors you happen to have.** The design makes "what
information is available" a first-class, introspectable input.

---

## 1. What is being estimated (the invariant target)

Unchanged by any capability:

- `contact_posterior[t]` — P(in contact) per body-pair (edge).
- `state_posterior[t, mode]` — distribution over {free, static, sliding, pivoting, rolling, impact}.
- `active_posterior[t, edge]` — which edges are simultaneously active (the contact graph).
- `impulses` / per-candidate `contact_force` — the force/impulse story (when observable).

This is the **narrow target**. Every capability below is just more (or sharper) evidence about it.

---

## 2. The two missing axes (and the ones already here)

| Axis | Today | Path |
| --- | --- | --- |
| **Geometry fidelity** | one flat plane, one fixed point per edge | ladder: flat → primitive → mesh |
| **Force observability** | none (kinematics only) | none → inferred (dynamics) → measured (F/T, tactile) |
| Measurement uncertainty | `meas_cov` → tempering (`use_uncertainty`) | already a gated factor — generalize the pattern |
| Material (μ, k, e) | `MaterialParams` hook **accepted but unused** in emissions | wire it as emission + compliance terms |
| Dynamics (mass/inertia) | `dynamics_id` exists, runs off **truth** `meta`, not wired to detection | wire as the *inferred-force* source + consistency |
| Energy / balance priors | `use_energy_prior` / `use_balance_prior` factors (exact path only) | already gated factors — fold into the registry |
| Contact-graph structure | `structure_inference` (exact 2^E + particle) | already present |

**Key realization from the codebase:** the "optional, capability-gated factor" pattern is
*already in use* in four places (tempering, energy, balance, compliance-force). The work is to
**(a) formalize that pattern into one registry**, and **(b) fill the two big gaps — per-frame
contact geometry and a force channel.**

---

## 3. The narrow waist: a `ContactGeometry` contract

### 3.1 What's actually wrong today (precisely)

`geometry.observe(moving, support, surface, contact_point_local, vel_smooth_time)`
(`geometry.py:365`) already does the right *kinematic* decomposition. Per frame it:

- rotates the **fixed local** `contact_point_local` by the moving body's quat → world point
  (`geometry.py:434`),
- rotates the **fixed local** `surface.normal` by the **support** body's quat → world normal
  (`geometry.py:440`),
- decomposes the relative twist into (gap, v_normal, v_tangent, omega_normal, omega_tangent).

The decomposition is fine. The defect is that **the contact point and normal are fixed
*local* specs.** For a flat, non-rotating floor that's exact. For a **sphere** (or any curved
or rotating support) it is wrong: a normal glued to the support's body frame **spins with the
support**, so projecting the real ~2.5 m/s relative velocity onto a whirling axis yields the
±24 m/s garbage and 6 phantom impacts seen on `ballA↔ballB`; on the cradle it cancels the real
closing velocity down to ~0.04 m/s and the clacks vanish. The *truth* avoids this because it
reads the **actual contact point and normal from the geometry every frame**
(`mj_geomDistance`, the candidate-corner attribution `mujoco_gen.py:806`).

### 3.2 The contract

Introduce a per-frame, world-frame contact description that `observe()` consumes:

```
ContactPoint  = { point: (3,), normal: (3,) unit, gap: float,
                  normal_sigma: float, gap_sigma: float }     # provenance/uncertainty
ContactFrame  = list[ContactPoint]                            # >1 for area/face contacts
ContactGeometry.resolve(pose_moving, pose_support) -> ContactFrame[T]   # one per recorded frame
```

`observe()` is refactored to: *ask the geometry for `(point, normal, gap)` per frame, then run
the existing twist decomposition unchanged.* The legacy `(surface, contact_point_local)` becomes
the configuration of **one** resolver (`FlatRegion`), so the refactor is bit-identical on every
current demo (regression lock — see §9 Phase 0).

### 3.3 The fidelity ladder of resolvers

| Resolver | Needs | World normal / point per frame |
| --- | --- | --- |
| **FlatRegion** (default, mesh-free) | a local plane + a tracked point (today's spec) | rotate local normal/point by support/moving quats — *exactly `observe()` today* |
| **SpherePlane** | radius of moving sphere; plane on support | normal = plane normal (world); gap = signed dist of center − r; point = foot |
| **SphereSphere** | both radii | **normal = (c_moving − c_support)/‖·‖ (world, position-derived)**; gap = ‖c−c‖ − r₁ − r₂; point on the line of centers |
| **BoxPlane / SphereBox / Capsule / Ellipsoid** | primitive params | closed-form nearest feature; **migrating** corner/face → fixes tumbling |
| **Mesh / SDF** | meshes (or signed-distance field) | GJK/EPA closest-point + normal + penetration |

All return the **same** `ContactFrame`, so everything downstream (emissions, HMM, active-set,
inverse dynamics) is untouched. `SphereSphere` alone turns `ballA↔ballB`'s "7 phantom impacts"
into the single real collision and restores the cradle's true closing velocities.

### 3.4 Multi-point contacts unify with what's already here

A box face is **four** contact points, not one. `resolve()` returning a *list* is exactly the
candidate-corner representation the codebase already produces truth-side
(`cand_points_local`, `cand_gap (K,T)`, `cand_normals_local`, `mujoco_gen.py:719–732`) and that
`dynamics_id.contact_wrench_map` already consumes (`dynamics_id.py:363`). So the resolver becomes
the **detection-side producer** of the point set the inverse-dynamics solver already eats — one
source of contact geometry for *both* the kinematic `observe()` and the dynamics. Per-point
evidence then flows through the existing active-set machinery (`structure_inference`).

### 3.5 Provenance feeds uncertainty (robustness)

Each `ContactPoint` carries `normal_sigma`/`gap_sigma`. A crude `FlatRegion` approximation
declares *large* normal uncertainty; an exact mesh declares *small*. These feed the **existing**
measurement-tempering path (`obs.meas_cov` → `uncertainty.emission_tempering`,
`uncertainty.py:156`), so a low-fidelity geometry is automatically *trusted less* rather than
over-committed. This is how "approximate flat regions" stay safe.

---

## 4. The capability registry

Declared per body and per edge; introspectable; logged.

```
ShapeDescriptor = None | Primitive(kind, params) | Mesh(verts, faces | sdf)
InertialParams  = None | { mass, inertia(3x3), com_local }          # already in dynamics_id
Sensors         = { force: None|Measured|Inferred, tactile: None|Array, ... }
Material        = None | MaterialParams                              # already in config
Capabilities(per edge) = { shapeA, shapeB, inertial, sensors, material, meas_cov? }
```

A **selector** maps `Capabilities → (ContactGeometry, [EvidenceFactor])`:

- pick the richest resolver the `(shapeA, shapeB)` pair supports, else `FlatRegion`;
- enable each evidence factor whose inputs are present (else it's simply absent — a no-op).

The result records *what was used and what was missing* (drives §8).

---

## 5. Evidence factors (the one unifying interface)

Today the optional factors are ad-hoc. Formalize them as a single protocol:

```
EvidenceFactor.contribute(obs, caps) -> log_factor   # (T, S) for per-state, (T, n_subsets) for active-set
                                                      # all-zeros == "inactive / unknown" (no-op)
```

The detector/graph **sums all available factors in log-space** — exactly what `graph.py:573`
already does for energy/balance. Catalog and code seam for each:

| Factor | Evidence | Code seam (exists?) |
| --- | --- | --- |
| Kinematic emission | poses → gap/twist | `emissions.log_emissions` (✅ always on) |
| Measurement tempering | `meas_cov` | `uncertainty.apply_tempering` (✅, gated `use_uncertainty`) |
| **Geometry uncertainty** | resolver `normal_sigma/gap_sigma` | feeds the tempering path (🔜 from §3.5) |
| **Material** | μ, stiffness, restitution | `emissions` builders take `material` **but ignore it** (🔜 wire) |
| **Force / tactile (measured)** | sensor stream | **new** per-state force term in each builder (🔜 §6) |
| **Force (inferred)** | mass/inertia + dynamics | `dynamics_id.solve_contact_implicit` (✅ exists, 🔜 wire to detection) |
| Energy consistency | masses + active set | `consistency.energy_log_factor` (✅, gated `use_energy_prior`) |
| Balance consistency | support polygon | `consistency.balance_log_factor` (✅, gated `use_balance_prior`) |
| Contact-graph structure | proximity / dwell | `structure_inference` (✅) |

Every factor is **normalized** (keep the Gaussian constants — `emissions.py:10`) so cross-state
likelihood *ratios* stay calibrated, and **no-op when absent** so the posterior never lies.

---

## 6. The force channel (the highest-value gap)

The cradle is the canonical case: momentum hops through **touching** balls as force impulses with
~0 relative motion — *unobservable from kinematics by theorem* (`THEORY.md` §6–§8). Two changes:

### 6.1 Carry force as an optional observation
Extend `ContactObservations` with `normal_force: (T,) | None` (and later
`tangent_force`/`contact_location` for tactile) — exactly mirroring the existing optional
`meas_cov` field (`types.py:78`). Absent → no factor.

### 6.2 A per-state force emission
Add one term per state builder (`emissions.py:292–508`, before each `return`), gated on
`obs.normal_force is not None`:

- **free** → `N(force; 0, σ₀²)` (≈ 0 force; a separated body carries none),
- **contact modes** → force ≥ 0 consistent with the twist and the **friction cone**
  (‖f_t‖ ≤ μ f_n, from `material`) — a Signorini/cone *prior* that also sharpens mode choice,
- **impact** → a force *spike* density (large, brief).

This is where a cradle clack — invisible kinematically — becomes a decisive force pulse.

### 6.3 Two sources, one factor
- **Measured**: a force/torque or tactile sensor populates `normal_force` directly.
- **Inferred** ("virtual force sensor"): `dynamics_id` already implements contact-implicit
  inverse dynamics — `required_wrench` (Newton-Euler, `dynamics_id.py:285`) +
  `contact_wrench_map` (`:363`) + a per-frame convex solve with Signorini + Coulomb cone
  (`_solve_frame`, `:501`). Today it runs from **truth** `meta["candidates"]`. Wire it to the
  **detection side**: the §3 `ContactGeometry` resolver supplies the candidate points/normals,
  `InertialParams` come from the registry, and the recovered per-candidate normal force becomes
  the `normal_force` channel. → the cradle is solvable **without a physical sensor**, given mass
  properties.

### 6.4 The bootstrap subtlety
Inferred force needs an **active set** (which candidates carry force), which needs detection — a
loop. Resolve by (a) seeding the complementarity mask from the kinematic detector's contact
posterior (`dynamics_id` already uses a gap-based mask, `:709`), then (b) iterating force ↔ state
EM-style to a fixed point. Note inferred force is a **whole-body** quantity (the wrench is shared
across the body's contacts), so it must run at the **scene/body** level and be *attributed back*
to edges via `contact_wrench_map`, not per-edge in isolation.

---

## 7. Graceful degradation & calibration (the "robust" guarantees)

1. **Floor is the validated baseline.** `FlatRegion` + no extra factors == today's pipeline,
   bit-for-bit (Phase 0 regression lock). Nothing the user *doesn't* provide can break it.
2. **Absence is honest.** Every factor no-ops when its inputs are missing; the posterior widens
   rather than fabricating a mode.
3. **Bad input can't dominate.** Declared `normal_sigma`/`gap_sigma`/`meas_cov` and sensor
   covariance down-weight noisy or low-fidelity evidence through the tempering path.
4. **More input never hurts.** Adding a (correctly-weighted, normalized) factor is a Bayesian
   update — it can only concentrate the posterior toward the truth in expectation.

---

## 8. Value of information (a robustness *bonus*)

Because the estimator (a) carries posterior uncertainty and (b) can run with/without each factor,
a thin diagnostic layer can tell the user **what to provide next**:

- per-frame ambiguity = entropy of `state_posterior`;
- counterfactual gain: re-run a window with a *hypothesized* factor (e.g. a force prior) and
  report the entropy drop;
- canned guidance: "frames 120–160 are sliding-vs-rolling ambiguous → a tactile pad resolves it";
  "this edge is force-transfer (cradle) → kinematics can't see it; add force (measured or inferred)."

Not core inference — but it closes the loop on "best with what you can give us."

---

## 9. Migration plan (phased, each gated & zero-regression)

| Phase | Deliverable | Unlocks | Risk |
| --- | --- | --- | --- |
| **0** | `ContactFrame`/`ContactGeometry` waist; `FlatRegion` wraps current spec; refactor `observe()`; capability-registry skeleton | bit-identical baseline + the seam | low (pure refactor; lock with 138 tests) |
| **1** | `SpherePlane`, `SphereSphere`; re-point ball-ball + cradle edges; geometry-provenance σ | 7 phantom impacts → 1; cradle closing velocities recovered | low (self-contained) |
| **2** | multi-point + nearest-feature primitives (`BoxPlane`, `Capsule`, `Ellipsoid`); unify with candidate-corner machinery | tumbling migrating-corner; box faces | med (multi-point emissions via `structure_inference`) |
| **3** | `Mesh`/`SDF` resolver (GJK/EPA, broad-phase, caching) | arbitrary shapes | med-high (collision geometry) |
| **4** | force channel: `ContactObservations.normal_force`; per-state force emission; wire measured **and** inferred (`dynamics_id`) sources; μ/Signorini priors from `material` | cradle / force-transfer; sharper modes | high (bootstrap EM; whole-body attribution) |
| **5** | unify tempering/energy/balance/force/material under one `EvidenceFactor` registry; value-of-information diagnostics | clean extensibility; VoI | low-med (refactor) |

Gate: every phase ships behind capability flags, defaults to the Phase-0 baseline, and must keep
the full suite + `verify_demos.py` green before the next.

---

## 10. Data-contract changes (precise, additive, backward-compatible)

- **New:** `ShapeDescriptor`, `ContactPoint`/`ContactFrame`, `ContactGeometry` (+ resolver
  classes), `Capabilities`/`Sensors`.
- **Extend `ContactObservations`** (`types.py:78`): `+ normal_force: ndarray|None = None`
  (and later tactile fields) — mirrors the optional `meas_cov`.
- **Extend `ContactEdge`** (`types.py:193`): `+ geometry: ContactGeometry | None`. When `None`,
  fall back to `surface` + `contact_point_local` (→ `FlatRegion`). **Legacy specs keep working.**
- **Extend `DetectorConfig`**: `+ force: ForceEmissionParams` (per-state force σ's, spike model),
  `+ geometry: GeometryParams`. Wire the already-present `material` arg through the emission
  builders.
- **Reuse as-is:** `InverseDynamicsParams`, `InertialParams`, `meas_cov`, `use_energy_prior`,
  `use_balance_prior`, `use_uncertainty`, `structure_inference`.

---

## 11. Risks & open questions

- **Inferred-force bootstrap** (active set ↔ detection circularity) — EM/iterate; seed from
  kinematic posterior. Convergence/identifiability on indeterminate sets (the §7 null space →
  compliance regularizer already studied in `dynamics.observability_demo`).
- **Whole-body vs per-edge force.** Inferred force is a body-level wrench; must attribute across
  a body's edges (scene-level pass), not per-edge. Pipeline placement TBD.
- **Primitive×primitive combinatorics** — dispatch table keyed on (kind, kind); unsupported pairs
  fall back to `FlatRegion` (never worse than today).
- **Mesh cost** — broad-phase + SDF caching; only pay it when a mesh is actually provided.
- **Multi-point emission semantics** — per-point modes vs an aggregated edge mode; lean on the
  existing per-candidate active-set inference.
- **Force-density calibration** — per-state force σ's, the impact spike model, tactile location
  likelihood; needs synthetic-truth calibration.

---

## 12. Testing strategy

- **Regression lock:** `FlatRegion` produces bit-identical `ContactObservations` on all demos
  (assert against current outputs) — guarantees the validated floor is untouched.
- **Per-resolver geometry unit tests:** sphere-sphere normal == center line; analytic gaps for
  sphere-plane / box-plane; nearest-feature correctness under rotation.
- **Phantom-impact regression:** `ballA↔ballB` atoms 7 → 1; `v_normal` one sign change, peak
  ≈ real closing speed.
- **Cradle:** inferred-force path recovers the clacks (impact frames detected ≈ truth).
- **Tumbling:** nearest-corner geometry lights the impact mode on each corner strike.
- **Force channel:** feed MuJoCo truth force as a "sensor" → contact IoU / mode accuracy improve;
  ablate to confirm graceful no-op when absent.
- **Value-of-information:** entropy-drop sanity on a hand-built ambiguous case.

---

### One-paragraph summary

Keep one Bayesian estimator with a fixed target (the contact state) and feed it through a single
narrow waist — a per-frame, world-frame `ContactFrame` produced by a **resolver chosen for
whatever shape knowledge exists** (flat → primitive → mesh) — plus a set of **optional evidence
factors** (measurement uncertainty, material, energy/balance, and a **force channel** that is
either measured or *inferred* from dynamics) that are summed in log-space and **no-op when
absent**. The validated flat-floor/kinematic path is the guaranteed floor; every capability the
user can provide is a sharper prior or an extra factor on top. Most of the scaffolding
(tempering, consistency priors, inverse dynamics, candidate-corner machinery, structure
inference) already exists — the work is to unify it behind a capability registry and fill the two
gaps: real per-frame contact geometry, and the force factor.

---
---

# PART II — Deep dives on the consequential problems

These are the four places where a sloppy decision now costs the most later: the force
**emission math** (must be proper densities or it silently biases the HMM), the **inferred-force
bootstrap** (circularity + whole-body attribution), **multi-point** semantics (per-point vs
aggregate), and the **primitive geometry math** (the thing that actually fixes the bugs).

## A. The force-emission factor (math)

The emission module's invariant (`emissions.py:8–13`): every per-state column is a **proper,
normalized density over the same observation space**, constants kept, so cross-state *log-ratios*
stay calibrated. The force factor must obey this. Force is physically `f_n ≥ 0`, so all force
densities live on `[0, ∞)` (a shared support across states).

**Normal-force density per state** (added as one more `lp = lp + …` term in each builder):

| State | Density on `f_n ∈ [0,∞)` | Why |
| --- | --- | --- |
| FREE | **Half-normal** `HN(σ_free)`: `log = ½log(2/π) − log σ_free − ½(f/σ_free)²` | separated ⇒ no load; mode at 0 |
| STATIC / SLIDING / PIVOTING / ROLLING | **Mixture** `w·HN(σ_free) + (1−w)·R(s_load)` | a contact may be **unloaded (a resting touch, f≈0)** *or* loaded — the density must allow both |
| IMPACT | **Rayleigh** `R(s_imp)`, `s_imp ≫ s_load` | a brief, large force spike; an impact is never unloaded ⇒ zero-at-0 is correct here |

**Corrected during Phase-4a build (the demonstration earned it).** The original spec made
sustained contact a pure **Rayleigh** (zero density at `f=0`). That is wrong for a
**touching-but-unloaded** contact: on the cradle, b3↔b4 is 96% resting touch with *median force
0.00 N* (hanging balls rest together carrying no load), and a Rayleigh-vs-FREE-half-normal
comparison pulled all 436 such frames to FREE — **collapsing contact recall 452 → 16.** A force
sensor measures **load**, and `f=0` is genuinely *ambiguous* (consistent with FREE *or* a resting
contact). So the sustained-contact density must be a **mixture** whose unloaded component
(`w·HN(σ_free)`) makes the contact-vs-FREE log-ratio at `f≈0` just `log w` (a small constant) —
near-**neutral**, so the **gap decides an unloaded touch** — while the loaded `R(s_load)` component
still pulls decisively to contact when force is appreciable. Result with the fix: impact 0→2
(clacks recovered) **and** contact recall preserved (835, no collapse), bit-identical when off.
Net rule: **high force ⇒ contact/impact; `f≈0` ⇒ uninformative (let the gap rule).** (`w = w_unloaded`,
default 0.5.)

**Tangential force (only if a tactile/F-T sensor gives `f_t`)** discriminates *within* contact via
the **friction cone**. Let `ρ = ‖f_t‖ / (μ f_n)` (μ from `material`):
- STATIC → `ρ` density on `[0,1)` favouring `ρ < 1` (strictly inside the cone),
- SLIDING → `ρ` peaked near `1` (on the cone boundary — kinetic friction),
- others → broad.

This is the *measured* version of the cone cross-check `dynamics.friction_stick_slip` already does
kinematically (`dynamics.py:123`) — with real tangential force it becomes direct evidence.

**Calibration of `s_load`, `σ_free`, `s_imp`** (Phase-4 sub-task, in priority order):
1. **Inertials known** → `s_load ≈ m·g` (static weight), `σ_free ≈` sensor noise floor,
   `s_imp ≈` a multiple of `s_load`.
2. **Material/scale only** → normalize the force observation by its own robust running scale
   (median of positive force) so the densities are *dimensionless* (`s_load = 1`); robust to
   unknown absolute calibration.
3. **EM** → treat `s_load` like `gap_bias`: re-estimate from the contact-responsibility-weighted
   mean force inside the existing EM loop (`model.py:274–286`).

**New config** `ForceEmissionParams { sigma_free, s_load, s_impact, use_tangential, cone_sharpness }`.
Gate the whole term on `obs.normal_force is not None` (no channel ⇒ no factor ⇒ today's behavior).

## B. The inferred-force pass + bootstrap

`dynamics_id` is a working contact-implicit solver; the open questions are *circularity* and
*whole-body attribution*. Both resolve cleanly:

**Circularity is milder than it looks.** `solve_contact_implicit` takes the active set from a
**gap-based complementarity mask** (`force only where |gap| < complementarity_gap`,
`dynamics_id.py:709`), *not* from the detector. So a first inferred-force pass needs **no**
detector output — gaps come from the geometry resolver. Sequence per recorded clip:

```
1. geometry.resolve(edge) -> ContactFrame[T]            # candidate points + normals + gaps (§D)
2. group edges by MOVING BODY                            # force is a whole-body quantity
3. for each body with InertialParams:
     w   = required_wrench(pose_body, inertials)         # (T,6)  Newton-Euler   [dynamics_id.py:285]
     G   = contact_wrench_map(pose_body, pts, normals)   # (T,6,3K)              [:363]
     res = solve_contact_implicit(w, G, gaps, mu, params)# Signorini + cone      [:625]
     # res.contact_normal_force : (T,K)
4. attribute each candidate k's force to the edge that owns it
     -> obs[edge].normal_force = sum of that edge's candidates' f_n     # (T,)
5. per-edge ContactDetector.detect(obs)                 # now sees the force channel (§A)
```

**Whole-body attribution is mandatory:** `required_wrench` is the body's *net* external wrench, so
the solve must pool **all** of a body's contacts (across all its edges) and then split the result
back. This is why the inferred-force pass lives at the **scene/body level, before** the per-edge
detect — a small reorder of `detect_scene` (§ Part III), reducing exactly to today when no body
has inertials.

**The EM refinement (optional, Phase-4b)** only matters for *statically indeterminate* sets (a box
on 4 corners: 12 unknowns, 6-DOF balance → a null space, `dynamics.observability_demo`). There the
gap mask under-determines the distribution; iterate
`detect → active-set posterior → re-mask solver → re-solve → re-detect` and lean on the
**compliance regularizer** (`force_regularization`, and `normal_force_from_penetration` when
stiffness is known) to pick the minimum-norm/compliance-consistent solution. For determinate
single-contact cases (the cradle, two balls) **one pass suffices** — no loop.

**Failure handling:** `solve_contact_implicit` returns a `wrench_residual` (`‖Gf − w‖`). A large
residual means the inferred force does not explain the motion (bad inertials, missing contact,
articulation) → **declare the inferred-force factor low-confidence** (inflate `σ`/down-weight via
tempering) rather than trust it. This keeps a wrong dynamics model from corrupting the estimate.

## C. Multi-point: aggregate for *mode*, per-point for *force*

A box face is K contact points but **one** kinematic contact mode. Resolution:

- **Kinematic `observe()` uses an aggregated representative**: the active-set-weighted centroid
  point and the (shared) face normal. One `ContactObservations` per edge → the per-edge mode
  detector is **unchanged**. (The truth side already aggregates per-edge this way,
  `_edge_frame_truth` `mujoco_gen.py:1414`.)
- **The force/structure layer uses the full K points**: `contact_wrench_map` needs all K (force
  distribution); per-candidate activity flows through the **existing** active-set inference
  (`structure_inference`, which already runs over candidates). So "per-point vs aggregated" is not
  a dilemma — it's a **split by purpose**: aggregate for the kinematic mode, keep all points for
  force and the active set. `ContactFrame` carries the list; `observe()` reduces it; the dynamics
  pass consumes it whole.

## D. Primitive geometry (the math that fixes the bugs)

All resolvers return world-frame `(point, normal, gap)` per frame; the normal must come from
**positions** for curved supports (not a body-fixed local vector — that's the spin bug).

- **FlatRegion** (default): `n_w = R_sup · n_local`, `p_w = sup_pos + R_sup·surf_pt`,
  `gap = (mov_pt − p_w)·n_w`, `point = mov_pt`. **Identical to `observe()` today** (`geometry.py:434–444`).
- **SpherePlane** (moving sphere r, support plane): `n_w =` world plane normal;
  `gap = (c − p_w)·n_w − r`; `point = c − r·n_w`. (`c` = sphere center = body origin.)
- **SphereSphere** (radii r₁ moving, r₂ support): `d = c₁ − c₂`, `dist = ‖d‖`,
  **`n_w = d/dist`** (position-derived ⇒ *no spin artifact*), `gap = dist − r₁ − r₂`,
  `point = c₁ − r₁·n_w` (the **moving** sphere's surface point — see note). ← this single
  resolver turns `ballA↔ballB`'s 7 phantom impacts into 1 (verified: peak |vₙ| 24→1.04 m/s,
  12→1 sign changes).
  **Note (corrected during build):** the contact `point` must be **moving-pinned**
  (`c₁ − r₁·n`), not support-pinned (`c₂ + r₂·n`). At contact the two coincide, but
  `observe()` recovers the *moving* body's velocity by differentiating this point's
  trajectory — a support-pinned point would track the support and **zero out the relative
  closing velocity** (0 impulses, no cradle recovery). Mirrors `SpherePlane`'s `c − r·n`.
- **BoxPlane** (8 corners, plane): per corner signed dist `dᵢ = (cornerᵢ − p_w)·n_w`;
  `gap = min dᵢ`; contact points = corners with `dᵢ ≤ gap + ε` (1 = tipping, 2 = edge, 4 = face);
  `normal = n_w`. ← migrating contact ⇒ fixes tumbling, and is naturally multi-point (§C).
- **SphereBox / Capsule / Ellipsoid** (Phase 2): closed-form or 1-D iterative closest point.
- **Mesh/SDF** (Phase 3): GJK (separation) + EPA (penetration); witness points → `point`,
  simplex → `normal`. Broad-phase + SDF cache; only paid when a mesh is supplied.

**Provenance σ per resolver** (feeds `meas_cov` → tempering, §3.5): `FlatRegion` declares a large
`normal_sigma` (it's an approximation); exact primitives declare small; mesh small. So fidelity
auto-modulates trust.

---

# PART III — Precise implementation plan

## III.1 New / extended data contracts (`types.py`, `config.py`)

```python
# --- shape & capability declaration -------------------------------------------------
@dataclass(frozen=True)
class Primitive:
    kind: Literal["sphere", "plane", "box", "capsule", "ellipsoid"]
    params: dict          # sphere:{r}; box:{half_extents(3)}; capsule:{r,h}; ...
@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray  # (V,3) body-local   (or)
    sdf: Callable | None  # body-local signed-distance field
ShapeDescriptor = Primitive | Mesh | None

@dataclass(frozen=True)
class InertialParams:     # already used by dynamics_id; promote to a shared type
    mass: float; inertia: np.ndarray; com_local: np.ndarray
@dataclass(frozen=True)
class Sensors:
    normal_force: Literal["none", "measured", "inferred"] = "none"
    tactile: bool = False
@dataclass(frozen=True)
class Capabilities:       # per edge (or resolved from per-body declarations)
    shape_moving: ShapeDescriptor = None
    shape_support: ShapeDescriptor = None
    inertial: InertialParams | None = None
    sensors: Sensors = Sensors()
    material: "MaterialParams | None" = None

# --- the narrow waist ----------------------------------------------------------------
@dataclass
class ContactPoint:
    point: np.ndarray      # (3,) world
    normal: np.ndarray     # (3,) world unit (support -> moving)
    gap: float             # signed; <0 penetration
    normal_sigma: float    # provenance / fidelity (m or rad)
    gap_sigma: float
ContactFrame = list[ContactPoint]          # >1 => area/face contact

class ContactGeometry(Protocol):
    def resolve(self, moving: PoseTrajectory, support: PoseTrajectory) -> list[ContactFrame]:
        """One ContactFrame per recorded frame (length T)."""

# --- extended existing contracts (ADDITIVE, backward-compatible) ---------------------
# ContactObservations  += normal_force: np.ndarray | None = None      # mirrors meas_cov
#                       += tangent_force: np.ndarray | None = None     # (T,2), tactile
# ContactEdge          += geometry: ContactGeometry | None = None      # None -> FlatRegion(surface, cpl)
# DetectorConfig       += force: ForceEmissionParams
#                       += geometry: GeometryParams      # eps for multi-point, fidelity sigmas
@dataclass
class ForceEmissionParams:
    sigma_free: float = 0.05      # normalized; or N noise floor when inertials known
    s_load: float = 1.0           # Rayleigh scale for sustained contact (EM/inertial-set)
    s_impact: float = 4.0
    use_tangential: bool = False
    cone_sharpness: float = 8.0
```

## III.2 The `observe()` refactor (Phase 0, zero-regression)

```python
def observe(moving, support, geometry: ContactGeometry, vel_smooth_time=0.05) -> ContactObservations:
    frames = geometry.resolve(moving, support)          # list[ContactFrame], length T
    point, normal, gap, nsig, gsig = _aggregate(frames) # (T,3),(T,3),(T,),(T,),(T,)  [§C reduce]
    # ... EXISTING twist decomposition, verbatim, using `point`,`normal` instead of the
    #     rotated local spec ...
    meas_cov = _geometry_meas_cov(nsig, gsig)            # provenance -> tempering (§3.5)
    return ContactObservations(..., meas_cov=meas_cov)

# back-compat shim so every current call site & demo is untouched:
def observe_legacy(moving, support, surface, contact_point_local, vel_smooth_time=0.05):
    return observe(moving, support, FlatRegion(surface, contact_point_local), vel_smooth_time)
```
`FlatRegion.resolve` reproduces `geometry.py:434–444` exactly ⇒ **bit-identical** observations.
Acceptance: snapshot every demo's `ContactObservations` before/after; assert equal. Lock with the
138-test suite + `verify_demos.py`.

## III.3 Force emission (Phase 4)

```python
# emissions.py — new proper densities on [0, inf)
def _log_half_normal(f, sigma): ...      # FREE
def _log_rayleigh(f, scale): ...         # contact / impact
# each builder, before `return lp`, gated:
if obs.normal_force is not None:
    lp = lp + _force_term(obs.normal_force, state, fp)      # fp: ForceEmissionParams
    if fp.use_tangential and obs.tangent_force is not None:
        lp = lp + _cone_term(obs, state, material, fp)
```
No new assembly seam — `log_emissions` (`emissions.py:533`) already stacks builders; the term
lives inside each builder exactly where the kinematic terms are.

## III.4 Inferred-force pass + `detect_scene` reorder (Phase 4)

```python
# graph.py detect_scene, NEW step between geometry-resolve and the per-edge detect loop:
def _infer_forces(scene, edges, caps, cfg):
    for body, body_edges in _group_by_moving_body(edges):
        ip = caps[body].inertial
        if ip is None: continue
        pts, normals, gaps, owner = _collect_candidates(body_edges)   # from each edge's ContactFrame
        w   = required_wrench(scene.bodies[body], ip, cfg.inverse_dynamics)
        G   = contact_wrench_map(scene.bodies[body], pts, normals)
        res = solve_contact_implicit(w, G, gaps, mu, cfg.inverse_dynamics)
        if np.median(res.wrench_residual) > tol: continue   # untrustworthy -> skip (down-weight)
        for e in body_edges:
            obs[e].normal_force = res.contact_normal_force[:, owner[e]].sum(axis=1)
```
Today's flow == this flow with all `inertial is None` (loop body skipped). Optional Phase-4b wraps
steps {infer_forces → detect → active-set} in an EM loop for indeterminate sets only.

## III.5 Phase-by-phase deliverables, files, acceptance

| Phase | Files touched | Acceptance test (must stay green: 138 suite + `verify_demos.py`) |
| --- | --- | --- |
| **0** waist + FlatRegion + registry skeleton | `types.py`, `geometry.py`, new `contact/geometry_resolvers.py`, `graph.py`, `model.py` (call site) | `ContactObservations` byte-identical on all demos; new `test_flatregion_identity.py` |
| **1** sphere primitives | `geometry_resolvers.py`, `demos_scenes_chain.py` (ball edges), cradle edges | `ballA↔ballB` atoms 7→1, `v_normal` 1 sign change; cradle closing-speed recovered; `test_sphere_geometry.py` (analytic normal/gap) |
| **2** box/capsule/ellipsoid + multi-point | `geometry_resolvers.py`, `geometry.py` (`_aggregate`), `tumbling_box` edge | tumbling impact mode fires per corner; `test_boxplane_nearest_feature.py` |
| **3** mesh/SDF | new `contact/mesh_collision.py` (GJK/EPA), `geometry_resolvers.py` | analytic-shape-vs-mesh agreement test |
| **4** force factor + inferred pass | `types.py`, `config.py`, `emissions.py`, `graph.py`, `model.py` | cradle clacks detected via inferred force; force-channel ablation no-ops; `test_force_emission.py` (proper-density integrates to 1), `test_inferred_force.py` |
| **4b** EM for indeterminate sets | `graph.py` | indeterminate rig force distribution converges; residual bounded |
| **5** unify factors + VoI | new `contact/factors.py` (registry), `uncertainty.py`/`consistency.py` adapters | refactor-only: identical numbers; `test_voi.py` entropy-drop sanity |

## III.6 Invariants enforced throughout

1. **Floor lock:** no shapes + no inertials + no sensors ⇒ byte-identical to today (Phase-0 test
   is the canary; every later phase re-runs it).
2. **Proper densities:** every new emission term integrates to 1 over its support (unit test per
   term) so log-ratios stay calibrated.
3. **No-op absence:** each factor gated on its input being present; gated off ⇒ no contribution.
4. **Down-weight, don't trust blindly:** geometry fidelity and dynamics residual both flow into
   the tempering weight, so low-quality evidence cannot dominate.
5. **Additive contracts:** all new fields default to `None`/off; no current call site changes
   meaning.

## III.7 Open items deferred (not blockers, flagged for Phase 4/5)

- Exact `s_load`/`σ` calibration curves and the impact-spike scale (needs synthetic-force study).
- Per-candidate *mode* (vs aggregated) — only if a use case needs per-corner modes, not just
  per-corner force.
- Articulated bodies in the inferred-force solve (current `dynamics_id` assumes a single rigid
  body with constant inertia).
- Tactile *location*/pressure-map likelihood (beyond scalar `f_t`).
