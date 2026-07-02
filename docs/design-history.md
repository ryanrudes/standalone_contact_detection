# DESIGN build history — the phase-by-phase evidence

The record of *how* [`DESIGN.md`](../DESIGN.md)'s capability-driven architecture was
built: the phased rollout, each phase's acceptance evidence, and the caveats found along
the way. Kept for provenance — the living design (thesis, contracts, deep dives, and the
implementation reference the code cites as `DESIGN.md III.x` / `PHASE n`) stays in
DESIGN.md itself. Module paths and test counts below are as of the build (pre-dating the
contact/oracle repackaging), preserved verbatim.

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
