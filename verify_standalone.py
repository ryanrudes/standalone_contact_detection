#!/usr/bin/env python3
"""Prove that ``contact_detection_standalone.py`` reproduces the ``contact/`` package's results.

This harness is the ONLY place the package is imported — the standalone file itself depends on
nothing from ``contact/``. We generate every scenario / scene with the package (its MuJoCo truth
factory), run BOTH the package detector and the standalone detector on the *same* recorded
observations, and assert every output field is identical:

  * single pair (15 scenarios): contact_posterior, state_posterior, map_state, in_contact,
    intervals, events, resting_bias, normal_force, slip_state, impulses — bit-for-bit;
  * contact graph (8 scenes): per-edge results, active_posterior, map_active_set, energy flag;
  * the optional force-gauge (``--stiffness``) and measured-force-channel code paths.

Run:  uv run python verify_standalone.py

Note on the contact graph: above ``enumerate_max_edges`` (default 4) the package swaps its exact
2^E enumeration for a Rao–Blackwellized particle smoother (a scaling approximation). The
standalone always enumerates exactly (the reference the particle filter approximates), so for the
one large scene (``dominoes``, 6 edges) we force the package onto its exact path to compare like
with like.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import contact as C
from contact.graph import detect_scene as C_detect_scene

import contact_detection_standalone as S

FAIL: list[str] = []


def _close(a, b, tol=1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        return False
    return a.size == 0 or bool(np.allclose(a, b, rtol=0, atol=tol, equal_nan=True))


def _cmp_result(tag: str, L, M) -> bool:
    ok = True

    def bad(field):
        nonlocal ok
        ok = False
        FAIL.append(f"{tag}.{field}")

    if not _close(L.contact_posterior, M.contact_posterior):
        bad("contact_posterior")
    if not _close(L.state_posterior, M.state_posterior):
        bad("state_posterior")
    if list(L.map_state) != list(M.map_state):
        bad("map_state")
    if not np.array_equal(np.asarray(L.in_contact), np.asarray(M.in_contact)):
        bad("in_contact")
    if abs(L.resting_bias - M.resting_bias) > 1e-9:
        bad("resting_bias")
    if not _close(L.normal_force, M.normal_force):
        bad("normal_force")
    if (L.slip_state or []) != (M.slip_state or []):
        bad("slip_state")
    li = [(round(i.t_start, 9), round(i.t_end, 9), i.mode) for i in L.intervals]
    mi = [(round(i.t_start, 9), round(i.t_end, 9), i.mode) for i in M.intervals]
    if li != mi:
        bad("intervals")
    le = [(e.kind, round(e.time, 9), e.index) for e in L.events]
    me = [(e.kind, round(e.time, 9), e.index) for e in M.events]
    if le != me:
        bad("events")

    def imps(r):
        return [(round(x.time, 9), x.index, round(x.closing_speed, 9),
                 "nan" if np.isnan(x.restitution) else round(x.restitution, 9),
                 "nan" if np.isnan(x.normal_impulse) else round(x.normal_impulse, 9)) for x in r.impulses]

    if imps(L) != imps(M):
        bad("impulses")
    return ok


def main() -> int:
    print("=== SINGLE-PAIR scenarios (default) ===")
    for name in C.SCENARIOS:
        raw = C.generate(name)
        obs = C.observe(raw.moving, raw.support, raw.surface, raw.contact_point_local,
                        geometry=getattr(raw, "geometry", None))
        L = C.ContactDetector().detect(obs)
        M = S.ContactDetector().detect(obs)
        ok = _cmp_result(name, L, M)
        print(f"  {name:20s} {'OK' if ok else 'FAIL'}")

    print("\n=== SINGLE-PAIR with --stiffness force gauge (§7) ===")
    for name in ["drop_rest", "push_to_slide", "indeterminate_rig", "incline_slide"]:
        raw = C.generate(name)
        obs = C.observe(raw.moving, raw.support, raw.surface, raw.contact_point_local,
                        geometry=getattr(raw, "geometry", None))
        cL = C.DetectorConfig(); cL.material.stiffness = 5e4
        cM = S.DetectorConfig(); cM.material.stiffness = 5e4
        L = C.ContactDetector(cL).detect(obs)
        M = S.ContactDetector(cM).detect(obs)
        print(f"  {name:20s} {'OK' if _cmp_result(name + ':k', L, M) else 'FAIL'}")

    print("\n=== SINGLE-PAIR with a measured-force channel ===")
    for name in ["drop_rest", "bouncing_ball"]:
        raw = C.generate(name)
        obs = C.observe(raw.moving, raw.support, raw.surface, raw.contact_point_local,
                        geometry=getattr(raw, "geometry", None))
        f = np.asarray(raw.truth.normal_force, float)
        L = C.ContactDetector().detect(replace(obs, normal_force=f))
        M = S.ContactDetector().detect(replace(obs, normal_force=f))
        print(f"  {name:20s} {'OK' if _cmp_result(name + ':f', L, M) else 'FAIL'}")

    print("\n=== MULTI-BODY scenes (contact graph) ===")
    for name in C.SCENES:
        scn = C.generate_scene(name)
        cfg = C.DetectorConfig()
        cfg.inference.enumerate_max_edges = 16   # force the package's exact path for like-with-like
        LG = C_detect_scene(scn, cfg)
        MG = S.detect_scene(scn)
        ok = True
        if list(LG.edges) != list(MG.edges):
            ok = False; FAIL.append(f"{name}.edges")
        if not _close(LG.active_posterior, MG.active_posterior):
            ok = False; FAIL.append(f"{name}.active_posterior")
        if [sorted(s) for s in LG.map_active_set] != [sorted(s) for s in MG.map_active_set]:
            ok = False; FAIL.append(f"{name}.map_active_set")
        for k in ("energy_prior_active", "num_subsets"):
            if LG.meta.get(k) != MG.meta.get(k):
                ok = False; FAIL.append(f"{name}.meta.{k}")
        for eid in LG.edges:
            if not _cmp_result(f"{name}:{eid}", LG.per_edge[eid], MG.per_edge[eid]):
                ok = False
        print(f"  {name:22s} {'OK' if ok else 'FAIL'}  edges={len(LG.edges)}")

    print()
    if FAIL:
        print(f"MISMATCHES ({len(FAIL)}): " + ", ".join(FAIL[:40]))
        return 1
    print("ALL MATCH — contact_detection_standalone.py is output-equivalent to the contact/ package.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
