#!/usr/bin/env python3
"""End-to-end contact-detection CLI (THEORY.md sections 9 & 10).

Ties the whole package together on one MuJoCo scenario, exercising the exact
THEORY.md s.9 workflow: simulate -> expose only the noisy observable channel
(poses) -> run the detector -> score the inferred posterior against the withheld
ground truth.

    generate(scenario)            # MuJoCo truth factory (s.9)
        -> geometry.observe(...)  # poses -> support-relative observations (s.1, s.3)
        -> ContactDetector().detect(...)  # the generative-HMM estimator (s.4-8)
        -> report.print_report / plot_result  # score & visualize (s.9)

Usage
-----
    uv run python detect.py --scenario drop_rest
    uv run python detect.py --scenario push_to_slide --no-plot
    uv run python detect.py --scenario drop_rest --stiffness 1e5

Run with no arguments for the ``drop_rest`` default.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from contact import (
    ContactDetector,
    DetectorConfig,
    SCENARIOS,
    contact_implicit_from_raw,
    generate,
    observe,
)
from oracle import report

#: Where the diagnostic figure is written (unless --no-plot): next to this script, like
#: detect_scene.py's default (the .png is git-ignored).
_PLOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contact_detection.png")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments (scenario / noise / plotting / material stiffness)."""
    parser = argparse.ArgumentParser(
        description="Run the probabilistic contact detector on a MuJoCo scenario "
        "and score it against ground truth (THEORY.md s.9/s.10)."
    )
    parser.add_argument(
        "--scenario",
        default="drop_rest",
        choices=SCENARIOS,
        help="MuJoCo scenario to simulate and detect on (default: drop_rest).",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=5e-4,
        help="Std-dev (m) of the Gaussian mocap position noise on the moving body "
        "(default: 5e-4).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the diagnostic figure.",
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        default=None,
        help="Contact material stiffness (N/m). When given, penetration becomes a "
        "calibrated force gauge (lambda = k*delta, THEORY.md s.7) and the estimated "
        "normal force is computed.",
    )
    parser.add_argument(
        "--discover-modes",
        action="store_true",
        help="Additionally run UNSUPERVISED contact-mode discovery (sticky HDP-HMM, "
        "THEORY.md s.8) on the same observations and print the discovered modes, their "
        "signatures, alignment to the canonical modes, and a comparison against the "
        "supervised MAP segmentation. Off by default; does not change detection.",
    )
    parser.add_argument(
        "--inverse-dynamics",
        action="store_true",
        help="Additionally run CONTACT-IMPLICIT INVERSE DYNAMICS (THEORY.md s.8, the "
        "north star): jointly recover the contact forces and active set that explain the "
        "observed motion under Newton-Euler with Signorini complementarity (s.2) and the "
        "Coulomb cone (s.7), and score the recovered total normal force at rest vs m*g and "
        "vs the MuJoCo summed corner force, the active-set timeline vs truth, and the mean "
        "wrench residual. Only runs on scenarios that expose the inertial/candidate "
        "metadata (the single-rigid-box-on-plane scenarios); skipped with a note "
        "otherwise. Off by default; does not change the kinematic detection.",
    )
    return parser.parse_args(argv)


def _run_inverse_dynamics(raw, config: DetectorConfig, name: str) -> None:
    """Run contact-implicit inverse dynamics on a scenario and report it (THEORY.md s.8).

    Only scenarios that expose ``raw.meta['inertial']`` AND ``raw.meta['candidates']``
    (the single-rigid-box-on-plane family: ``drop_rest``, ``drop_rest_liftoff``,
    ``push_to_slide``, ``moving_support``, ``indeterminate_rig``) carry the mass/inertia
    and candidate point-contacts the Newton-Euler solve needs. Spheres and the multi-body
    scenes do not; for those we print a clear skip message rather than failing.
    """
    meta = raw.meta or {}
    if "inertial" not in meta or "candidates" not in meta:
        print()
        print(
            f"--inverse-dynamics: scenario '{name}' exposes no inertial/candidate "
            "metadata (needs meta['inertial'] and meta['candidates']); skipping the "
            "contact-implicit inverse dynamics. It runs on the rigid-box-on-plane "
            "scenarios (drop_rest, drop_rest_liftoff, push_to_slide, moving_support, "
            "indeterminate_rig)."
        )
        return

    print()
    id_result = contact_implicit_from_raw(raw, config)
    id_scores = report.print_inverse_dynamics(id_result, meta, name=name)
    print("Inverse-dynamics score dict:", id_scores)


def _report_discovered_modes(detector, obs, result) -> None:
    """Run unsupervised mode discovery and print it against the supervised MAP modes.

    THEORY.md s.8 research surface: fit a sticky HDP-HMM to the per-frame twist feature
    (label-free) and report (a) how many modes it found, (b) each mode's mean physical
    signature ``[gap, |v_n|, |v_t|, |omega_n|, |omega_t|]`` and its nearest canonical
    mode (validation-only alignment), and (c) for each discovered mode the distribution
    of supervised :meth:`ContactDetector.detect` MAP labels over its frames -- so one can
    see whether the unsupervised vocabulary rediscovers the supervised regimes.
    """
    disc = detector.discover_modes(obs)
    print()
    print("Unsupervised mode discovery (sticky HDP-HMM, THEORY.md s.8):")
    print(f"  discovered {disc.n_modes} mode(s) over {len(disc.labels)} frames")
    map_state = list(result.map_state)
    for mode_id in sorted(disc.signatures):
        sig = disc.signatures[mode_id]
        canon = disc.alignment.get(mode_id, "?")
        frac = float(np.mean(disc.labels == mode_id))
        print(
            f"  mode {mode_id}: aligns->{canon:<8} occupancy={frac:6.1%}  "
            f"signature[gap,|v_n|,|v_t|,|w_n|,|w_t|]="
            f"[{sig[0]:+.4f}, {sig[1]:.3f}, {sig[2]:.3f}, {sig[3]:.3f}, {sig[4]:.3f}]"
        )
        # Distribution of supervised MAP labels over this discovered mode's frames.
        idx = np.flatnonzero(disc.labels == mode_id)
        sup_counts: dict[str, int] = {}
        for i in idx:
            if 0 <= i < len(map_state):
                lbl = map_state[i]
                sup_counts[lbl] = sup_counts.get(lbl, 0) + 1
        if sup_counts:
            total = sum(sup_counts.values())
            parts = ", ".join(
                f"{lbl} {c / total:.0%}"
                for lbl, c in sorted(sup_counts.items(), key=lambda kv: -kv[1])
            )
            print(f"            supervised MAP modes on these frames: {parts}")


def main(argv: list[str] | None = None) -> None:
    """Generate one scenario, detect contacts, report scores, and optionally plot."""
    args = _parse_args(argv)

    # --- 1. MuJoCo truth factory: simulate and label (THEORY.md s.9). ---
    raw = generate(args.scenario, noise_m=args.noise)

    # --- 2. Poses -> support-relative observations (THEORY.md s.1, s.3). ---
    # Smoothing time before differentiation comes from the detector config so the
    # observation pipeline and the emission scales agree on the same time constant.
    config = DetectorConfig()
    if args.stiffness is not None:
        config.material.stiffness = args.stiffness

    obs = observe(
        raw.moving,
        raw.support,
        raw.surface,
        contact_point_local=raw.contact_point_local,
        vel_smooth_time=config.vel_smooth_time,
        geometry=getattr(raw, "geometry", None),
    )

    # --- 3. Run the generative-HMM detector (THEORY.md s.4-8). ---
    detector = ContactDetector(config)
    result = detector.detect(obs)

    # --- 4. Score against the withheld truth + report (THEORY.md s.9). ---
    report.print_report(args.scenario, result, raw.truth)
    scores = report.score(result, raw.truth)
    print("Score dict:", scores)

    # --- impact-atom summary (THEORY.md s.6): count + measured restitutions. ---
    # The impulses are the matched-filter velocity-step atoms of the force measure. We
    # print how many were found and the restitution e of each that resolved a bounce
    # (NaN ones -- plastic / unresolved landings -- are reported as n/a). The impulse
    # magnitude needs the moving body's mass, which the observable channel does not
    # carry (s.7), so it is NaN here and surfaced in the full report above.
    n_imp = len(result.impulses)
    e_vals = [imp.restitution for imp in result.impulses if not np.isnan(imp.restitution)]
    print(f"Detected impacts: {n_imp}")
    if e_vals:
        e_str = ", ".join(f"{e:.3f}" for e in e_vals)
        print(f"  measured restitutions: {e_str}")

    # --- material/dynamics summary (THEORY.md s.7), only when --stiffness was given. ---
    # With a stiffness the detector turns penetration into a calibrated force gauge
    # (lambda = k*delta) and labels each contact frame stick/slip via the Coulomb cone.
    if args.stiffness is not None and result.normal_force is not None:
        nf = np.asarray(result.normal_force, dtype=float)
        if np.any(np.isfinite(nf)):
            print(
                f"Estimated normal force (k={args.stiffness:g} N/m): "
                f"peak {float(np.nanmax(nf)):.2f} N"
            )

    # --- unsupervised mode-discovery surface (THEORY.md s.8), opt-in via --discover-modes.
    if args.discover_modes:
        _report_discovered_modes(detector, obs, result)

    # --- contact-implicit inverse dynamics (THEORY.md s.8 north star), opt-in. ----------
    # The dual, dynamics-first path: recover the contact forces + active set that explain
    # the OBSERVED motion under Newton-Euler with Signorini + Coulomb, and score them
    # against the withheld physical truth (m*g, MuJoCo corner forces, the truth active
    # set). Runs only on scenarios that carry the inertial/candidate metadata.
    if args.inverse_dynamics:
        _run_inverse_dynamics(raw, config, args.scenario)

    if not args.no_plot:
        report.plot_result(obs, result, raw.truth, _PLOT_PATH)
        print(f"Wrote plot to {_PLOT_PATH}")


if __name__ == "__main__":
    main()
