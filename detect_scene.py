#!/usr/bin/env python3
"""CLI: detect the joint contact-graph structure of a multi-body scene (THEORY.md §8).

This is the scene-level analogue of the single-pair detection entrypoint, exercising
rung 5 of the pragmatic ladder (THEORY.md §10): it takes one of the multi-body SCENES
(``oracle.SCENES``), runs the full graph detector
(``contact.graph.detect_scene``) over its candidate edges, and reports

  * per edge: the detected contact intervals + dominant twist-subspace mode (§3), scored
    against that edge's withheld ground truth (``oracle.report.score`` per edge, §9);
  * the joint MAP active-set timeline (which edges are simultaneously active over time,
    §8) against the ground-truth active set;

and (unless ``--no-plot``) writes the contact-graph diagnostic figure to
``contact_graph.png`` (one row per edge + an active-set strip).

The headline check (THEORY.md §1/§8): in ``person_on_skateboard`` the ``person_board``
edge must read as a sustained STATIC contact even though both bodies scream across the
world at ~1.2 m/s -- the relative-frame payoff. The graph detector measures every edge in
its support's frame, so a foot rigidly riding the deck reads ~0 relative motion.

Emission velocity scales (THEORY.md §4). The package defaults are tuned for typical
optical mocap of a *foot* (slide speeds ~0.1-0.2 m/s). A skateboard scene moves an order
of magnitude faster, so we widen the emission's tangential-velocity scales to span the
scene's actual observed motion before detecting -- a physically-interpretable config
choice (§9: keep the model physically parameterized, not overfit to one number), not a
change to the validated single-pair detector. The widening is derived from the scene's own
per-edge tangential speeds, so it generalizes beyond the two built-in scenes.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from contact import geometry
import oracle
from contact.config import DetectorConfig
from contact.graph import _resolve_support, build_candidate_edges, detect_scene
from oracle.report import plot_graph, print_graph_report
from contact.types import MultiBodyScene

#: Where the contact-graph figure is written (repo root), mirroring the single-pair PNG.
_DEFAULT_PLOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contact_graph.png")


def _scene_velocity_scale(scene: MultiBodyScene) -> float:
    """Robust upper estimate of the fastest support-relative tangential speed in the scene.

    For every candidate edge we run the same support-relative :func:`contact.geometry.observe`
    the detector uses (synthesizing the implicit static world floor where the support is
    ``"world"``, THEORY.md §1) and take a high percentile of ``|v_tangent|``. The maximum
    over edges is a scale for how fast contacts actually slide in this scene -- used to size
    the emission's sliding/free velocity priors so a genuine fast slide is not mistaken for
    free flight (THEORY.md §3/§4). Returns 0.0 if nothing can be measured.
    """
    scale = 0.0
    for edge in scene.edges:
        moving = scene.bodies.get(edge.moving_body)
        support = _resolve_support(scene, edge.support_body, moving)
        if moving is None or support is None:
            continue
        try:
            obs = geometry.observe(moving, support, edge.surface, edge.contact_point_local,
                                   geometry=getattr(edge, "geometry", None))
            vt = np.linalg.norm(np.asarray(obs.v_tangent, dtype=float), axis=1)
            if vt.size:
                scale = max(scale, float(np.percentile(vt, 90)))
        except Exception:
            continue
    return scale


def _config_for_scene(scene: MultiBodyScene) -> DetectorConfig:
    """A :class:`DetectorConfig` whose emission velocity scales span the scene's motion.

    Starts from the package defaults (tuned for slow mocap-of-a-foot) and, when the scene
    slides materially faster than the default ``slide_speed``, widens the sliding speed
    scale and the FREE velocity prior to cover the observed range (THEORY.md §4: the
    emission must place real mass where the data actually is). Slow/static scenes (e.g.
    ``box_on_two_blocks``) are left at the defaults. Everything else (the temporal prior,
    the active-set dwell, the consistency priors) is the package default.
    """
    # Emission velocity scaling is now done PER EDGE inside detect_scene (each edge fit to its
    # own tangential motion), which is strictly better than one scene-wide value: it stops the
    # fastest edge from inflating the sliding scale for slow edges (which made them read FREE).
    return DetectorConfig()


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the graph detector on the chosen scene, and report."""
    parser = argparse.ArgumentParser(
        description="Detect the joint contact-graph active-set structure of a multi-body "
        "scene (THEORY.md §8) and score it against the simulator's withheld truth.",
    )
    parser.add_argument(
        "--scene",
        default="person_on_skateboard",
        choices=oracle.SCENES,
        help="which multi-body scene to generate and detect (default: person_on_skateboard).",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=5e-4,
        help="std (m) of i.i.d. Gaussian mocap position noise added to every body "
        "(THEORY.md §4/§9). Default 5e-4.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for the mocap noise (default 0)."
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip writing the contact-graph diagnostic PNG.",
    )
    parser.add_argument(
        "--plot-path",
        default=_DEFAULT_PLOT_PATH,
        help=f"where to write the figure (default: {_DEFAULT_PLOT_PATH}).",
    )
    args = parser.parse_args(argv)

    # --- generate the scene (simulate -> noised poses + withheld per-edge truth, §9) ---
    scene = oracle.generate_scene(args.scene, seed=args.seed, noise_m=args.noise)

    # --- broad-phase (§8): identity on the simulator's vouched edges, a guard otherwise.
    candidate = build_candidate_edges(scene)
    pruned = [e.edge_id for e in scene.edges if e.edge_id not in {c.edge_id for c in candidate}]
    if pruned:
        print(f"Broad-phase pruned edges (never within proximity_gap): {pruned}")

    # --- detect: per-edge single-pair HMMs fused into the joint active-set posterior. ---
    cfg = _config_for_scene(scene)
    if cfg.emission.slide_speed != DetectorConfig().emission.slide_speed:
        print(
            f"Emission velocity scales widened to the scene's motion "
            f"(slide_speed={cfg.emission.slide_speed:.2f} m/s, "
            f"free_vel_sigma={cfg.emission.free_vel_sigma:.2f} m/s)."
        )
    result = detect_scene(scene, cfg)

    # --- report: per-edge intervals/mode/score + joint active-set timeline vs truth. ----
    print_graph_report(scene, result)

    # --- the THEORY.md §1/§8 headline call-out for the skateboard scene. --------------
    if args.scene == "person_on_skateboard" and "person_board" in result.per_edge:
        pb = result.per_edge["person_board"]
        modes = {m for m in pb.map_state if m != "free"}
        p_contact = float(np.asarray(pb.contact_posterior, dtype=float).mean())
        sustained_static = modes == {"static"} and p_contact > 0.95
        flag = "OK" if sustained_static else "CHECK"
        print(
            f"[{flag}] person_board: mean P(contact)={p_contact:.3f}, contact mode(s)={sorted(modes)} "
            f"-- a sustained STATIC contact despite ~1.2 m/s world motion "
            f"(the THEORY.md §1/§8 relative-frame payoff)."
        )

    # --- plot (guarded; matplotlib optional) -------------------------------------------
    if not args.no_plot:
        plot_graph(scene, result, args.plot_path)
        print(f"Wrote contact-graph figure to {args.plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
