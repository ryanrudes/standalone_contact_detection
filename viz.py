#!/usr/bin/env python3
"""Render a synced, real-time, side-by-side animation of the contact pipeline.

    uv run python viz.py drop_rest_liftoff                 # single-pair scenario
    uv run python viz.py push_to_slide --stiffness 80000   # + a normal-force panel
    uv run python viz.py person_on_skateboard              # multi-body scene (auto-detected)
    uv run python viz.py rolling_ball --out media/roll.gif --fps 30

The left panel is the actual MuJoCo-rendered world; the right panels are the live gap,
speeds, contact posterior + mode ribbon + events (and per-edge active set for scenes),
all locked to one moving playhead and playing at real time. Output format follows the
--out extension (.mp4 via ffmpeg, .gif via pillow).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contact import mujoco_gen
from contact.config import DetectorConfig
from oracle.visualize import animate_scene, animate_scenario


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-by-side real-time contact-pipeline animation.")
    ap.add_argument("name", help=f"scenario {mujoco_gen.SCENARIOS} or scene {mujoco_gen.SCENES}")
    ap.add_argument("--out", default=None, help="output path (.mp4 or .gif); default media/<name>.mp4")
    ap.add_argument("--fps", type=int, default=50, help="playback fps (real-time; default 50)")
    ap.add_argument("--hz", type=float, default=100.0, help="simulation/record rate (default 100)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stiffness", type=float, default=None,
                    help="contact stiffness (N/m) -> adds a normal-force panel (scenarios only)")
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=560)
    ap.add_argument("--distance", type=float, default=None, help="camera distance override")
    ap.add_argument("--pairs", action="store_true",
                    help="render one per-contact-pair video (fixed camera on the two tracked "
                         "bodies, others faded, slow-mo at that pair's events) -> media/pairs/")
    ap.add_argument("--force", action="store_true",
                    help="with --pairs: feed the FORCE channel (truth force as a stand-in sensor) "
                         "so force-mediated contacts kinematics can't see (the cradle clacks) light "
                         "up as IMPACT. Writes __force-suffixed videos beside the kinematic ones.")
    ap.add_argument("--events", action="store_true",
                    help="render one zoomed slow-mo clip per contact event -> media/events/")
    args = ap.parse_args()

    is_scene = args.name in mujoco_gen.SCENES
    if not is_scene and args.name not in mujoco_gen.SCENARIOS:
        ap.error(f"unknown name '{args.name}'. scenarios={mujoco_gen.SCENARIOS} scenes={mujoco_gen.SCENES}")

    config = DetectorConfig()
    if args.stiffness is not None:
        config.material.stiffness = args.stiffness
    cfg = config if args.stiffness is not None else None

    if args.pairs:
        from oracle.visualize import animate_pairs
        paths = animate_pairs(args.name, seed=args.seed, hz=args.hz, config=cfg, use_force=args.force)
        print("Wrote per-pair contact videos" + (" (force channel on)" if args.force else "") + ":")
        for p in paths:
            print(f"  {p}")
        return

    if args.events:
        from oracle.visualize import animate_event_clips
        paths = animate_event_clips(args.name, outdir="media/events", seed=args.seed, hz=args.hz, config=cfg)
        print("Wrote event clips:")
        for p in paths:
            print(f"  {p}")
        return

    out = args.out or os.path.join("media", "overviews", f"{args.name}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    cam = {} if args.distance is None else {"distance": args.distance}

    if is_scene:
        animate_scene(args.name, out, seed=args.seed, hz=args.hz, fps=args.fps,
                      width=args.width, height=args.height, **cam)
    else:
        animate_scenario(args.name, out, seed=args.seed, hz=args.hz, fps=args.fps,
                         config=config, width=args.width, height=args.height, **cam)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
