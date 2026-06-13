from __future__ import annotations

import argparse
import sys

from .app import ParticleHeartV2App
from .constants import QUALITY_DEFAULTS, QUALITY_PARTICLE_COUNTS, THEME_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D Particle Heart v2 - Cinematic real-time particle art",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--particles",
        type=int,
        default=70000,
        help="Total particle count (default 70000, max ~500000)",
    )
    parser.add_argument(
        "--bloom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable HDR bloom post-processing",
    )
    parser.add_argument(
        "--trails",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable motion trail accumulation",
    )
    parser.add_argument(
        "--pedestal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable particle array pedestal below the heart",
    )
    parser.add_argument("--fps", action="store_true", help="Show FPS counter in window title")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Headless render test, exit after a few frames",
    )
    parser.add_argument(
        "--theme",
        type=int,
        default=0,
        choices=range(len(THEME_NAMES)),
        help=f"Color theme: {', '.join(f'{i}={n}' for i, n in enumerate(THEME_NAMES))}",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Quality preset: low (fewer particles, faster), medium (balanced), high (dense particles, full effects)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    has_particles = any(a.startswith("--particles") for a in sys.argv[1:])
    has_bloom = any(a.startswith("--bloom") for a in sys.argv[1:])
    has_trails = any(a.startswith("--trails") for a in sys.argv[1:])

    if not has_particles:
        args.particles = QUALITY_PARTICLE_COUNTS.get(args.quality, 70000)
    if not has_bloom:
        args.bloom = QUALITY_DEFAULTS[args.quality]["bloom"]
    if not has_trails:
        args.trails = QUALITY_DEFAULTS[args.quality]["trails"]

    app = ParticleHeartV2App(args)
    if args.self_test:
        app.self_test()
        print("Self-test passed.")
    else:
        app.run()
