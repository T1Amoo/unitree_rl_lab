#!/usr/bin/env python3
"""Generate the A1 table-tennis MJCF used by sim2sim."""

from __future__ import annotations

import argparse
from pathlib import Path

from a1_scene import DEFAULT_MESHDIR, DEFAULT_SCENE_XML, DEFAULT_URDF, build_scene_xml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--meshdir", type=Path, default=DEFAULT_MESHDIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_SCENE_XML)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = build_scene_xml(args.urdf, args.meshdir, args.out)
    print(out)


if __name__ == "__main__":
    main()
