#!/usr/bin/python3
"""Print the first-pass A1 joint-ID command matrix."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def tag_float(value: float | str) -> str:
    text = f"{float(value):g}" if isinstance(value, (float, int)) else str(value)
    return text.replace("-", "m")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=Path("系统辨识"))
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--kp", type=float, default=120.0, help="Gain tag for the tested joint.")
    ap.add_argument("--kd", type=float, default=3.5, help="Gain tag for the tested joint.")
    ap.add_argument("--param-tag", default="", help="Override experiment folder name.")
    ap.add_argument("--real-kp", default="", help="7-value kp metadata passed to record_real_sine.py.")
    ap.add_argument("--real-kd", default="", help="7-value kd metadata passed to record_real_sine.py.")
    ap.add_argument("--amplitude", type=float, default=0.12)
    ap.add_argument("--freqs", default="0.5,1.0,1.5")
    ap.add_argument("--cycles", type=float, default=8.0)
    ap.add_argument("--joints", default="1,2,3,4,5,6,7")
    ap.add_argument("--recorder", choices=["fsm", "direct"], default="fsm")
    ap.add_argument("--mkdirs", action="store_true", help="Create the directory tree.")
    ap.add_argument("--dry-run", action="store_true", help="Omit --enable-servo from commands.")
    args = ap.parse_args()

    freqs = [float(x) for x in args.freqs.replace(",", " ").split()]
    joints = [int(x) for x in args.joints.replace(",", " ").split()]
    experiment = args.param_tag or f"kp{tag_float(args.kp)}_kd{tag_float(args.kd)}"
    print("# Source ROS first:")
    print("source /opt/ros/humble/setup.bash")
    print("source unitree_rl_lab/deploy/robots/a1_h1/install/setup.bash")
    if args.recorder == "fsm":
        print("# Start a1_tt_fsm_supervisor with test_enabled:=true, press FixStand, then press R2/test.")
    else:
        print("# Stop a1_policy_bridge_cpp/a1_tt_fsm_supervisor before enabling servo.")
    for joint in joints:
        base = args.output_root / f"joint{joint}" / args.date / experiment
        real_dir = base / "real"
        if args.mkdirs:
            for subdir in ("real", "sim", "merged", "summary", "plots", "logs"):
                (base / subdir).mkdir(parents=True, exist_ok=True)
        for freq in freqs:
            out = real_dir / f"j{joint}_f{freq:g}hz.csv"
            enable = "" if args.dry_run else " --enable-servo"
            metadata = f" --param-tag {experiment}"
            if args.real_kp:
                metadata += f" --real-kp \"{args.real_kp}\""
            if args.real_kd:
                metadata += f" --real-kd \"{args.real_kd}\""
            if args.recorder == "fsm":
                print(
                    "/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/record_fsm_test.py "
                    f"--joint {joint} --freq {freq:g} --amplitude {args.amplitude:g} "
                    f"--cycles {args.cycles:g} --output {out}{metadata}"
                )
            else:
                print(
                    "/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/record_real_sine.py "
                    f"--joint {joint} --freq {freq:g} --amplitude {args.amplitude:g} "
                    f"--cycles {args.cycles:g} --output {out}{metadata}{enable}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
