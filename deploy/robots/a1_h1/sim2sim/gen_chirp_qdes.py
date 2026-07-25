#!/usr/bin/env python3
"""Generate a chirp q_des CSV for executor identification (Step D).

Produces a 50 Hz q_des sequence: a static warm-up at the fixstand default pose
followed by a linear-frequency-sweep (chirp) applied jointly to all 7 joints
(with a small per-joint phase offset so joints are not exactly in-phase).

Optionally runs the mujoco damiao_mit and direct_response replays in-process
(--run-compare) and saves a RMSE comparison to --out-dir.

Usage
-----
    # Just generate the CSV:
    python gen_chirp_qdes.py --out /tmp/a1_chirp_qdes.csv

    # Generate + replay + compare (Step D):
    python gen_chirp_qdes.py \\
        --out /tmp/a1_chirp_qdes.csv \\
        --run-compare \\
        --out-dir /media/woan/.../chirp_comparison/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Add sim2sim directory to path so we can import local modules when running
# in --run-compare mode.
sys.path.insert(0, str(Path(__file__).parent))

# Default fixstand joint angles (matches a1_scene.DEFAULT_QPOS right-arm joints).
DEFAULT_RIGHT_Q = np.array([0.569, -0.692, 0.717, 1.13, -1.24, 0.0314, 0.772], dtype=np.float64)

CTRL_HZ = 50.0


def generate_chirp(
    *,
    out_csv: Path,
    step_duration_s: float = 2.0,
    chirp_duration_s: float = 8.0,
    amp_rad: float = 0.08,
    freq_start_hz: float = 0.1,
    freq_end_hz: float = 3.0,
) -> np.ndarray:
    """Write q_des_1..7 CSV and return the (N, 7) float64 array."""
    dt = 1.0 / CTRL_HZ
    n_step = int(round(step_duration_s * CTRL_HZ))
    n_chirp = int(round(chirp_duration_s * CTRL_HZ))

    t = np.arange(n_chirp) * dt
    # Linear frequency sweep (chirp): instantaneous frequency rises from freq_start to freq_end.
    k = (freq_end_hz - freq_start_hz) / chirp_duration_s
    rows = np.empty((n_step + n_chirp, 7), dtype=np.float64)

    # Warm-up: hold default pose.
    rows[:n_step] = DEFAULT_RIGHT_Q

    # Chirp: same sweep on all joints, but with a small per-joint phase offset.
    for j in range(7):
        phase_offset = j * 0.15  # ~8.6 deg offset between adjacent joints
        phase = 2.0 * np.pi * (freq_start_hz * t + 0.5 * k * t ** 2) + phase_offset
        rows[n_step:, j] = DEFAULT_RIGHT_Q[j] + amp_rad * np.sin(phase)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    q_des_cols = [f"q_des_{i}" for i in range(1, 8)]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=q_des_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({f"q_des_{i}": float(row[i - 1]) for i in range(1, 8)})
    n = len(rows)
    print(f"[gen_chirp] wrote {out_csv}  ({n} ticks at {CTRL_HZ:.0f} Hz, {n / CTRL_HZ:.1f} s)", flush=True)
    return rows


def run_compare(chirp_csv: Path, out_dir: Path) -> dict:
    """Replay chirp through damiao_mit and direct_response, then compare."""
    from a1_scene import PHYSICS_DT, build_scene_xml, load_scene
    from policy_io import A1PolicyIO
    from replay_qdes_damiao_mit import DECIMATION, _load_q_des, _replay_damiao_mit, _replay_direct_response
    from compare_isaac_vs_mujoco import compare
    import mujoco

    model, data, xml_path = load_scene(build_scene_xml())
    io = A1PolicyIO(model, data, use_predictor=False)
    print(f"[gen_chirp] scene: {xml_path}", flush=True)

    q_des_seq = _load_q_des(chirp_csv)
    print(f"[gen_chirp] loaded {len(q_des_seq)} chirp ticks", flush=True)

    # damiao_mit replay
    q_damiao = _replay_damiao_mit(q_des_seq, io, model, data)
    damiao_csv = chirp_csv.parent / "a1_chirp_mujoco_damiao_mit.csv"
    damiao_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    cols = [f"q_{i}" for i in range(1, 8)]
    with damiao_csv.open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for q_row in q_damiao:
            writer.writerow({f"q_{i}": float(q_row[i - 1]) for i in range(1, 8)})
    print(f"[gen_chirp] damiao_mit trace -> {damiao_csv}", flush=True)

    # direct_response baseline
    q_direct = _replay_direct_response(q_des_seq, io)
    direct_csv = chirp_csv.parent / "a1_chirp_mujoco_direct.csv"
    with direct_csv.open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for q_row in q_direct:
            writer.writerow({f"q_{i}": float(q_row[i - 1]) for i in range(1, 8)})
    print(f"[gen_chirp] direct_response trace -> {direct_csv}", flush=True)

    metrics = compare(
        damiao_csv,
        direct_csv,
        label_a="damiao_mit",
        label_b="direct_response",
        out_dir=out_dir,
        tag="chirp",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/a1_chirp_qdes.csv"),
        help="Output chirp q_des CSV path.",
    )
    ap.add_argument("--step-duration-s", type=float, default=2.0, help="Hold default pose for N seconds before chirp.")
    ap.add_argument("--chirp-duration-s", type=float, default=8.0, help="Duration of the chirp sweep in seconds.")
    ap.add_argument("--amp-rad", type=float, default=0.08, help="Chirp amplitude in radians.")
    ap.add_argument("--freq-start-hz", type=float, default=0.1, help="Chirp start frequency.")
    ap.add_argument("--freq-end-hz", type=float, default=3.0, help="Chirp end frequency.")
    ap.add_argument("--run-compare", action="store_true", help="Also run mujoco replays and compare damiao_mit vs direct_response.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/"
            "系统辨识/sim2real/2026-07-25/damiao_mit_isaac_align"
        ),
        help="Output directory for comparison figures and metrics (used with --run-compare).",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_chirp(
        out_csv=args.out,
        step_duration_s=args.step_duration_s,
        chirp_duration_s=args.chirp_duration_s,
        amp_rad=args.amp_rad,
        freq_start_hz=args.freq_start_hz,
        freq_end_hz=args.freq_end_hz,
    )
    if args.run_compare:
        run_compare(args.out, args.out_dir)
