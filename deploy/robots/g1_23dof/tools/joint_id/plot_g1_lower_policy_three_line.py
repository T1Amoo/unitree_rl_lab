#!/usr/bin/env python3
"""Plot G1 joint-ID target/real/lower-policy curves only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from g1_joint_id_common import RIGHT_ARM_POLICY_INDICES, RIGHT_ARM_JOINT_NAMES, write_json


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def range_stats(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"min": math.nan, "max": math.nan, "ptp": math.nan, "mean": math.nan}
    return {"min": float(np.min(x)), "max": float(np.max(x)), "ptp": float(np.ptp(x)), "mean": float(np.mean(x))}


def error_stats(diff: np.ndarray) -> dict[str, float]:
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {"bias_rad": math.nan, "rmse_rad": math.nan, "max_abs_rad": math.nan, "mean_abs_rad": math.nan}
    return {
        "bias_rad": float(np.mean(diff)),
        "rmse_rad": float(np.sqrt(np.mean(diff**2))),
        "max_abs_rad": float(np.max(np.abs(diff))),
        "mean_abs_rad": float(np.mean(np.abs(diff))),
    }


def quat_to_rpy(w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--sim-lower", type=Path, required=True)
    ap.add_argument("--joint", type=int, required=True, choices=range(1, 6))
    ap.add_argument("--plot", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--merged", type=Path, default=None)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    real_rows = read_rows(args.real)
    sim_rows = read_rows(args.sim_lower)
    if not real_rows or not sim_rows:
        raise SystemExit("empty input")
    n = min(len(real_rows), len(sim_rows))
    real_rows = real_rows[:n]
    sim_rows = sim_rows[:n]

    policy_i = RIGHT_ARM_POLICY_INDICES[args.joint - 1] + 1
    t = np.array([get_float(r, "t") for r in real_rows], dtype=float)
    stage = np.array([r.get("stage", "") for r in real_rows])
    chirp_mask = stage == "chirp"
    target = np.array([get_float(r, f"target_q{policy_i}") for r in real_rows], dtype=float)
    real = np.array([get_float(r, f"actual_q{policy_i}") for r in real_rows], dtype=float)
    lower = np.array([get_float(r, f"sim_q{policy_i}") for r in sim_rows], dtype=float)

    root_x = np.array([get_float(r, "root_x") for r in sim_rows], dtype=float)
    root_y = np.array([get_float(r, "root_y") for r in sim_rows], dtype=float)
    root_z = np.array([get_float(r, "root_z") for r in sim_rows], dtype=float)
    roll, pitch, yaw = quat_to_rpy(
        np.array([get_float(r, "root_qw") for r in sim_rows], dtype=float),
        np.array([get_float(r, "root_qx") for r in sim_rows], dtype=float),
        np.array([get_float(r, "root_qy") for r in sim_rows], dtype=float),
        np.array([get_float(r, "root_qz") for r in sim_rows], dtype=float),
    )
    root_disp = np.sqrt((root_x - root_x[0]) ** 2 + (root_y - root_y[0]) ** 2 + (root_z - root_z[0]) ** 2)
    fall_threshold_m = 0.3
    fall_mask = root_z < fall_threshold_m
    first_fall_i = int(np.argmax(fall_mask)) if bool(np.any(fall_mask)) else -1

    summary = {
        "real_csv": str(args.real),
        "sim_lower_policy_csv": str(args.sim_lower),
        "plot": str(args.plot),
        "merged_csv": str(args.merged) if args.merged else "",
        "joint": args.joint,
        "joint_name": RIGHT_ARM_JOINT_NAMES[args.joint - 1],
        "rows": n,
        "stage_counts": {name: int(np.sum(stage == name)) for name in sorted(set(stage.tolist()))},
        "finite_target_q": int(np.sum(np.isfinite(target))),
        "finite_real_q": int(np.sum(np.isfinite(real))),
        "finite_lower_policy_q": int(np.sum(np.isfinite(lower))),
        "chirp_t_range": [float(np.min(t[chirp_mask])), float(np.max(t[chirp_mask]))] if np.any(chirp_mask) else [],
        "target_q_range_chirp": range_stats(target[chirp_mask]),
        "real_q_range_chirp": range_stats(real[chirp_mask]),
        "lower_policy_q_range_chirp": range_stats(lower[chirp_mask]),
        "target_vs_real_chirp": error_stats(target[chirp_mask] - real[chirp_mask]),
        "lower_policy_vs_real_chirp": error_stats(lower[chirp_mask] - real[chirp_mask]),
        "lower_policy_vs_target_chirp": error_stats(lower[chirp_mask] - target[chirp_mask]),
        "fall_threshold_m": fall_threshold_m,
        "fell_any": bool(np.any(fall_mask)),
        "fell_chirp": bool(np.any(fall_mask & chirp_mask)),
        "first_fall_t": float(t[first_fall_i]) if first_fall_i >= 0 else math.nan,
        "first_fall_stage": str(stage[first_fall_i]) if first_fall_i >= 0 else "",
        "lower_policy_root_motion": {
            "max_displacement_m": float(np.nanmax(root_disp)),
            "chirp_max_displacement_m": float(np.nanmax(root_disp[chirp_mask])),
            "x": range_stats(root_x[chirp_mask]),
            "y": range_stats(root_y[chirp_mask]),
            "z": range_stats(root_z[chirp_mask]),
            "roll_rad": range_stats(roll[chirp_mask]),
            "pitch_rad": range_stats(pitch[chirp_mask]),
            "yaw_rad": range_stats(yaw[chirp_mask]),
        },
    }
    write_json(args.summary, summary)

    if args.merged is not None:
        args.merged.parent.mkdir(parents=True, exist_ok=True)
        with args.merged.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "t",
                "stage",
                "joint",
                "joint_name",
                "target_q",
                "real_q",
                "lower_policy_q",
                "real_minus_target",
                "lower_policy_minus_real",
                "lower_policy_minus_target",
                "lower_policy_root_x",
                "lower_policy_root_y",
                "lower_policy_root_z",
                "lower_policy_root_roll",
                "lower_policy_root_pitch",
                "lower_policy_root_yaw",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(n):
                writer.writerow(
                    {
                        "t": t[i],
                        "stage": stage[i],
                        "joint": args.joint,
                        "joint_name": RIGHT_ARM_JOINT_NAMES[args.joint - 1],
                        "target_q": target[i],
                        "real_q": real[i],
                        "lower_policy_q": lower[i],
                        "real_minus_target": real[i] - target[i],
                        "lower_policy_minus_real": lower[i] - real[i],
                        "lower_policy_minus_target": lower[i] - target[i],
                        "lower_policy_root_x": root_x[i],
                        "lower_policy_root_y": root_y[i],
                        "lower_policy_root_z": root_z[i],
                        "lower_policy_root_roll": roll[i],
                        "lower_policy_root_pitch": pitch[i],
                        "lower_policy_root_yaw": yaw[i],
                    }
                )

    import matplotlib.pyplot as plt

    if np.any(chirp_mask):
        zoom_start = max(float(np.max(t[chirp_mask])) - 10.0, float(np.min(t[chirp_mask])))
        zoom_mask = chirp_mask & (t >= zoom_start)
    else:
        zoom_mask = np.ones_like(t, dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    for ax, mask, title in (
        (axes[0], np.ones_like(t, dtype=bool), "full trial"),
        (axes[1], zoom_mask, "last 10 s active chirp"),
    ):
        ax.plot(t[mask], target[mask], label="target", linewidth=1.2)
        ax.plot(t[mask], real[mask], label="real", linewidth=1.0)
        ax.plot(t[mask], lower[mask], label="lower_policy", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel("q (rad)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=3, fontsize=9)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"G1 joint{args.joint} {RIGHT_ARM_JOINT_NAMES[args.joint - 1]}")
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=160)
    plt.close(fig)

    print(f"wrote {args.summary}")
    if args.merged is not None:
        print(f"wrote {args.merged}")
    print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
