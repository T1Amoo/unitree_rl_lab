#!/usr/bin/env python3
"""Compare two joint-angle trace CSVs and compute per-joint RMSE.

Primary use: compare Isaac ground-truth q_1..7 from a play trace against a
MuJoCo replay q_1..7 to verify that the mujoco damiao_mit executor reproduces
the IsaacLab training executor response.

Secondary use: compare damiao_mit vs direct_response on a chirp sequence to
measure torque-chain tracking error (the "is the torque layer necessary?" check).

Outputs to --out-dir:
  metrics.json      Per-joint RMSE and summary stats.
  *_overlay.png     7-subplot joint-angle overlay figure.

Usage
-----
    # Real-trace alignment (Steps A-C):
    python compare_isaac_vs_mujoco.py \\
        --trace-a /tmp/a1_v7_isaac_trace.csv    --label-a Isaac \\
        --trace-b /tmp/a1_v7_mujoco_damiao_mit.csv --label-b MuJoCo_damiao_mit \\
        --out-dir /media/woan/.../damiao_mit_isaac_align/

    # Chirp comparison (Step D):
    python compare_isaac_vs_mujoco.py \\
        --trace-a /tmp/a1_chirp_mujoco_damiao_mit.csv --label-a damiao_mit \\
        --trace-b /tmp/a1_chirp_mujoco_direct.csv     --label-b direct_response \\
        --out-dir /media/woan/.../chirp_comparison/ \\
        --tag chirp
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

WARMUP_ROWS = 5  # skip first N rows for RMSE to avoid initialisation transient


def _load_q(csv_path: Path, prefix: str = "q_") -> np.ndarray:
    """Load q_1..7 (or any 7-joint columns) from a CSV.  Returns (N, 7) float64."""
    cols = [f"{prefix}{i}" for i in range(1, 8)]
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    missing = set(cols) - set(rows[0].keys())
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    return np.array([[float(row[col]) for col in cols] for row in rows], dtype=np.float64)


def compare(
    trace_a: Path,
    trace_b: Path,
    label_a: str,
    label_b: str,
    out_dir: Path,
    tag: str = "comparison",
) -> dict:
    q_a = _load_q(trace_a, prefix="q_")
    q_b = _load_q(trace_b, prefix="q_")

    n = min(len(q_a), len(q_b))
    if n < WARMUP_ROWS + 1:
        raise ValueError(f"Trace too short ({n} ticks) for warmup={WARMUP_ROWS}")
    q_a = q_a[:n]
    q_b = q_b[:n]

    free = slice(WARMUP_ROWS, n)
    err = q_a[free] - q_b[free]
    rmse_per_joint = np.sqrt(np.mean(err ** 2, axis=0))

    metrics: dict = {
        f"rmse_J{i + 1}_rad": float(rmse_per_joint[i]) for i in range(7)
    }
    metrics["label_a"] = label_a
    metrics["label_b"] = label_b
    metrics["warmup_rows_skipped"] = WARMUP_ROWS
    metrics["n_compared"] = int(n - WARMUP_ROWS)
    metrics["n_total"] = int(n)
    metrics["max_abs_err_rad"] = float(np.max(np.abs(err)))
    metrics["mean_abs_err_rad"] = float(np.mean(np.abs(err)))

    print(f"[compare] {label_a} vs {label_b}  (n={n - WARMUP_ROWS} ticks, warmup={WARMUP_ROWS})", flush=True)
    for i in range(7):
        print(f"  J{i + 1}: RMSE={rmse_per_joint[i]:.4f} rad", flush=True)
    print(f"  max_abs={metrics['max_abs_err_rad']:.4f}  mean_abs={metrics['mean_abs_err_rad']:.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"{tag}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[compare] metrics -> {metrics_path}", flush=True)

    # 7-subplot overlay figure
    ticks = np.arange(n)
    fig, axes = plt.subplots(7, 1, figsize=(13, 17), sharex=True)
    colors = {"a": "steelblue", "b": "#e05a4c"}
    for i in range(7):
        ax = axes[i]
        ax.plot(ticks, q_a[:, i], label=label_a, color=colors["a"], linewidth=1.1)
        ax.plot(ticks, q_b[:, i], label=label_b, color=colors["b"], linewidth=1.1, linestyle="--")
        ax.axvline(WARMUP_ROWS, color="#888888", linestyle=":", linewidth=0.8, label="warmup end")
        ax.set_ylabel(f"J{i + 1} [rad]", fontsize=8)
        ax.set_title(f"Joint {i + 1}  |  RMSE = {rmse_per_joint[i]:.4f} rad", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", ncol=3)
        ax.grid(True, linewidth=0.4, alpha=0.5)
    axes[-1].set_xlabel("control tick (50 Hz)", fontsize=9)
    fig.suptitle(
        f"{label_a} vs {label_b} — joint angle overlay\n"
        f"mean RMSE = {float(np.mean(rmse_per_joint)):.4f} rad  |  "
        f"max_abs = {metrics['max_abs_err_rad']:.4f} rad",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = out_dir / f"{tag}_overlay.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] figure -> {fig_path}", flush=True)

    return metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trace-a", type=Path, required=True, help="First trace CSV (with q_1..7 columns).")
    ap.add_argument("--trace-b", type=Path, required=True, help="Second trace CSV (with q_1..7 columns).")
    ap.add_argument("--label-a", type=str, default="trace_a", help="Display label for trace A.")
    ap.add_argument("--label-b", type=str, default="trace_b", help="Display label for trace B.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/"
            "系统辨识/sim2real/2026-07-25/damiao_mit_isaac_align"
        ),
        help="Directory for metrics.json and overlay figure.",
    )
    ap.add_argument("--tag", type=str, default="comparison", help="File name prefix for outputs.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare(args.trace_a, args.trace_b, args.label_a, args.label_b, args.out_dir, tag=args.tag)
