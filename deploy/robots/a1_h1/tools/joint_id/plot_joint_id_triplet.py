#!/usr/bin/python3
"""Plot 0.5/1.0/1.5 Hz target-real-sim traces as one vertical figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_series(path: Path, joint: int) -> tuple[list[float], list[float], list[float], list[float]]:
    t: list[float] = []
    target: list[float] = []
    real: list[float] = []
    sim: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t.append(float(row["t"]))
            if "target_q" in row:
                target.append(float(row["target_q"]))
                real.append(float(row["real_q"]))
                sim.append(float(row["sim_q"]))
            else:
                target.append(float(row[f"target_q{joint}"]))
                real.append(float(row[f"actual_q{joint}"]))
                sim.append(float(row[f"sim_q{joint}"]))
    if t:
        t0 = t[0]
        t = [x - t0 for x in t]
    return t, target, real, sim


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True, help="Merged CSV files ordered by frequency.")
    ap.add_argument("--labels", nargs="+", default=["0.5 Hz", "1.0 Hz", "1.5 Hz"])
    ap.add_argument("--joint", type=int, required=True, choices=range(1, 8))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--target-label", default="target")
    ap.add_argument("--real-label", default="real")
    ap.add_argument("--sim-label", default="sim")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    if len(args.inputs) != len(args.labels):
        raise SystemExit("--inputs and --labels must have the same length")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(args.inputs), 1, figsize=(12, 3.2 * len(args.inputs)), sharex=False)
    if len(args.inputs) == 1:
        axes = [axes]

    for ax, path, label in zip(axes, args.inputs, args.labels):
        t, target, real, sim = read_series(path, args.joint)
        ax.plot(t, target, label=args.target_label, linewidth=1.6)
        ax.plot(t, real, label=args.real_label, linewidth=1.2)
        ax.plot(t, sim, label=args.sim_label, linewidth=1.2)
        ax.set_title(label)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"joint{args.joint} q (rad)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
