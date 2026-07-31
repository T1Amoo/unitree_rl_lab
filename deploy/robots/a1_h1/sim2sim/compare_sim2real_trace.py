#!/usr/bin/env python3
"""Replay a sim2sim CSV trace through the sim2real deploy runtime."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
A1_H1_DIR = THIS_DIR.parent
LEGACY_SRC = A1_H1_DIR / "sim2real_python_legacy"
sys.path.insert(0, str(LEGACY_SRC))

from sim2real_bridge.policy_runtime import A1DeployPolicy, DEFAULT_POLICY  # noqa: E402


DEFAULT_MAX_DELTA = np.array([0.050, 0.050, 0.050, 0.100, 0.100, 0.100, 0.100], dtype=np.float64)


@dataclass
class DiffStats:
    rows: int = 0
    valid_rows: int = 0
    max_raw_action: float = 0.0
    max_raw_q_des: float = 0.0
    max_q_des: float = 0.0
    max_ball_pred: float = 0.0
    max_delta_over: float = 0.0
    worst_row: int = -1
    worst_field: str = ""

    def update(self, row_index: int, field: str, value: float) -> None:
        if value > getattr(self, field):
            setattr(self, field, value)
            if field in ("max_raw_action", "max_raw_q_des", "max_q_des", "max_ball_pred", "max_delta_over"):
                self.worst_row = row_index
                self.worst_field = field


def _vec(row: dict[str, str], prefix: str, n: int) -> np.ndarray:
    return np.array([float(row[f"{prefix}_{i}"]) for i in range(1, n + 1)], dtype=np.float64)


def _optional_vec(row: dict[str, str], prefix: str, n: int) -> np.ndarray | None:
    keys = [f"{prefix}_{i}" for i in range(1, n + 1)]
    if not all(k in row for k in keys):
        return None
    return np.array([float(row[k]) for k in keys], dtype=np.float64)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _limit_delta(q_des: np.ndarray, previous_q_des: np.ndarray, max_delta: np.ndarray | None) -> np.ndarray:
    if max_delta is None or np.all(max_delta <= 0.0):
        return q_des.copy()
    delta = np.clip(q_des - previous_q_des, -max_delta, max_delta)
    return previous_q_des + delta


def _trace_uses_bridge_limit(row: dict[str, str]) -> bool:
    value = row.get("bridge_qdes_limit")
    if value is None:
        return False
    return float(value) >= 0.5


def compare(args: argparse.Namespace) -> DiffStats:
    with args.trace.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty trace: {args.trace}")

    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    policy = A1DeployPolicy(args.policy, args.predictor, use_predictor=not args.no_predictor)
    first_q = _vec(rows[0], "q", 7)
    if args.reset_to_first_q:
        policy.reset(first_q)
        last_pub_q = first_q.copy()
    else:
        last_pub_q = policy.last_q_des.copy()

    stats = DiffStats()
    valid_samples: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        q = _vec(row, "q", 7)
        dq = _vec(row, "dq", 7)
        ball = _vec(row, "ball", 3)
        ball_vel = _vec(row, "ball_vel", 3)
        valid_ball = float(row["valid_ball"]) >= 0.5

        step = policy.step(q, dq, ball, ball_vel, valid_ball, args.dt)
        trace_action = _vec(row, "action", 7)
        trace_q_des = _vec(row, "q_des", 7)
        trace_ball_pred = _vec(row, "ball_pred", 3)
        trace_raw_q_des = _optional_vec(row, "raw_q_des", 7)
        max_delta = DEFAULT_MAX_DELTA if _trace_uses_bridge_limit(row) else None
        if args.force_bridge_limit:
            max_delta = DEFAULT_MAX_DELTA
        elif args.no_bridge_limit:
            max_delta = None

        limited_q_des = _limit_delta(step.q_des, last_pub_q, max_delta)
        delta_over = 0.0
        if max_delta is not None:
            delta_over = float(np.max(np.maximum(np.abs(limited_q_des - last_pub_q) - max_delta, 0.0)))
        last_pub_q = limited_q_des.copy()

        if row_index > args.warmup:
            stats.rows += 1
            stats.valid_rows += int(valid_ball)
            stats.update(row_index, "max_raw_action", _max_abs(step.raw_action, trace_action))
            if trace_raw_q_des is not None:
                stats.update(row_index, "max_raw_q_des", _max_abs(step.q_des, trace_raw_q_des))
            stats.update(row_index, "max_q_des", _max_abs(limited_q_des, trace_q_des))
            stats.update(row_index, "max_ball_pred", _max_abs(step.ball_pred, trace_ball_pred))
            stats.update(row_index, "max_delta_over", delta_over)

        if valid_ball:
            valid_samples.append(
                "row={row} raw={raw:.3e} raw_q={raw_q:.3e} q={qdiff:.3e} pred={pred:.3e} gate=valid".format(
                    row=row_index,
                    raw=_max_abs(step.raw_action, trace_action),
                    raw_q=0.0 if trace_raw_q_des is None else _max_abs(step.q_des, trace_raw_q_des),
                    qdiff=_max_abs(limited_q_des, trace_q_des),
                    pred=_max_abs(step.ball_pred, trace_ball_pred),
                )
            )

    print(
        "[compare_sim2real_trace] "
        f"trace={args.trace} rows={stats.rows} valid_rows={stats.valid_rows} warmup={args.warmup} "
        f"max_raw_action={stats.max_raw_action:.3e} "
        f"max_raw_q_des={stats.max_raw_q_des:.3e} "
        f"max_q_des={stats.max_q_des:.3e} "
        f"max_ball_pred={stats.max_ball_pred:.3e} "
        f"max_delta_over={stats.max_delta_over:.3e} "
        f"worst={stats.worst_field}@row{stats.worst_row}"
    )
    samples: list[str] = []
    if valid_samples:
        count = min(args.samples, len(valid_samples))
        sample_indexes = np.linspace(0, len(valid_samples) - 1, count, dtype=int)
        samples = [valid_samples[int(i)] for i in sample_indexes]
    for sample in samples:
        print(f"[compare_sim2real_trace] sample {sample}")

    max_diff = max(stats.max_raw_action, stats.max_raw_q_des, stats.max_q_des, stats.max_ball_pred, stats.max_delta_over)
    if max_diff > args.tolerance:
        raise SystemExit(
            f"compare failed: max diff {max_diff:.3e} exceeds tolerance {args.tolerance:.3e}"
        )
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path, help="CSV produced by run_a1_tt_sim2sim.py --trace-csv.")
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--predictor", type=Path, default=None)
    ap.add_argument("--no-predictor", action="store_true")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=5, help="Ignore the first N rows for pass/fail statistics.")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--samples", type=int, default=8, help="Print up to N valid-frame sample diffs.")
    ap.add_argument("--tolerance", type=float, default=2e-5)
    ap.add_argument("--reset-to-first-q", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--force-bridge-limit", action="store_true")
    ap.add_argument("--no-bridge-limit", action="store_true")
    return ap.parse_args(argv)


if __name__ == "__main__":
    compare(parse_args())
