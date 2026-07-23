#!/usr/bin/python3
"""Merge target, real response, and simulated response into analysis tables."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from joint_id_common import get_float, open_csv_writer, wrap_to_pi


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fit_sine(t: np.ndarray, y: np.ndarray, freq_hz: float) -> tuple[float, float, float]:
    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]
    if t.size < 8:
        return math.nan, math.nan, math.nan
    omega = 2.0 * math.pi * freq_hz
    a = np.column_stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)])
    coeff, *_ = np.linalg.lstsq(a, y, rcond=None)
    offset, sin_c, cos_c = [float(x) for x in coeff]
    amp = math.hypot(sin_c, cos_c)
    phase = math.atan2(cos_c, sin_c)
    return offset, amp, phase


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--sim", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="Merged sample-level CSV.")
    ap.add_argument("--summary", type=Path, default=Path("joint_id_summary.csv"))
    ap.add_argument("--ignore-start-s", type=float, default=1.0)
    ap.add_argument("--only-phase", default="sine")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    real_rows = read_rows(args.real)
    sim_rows = read_rows(args.sim)
    if not real_rows:
        raise SystemExit(f"empty real CSV: {args.real}")
    if not sim_rows:
        raise SystemExit(f"empty sim CSV: {args.sim}")

    joint = int(float(real_rows[0].get("joint_index", "1")))
    freq = get_float(real_rows[0], "freq_hz")
    t_real = np.array([get_float(r, "t") for r in real_rows], dtype=float)
    t_sim = np.array([get_float(r, "t") for r in sim_rows], dtype=float)
    target = np.array([get_float(r, f"target_q{joint}") for r in real_rows], dtype=float)
    real = np.array([get_float(r, f"actual_q{joint}") for r in real_rows], dtype=float)
    sim = np.array([get_float(r, f"sim_q{joint}") for r in sim_rows], dtype=float)
    sim_interp = np.interp(t_real, t_sim, sim)

    fields = [
        "run_id",
        "t",
        "phase",
        "joint_index",
        "joint_name",
        "freq_hz",
        "amplitude_rad",
        "target_q",
        "real_q",
        "sim_q",
        "real_minus_target",
        "sim_minus_target",
        "real_minus_sim",
    ]
    writer, handle = open_csv_writer(args.output, fields)
    try:
        for i, row in enumerate(real_rows):
            writer.writerow(
                {
                    "run_id": row.get("run_id", ""),
                    "t": t_real[i],
                    "phase": row.get("phase", ""),
                    "joint_index": joint,
                    "joint_name": row.get("joint_name", ""),
                    "freq_hz": freq,
                    "amplitude_rad": row.get("amplitude_rad", ""),
                    "target_q": target[i],
                    "real_q": real[i],
                    "sim_q": sim_interp[i],
                    "real_minus_target": real[i] - target[i],
                    "sim_minus_target": sim_interp[i] - target[i],
                    "real_minus_sim": real[i] - sim_interp[i],
                }
            )
    finally:
        handle.close()

    phase = [r.get("phase", "") for r in real_rows]
    mask = np.isfinite(t_real) & (t_real >= t_real[0] + args.ignore_start_s)
    if args.only_phase:
        mask &= np.array([p == args.only_phase for p in phase], dtype=bool)
    if mask.sum() < 8:
        mask = np.isfinite(t_real)

    t_fit = t_real[mask] - t_real[mask][0]
    target_fit = target[mask]
    real_fit = real[mask]
    sim_fit = sim_interp[mask]
    _, target_amp, target_phase = fit_sine(t_fit, target_fit, freq)
    _, real_amp, real_phase = fit_sine(t_fit, real_fit, freq)
    _, sim_amp, sim_phase = fit_sine(t_fit, sim_fit, freq)
    real_rmse = float(np.sqrt(np.nanmean((real_fit - target_fit) ** 2)))
    sim_rmse = float(np.sqrt(np.nanmean((sim_fit - target_fit) ** 2)))
    real_sim_rmse = float(np.sqrt(np.nanmean((real_fit - sim_fit) ** 2)))

    summary_fields = [
        "run_id",
        "joint_index",
        "joint_name",
        "freq_hz",
        "target_amp",
        "real_amp",
        "sim_amp",
        "real_gain",
        "sim_gain",
        "real_phase_lag_rad",
        "sim_phase_lag_rad",
        "real_rmse_to_target",
        "sim_rmse_to_target",
        "real_sim_rmse",
        "n_fit",
    ]
    summary_writer, summary_handle = open_csv_writer(args.summary, summary_fields)
    try:
        summary_writer.writerow(
            {
                "run_id": real_rows[0].get("run_id", ""),
                "joint_index": joint,
                "joint_name": real_rows[0].get("joint_name", ""),
                "freq_hz": freq,
                "target_amp": target_amp,
                "real_amp": real_amp,
                "sim_amp": sim_amp,
                "real_gain": real_amp / target_amp if target_amp > 1e-9 else math.nan,
                "sim_gain": sim_amp / target_amp if target_amp > 1e-9 else math.nan,
                "real_phase_lag_rad": wrap_to_pi(real_phase - target_phase),
                "sim_phase_lag_rad": wrap_to_pi(sim_phase - target_phase),
                "real_rmse_to_target": real_rmse,
                "sim_rmse_to_target": sim_rmse,
                "real_sim_rmse": real_sim_rmse,
                "n_fit": int(mask.sum()),
            }
        )
    finally:
        summary_handle.close()
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
