#!/usr/bin/python3
"""Fit local bias, gain, and phase lag for a chirp joint-ID response."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


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


def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def fit_local_sine(t: np.ndarray, y: np.ndarray, freq_hz: float) -> tuple[float, float, float, float]:
    """Fit y ~= offset + a*sin(wt) + b*cos(wt). Return offset, amplitude, phase, r2."""
    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]
    if t.size < 12:
        return math.nan, math.nan, math.nan, math.nan
    omega = 2.0 * math.pi * freq_hz
    x = np.column_stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)])
    coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
    offset, sin_c, cos_c = [float(v) for v in coeff]
    pred = x @ coeff
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else math.nan
    return offset, math.hypot(sin_c, cos_c), math.atan2(cos_c, sin_c), r2


def load_merged(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rows = read_rows(path)
    if not rows:
        raise SystemExit(f"empty merged CSV: {path}")
    t = np.array([get_float(r, "relative_s") for r in rows], dtype=float)
    target = np.array([get_float(r, "target_raw_q") for r in rows], dtype=float)
    series = {
        "cmd": np.array([get_float(r, "cmd_q") for r in rows], dtype=float),
        "real": np.array([get_float(r, "real_actual_q") for r in rows], dtype=float),
    }
    return t, target, series


def load_sim(path: Path, joint: int, t_ref: np.ndarray, target_ref: np.ndarray) -> tuple[np.ndarray, float]:
    rows = read_rows(path)
    if not rows:
        raise SystemExit(f"empty sim CSV: {path}")
    t_sim = np.array([get_float(r, "t") for r in rows], dtype=float)
    target_sim = np.array([get_float(r, f"target_q{joint}") for r in rows], dtype=float)
    q_sim = np.array([get_float(r, f"sim_q{joint}") for r in rows], dtype=float)
    best_err = math.inf
    best_offset = 0.0
    for offset in np.arange(0.0, 5.0001, 0.0005):
        interp_target = np.interp(t_ref + offset, t_sim, target_sim)
        err = float(np.nanmean((interp_target - target_ref) ** 2))
        if err < best_err:
            best_err = err
            best_offset = float(offset)
    return np.interp(t_ref + best_offset, t_sim, q_sim), best_offset


def parse_sim_arg(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--sim expects label=/path/to/sim.csv")
    label, path = text.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("sim label cannot be empty")
    return label, Path(path)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--joint", type=int, default=1, choices=range(1, 8))
    ap.add_argument("--sim", action="append", type=parse_sim_arg, default=[])
    ap.add_argument("--freq-start", type=float, default=0.1)
    ap.add_argument("--freq-end", type=float, default=2.0)
    ap.add_argument("--step-s", type=float, default=0.25)
    ap.add_argument("--window-cycles", type=float, default=2.0)
    ap.add_argument("--min-window-s", type=float, default=2.0)
    ap.add_argument("--max-window-s", type=float, default=4.0)
    ap.add_argument("--min-target-r2", type=float, default=0.88)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--plot", type=Path, required=True)
    ap.add_argument("--title", default="")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    t, target, series = load_merged(args.merged)
    sim_offsets: dict[str, float] = {}
    for label, path in args.sim:
        series[label], sim_offsets[label] = load_sim(path, args.joint, t, target)

    t0 = float(np.nanmin(t))
    duration = float(np.nanmax(t) - t0)
    if duration <= 0.0:
        raise SystemExit("invalid time span")

    centers = np.arange(t0 + args.max_window_s * 0.5, t0 + duration - args.max_window_s * 0.5, args.step_s)
    rows_out: list[dict[str, float | str | int]] = []
    for center in centers:
        alpha = (center - t0) / duration
        freq = args.freq_start + (args.freq_end - args.freq_start) * alpha
        window_s = min(max(args.min_window_s, args.window_cycles / max(freq, 1e-6)), args.max_window_s)
        mask = (t >= center - 0.5 * window_s) & (t <= center + 0.5 * window_s)
        if int(mask.sum()) < 20:
            continue
        target_offset, target_amp, target_phase, target_r2 = fit_local_sine(t[mask], target[mask], freq)
        if not np.isfinite(target_amp) or target_amp < 1e-6 or target_r2 < args.min_target_r2:
            continue
        for label, values in series.items():
            offset, amp, phase, r2 = fit_local_sine(t[mask], values[mask], freq)
            phase_lag_rad = wrap_to_pi(target_phase - phase)
            rows_out.append(
                {
                    "series": label,
                    "center_s": center,
                    "freq_hz": freq,
                    "window_s": window_s,
                    "target_offset": target_offset,
                    "target_amp": target_amp,
                    "target_r2": target_r2,
                    "response_offset": offset,
                    "response_amp": amp,
                    "response_r2": r2,
                    "bias_rad": offset - target_offset,
                    "gain": amp / target_amp,
                    "phase_lag_rad": phase_lag_rad,
                    "phase_lag_s": phase_lag_rad / (2.0 * math.pi * freq),
                    "sim_align_offset_s": sim_offsets.get(label, math.nan),
                }
            )

    if not rows_out:
        raise SystemExit("no valid fit windows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_out[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    import matplotlib.pyplot as plt

    labels = list(series.keys())
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for label in labels:
        data = [r for r in rows_out if r["series"] == label]
        freq = np.array([float(r["freq_hz"]) for r in data])
        bias = np.array([float(r["bias_rad"]) for r in data])
        gain = np.array([float(r["gain"]) for r in data])
        lag_ms = np.array([1000.0 * float(r["phase_lag_s"]) for r in data])
        r2 = np.array([float(r["response_r2"]) for r in data])
        axes[0].plot(freq, bias, label=label)
        axes[1].plot(freq, gain, label=label)
        axes[2].plot(freq, lag_ms, label=label)
        axes[3].plot(freq, r2, label=label)

    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("DC bias (rad)")
    axes[1].set_ylabel("gain")
    axes[2].set_ylabel("phase lag (ms)")
    axes[3].set_ylabel("fit R2")
    axes[3].set_xlabel("frequency (Hz)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=160)

    print(f"wrote {args.output}")
    print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
