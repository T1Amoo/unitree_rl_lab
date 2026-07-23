#!/usr/bin/env python3
"""Validate, merge, fit, and plot G1 joint-ID real/sim chirp results."""

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


def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def fit_local_sine(t: np.ndarray, y: np.ndarray, freq_hz: float) -> tuple[float, float, float, float]:
    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]
    if t.size < 20:
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


def quat_to_rpy(w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


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
        "rmse_rad": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_rad": float(np.max(np.abs(diff))),
        "mean_abs_rad": float(np.mean(np.abs(diff))),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--sim-fixed", type=Path, required=True)
    ap.add_argument("--sim-held", type=Path, default=None)
    ap.add_argument("--sim-free", type=Path, default=None)
    ap.add_argument(
        "--sim-extra",
        nargs=2,
        action="append",
        default=[],
        metavar=("LABEL", "CSV"),
        help="optional additional sim curve, e.g. --sim-extra lower_policy path.csv",
    )
    ap.add_argument("--joint", type=int, required=True, choices=range(1, 6))
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--fit-output", type=Path, required=True)
    ap.add_argument("--plot", type=Path, required=True)
    ap.add_argument("--freq-start", type=float, default=0.1)
    ap.add_argument("--freq-end", type=float, default=2.0)
    ap.add_argument("--window-cycles", type=float, default=2.0)
    ap.add_argument("--min-window-s", type=float, default=2.0)
    ap.add_argument("--max-window-s", type=float, default=4.0)
    ap.add_argument("--step-s", type=float, default=0.25)
    return ap


def clean_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in label.strip()).strip("_") or "extra"


def main() -> int:
    args = build_arg_parser().parse_args()
    real_rows = read_rows(args.real)
    fixed_rows = read_rows(args.sim_fixed)
    other_path = args.sim_held or args.sim_free
    other_label = "held" if args.sim_held else "free"
    if other_path is None:
        raise SystemExit("one of --sim-held or --sim-free is required")
    other_rows = read_rows(other_path)
    extra_inputs = [(clean_label(label), Path(path)) for label, path in args.sim_extra]
    extra_rows_by_label = {label: read_rows(path) for label, path in extra_inputs}
    if not real_rows or not fixed_rows or not other_rows or any(not rows for rows in extra_rows_by_label.values()):
        raise SystemExit("empty input")
    n = min(len(real_rows), len(fixed_rows), len(other_rows), *(len(rows) for rows in extra_rows_by_label.values()))
    real_rows = real_rows[:n]
    fixed_rows = fixed_rows[:n]
    other_rows = other_rows[:n]
    extra_rows_by_label = {label: rows[:n] for label, rows in extra_rows_by_label.items()}

    policy_i = RIGHT_ARM_POLICY_INDICES[args.joint - 1] + 1
    t = np.array([get_float(r, "t") for r in real_rows], dtype=float)
    stage = np.array([r.get("stage", "") for r in real_rows])
    chirp_mask = stage == "chirp"
    target = np.array([get_float(r, f"target_q{policy_i}") for r in real_rows], dtype=float)
    real = np.array([get_float(r, f"actual_q{policy_i}") for r in real_rows], dtype=float)
    fixed = np.array([get_float(r, f"sim_q{policy_i}") for r in fixed_rows], dtype=float)
    other = np.array([get_float(r, f"sim_q{policy_i}") for r in other_rows], dtype=float)
    extra = {
        label: np.array([get_float(r, f"sim_q{policy_i}") for r in rows], dtype=float)
        for label, rows in extra_rows_by_label.items()
    }

    root_x = np.array([get_float(r, "root_x") for r in other_rows], dtype=float)
    root_y = np.array([get_float(r, "root_y") for r in other_rows], dtype=float)
    root_z = np.array([get_float(r, "root_z") for r in other_rows], dtype=float)
    qw = np.array([get_float(r, "root_qw") for r in other_rows], dtype=float)
    qx = np.array([get_float(r, "root_qx") for r in other_rows], dtype=float)
    qy = np.array([get_float(r, "root_qy") for r in other_rows], dtype=float)
    qz = np.array([get_float(r, "root_qz") for r in other_rows], dtype=float)
    roll, pitch, yaw = quat_to_rpy(qw, qx, qy, qz)
    extra_roots = {}
    for label, rows in extra_rows_by_label.items():
        ex = np.array([get_float(r, "root_x") for r in rows], dtype=float)
        ey = np.array([get_float(r, "root_y") for r in rows], dtype=float)
        ez = np.array([get_float(r, "root_z") for r in rows], dtype=float)
        eqw = np.array([get_float(r, "root_qw") for r in rows], dtype=float)
        eqx = np.array([get_float(r, "root_qx") for r in rows], dtype=float)
        eqy = np.array([get_float(r, "root_qy") for r in rows], dtype=float)
        eqz = np.array([get_float(r, "root_qz") for r in rows], dtype=float)
        eroll, epitch, eyaw = quat_to_rpy(eqw, eqx, eqy, eqz)
        extra_roots[label] = (ex, ey, ez, eroll, epitch, eyaw)

    args.merged.parent.mkdir(parents=True, exist_ok=True)
    with args.merged.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "t", "stage", "joint", "joint_name", "target_q", "real_q",
            "sim_fixed_q", f"sim_{other_label}_q", f"fixed_minus_{other_label}",
            "real_minus_target", "fixed_minus_real", f"{other_label}_minus_real",
            f"{other_label}_root_x", f"{other_label}_root_y", f"{other_label}_root_z",
            f"{other_label}_root_roll", f"{other_label}_root_pitch", f"{other_label}_root_yaw",
        ]
        for label in extra:
            fieldnames += [
                f"sim_{label}_q",
                f"fixed_minus_{label}",
                f"{label}_minus_real",
                f"{label}_root_x",
                f"{label}_root_y",
                f"{label}_root_z",
                f"{label}_root_roll",
                f"{label}_root_pitch",
                f"{label}_root_yaw",
            ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            out = {
                "t": t[i],
                "stage": stage[i],
                "joint": args.joint,
                "joint_name": RIGHT_ARM_JOINT_NAMES[args.joint - 1],
                "target_q": target[i],
                "real_q": real[i],
                "sim_fixed_q": fixed[i],
                f"sim_{other_label}_q": other[i],
                f"fixed_minus_{other_label}": fixed[i] - other[i],
                "real_minus_target": real[i] - target[i],
                "fixed_minus_real": fixed[i] - real[i],
                f"{other_label}_minus_real": other[i] - real[i],
                f"{other_label}_root_x": root_x[i],
                f"{other_label}_root_y": root_y[i],
                f"{other_label}_root_z": root_z[i],
                f"{other_label}_root_roll": roll[i],
                f"{other_label}_root_pitch": pitch[i],
                f"{other_label}_root_yaw": yaw[i],
            }
            for label, values in extra.items():
                ex, ey, ez, eroll, epitch, eyaw = extra_roots[label]
                out.update({
                    f"sim_{label}_q": values[i],
                    f"fixed_minus_{label}": fixed[i] - values[i],
                    f"{label}_minus_real": values[i] - real[i],
                    f"{label}_root_x": ex[i],
                    f"{label}_root_y": ey[i],
                    f"{label}_root_z": ez[i],
                    f"{label}_root_roll": eroll[i],
                    f"{label}_root_pitch": epitch[i],
                    f"{label}_root_yaw": eyaw[i],
                })
            writer.writerow(out)

    diff = fixed - other
    active_diff = diff[chirp_mask]
    root_disp = np.sqrt((root_x - root_x[0]) ** 2 + (root_y - root_y[0]) ** 2 + (root_z - root_z[0]) ** 2)
    summary = {
        "real_csv": str(args.real),
        "sim_fixed_csv": str(args.sim_fixed),
        f"sim_{other_label}_csv": str(other_path),
        "sim_extra_csv": {label: str(path) for label, path in extra_inputs},
        "merged_csv": str(args.merged),
        "joint": args.joint,
        "joint_name": RIGHT_ARM_JOINT_NAMES[args.joint - 1],
        "rows": n,
        "stage_counts": {s: int(np.sum(stage == s)) for s in sorted(set(stage.tolist()))},
        "finite_actual_q": int(np.sum(np.isfinite(real))),
        "finite_target_q": int(np.sum(np.isfinite(target))),
        "finite_fixed_q": int(np.sum(np.isfinite(fixed))),
        f"finite_{other_label}_q": int(np.sum(np.isfinite(other))),
        "finite_extra_q": {label: int(np.sum(np.isfinite(values))) for label, values in extra.items()},
        "chirp_t_range": [float(np.min(t[chirp_mask])), float(np.max(t[chirp_mask]))] if np.any(chirp_mask) else [],
        "target_q_range_chirp": range_stats(target[chirp_mask]),
        "real_q_range_chirp": range_stats(real[chirp_mask]),
        "sim_fixed_q_range_chirp": range_stats(fixed[chirp_mask]),
        f"sim_{other_label}_q_range_chirp": range_stats(other[chirp_mask]),
        "sim_extra_q_range_chirp": {label: range_stats(values[chirp_mask]) for label, values in extra.items()},
        f"fixed_vs_{other_label}_chirp": {
            "rmse_rad": float(np.sqrt(np.nanmean(active_diff ** 2))),
            "max_abs_rad": float(np.nanmax(np.abs(active_diff))),
            "mean_abs_rad": float(np.nanmean(np.abs(active_diff))),
        },
        "fixed_vs_extra_chirp": {
            label: error_stats((fixed - values)[chirp_mask])
            for label, values in extra.items()
        },
        "target_vs_real_chirp": error_stats(target[chirp_mask] - real[chirp_mask]),
        "fixed_vs_real_chirp": error_stats(fixed[chirp_mask] - real[chirp_mask]),
        f"{other_label}_vs_real_chirp": error_stats(other[chirp_mask] - real[chirp_mask]),
        "extra_vs_real_chirp": {
            label: error_stats(values[chirp_mask] - real[chirp_mask])
            for label, values in extra.items()
        },
        "stage_error_stats": {
            name: {
                "target_vs_real": error_stats(target[stage == name] - real[stage == name]),
                "fixed_vs_real": error_stats(fixed[stage == name] - real[stage == name]),
                f"{other_label}_vs_real": error_stats(other[stage == name] - real[stage == name]),
                **{
                    f"{label}_vs_real": error_stats(values[stage == name] - real[stage == name])
                    for label, values in extra.items()
                },
            }
            for name in sorted(set(stage.tolist()))
        },
        f"{other_label}_root_motion": {
            "max_displacement_m": float(np.nanmax(root_disp)),
            "chirp_max_displacement_m": float(np.nanmax(root_disp[chirp_mask])),
            "x": range_stats(root_x[chirp_mask]),
            "y": range_stats(root_y[chirp_mask]),
            "z": range_stats(root_z[chirp_mask]),
            "roll_rad": range_stats(roll[chirp_mask]),
            "pitch_rad": range_stats(pitch[chirp_mask]),
            "yaw_rad": range_stats(yaw[chirp_mask]),
        },
        "extra_root_motion": {},
    }
    for label, root_values in extra_roots.items():
        ex, ey, ez, eroll, epitch, eyaw = root_values
        disp = np.sqrt((ex - ex[0]) ** 2 + (ey - ey[0]) ** 2 + (ez - ez[0]) ** 2)
        summary["extra_root_motion"][label] = {
            "max_displacement_m": float(np.nanmax(disp)),
            "chirp_max_displacement_m": float(np.nanmax(disp[chirp_mask])),
            "x": range_stats(ex[chirp_mask]),
            "y": range_stats(ey[chirp_mask]),
            "z": range_stats(ez[chirp_mask]),
            "roll_rad": range_stats(eroll[chirp_mask]),
            "pitch_rad": range_stats(epitch[chirp_mask]),
            "yaw_rad": range_stats(eyaw[chirp_mask]),
        }

    t_chirp = t[chirp_mask]
    if t_chirp.size > 0:
        t0 = float(np.min(t_chirp))
        duration = float(np.max(t_chirp) - t0)
    else:
        t0 = float(np.min(t))
        duration = float(np.max(t) - t0)
    series = {"real": real, "sim_fixed": fixed, f"sim_{other_label}": other}
    series.update({f"sim_{label}": values for label, values in extra.items()})
    fit_rows: list[dict[str, float | str]] = []
    if duration > 0.0:
        centers = np.arange(t0 + args.max_window_s * 0.5, t0 + duration - args.max_window_s * 0.5, args.step_s)
        for center in centers:
            alpha = (center - t0) / duration
            freq = args.freq_start + (args.freq_end - args.freq_start) * alpha
            window_s = min(max(args.min_window_s, args.window_cycles / max(freq, 1e-6)), args.max_window_s)
            mask = chirp_mask & (t >= center - 0.5 * window_s) & (t <= center + 0.5 * window_s)
            if int(mask.sum()) < 50:
                continue
            target_offset, target_amp, target_phase, target_r2 = fit_local_sine(t[mask] - t0, target[mask], freq)
            if not np.isfinite(target_amp) or target_amp < 1e-6:
                continue
            for label, values in series.items():
                offset, amp, phase, r2 = fit_local_sine(t[mask] - t0, values[mask], freq)
                phase_lag_rad = wrap_to_pi(target_phase - phase)
                fit_rows.append({
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
                })

    args.fit_output.parent.mkdir(parents=True, exist_ok=True)
    with args.fit_output.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(fit_rows[0].keys()) if fit_rows else ["series"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fit_rows)
    summary["fit_csv"] = str(args.fit_output)
    summary["fit_windows"] = len(fit_rows)
    write_json(args.summary, summary)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=False)
    axes[0].plot(t, target, label="target", linewidth=1.2)
    axes[0].plot(t, real, label="real", linewidth=1.0)
    axes[0].plot(t, fixed, label="sim fixed", linewidth=1.0)
    axes[0].plot(t, other, label=f"sim {other_label}", linewidth=1.0)
    for label, values in extra.items():
        axes[0].plot(t, values, label=f"sim {label}", linewidth=1.0)
    axes[0].set_title(f"G1 joint{args.joint} {RIGHT_ARM_JOINT_NAMES[args.joint - 1]}")
    axes[0].set_ylabel("q (rad)")
    axes[0].legend(loc="best", ncol=4, fontsize=8)

    if np.any(chirp_mask):
        zoom_start = max(float(np.max(t[chirp_mask])) - 10.0, float(np.min(t[chirp_mask])))
        zoom = chirp_mask & (t >= zoom_start)
    else:
        zoom = np.ones_like(t, dtype=bool)
    for y, label in ((target, "target"), (real, "real"), (fixed, "sim fixed"), (other, f"sim {other_label}")):
        axes[1].plot(t[zoom], y[zoom], label=label, linewidth=1.0)
    for label, values in extra.items():
        axes[1].plot(t[zoom], values[zoom], label=f"sim {label}", linewidth=1.0)
    axes[1].set_title("last 10 s active chirp")
    axes[1].set_ylabel("q (rad)")
    axes[1].legend(loc="best", ncol=4, fontsize=8)

    axes[2].plot(t[chirp_mask], (real - target)[chirp_mask], label="real-target", linewidth=1.0)
    axes[2].plot(t[chirp_mask], (fixed - real)[chirp_mask], label="fixed-real", linewidth=1.0)
    axes[2].plot(t[chirp_mask], (other - real)[chirp_mask], label=f"{other_label}-real", linewidth=1.0)
    axes[2].plot(t[chirp_mask], (fixed - other)[chirp_mask], label=f"fixed-{other_label}", linewidth=1.0)
    for label, values in extra.items():
        axes[2].plot(t[chirp_mask], (values - real)[chirp_mask], label=f"{label}-real", linewidth=1.0)
        axes[2].plot(t[chirp_mask], (fixed - values)[chirp_mask], label=f"fixed-{label}", linewidth=0.9)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_title("residuals during chirp")
    axes[2].set_ylabel("rad")
    axes[2].legend(loc="best", ncol=4, fontsize=8)

    axes[3].plot(t[chirp_mask], root_x[chirp_mask] - root_x[0], label=f"{other_label} root dx")
    axes[3].plot(t[chirp_mask], root_y[chirp_mask] - root_y[0], label=f"{other_label} root dy")
    axes[3].plot(t[chirp_mask], root_z[chirp_mask] - root_z[0], label=f"{other_label} root dz")
    for label, root_values in extra_roots.items():
        ex, ey, ez, *_ = root_values
        axes[3].plot(t[chirp_mask], ex[chirp_mask] - ex[0], label=f"{label} root dx", linewidth=0.9)
        axes[3].plot(t[chirp_mask], ey[chirp_mask] - ey[0], label=f"{label} root dy", linewidth=0.9)
        axes[3].plot(t[chirp_mask], ez[chirp_mask] - ez[0], label=f"{label} root dz", linewidth=0.9)
    axes[3].set_title("root translation")
    axes[3].set_ylabel("m")
    axes[3].legend(loc="best", ncol=3, fontsize=8)

    if fit_rows:
        fit_labels = ["real", "sim_fixed", f"sim_{other_label}"] + [f"sim_{label}" for label in extra]
        for label in fit_labels:
            data = [r for r in fit_rows if r["series"] == label]
            freq = np.array([float(r["freq_hz"]) for r in data])
            gain = np.array([float(r["gain"]) for r in data])
            axes[4].plot(freq, gain, label=label)
        axes[4].axhline(1.0, color="black", linewidth=0.8)
    axes[4].set_title("local sine gain")
    axes[4].set_xlabel("frequency (Hz)")
    axes[4].set_ylabel("gain")
    axes[4].legend(loc="best", fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=160)

    print(f"wrote {args.merged}")
    print(f"wrote {args.summary}")
    print(f"wrote {args.fit_output}")
    print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
