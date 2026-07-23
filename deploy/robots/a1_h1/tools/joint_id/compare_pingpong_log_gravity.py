#!/usr/bin/python3
"""Compare A1 pingpong servo logs against IsaacLab gravity torque."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Pingpong servo CSV with parameter comments.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--sample-stride", type=int, default=1, help="Evaluate every Nth row in Isaac.")
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap after stride. 0 means no cap.")
    ap.add_argument("--legged-lab-root", type=Path, default=Path("Pingpong_TTRL/legged_lab"))
    ap.add_argument("--dt", type=float, default=0.005)
    AppLauncher.add_app_launcher_args(ap)
    return ap


args_cli = build_arg_parser().parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app = AppLauncher(args_cli).app

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402


JOINTS = [f"j{i}" for i in range(1, 8)]
BASE_FIELDS = ["sample_id", "ros_time_s", "stage"]
SERIES_PREFIXES = ("q_plan", "dq_plan", "q_actual", "dq_actual", "tau_actual", "tau_ff")


def parse_header_params(path: Path) -> dict[str, object]:
    params: dict[str, object] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            text = line[1:].strip()
            if not text or text.startswith("===") or ":" not in text:
                continue
            key, value = text.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                parsed: object = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
            params[key] = parsed
    return params


def list7_from_params(params: dict[str, object], key: str, default: list[float]) -> list[float]:
    value = params.get(key, default)
    if isinstance(value, str):
        cleaned = value.strip().strip("[]()")
        out = [float(x) for x in cleaned.replace(",", " ").split()]
    elif isinstance(value, (list, tuple)):
        out = [float(x) for x in value]
    else:
        out = [float(value)] * 7
    if len(out) != 7:
        raise ValueError(f"{key} must contain 7 values, got {len(out)}: {value!r}")
    return out


def get_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_log(path: Path, sample_stride: int, max_samples: int) -> tuple[list[dict[str, object]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        data_lines = [line for line in f if not line.startswith("#")]
    reader = csv.DictReader(data_lines)
    if reader.fieldnames is None:
        raise SystemExit(f"missing CSV header: {path}")

    required = list(BASE_FIELDS)
    for prefix in SERIES_PREFIXES:
        required.extend(f"{prefix}_{i}" for i in range(7))
    missing = [col for col in required if col not in reader.fieldnames]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    for raw_idx, row in enumerate(reader):
        if raw_idx % sample_stride != 0:
            continue
        if any(row.get(col) is None or row.get(col) == "" for col in required):
            continue
        out: dict[str, object] = {}
        out["sample_id"] = int(float(row["sample_id"]))
        out["ros_time_s"] = get_float(row, "ros_time_s")
        out["stage"] = row.get("stage", "")
        for name in reader.fieldnames:
            if name in out or name == "stage":
                continue
            value = row.get(name, "")
            if value is None or value == "":
                out[name] = ""
                continue
            try:
                out[name] = float(value)
            except ValueError:
                out[name] = value
        rows.append(out)
        if max_samples > 0 and len(rows) >= max_samples:
            break
    return rows, list(reader.fieldnames)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return math.nan
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def summarize_values(values_a: list[float], values_b: list[float]) -> dict[str, float | int]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diff = a - b
    mask = np.isfinite(diff)
    if int(mask.sum()) == 0:
        return {"n": 0, "bias": math.nan, "mae": math.nan, "rmse": math.nan, "corr": math.nan}
    return {
        "n": int(mask.sum()),
        "bias": float(np.nanmean(diff)),
        "mae": float(np.nanmean(np.abs(diff))),
        "rmse": float(np.sqrt(np.nanmean(diff * diff))),
        "corr": corrcoef(a, b),
    }


def summarize_pair(
    long_rows: list[dict[str, object]],
    lhs: str,
    rhs: str,
    group_cols: list[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], tuple[list[float], list[float]]] = {}
    for row in long_rows:
        key = tuple(row[col] for col in group_cols)
        if key not in grouped:
            grouped[key] = ([], [])
        grouped[key][0].append(float(row[lhs]))
        grouped[key][1].append(float(row[rhs]))

    out: list[dict[str, object]] = []
    for key in sorted(grouped.keys(), key=lambda x: tuple(str(v) for v in x)):
        item = {col: key[idx] for idx, col in enumerate(group_cols)}
        a_values, b_values = grouped[key]
        item.update(summarize_values(a_values, b_values))
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_long_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    long_rows: list[dict[str, object]] = []
    for row in rows:
        speed_norm = math.sqrt(sum(float(row[f"dq_actual_{i}"]) ** 2 for i in range(7)))
        low_speed = speed_norm < 0.25
        for i, joint in enumerate(JOINTS):
            tau_ff = float(row[f"tau_ff_{i}"])
            tau_actual = float(row[f"tau_actual_{i}"])
            tau_pdff = float(row[f"tau_pdff_{i}"])
            tau_isaac = float(row[f"tau_isaac_g_{i}"])
            long_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "ros_time_s": row["ros_time_s"],
                    "stage": row["stage"],
                    "joint": joint,
                    "q_plan": row[f"q_plan_{i}"],
                    "dq_plan": row[f"dq_plan_{i}"],
                    "q_actual": row[f"q_actual_{i}"],
                    "dq_actual": row[f"dq_actual_{i}"],
                    "tau_actual": tau_actual,
                    "tau_ff": tau_ff,
                    "tau_pd": row[f"tau_pd_{i}"],
                    "tau_pdff": tau_pdff,
                    "tau_isaac_g": tau_isaac,
                    "ff_minus_isaac_g": tau_ff - tau_isaac,
                    "actual_minus_pdff": tau_actual - tau_pdff,
                    "actual_minus_isaac_g": tau_actual - tau_isaac,
                    "low_speed_all_joints": int(low_speed),
                }
            )
    return long_rows


def write_summary(
    rows: list[dict[str, object]],
    long_rows: list[dict[str, object]],
    output_dir: Path,
    params: dict[str, object],
) -> dict[str, object]:
    long_fields = [
        "sample_id",
        "ros_time_s",
        "stage",
        "joint",
        "q_plan",
        "dq_plan",
        "q_actual",
        "dq_actual",
        "tau_actual",
        "tau_ff",
        "tau_pd",
        "tau_pdff",
        "tau_isaac_g",
        "ff_minus_isaac_g",
        "actual_minus_pdff",
        "actual_minus_isaac_g",
        "low_speed_all_joints",
    ]
    long_path = output_dir / "pingpong_log_12_torque_compare_long.csv"
    write_csv(long_path, long_rows, long_fields)

    metric_rows: list[dict[str, object]] = []
    comparisons = (
        ("tau_ff", "tau_isaac_g", "tau_ff_vs_isaac_g"),
        ("tau_actual", "tau_pdff", "tau_actual_vs_pdff"),
        ("tau_actual", "tau_isaac_g", "tau_actual_vs_isaac_g"),
    )
    for lhs, rhs, label in comparisons:
        for item in summarize_pair(long_rows, lhs, rhs, ["joint", "stage"]):
            item = {"comparison": label, **item}
            metric_rows.append(item)
        low_rows = [row for row in long_rows if int(row["low_speed_all_joints"]) == 1]
        for item in summarize_pair(low_rows, lhs, rhs, ["joint"]):
            item = {"comparison": label, "stage": "low_speed_all_joints", **item}
            metric_rows.append(item)
        for item in summarize_pair(long_rows, lhs, rhs, ["joint"]):
            item = {"comparison": label, "stage": "all", **item}
            metric_rows.append(item)

    metric_fields = ["comparison", "joint", "stage", "n", "bias", "mae", "rmse", "corr"]
    metrics_path = output_dir / "pingpong_log_12_torque_compare_metrics.csv"
    write_csv(metrics_path, metric_rows, metric_fields)

    stage_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        stage_counts[str(row["stage"])] += 1

    key_metrics: list[dict[str, object]] = []
    keep_stages = {"all", "tracking", "to_hit", "follow-through", "low_speed_all_joints"}
    keep_comparisons = {"tau_ff_vs_isaac_g", "tau_actual_vs_pdff"}
    for row in metric_rows:
        if row["stage"] in keep_stages and row["comparison"] in keep_comparisons:
            item = dict(row)
            for key in ("bias", "rmse", "corr"):
                value = item[key]
                item[key] = None if not math.isfinite(float(value)) else round(float(value), 4)
            key_metrics.append(item)

    report = {
        "input": str(args_cli.input),
        "sample_count": len(rows),
        "stages": dict(stage_counts),
        "kps": list7_from_params(params, "kps", [math.nan] * 7),
        "kds": list7_from_params(params, "kds", [math.nan] * 7),
        "torque_ff_scale": list7_from_params(params, "torque_ff_scale", [1.0] * 7),
        "id_gravity_scale": params.get("id_gravity_scale"),
        "id_inertia_scale": params.get("id_inertia_scale"),
        "id_friction_scale": params.get("id_friction_scale"),
        "paths": {
            "wide_csv": str(output_dir / "pingpong_log_12_isaac_gravity_compare.csv"),
            "long_csv": str(long_path),
            "metrics_csv": str(metrics_path),
            "plots": str(output_dir / "plots"),
        },
        "key_metrics": key_metrics,
    }
    report_path = output_dir / "pingpong_log_12_torque_compare_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def col(rows: list[dict[str, object]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def make_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    t = col(rows, "ros_time_s")
    stages = np.asarray([str(row["stage"]) for row in rows])
    labels = sorted(set(stages.tolist()))

    for i, joint in enumerate(JOINTS):
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        axes[0].plot(t, col(rows, f"tau_actual_{i}"), label="tau_actual", lw=1.0)
        axes[0].plot(t, col(rows, f"tau_pdff_{i}"), label="PD+FF", lw=1.0)
        axes[0].plot(t, col(rows, f"tau_ff_{i}"), label="tau_ff", lw=1.0)
        axes[0].plot(t, col(rows, f"tau_isaac_g_{i}"), label="Isaac gravity", lw=1.0)
        axes[0].set_ylabel("torque Nm")
        axes[0].legend(loc="upper right", ncol=4, fontsize=8)
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(t, col(rows, f"tau_ff_minus_isaac_g_{i}"), label="FF - Isaac g", lw=0.9)
        axes[1].plot(t, col(rows, f"tau_actual_minus_pdff_{i}"), label="actual - PDFF", lw=0.9)
        axes[1].axhline(0.0, color="black", lw=0.7)
        axes[1].set_ylabel("residual Nm")
        axes[1].legend(loc="upper right", ncol=2, fontsize=8)
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(t, col(rows, f"q_plan_{i}"), label="q_plan", lw=0.9)
        axes[2].plot(t, col(rows, f"q_actual_{i}"), label="q_actual", lw=0.9)
        axes[2].set_ylabel("q rad")
        axes[2].set_xlabel("ros_time_s")
        axes[2].legend(loc="upper right", ncol=2, fontsize=8)
        axes[2].grid(True, alpha=0.25)
        fig.suptitle(f"pingpong_log_12 {joint} torque compare")
        fig.tight_layout()
        fig.savefig(plot_dir / f"pingpong_log_12_{joint}_torque_timeseries.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        pairs = (
            (axes[0], f"tau_isaac_g_{i}", f"tau_ff_{i}", "tau_ff vs Isaac gravity"),
            (axes[1], f"tau_pdff_{i}", f"tau_actual_{i}", "tau_actual vs PD+FF"),
        )
        for ax, x_col, y_col, title in pairs:
            x = col(rows, x_col)
            y = col(rows, y_col)
            for label in labels:
                mask = stages == label
                ax.scatter(x[mask], y[mask], s=6, alpha=0.45, label=label)
            vals = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
            if vals.size:
                lo = float(vals.min())
                hi = float(vals.max())
                pad = max((hi - lo) * 0.05, 1e-3)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=0.8)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
        axes[1].legend(loc="best", fontsize=7)
        fig.suptitle(f"pingpong_log_12 {joint} scatter")
        fig.tight_layout()
        fig.savefig(plot_dir / f"pingpong_log_12_{joint}_torque_scatter.png", dpi=150)
        plt.close(fig)


def main() -> int:
    input_path = args_cli.input.resolve()
    output_dir = args_cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args_cli.sample_stride < 1:
        raise SystemExit("--sample-stride must be >= 1")
    params = parse_header_params(input_path)
    rows, input_fields = read_log(input_path, args_cli.sample_stride, args_cli.max_samples)
    if not rows:
        raise SystemExit(f"empty input after sampling: {input_path}")

    kps = list7_from_params(params, "kps", [0.0] * 7)
    kds = list7_from_params(params, "kds", [0.0] * 7)
    for row in rows:
        for i in range(7):
            tau_pd = kps[i] * (float(row[f"q_plan_{i}"]) - float(row[f"q_actual_{i}"]))
            tau_pd += kds[i] * (float(row[f"dq_plan_{i}"]) - float(row[f"dq_actual_{i}"]))
            row[f"tau_pd_{i}"] = tau_pd
            row[f"tau_pdff_{i}"] = tau_pd + float(row[f"tau_ff_{i}"])

    legged_lab_root = args_cli.legged_lab_root.resolve()
    if not legged_lab_root.exists():
        raise SystemExit(f"missing legged_lab root: {legged_lab_root}")
    sys.path.insert(0, str(legged_lab_root))
    from assets.a1.a1 import A1_RIGHT_ARM_JOINTS, A1_TT_CFG  # noqa: PLC0415

    cfg = copy.deepcopy(A1_TT_CFG).replace(prim_path="/World/Robot")
    cfg.spawn.articulation_props.fix_root_link = True
    joint_pos = dict(cfg.init_state.joint_pos)
    for i in range(7):
        joint_pos[f"r{i + 1}"] = float(rows[0][f"q_actual_{i}"])
    cfg.init_state.joint_pos = joint_pos

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    robot = Articulation(cfg)
    sim.reset()
    right_ids = [robot.joint_names.index(name) for name in A1_RIGHT_ARM_JOINTS]

    for row_idx, row in enumerate(rows):
        q = torch.tensor(
            [[float(row[f"q_actual_{i}"]) for i in range(7)]], dtype=torch.float32, device=robot.device
        )
        dq = torch.tensor(
            [[float(row[f"dq_actual_{i}"]) for i in range(7)]], dtype=torch.float32, device=robot.device
        )
        robot.write_joint_state_to_sim(q, dq, joint_ids=right_ids)
        robot.update(0.0)
        if hasattr(robot.root_physx_view, "get_gravity_compensation_forces"):
            grav = robot.root_physx_view.get_gravity_compensation_forces()[0, right_ids]
        else:
            grav = robot.root_physx_view.get_generalized_gravity_forces()[0, right_ids]
        grav_values = grav.detach().cpu().numpy().tolist()
        for i in range(7):
            row[f"tau_isaac_g_{i}"] = float(grav_values[i])
            row[f"tau_ff_minus_isaac_g_{i}"] = float(row[f"tau_ff_{i}"]) - float(row[f"tau_isaac_g_{i}"])
            row[f"tau_actual_minus_isaac_g_{i}"] = float(row[f"tau_actual_{i}"]) - float(row[f"tau_isaac_g_{i}"])
            row[f"tau_actual_minus_pdff_{i}"] = float(row[f"tau_actual_{i}"]) - float(row[f"tau_pdff_{i}"])
        if row_idx % 1000 == 0:
            print(f"evaluated {row_idx}/{len(rows)}", flush=True)

    computed_prefixes = (
        "tau_pd",
        "tau_pdff",
        "tau_isaac_g",
        "tau_ff_minus_isaac_g",
        "tau_actual_minus_isaac_g",
        "tau_actual_minus_pdff",
    )
    computed_fields = [f"{prefix}_{i}" for prefix in computed_prefixes for i in range(7)]
    wide_fields = list(dict.fromkeys(input_fields + computed_fields))
    wide_path = output_dir / "pingpong_log_12_isaac_gravity_compare.csv"
    write_csv(wide_path, rows, wide_fields)

    long_rows = build_long_rows(rows)
    report = write_summary(rows, long_rows, output_dir, params)
    make_plots(rows, output_dir)

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if os.environ.get("A1_GRAVITY_COMPARE_CLOSE_APP") == "1":
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
