#!/usr/bin/python3
"""Compare a single-arm USD gravity torque against RBDL and production Isaac logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usd", type=Path, required=True)
    ap.add_argument("--wide-csv", type=Path, required=True)
    ap.add_argument("--rbdl-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--gravity", nargs=3, type=float, default=(9.81, 0.0, 0.0))
    ap.add_argument("--sample-stride", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=0)
    AppLauncher.add_app_launcher_args(ap)
    return ap


args_cli = build_arg_parser().parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app = AppLauncher(args_cli).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402


JOINT_COUNT = 7
JOINT_NAMES = [f"joint{i}_a1_r" for i in range(1, JOINT_COUNT + 1)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def corrcoef(a: list[float], b: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx < 1e-24 or vy < 1e-24:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def summarize(values_a: list[float], values_b: list[float]) -> dict[str, float | int]:
    pairs = [(x, y) for x, y in zip(values_a, values_b) if math.isfinite(x) and math.isfinite(y)]
    if not pairs:
        return {"n": 0, "bias": math.nan, "mae": math.nan, "rmse": math.nan, "corr": math.nan}
    diff = [x - y for x, y in pairs]
    return {
        "n": len(pairs),
        "bias": sum(diff) / len(diff),
        "mae": sum(abs(x) for x in diff) / len(diff),
        "rmse": math.sqrt(sum(x * x for x in diff) / len(diff)),
        "corr": corrcoef(values_a, values_b),
    }


def metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    comparisons = [
        ("tau_usd_g", "tau_rbdl_g", "usd_vs_single_arm_rbdl_raw"),
        ("tau_usd_g", "tau_rbdl_g_scaled", "usd_vs_single_arm_rbdl_scaled"),
        ("tau_usd_g", "tau_prod_isaac_g", "usd_vs_production_isaac"),
        ("tau_rbdl_g_scaled", "tau_prod_isaac_g", "single_arm_rbdl_scaled_vs_production_isaac"),
        ("tau_ff", "tau_rbdl_g_scaled", "tau_ff_vs_single_arm_rbdl_scaled"),
        ("tau_ff", "tau_usd_g", "tau_ff_vs_single_arm_usd"),
    ]
    stages = sorted({str(row["stage"]) for row in rows})
    groups = [("all", rows), ("low_speed_all_joints", [r for r in rows if int(r["low_speed_all_joints"])])]
    groups.extend((stage, [r for r in rows if r["stage"] == stage]) for stage in stages)
    for lhs, rhs, label in comparisons:
        for stage, stage_rows in groups:
            if not stage_rows:
                continue
            for j in range(JOINT_COUNT):
                lhs_values = [float(row[f"{lhs}_{j}"]) for row in stage_rows]
                rhs_values = [float(row[f"{rhs}_{j}"]) for row in stage_rows]
                item: dict[str, object] = {
                    "comparison": label,
                    "stage": stage,
                    "joint": f"j{j + 1}",
                }
                item.update(summarize(lhs_values, rhs_values))
                out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if args_cli.sample_stride < 1:
        raise SystemExit("--sample-stride must be >= 1")
    wide_rows = read_csv(args_cli.wide_csv)
    rbdl_rows_by_id = {row["sample_id"]: row for row in read_csv(args_cli.rbdl_csv)}
    selected_rows = []
    for raw_idx, row in enumerate(wide_rows):
        if raw_idx % args_cli.sample_stride:
            continue
        if row.get("sample_id") not in rbdl_rows_by_id:
            continue
        selected_rows.append(row)
        if args_cli.max_samples > 0 and len(selected_rows) >= args_cli.max_samples:
            break
    if not selected_rows:
        raise SystemExit("no rows selected")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, gravity=tuple(args_cli.gravity))
    )
    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(args_cli.usd.resolve()),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-7]_a1_r"],
                effort_limit_sim=1.0e9,
                velocity_limit_sim=1.0e9,
                stiffness=0.0,
                damping=0.0,
            )
        },
    )
    robot = Articulation(cfg)
    sim.reset()
    joint_ids, found_joint_names = robot.find_joints(JOINT_NAMES, preserve_order=True)

    out_rows: list[dict[str, object]] = []
    for row_idx, row in enumerate(selected_rows):
        rbdl_row = rbdl_rows_by_id[row["sample_id"]]
        q = torch.tensor(
            [[f(row, f"q_actual_{i}") for i in range(JOINT_COUNT)]], dtype=torch.float32, device=robot.device
        )
        dq = torch.zeros_like(q)
        robot.write_joint_state_to_sim(q, dq, joint_ids=joint_ids)
        robot.update(0.0)
        grav = robot.root_physx_view.get_gravity_compensation_forces()[0, joint_ids].detach().cpu().tolist()

        speed_norm = math.sqrt(sum(f(row, f"dq_actual_{i}") ** 2 for i in range(JOINT_COUNT)))
        out: dict[str, object] = {
            "sample_id": int(float(row["sample_id"])),
            "ros_time_s": f(row, "ros_time_s"),
            "stage": row.get("stage", ""),
            "low_speed_all_joints": int(speed_norm < 0.25),
        }
        for i in range(JOINT_COUNT):
            out[f"q_actual_{i}"] = f(row, f"q_actual_{i}")
            out[f"tau_usd_g_{i}"] = float(grav[i])
            out[f"tau_rbdl_g_{i}"] = f(rbdl_row, f"tau_rbdl_g_{i}")
            out[f"tau_rbdl_g_scaled_{i}"] = f(rbdl_row, f"tau_rbdl_g_scaled_{i}")
            out[f"tau_prod_isaac_g_{i}"] = f(rbdl_row, f"tau_isaac_g_{i}")
            out[f"tau_ff_{i}"] = f(rbdl_row, f"tau_ff_{i}")
            out[f"usd_minus_rbdl_{i}"] = out[f"tau_usd_g_{i}"] - out[f"tau_rbdl_g_{i}"]
            out[f"usd_minus_rbdl_scaled_{i}"] = out[f"tau_usd_g_{i}"] - out[f"tau_rbdl_g_scaled_{i}"]
            out[f"usd_minus_prod_isaac_{i}"] = out[f"tau_usd_g_{i}"] - out[f"tau_prod_isaac_g_{i}"]
        out_rows.append(out)
        if row_idx % 1000 == 0:
            print(f"evaluated {row_idx}/{len(selected_rows)}", flush=True)

    output_dir = args_cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wide_fields = ["sample_id", "ros_time_s", "stage", "low_speed_all_joints"]
    for prefix in [
        "q_actual",
        "tau_usd_g",
        "tau_rbdl_g",
        "tau_rbdl_g_scaled",
        "tau_prod_isaac_g",
        "tau_ff",
        "usd_minus_rbdl",
        "usd_minus_rbdl_scaled",
        "usd_minus_prod_isaac",
    ]:
        wide_fields.extend(f"{prefix}_{i}" for i in range(JOINT_COUNT))
    wide_path = output_dir / "single_arm_usd_trajectory_gravity_compare.csv"
    write_csv(wide_path, out_rows, wide_fields)

    metrics = metric_rows(out_rows)
    metric_fields = ["comparison", "stage", "joint", "n", "bias", "mae", "rmse", "corr"]
    metrics_path = output_dir / "single_arm_usd_trajectory_gravity_metrics.csv"
    write_csv(metrics_path, metrics, metric_fields)

    key = [
        row
        for row in metrics
        if row["stage"] in {"all", "low_speed_all_joints"}
        and row["comparison"]
        in {
            "usd_vs_single_arm_rbdl_raw",
            "usd_vs_single_arm_rbdl_scaled",
            "usd_vs_production_isaac",
            "tau_ff_vs_single_arm_usd",
        }
    ]
    report = {
        "usd": str(args_cli.usd),
        "wide_csv": str(args_cli.wide_csv),
        "rbdl_csv": str(args_cli.rbdl_csv),
        "gravity": list(args_cli.gravity),
        "sample_count": len(out_rows),
        "joint_names": list(found_joint_names),
        "outputs": {
            "wide_csv": str(wide_path),
            "metrics_csv": str(metrics_path),
        },
        "key_metrics": key,
    }
    report_path = output_dir / "single_arm_usd_trajectory_gravity_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
