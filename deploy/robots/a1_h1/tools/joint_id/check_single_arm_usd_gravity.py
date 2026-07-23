#!/usr/bin/python3
"""Check gravity torque from a single-arm USD against a reference CSV row."""

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
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gravity", nargs=3, type=float, default=(9.81, 0.0, 0.0))
    ap.add_argument("--row-index", type=int, default=0)
    AppLauncher.add_app_launcher_args(ap)
    return ap


args_cli = build_arg_parser().parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app = AppLauncher(args_cli).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402


def read_row(path: Path, index: int) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if row_idx == index:
                return row
    raise IndexError(f"row index {index} outside {path}")


def main() -> int:
    wide_row = read_row(args_cli.wide_csv, args_cli.row_index)
    rbdl_row = read_row(args_cli.rbdl_csv, args_cli.row_index)
    q = [float(wide_row[f"q_actual_{i}"]) for i in range(7)]
    rbdl_g = [float(rbdl_row[f"tau_rbdl_g_{i}"]) for i in range(7)]

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
    joint_ids, joint_names = robot.find_joints(
        [
            "joint1_a1_r",
            "joint2_a1_r",
            "joint3_a1_r",
            "joint4_a1_r",
            "joint5_a1_r",
            "joint6_a1_r",
            "joint7_a1_r",
        ],
        preserve_order=True,
    )
    q_t = torch.tensor([q], dtype=torch.float32, device=robot.device)
    dq_t = torch.zeros_like(q_t)
    robot.write_joint_state_to_sim(q_t, dq_t, joint_ids=joint_ids)
    robot.update(0.0)
    grav = robot.root_physx_view.get_gravity_compensation_forces()[0, joint_ids].detach().cpu().tolist()
    diff = [grav[i] - rbdl_g[i] for i in range(7)]
    rmse = math.sqrt(sum(x * x for x in diff) / len(diff))
    report = {
        "usd": str(args_cli.usd),
        "gravity": list(args_cli.gravity),
        "row_index": args_cli.row_index,
        "joint_names": list(joint_names),
        "q": q,
        "usd_gravity_torque": grav,
        "rbdl_gravity_torque": rbdl_g,
        "usd_minus_rbdl": diff,
        "rmse_vs_rbdl": rmse,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
