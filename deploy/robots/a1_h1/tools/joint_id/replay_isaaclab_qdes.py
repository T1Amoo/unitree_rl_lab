#!/usr/bin/python3
"""Replay recorded q_des targets through the A1 IsaacLab articulation."""

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="CSV produced by record_fsm_test.py.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--joint", type=int, default=1, choices=range(1, 8))
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--kp", default="120 200 200 90 90 90 90")
    ap.add_argument("--kd", default="3.5 3.5 3.5 0.5 0.5 0.5 0.5")
    ap.add_argument("--effort-limit", default="28 28 28 8 8 8 8")
    ap.add_argument("--velocity-limit", default="8 8 8 20 20 20 20")
    ap.add_argument(
        "--actuator-model",
        choices=["implicit", "ideal_pd", "dc_motor"],
        default="implicit",
        help="Right-arm actuator model used for replay. implicit preserves the original A1_TT_CFG behavior.",
    )
    ap.add_argument(
        "--saturation-effort",
        type=float,
        default=56.0,
        help="DCMotor stall/peak torque used when --actuator-model=dc_motor.",
    )
    ap.add_argument(
        "--armature",
        default=None,
        help="Optional seven-value right-arm armature override. Defaults to the asset config.",
    )
    ap.add_argument(
        "--friction",
        default=None,
        help="Optional seven-value right-arm joint friction override. Defaults to the asset/USD config.",
    )
    ap.add_argument(
        "--ground-static-friction",
        type=float,
        default=None,
        help="Optional ground material static friction override.",
    )
    ap.add_argument(
        "--ground-dynamic-friction",
        type=float,
        default=None,
        help="Optional ground material dynamic friction override.",
    )
    ap.add_argument(
        "--initial",
        choices=["target", "real"],
        default="target",
        help="Initial right-arm q in IsaacLab. target uses the first q_des, i.e. fixstand.",
    )
    ap.add_argument(
        "--lock-non-test-joints",
        action="store_true",
        help="Kinematically reset non-tested right-arm joints to their q_des each sim step for isolation tests.",
    )
    ap.add_argument(
        "--right-arm-body-mass-scale",
        type=float,
        default=1.0,
        help="Diagnostic scale applied to PhysX masses and inertias for bodies whose names start with Link_r.",
    )
    ap.add_argument("--legged-lab-root", type=Path, default=Path("Pingpong_TTRL/legged_lab"))
    AppLauncher.add_app_launcher_args(ap)
    return ap


args_cli = build_arg_parser().parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app = AppLauncher(args_cli).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402


def parse_list7(text: str) -> list[float]:
    values = [float(x) for x in text.strip().strip("[]()").replace(",", " ").split()]
    if len(values) != 7:
        raise ValueError(f"expected 7 values, got {len(values)}: {text!r}")
    return values


def get_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def sample_fieldnames(prefixes: list[str]) -> list[str]:
    fields = [
        "source",
        "run_id",
        "stamp",
        "t",
        "phase",
        "joint_index",
        "joint_name",
        "freq_hz",
        "amplitude_rad",
        "sample_index",
    ]
    for prefix in prefixes:
        fields.extend(f"{prefix}{i}" for i in range(1, 8))
    return fields


def interp_series(times: list[float], series: list[list[float]], t: float) -> list[float]:
    if t <= times[0]:
        return list(series[0])
    if t >= times[-1]:
        return list(series[-1])
    hi = bisect.bisect_left(times, t)
    lo = hi - 1
    t0 = times[lo]
    t1 = times[hi]
    alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    return [series[lo][i] * (1.0 - alpha) + series[hi][i] * alpha for i in range(7)]


def apply_right_arm_body_mass_scale(robot: Articulation, scale: float, joint_ids: list[int]) -> dict:
    if scale <= 0.0:
        raise ValueError(f"--right-arm-body-mass-scale must be positive, got {scale}")

    body_ids = [i for i, name in enumerate(robot.body_names) if name.startswith("Link_r")]
    mass_before = robot.root_physx_view.get_masses().clone()
    inertia_before = robot.root_physx_view.get_inertias().clone()
    mass_matrix_before = robot.root_physx_view.get_generalized_mass_matrices()[0].detach().cpu()

    if body_ids and abs(scale - 1.0) > 1e-12:
        env_ids = torch.arange(mass_before.shape[0], dtype=torch.int64, device="cpu")
        body_ids_t = torch.tensor(body_ids, dtype=torch.int64, device="cpu")
        masses = mass_before.clone()
        inertias = inertia_before.clone()
        masses[:, body_ids_t] *= scale
        inertias[:, body_ids_t] *= scale
        robot.root_physx_view.set_masses(masses, env_ids)
        robot.root_physx_view.set_inertias(inertias, env_ids)

    mass_after = robot.root_physx_view.get_masses().clone()
    inertia_after = robot.root_physx_view.get_inertias().clone()
    mass_matrix_after = robot.root_physx_view.get_generalized_mass_matrices()[0].detach().cpu()
    return {
        "right_arm_body_mass_scale": scale,
        "matched_body_ids": body_ids,
        "matched_body_names": [robot.body_names[i] for i in body_ids],
        "matched_mass_before": [float(mass_before[0, i].detach().cpu()) for i in body_ids],
        "matched_mass_after": [float(mass_after[0, i].detach().cpu()) for i in body_ids],
        "matched_total_mass_before": float(mass_before[0, body_ids].sum().detach().cpu()) if body_ids else 0.0,
        "matched_total_mass_after": float(mass_after[0, body_ids].sum().detach().cpu()) if body_ids else 0.0,
        "matched_inertia_trace_before": [
            float(inertia_before[0, i].reshape(3, 3).trace().detach().cpu()) for i in body_ids
        ],
        "matched_inertia_trace_after": [
            float(inertia_after[0, i].reshape(3, 3).trace().detach().cpu()) for i in body_ids
        ],
        "right_joint_mass_diag_before": [float(mass_matrix_before[i, i]) for i in joint_ids],
        "right_joint_mass_diag_after": [float(mass_matrix_after[i, i]) for i in joint_ids],
    }


def main() -> int:
    input_path = args_cli.input
    output_path = args_cli.output
    with input_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty input: {input_path}")

    times = [get_float(row, "t", 0.0) for row in rows]
    target_q = [[get_float(row, f"target_q{i}", 0.0) for i in range(1, 8)] for row in rows]
    target_dq = [[get_float(row, f"target_dq{i}", 0.0) for i in range(1, 8)] for row in rows]
    if args_cli.initial == "real":
        initial_q = [get_float(rows[0], f"actual_q{i}", target_q[0][i - 1]) for i in range(1, 8)]
    else:
        initial_q = list(target_q[0])

    legged_lab_root = args_cli.legged_lab_root.resolve()
    if not legged_lab_root.exists():
        raise SystemExit(f"missing legged_lab root: {legged_lab_root}")
    sys.path.insert(0, str(legged_lab_root))
    from assets.a1.a1 import A1_TT_CFG, A1_RIGHT_ARM_JOINTS  # noqa: PLC0415
    from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg  # noqa: PLC0415

    kp = parse_list7(args_cli.kp)
    kd = parse_list7(args_cli.kd)
    effort = parse_list7(args_cli.effort_limit)
    velocity = parse_list7(args_cli.velocity_limit)
    armature = parse_list7(args_cli.armature) if args_cli.armature is not None else None
    friction = parse_list7(args_cli.friction) if args_cli.friction is not None else None

    cfg = copy.deepcopy(A1_TT_CFG).replace(prim_path="/World/Robot")
    cfg.spawn.articulation_props.fix_root_link = True
    joint_pos = dict(cfg.init_state.joint_pos)
    for i, value in enumerate(initial_q, start=1):
        joint_pos[f"r{i}"] = value
    cfg.init_state.joint_pos = joint_pos
    stiffness = {name: kp[i] for i, name in enumerate(A1_RIGHT_ARM_JOINTS)}
    damping = {name: kd[i] for i, name in enumerate(A1_RIGHT_ARM_JOINTS)}
    effort_limit = {name: effort[i] for i, name in enumerate(A1_RIGHT_ARM_JOINTS)}
    velocity_limit = {name: velocity[i] for i, name in enumerate(A1_RIGHT_ARM_JOINTS)}
    if args_cli.actuator_model == "dc_motor":
        cfg.actuators["right_arm"] = DCMotorCfg(
            joint_names_expr=A1_RIGHT_ARM_JOINTS,
            effort_limit=effort_limit,
            velocity_limit=velocity_limit,
            velocity_limit_sim=velocity_limit,
            stiffness=stiffness,
            damping=damping,
            saturation_effort=args_cli.saturation_effort,
        )
    elif args_cli.actuator_model == "ideal_pd":
        cfg.actuators["right_arm"] = IdealPDActuatorCfg(
            joint_names_expr=A1_RIGHT_ARM_JOINTS,
            effort_limit=effort_limit,
            velocity_limit=velocity_limit,
            velocity_limit_sim=velocity_limit,
            stiffness=stiffness,
            damping=damping,
        )
    else:
        cfg.actuators["right_arm"].stiffness = stiffness
        cfg.actuators["right_arm"].damping = damping
        cfg.actuators["right_arm"].effort_limit_sim = effort_limit
        cfg.actuators["right_arm"].velocity_limit_sim = velocity_limit
    if armature is not None:
        cfg.actuators["right_arm"].armature = {
            name: armature[i] for i, name in enumerate(A1_RIGHT_ARM_JOINTS)
        }
    if friction is not None:
        cfg.actuators["right_arm"].friction = {
            name: friction[i] for i, name in enumerate(A1_RIGHT_ARM_JOINTS)
        }

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device))
    if args_cli.ground_static_friction is not None or args_cli.ground_dynamic_friction is not None:
        ground = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5
                if args_cli.ground_static_friction is None
                else args_cli.ground_static_friction,
                dynamic_friction=0.5
                if args_cli.ground_dynamic_friction is None
                else args_cli.ground_dynamic_friction,
                friction_combine_mode="min",
            )
        )
    else:
        ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    robot = Articulation(cfg)
    sim.reset()

    right_ids = [robot.joint_names.index(name) for name in A1_RIGHT_ARM_JOINTS]
    mass_scale_report = apply_right_arm_body_mass_scale(
        robot, args_cli.right_arm_body_mass_scale, right_ids
    )
    test_joint_zero_based = args_cli.joint - 1
    locked_local_ids = [i for i in range(7) if i != test_joint_zero_based]
    locked_joint_ids = [right_ids[i] for i in locked_local_ids]

    def lock_non_test_joints(q_values: list[float]) -> None:
        if not args_cli.lock_non_test_joints:
            return
        q_lock = torch.tensor(
            [[q_values[i] for i in locked_local_ids]], dtype=torch.float32, device=robot.device
        )
        dq_lock = torch.zeros_like(q_lock)
        robot.write_joint_state_to_sim(q_lock, dq_lock, joint_ids=locked_joint_ids)

    q_init = torch.tensor([initial_q], dtype=torch.float32, device=robot.device)
    dq_init = torch.zeros_like(q_init)
    robot.write_joint_state_to_sim(q_init, dq_init, joint_ids=right_ids)
    robot.update(args_cli.dt)
    if abs(args_cli.right_arm_body_mass_scale - 1.0) > 1e-12:
        mass_scale_report["right_joint_mass_diag_after_q_init"] = [
            float(robot.root_physx_view.get_generalized_mass_matrices()[0].detach().cpu()[i, i])
            for i in right_ids
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sample_fieldnames(
        ["target_q", "target_dq", "sim_q", "sim_dq", "computed_tau", "applied_tau"]
    )
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        sim_t = 0.0
        dt = args_cli.dt
        for sample_index, row in enumerate(rows):
            target_time = times[sample_index]
            while sim_t + 0.5 * dt < target_time:
                q_cmd = interp_series(times, target_q, sim_t)
                q_cmd_t = torch.tensor([q_cmd], dtype=torch.float32, device=robot.device)
                lock_non_test_joints(q_cmd)
                robot.set_joint_position_target(q_cmd_t, joint_ids=right_ids)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(dt)
                lock_non_test_joints(q_cmd)
                robot.update(0.0)
                sim_t += dt

            lock_non_test_joints(target_q[sample_index])
            robot.update(0.0)
            q_out = robot.data.joint_pos[0, right_ids].detach().cpu().tolist()
            dq_out = robot.data.joint_vel[0, right_ids].detach().cpu().tolist()
            computed_tau_out = robot.data.computed_torque[0, right_ids].detach().cpu().tolist()
            applied_tau_out = robot.data.applied_torque[0, right_ids].detach().cpu().tolist()
            row_out: dict[str, float | int | str] = {
                "source": "isaaclab",
                "run_id": row.get("run_id", ""),
                "stamp": row.get("stamp", ""),
                "t": target_time,
                "phase": row.get("phase", ""),
                "joint_index": row.get("joint_index", args_cli.joint),
                "joint_name": row.get("joint_name", f"joint{args_cli.joint}-a1_r"),
                "freq_hz": row.get("freq_hz", ""),
                "amplitude_rad": row.get("amplitude_rad", ""),
                "sample_index": sample_index,
            }
            for i in range(7):
                row_out[f"target_q{i + 1}"] = target_q[sample_index][i]
                row_out[f"target_dq{i + 1}"] = target_dq[sample_index][i]
                row_out[f"sim_q{i + 1}"] = q_out[i]
                row_out[f"sim_dq{i + 1}"] = dq_out[i]
                row_out[f"computed_tau{i + 1}"] = computed_tau_out[i]
                row_out[f"applied_tau{i + 1}"] = applied_tau_out[i]
            writer.writerow(row_out)

    if abs(args_cli.right_arm_body_mass_scale - 1.0) > 1e-12:
        output_path.with_suffix(".mass_scale.json").write_text(
            json.dumps(mass_scale_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(f"wrote {output_path}", flush=True)
    if os.environ.get("A1_JOINT_ID_SKIP_APP_CLOSE") == "1":
        return 0
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
