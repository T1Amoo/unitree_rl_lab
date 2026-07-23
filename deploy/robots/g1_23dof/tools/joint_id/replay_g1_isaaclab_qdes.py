#!/usr/bin/env python3
"""Replay a G1 joint-ID q_des CSV through the current IsaacLab G1 implicit actuator."""

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

import numpy as np
from isaaclab.app import AppLauncher


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--joint", type=int, required=True, choices=range(1, 6))
    ap.add_argument("--root-mode", choices=["fixed", "free", "held", "lower_policy"], default="fixed")
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--initial", choices=["target", "real"], default="real")
    ap.add_argument("--legged-lab-root", type=Path, default=Path("Pingpong_TTRL/legged_lab"))
    ap.add_argument(
        "--lower-policy-onnx",
        type=Path,
        default=Path("unitree_rl_lab/deploy/robots/g1_23dof/config/policy/velocity/exported/policy.onnx"),
    )
    ap.add_argument("--lower-policy-control-dt", type=float, default=0.02)
    ap.add_argument("--lower-policy-command", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    AppLauncher.add_app_launcher_args(ap)
    return ap


args_cli = build_arg_parser().parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app = AppLauncher(args_cli).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from g1_joint_id_common import (  # noqa: E402
    DEFAULT_JOINT_POS_23,
    JOINT_IDS_MAP,
    JOINT_LIMITS_23,
    POLICY_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_ARM_POLICY_INDICES,
    add_vector,
    open_csv_writer,
    write_json,
)


LOWER_BODY_POLICY_INDICES = list(range(13))
LEFT_ARM_POLICY_INDICES = list(range(13, 18))
RIGHT_ARM_MANUAL_INDICES = list(range(18, 23))
POLICY_OBS_HISTORY = 10
POLICY_FRAME_DIM = 80
POLICY_ACTION_SCALE = 0.25


class LowerBodyVelocityPolicy:
    """50 Hz velocity policy wrapper used only for legs+waist stabilization."""

    def __init__(self, policy_path: Path, command: list[float], control_dt: float) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        if not policy_path.exists():
            raise SystemExit(f"missing lower-body policy: {policy_path}")
        if control_dt <= 0.0:
            raise SystemExit(f"invalid --lower-policy-control-dt: {control_dt}")
        self.policy_path = policy_path
        self.command = np.asarray(command, dtype=np.float32)
        self.control_dt = float(control_dt)
        self.next_update_t = 0.0
        self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        self.default_q = np.asarray(DEFAULT_JOINT_POS_23, dtype=np.float32)
        self.joint_limits = np.asarray(JOINT_LIMITS_23, dtype=np.float32)
        self.last_action = np.zeros(23, dtype=np.float32)
        self.action = np.zeros(23, dtype=np.float32)
        self.q_target = self.default_q.copy()
        self.obs_history: list[np.ndarray] = []
        self.control_updates = 0

    def _build_obs_frame(self, robot, policy_joint_ids: list[int]) -> np.ndarray:
        root_ang_vel = robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
        projected_gravity = robot.data.projected_gravity_b[0].detach().cpu().numpy().astype(np.float32)
        q = robot.data.joint_pos[0, policy_joint_ids].detach().cpu().numpy().astype(np.float32)
        dq = robot.data.joint_vel[0, policy_joint_ids].detach().cpu().numpy().astype(np.float32)
        command = self.command.copy()
        gait_phase = np.zeros(2, dtype=np.float32)
        if float(np.linalg.norm(command)) > 0.1:
            phase = (self.next_update_t % 0.8) / 0.8
            gait_phase[:] = [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)]
        frame = np.concatenate(
            [
                gait_phase,
                root_ang_vel,
                projected_gravity,
                command,
                q - self.default_q,
                dq,
                self.last_action,
            ]
        ).astype(np.float32)
        if frame.shape != (POLICY_FRAME_DIM,):
            raise RuntimeError(f"bad policy obs frame shape: {frame.shape}")
        return frame

    def maybe_update(self, robot, policy_joint_ids: list[int], sim_t: float) -> None:
        if sim_t + 1e-9 < self.next_update_t:
            return
        frame = self._build_obs_frame(robot, policy_joint_ids)
        if not self.obs_history:
            self.obs_history = [frame.copy() for _ in range(POLICY_OBS_HISTORY)]
        else:
            self.obs_history = (self.obs_history + [frame])[-POLICY_OBS_HISTORY:]
        obs = np.concatenate(self.obs_history, axis=0).reshape(1, -1).astype(np.float32)
        action = self.session.run(None, {"obs": obs})[0][0].astype(np.float32)
        if action.shape != (23,):
            raise RuntimeError(f"bad policy action shape: {action.shape}")
        q_target = self.default_q + POLICY_ACTION_SCALE * action
        q_target = np.clip(q_target, self.joint_limits[:, 0], self.joint_limits[:, 1]).astype(np.float32)
        self.action = action
        self.last_action = action.copy()
        self.q_target = q_target
        self.control_updates += 1
        while self.next_update_t <= sim_t + 1e-9:
            self.next_update_t += self.control_dt

    def set_applied_target(self, q_cmd: list[float]) -> None:
        # Kept for compatibility with older local runs.  The velocity policy was
        # trained with previous network action in this observation slot, not the
        # post-override joint target sent by the joint-ID replay.
        return


def get_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


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
    return [series[lo][i] * (1.0 - alpha) + series[hi][i] * alpha for i in range(23)]


def sim_fieldnames() -> list[str]:
    fields = [
        "source",
        "run_id",
        "t",
        "stage",
        "sample_index",
        "joint_index",
        "joint_name",
        "root_mode",
        "root_x",
        "root_y",
        "root_z",
        "root_qw",
        "root_qx",
        "root_qy",
        "root_qz",
        "root_vx",
        "root_vy",
        "root_vz",
        "root_wx",
        "root_wy",
        "root_wz",
    ]
    for prefix in ("target_q", "target_dq", "sim_q", "sim_dq"):
        for i in range(1, 24):
            fields.append(f"{prefix}{i}")
    for prefix in ("policy_action", "policy_target_q"):
        for i in range(1, 24):
            fields.append(f"{prefix}{i}")
    for prefix in ("right_sim_q", "right_sim_dq"):
        for i in range(1, 6):
            fields.append(f"{prefix}{i}")
    return fields


def load_manifest(input_path: Path) -> dict:
    manifest_path = input_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def apply_manifest_gains(cfg, manifest: dict) -> None:
    kp = manifest.get("kp23") or []
    kd = manifest.get("kd23") or []
    if len(kp) != 23 or len(kd) != 23:
        return
    by_name_kp = {name: float(kp[i]) for i, name in enumerate(POLICY_JOINT_NAMES)}
    by_name_kd = {name: float(kd[i]) for i, name in enumerate(POLICY_JOINT_NAMES)}
    groups = {
        "legs": POLICY_JOINT_NAMES[0:4] + POLICY_JOINT_NAMES[6:10],
        "feet": POLICY_JOINT_NAMES[4:6] + POLICY_JOINT_NAMES[10:12],
        "waist_yaw": [POLICY_JOINT_NAMES[12]],
        "arms": POLICY_JOINT_NAMES[13:23],
    }
    for group_name, joint_names in groups.items():
        if group_name not in cfg.actuators:
            continue
        cfg.actuators[group_name].stiffness = {name: by_name_kp[name] for name in joint_names}
        cfg.actuators[group_name].damping = {name: by_name_kd[name] for name in joint_names}


def compose_lower_policy_command(
    lower_policy: LowerBodyVelocityPolicy,
    q_record: list[float],
    dq_record: list[float],
    joint: int,
) -> tuple[list[float], list[float]]:
    q_cmd = list(DEFAULT_JOINT_POS_23)
    dq_cmd = [0.0] * 23
    policy_q = lower_policy.q_target.tolist()
    for i in LOWER_BODY_POLICY_INDICES:
        q_cmd[i] = float(policy_q[i])
    for i in LEFT_ARM_POLICY_INDICES + RIGHT_ARM_MANUAL_INDICES:
        q_cmd[i] = float(DEFAULT_JOINT_POS_23[i])
    tested_i = RIGHT_ARM_POLICY_INDICES[joint - 1]
    q_cmd[tested_i] = float(q_record[tested_i])
    dq_cmd[tested_i] = float(dq_record[tested_i])
    return q_cmd, dq_cmd


def main() -> int:
    with args_cli.input.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty input: {args_cli.input}")

    times = [get_float(row, "t", 0.0) for row in rows]
    target_q = [[get_float(row, f"target_q{i}", 0.0) for i in range(1, 24)] for row in rows]
    target_dq = [[get_float(row, f"target_dq{i}", 0.0) for i in range(1, 24)] for row in rows]
    if args_cli.initial == "real":
        initial_q = [
            get_float(rows[0], f"actual_q{i}", target_q[0][i - 1])
            for i in range(1, 24)
        ]
    else:
        initial_q = list(target_q[0])

    legged_lab_root = args_cli.legged_lab_root.resolve()
    if not legged_lab_root.exists():
        raise SystemExit(f"missing legged_lab root: {legged_lab_root}")
    sys.path.insert(0, str(legged_lab_root.parent))
    sys.path.insert(0, str(legged_lab_root))
    from legged_lab.assets.unitree.g1 import G1_TT_CFG  # noqa: PLC0415

    manifest = load_manifest(args_cli.input)
    cfg = copy.deepcopy(G1_TT_CFG).replace(prim_path="/World/Robot")
    cfg.spawn.articulation_props.fix_root_link = args_cli.root_mode == "fixed"
    apply_manifest_gains(cfg, manifest)
    lower_policy = None
    if args_cli.root_mode == "lower_policy":
        lower_policy = LowerBodyVelocityPolicy(
            args_cli.lower_policy_onnx.resolve(),
            [float(x) for x in args_cli.lower_policy_command],
            args_cli.lower_policy_control_dt,
        )
        tested_i = RIGHT_ARM_POLICY_INDICES[args_cli.joint - 1]
        policy_initial_q = list(DEFAULT_JOINT_POS_23)
        policy_initial_q[tested_i] = initial_q[tested_i]
        initial_q = policy_initial_q

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(cfg)
    sim.reset()

    policy_joint_ids = [robot.joint_names.index(name) for name in POLICY_JOINT_NAMES]
    q_init = torch.tensor([initial_q], dtype=torch.float32, device=robot.device)
    dq_init = torch.zeros_like(q_init)
    robot.write_joint_state_to_sim(q_init, dq_init, joint_ids=policy_joint_ids)
    robot.update(args_cli.dt)
    held_root_state = robot.data.root_state_w.clone()

    def hold_root_if_needed() -> None:
        if args_cli.root_mode != "held":
            return
        root = held_root_state.clone()
        root[:, 7:] = 0.0
        robot.write_root_state_to_sim(root)

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    writer, handle = open_csv_writer(args_cli.output, sim_fieldnames())
    try:
        sim_t = 0.0
        dt = args_cli.dt
        for sample_index, row in enumerate(rows):
            target_time = times[sample_index]
            while sim_t + 0.5 * dt < target_time:
                q_record = interp_series(times, target_q, sim_t)
                dq_record = interp_series(times, target_dq, sim_t)
                if lower_policy is not None:
                    lower_policy.maybe_update(robot, policy_joint_ids, sim_t)
                    q_cmd, _ = compose_lower_policy_command(lower_policy, q_record, dq_record, args_cli.joint)
                    lower_policy.set_applied_target(q_cmd)
                else:
                    q_cmd = q_record
                q_cmd_t = torch.tensor([q_cmd], dtype=torch.float32, device=robot.device)
                robot.set_joint_position_target(q_cmd_t, joint_ids=policy_joint_ids)
                hold_root_if_needed()
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(dt)
                hold_root_if_needed()
                robot.update(0.0)
                sim_t += dt

            hold_root_if_needed()
            robot.update(0.0)
            if lower_policy is not None:
                q_out_cmd, dq_out_cmd = compose_lower_policy_command(
                    lower_policy,
                    target_q[sample_index],
                    target_dq[sample_index],
                    args_cli.joint,
                )
                policy_action = lower_policy.action.tolist()
                policy_target_q = lower_policy.q_target.tolist()
            else:
                q_out_cmd = target_q[sample_index]
                dq_out_cmd = target_dq[sample_index]
                policy_action = [math.nan] * 23
                policy_target_q = [math.nan] * 23
            q_out = robot.data.joint_pos[0, policy_joint_ids].detach().cpu().tolist()
            dq_out = robot.data.joint_vel[0, policy_joint_ids].detach().cpu().tolist()
            root = robot.data.root_state_w[0].detach().cpu().tolist()
            out: dict[str, float | int | str] = {
                "source": "isaaclab_g1",
                "run_id": row.get("run_id", ""),
                "t": target_time,
                "stage": row.get("stage", ""),
                "sample_index": sample_index,
                "joint_index": args_cli.joint,
                "joint_name": RIGHT_ARM_JOINT_NAMES[args_cli.joint - 1],
                "root_mode": args_cli.root_mode,
                "root_x": root[0],
                "root_y": root[1],
                "root_z": root[2],
                "root_qw": root[3],
                "root_qx": root[4],
                "root_qy": root[5],
                "root_qz": root[6],
                "root_vx": root[7],
                "root_vy": root[8],
                "root_vz": root[9],
                "root_wx": root[10],
                "root_wy": root[11],
                "root_wz": root[12],
            }
            add_vector(out, "target_q", q_out_cmd, 23)
            add_vector(out, "target_dq", dq_out_cmd, 23)
            add_vector(out, "sim_q", q_out, 23)
            add_vector(out, "sim_dq", dq_out, 23)
            add_vector(out, "policy_action", policy_action, 23)
            add_vector(out, "policy_target_q", policy_target_q, 23)
            add_vector(out, "right_sim_q", [q_out[i] for i in RIGHT_ARM_POLICY_INDICES], 5)
            add_vector(out, "right_sim_dq", [dq_out[i] for i in RIGHT_ARM_POLICY_INDICES], 5)
            writer.writerow(out)
    finally:
        handle.close()

    write_json(
        args_cli.output.with_suffix(".params.json"),
        {
            "input": str(args_cli.input),
            "output": str(args_cli.output),
            "joint": args_cli.joint,
            "root_mode": args_cli.root_mode,
            "dt": args_cli.dt,
            "initial": args_cli.initial,
            "source_asset": "legged_lab.assets.unitree.g1.G1_TT_CFG",
            "actuator_model": "ImplicitActuatorCfg",
            "gains_source": str(args_cli.input.with_suffix(".manifest.json")),
            "lower_policy_onnx": str(args_cli.lower_policy_onnx) if lower_policy is not None else "",
            "lower_policy_control_dt": args_cli.lower_policy_control_dt if lower_policy is not None else None,
            "lower_policy_command": args_cli.lower_policy_command if lower_policy is not None else [],
            "lower_policy_control_updates": lower_policy.control_updates if lower_policy is not None else 0,
            "lower_policy_initial_override": "default lower body and non-tested upper-body joints; tested joint keeps input initial q"
            if lower_policy is not None
            else "",
            "lower_policy_indices": LOWER_BODY_POLICY_INDICES if lower_policy is not None else [],
            "left_arm_manual_indices": LEFT_ARM_POLICY_INDICES if lower_policy is not None else [],
            "right_arm_manual_indices": RIGHT_ARM_MANUAL_INDICES if lower_policy is not None else [],
            "manual_upper_body": "left arm default; right arm default except tested joint from input q_des"
            if lower_policy is not None
            else "",
            "kp23": manifest.get("kp23", []),
            "kd23": manifest.get("kd23", []),
            "joint_ids_map": JOINT_IDS_MAP,
        },
    )
    print(f"wrote {args_cli.output}", flush=True)
    if os.environ.get("G1_JOINT_ID_APP_CLOSE") == "1":
        app.close()
        return 0
    os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
