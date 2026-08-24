# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Bounded visual and numerical checks for the WOAN4310 quadruped asset.

This script intentionally drives the position action with zeros.  Since the
task uses ``use_default_offset=True``, zero action means the nominal joint
targets (hip=0.0, thigh=0.8, knee=-1.5 rad), not zero joint position.
"""

# Launch Isaac Sim before importing Isaac Lab modules.

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Inspect the WOAN4310 stance and positive joint directions."
)
parser.add_argument(
    "--task",
    type=str,
    default="Unitree-WOAN4310-Velocity",
    help="Registered task name.",
)
parser.add_argument(
    "--mode",
    choices=("stance", "joint-directions"),
    default="stance",
    help="Floating-base stance check or fixed-base positive joint-direction probe.",
)
parser.add_argument(
    "--steps", type=int, default=500, help="Policy steps for the stance check."
)
parser.add_argument(
    "--pulse", type=float, default=0.4, help="Raw action pulse; 0.4 maps to +0.1 rad."
)
parser.add_argument(
    "--real-time", action="store_true", help="Throttle simulation to wall-clock time."
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Directory for screenshots and report.json.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", help="Use USD I/O instead of Fabric."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Viewport capture needs rendering even when a caller explicitly selects a non-GUI renderer.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# The remaining imports require a running Isaac Sim application.

import json
import math
import time
from datetime import datetime

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import euler_xyz_from_quat

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


FOOT_NAMES = ("ZQ5_foot", "ZH5_foot", "YQ5_foot", "YH5_foot")
PROBE_SUFFIXES = {"hip_abduction": "2", "hip_pitch": "3", "knee": "4"}


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / "inspection" / "woan4310" / f"{timestamp}_{args_cli.mode}"


def _make_deterministic_cfg():
    """Return the registered play configuration with all inspection randomness removed."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env_cfg.episode_length_s = 60.0
    env_cfg.viewer.resolution = (1280, 720)

    terrain_generator = env_cfg.scene.terrain.terrain_generator
    if terrain_generator is not None:
        terrain_generator.num_rows = 1
        terrain_generator.num_cols = 1
        terrain_generator.curriculum = False

    command_ranges = env_cfg.commands.base_velocity.ranges
    limit_ranges = env_cfg.commands.base_velocity.limit_ranges
    for ranges in (command_ranges, limit_ranges):
        ranges.lin_vel_x = (0.0, 0.0)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.debug_vis = False

    material_params = env_cfg.events.physics_material.params
    material_params["static_friction_range"] = (1.0, 1.0)
    material_params["dynamic_friction_range"] = (1.0, 1.0)
    material_params["restitution_range"] = (0.0, 0.0)
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.push_robot = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.curriculum.ang_vel_cmd_levels = None

    if args_cli.mode == "joint-directions":
        env_cfg.scene.robot.spawn.fix_base = True
        env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.65)

    return env_cfg


def _tensor_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def _snapshot(base_env) -> dict:
    robot = base_env.scene["robot"]
    contact_sensor = base_env.scene.sensors["contact_forces"]
    roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w[:1])
    feet = {}
    for foot_name in FOOT_NAMES:
        body_id = robot.body_names.index(foot_name)
        sensor_id = contact_sensor.body_names.index(foot_name)
        force = contact_sensor.data.net_forces_w[0, sensor_id]
        feet[foot_name] = {
            "position_w": _tensor_list(robot.data.body_pos_w[0, body_id]),
            "contact_force_w": _tensor_list(force),
            "contact_force_norm": float(torch.linalg.norm(force).item()),
        }

    action_term = base_env.action_manager.get_term("JointPositionAction")
    return {
        "base_position_w": _tensor_list(robot.data.root_pos_w[0]),
        "base_quaternion_wxyz": _tensor_list(robot.data.root_quat_w[0]),
        "base_rpy_rad": [float(roll[0]), float(pitch[0]), float(yaw[0])],
        "base_linear_velocity_w": _tensor_list(robot.data.root_lin_vel_w[0]),
        "base_angular_velocity_w": _tensor_list(robot.data.root_ang_vel_w[0]),
        "joint_names": list(robot.joint_names),
        "joint_position_rad": _tensor_list(robot.data.joint_pos[0]),
        "joint_velocity_rad_s": _tensor_list(robot.data.joint_vel[0]),
        "applied_torque_nm": _tensor_list(robot.data.applied_torque[0]),
        "action_joint_names": list(action_term._joint_names),
        "raw_action": _tensor_list(action_term.raw_actions[0]),
        "processed_joint_target_rad": _tensor_list(action_term.processed_actions[0]),
        "feet": feet,
    }


def _capture(base_env, output_path: Path, eye_offset=(0.78, 0.68, 0.42)) -> None:
    """Capture the active viewport after positioning it relative to the robot root."""
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    root = base_env.scene["robot"].data.root_pos_w[0].detach().cpu()
    eye = (root + torch.tensor(eye_offset)).tolist()
    target = (root + torch.tensor((0.0, 0.0, -0.13))).tolist()
    base_env.sim.set_camera_view(eye=eye, target=target)

    viewport = None
    for _ in range(60):
        simulation_app.update()
        viewport = get_active_viewport()
        if viewport is not None:
            break
    if viewport is None:
        raise RuntimeError("No active Isaac Sim viewport is available for capture.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture_viewport_to_file(viewport, file_path=str(output_path.resolve()))
    for _ in range(180):
        simulation_app.update()
        if output_path.is_file() and output_path.stat().st_size > 0:
            return
    raise RuntimeError(f"Viewport capture did not finish: {output_path}")


def _step(base_env, action: torch.Tensor, *, real_time: bool) -> tuple[bool, bool]:
    start = time.perf_counter()
    _, _, terminated, truncated, _ = base_env.step(action)
    if real_time:
        sleep_s = base_env.unwrapped.step_dt - (time.perf_counter() - start)
        if sleep_s > 0.0:
            time.sleep(sleep_s)
    return bool(terminated[0].item()), bool(truncated[0].item())


def _run_stance(base_env, output_dir: Path) -> dict:
    action_dim = base_env.action_manager.total_action_dim
    action = torch.zeros((1, action_dim), device=base_env.device)
    termination_count = 0
    timeout_count = 0
    min_base_z = math.inf
    max_abs_roll_pitch = 0.0
    snapshots = {}

    for step in range(args_cli.steps):
        terminated, truncated = _step(base_env, action, real_time=args_cli.real_time)
        termination_count += int(terminated)
        timeout_count += int(truncated)
        snapshot = _snapshot(base_env)
        min_base_z = min(min_base_z, snapshot["base_position_w"][2])
        max_abs_roll_pitch = max(
            max_abs_roll_pitch, *(abs(v) for v in snapshot["base_rpy_rad"][:2])
        )
        if step == min(99, args_cli.steps - 1):
            snapshots["settled"] = snapshot
            _capture(base_env, output_dir / "stance_settled.png")

    snapshots["final"] = _snapshot(base_env)
    _capture(base_env, output_dir / "stance_final.png")
    return {
        "steps": args_cli.steps,
        "simulated_seconds": args_cli.steps * base_env.step_dt,
        "termination_count": termination_count,
        "timeout_count": timeout_count,
        "min_base_z": min_base_z,
        "max_abs_roll_pitch_rad": max_abs_roll_pitch,
        "snapshots": snapshots,
    }


def _foot_positions(snapshot: dict) -> torch.Tensor:
    return torch.tensor([snapshot["feet"][name]["position_w"] for name in FOOT_NAMES])


def _write_action_pose(base_env, action: torch.Tensor) -> None:
    """Apply the action mapping kinematically, without PD/gravity transients."""
    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("JointPositionAction")
    base_env.action_manager.process_action(action)

    joint_pos = robot.data.default_joint_pos.clone()
    for action_index, joint_name in enumerate(action_term._joint_names):
        robot_index = robot.joint_names.index(joint_name)
        joint_pos[:, robot_index] = action_term.processed_actions[:, action_index]
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    base_env.scene.write_data_to_sim()
    base_env.sim.step(render=True)
    base_env.scene.update(base_env.physics_dt)


def _run_joint_directions(base_env, output_dir: Path) -> dict:
    action_term = base_env.action_manager.get_term("JointPositionAction")
    joint_names = list(action_term._joint_names)
    action = torch.zeros(
        (1, base_env.action_manager.total_action_dim), device=base_env.device
    )

    _write_action_pose(base_env, action)
    baseline = _snapshot(base_env)
    baseline_feet = _foot_positions(baseline)
    _capture(base_env, output_dir / "joint_default.png")

    probes = {}
    for probe_name, suffix in PROBE_SUFFIXES.items():
        probe_indices = [
            index for index, name in enumerate(joint_names) if name.endswith(suffix)
        ]
        if len(probe_indices) != 4:
            raise RuntimeError(
                f"Expected four action joints ending in {suffix}, got {probe_indices}."
            )

        action.zero_()
        action[:, probe_indices] = args_cli.pulse
        _write_action_pose(base_env, action)
        pulsed = _snapshot(base_env)
        pulsed_feet = _foot_positions(pulsed)
        _capture(base_env, output_dir / f"joint_positive_{probe_name}.png")

        probes[probe_name] = {
            "raw_action": args_cli.pulse,
            "expected_target_delta_rad": args_cli.pulse * 0.25,
            "joint_names": [joint_names[index] for index in probe_indices],
            "measured_joint_delta_rad": [
                pulsed["joint_position_rad"][robot_index]
                - baseline["joint_position_rad"][robot_index]
                for robot_index in (
                    base_env.scene["robot"].joint_names.index(joint_names[index])
                    for index in probe_indices
                )
            ],
            "foot_delta_w": (pulsed_feet - baseline_feet).tolist(),
            "snapshot": pulsed,
        }

        action.zero_()
        _write_action_pose(base_env, action)

    return {"baseline": baseline, "probes": probes}


def main() -> None:
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")

    output_dir = (args_cli.output_dir or _default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = _make_deterministic_cfg()
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    try:
        env.reset()
        metadata = {
            "task": args_cli.task,
            "mode": args_cli.mode,
            "device": str(base_env.device),
            "physics_dt": base_env.physics_dt,
            "policy_dt": base_env.step_dt,
            "num_bodies": base_env.scene["robot"].num_bodies,
            "num_joints": base_env.scene["robot"].num_joints,
            "action_dim": base_env.action_manager.total_action_dim,
            "body_names": list(base_env.scene["robot"].body_names),
            "joint_names": list(base_env.scene["robot"].joint_names),
        }
        result = (
            _run_stance(base_env, output_dir)
            if args_cli.mode == "stance"
            else _run_joint_directions(base_env, output_dir)
        )
        report = {"metadata": metadata, "result": result}
        report_path = output_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[INFO] WOAN4310 inspection report: {report_path}")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
