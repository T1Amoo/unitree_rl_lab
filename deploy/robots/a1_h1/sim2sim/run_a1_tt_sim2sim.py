#!/usr/bin/env python3
"""Run the A1/H1 table-tennis policy in MuJoCo.

This is the sim2sim half of the deployment path:
policy.onnx -> action -> q_des -> MIT-PD torque, matching the real DAMIAO SDK
call shape used later by sim2real:

    control_mit(motor, kp, kd, q_des, dq_des, tau_ff)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import mujoco
import numpy as np

from a1_scene import BALL_RADIUS, PHYSICS_DT, ball_addresses, body_id, load_scene, set_ball_state
from ball_gate import BallGateConfig, BallGateOutput, BallValidityGate
from policy_io import (
    A1PolicyIO,
    DEFAULT_POLICY,
    DEFAULT_RIGHT_Q,
    DAMIAO_DQ_LIMIT,
    DAMIAO_EFFORT,
    EFFORT,
    HIT_PLANE_X,
    ISAAC_PLAY_SERVO_VEL,
    VEL_LIMIT,
    OnnxPolicy,
)
from serve import Serve


DECIMATION = 10
CONTROL_DT = PHYSICS_DT * DECIMATION
IDENTITY_MAT = np.eye(3, dtype=np.float64).ravel()
PARKED_BALL_POS = np.array([1.75, 1.35, 0.20], dtype=np.float64)
PARKED_BALL_VEL = np.zeros(3, dtype=np.float64)


def _ball_paddle_contact_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
    ball_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    paddle_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_blade")
    if ball_gid < 0 or paddle_gid < 0:
        return None
    for i in range(data.ncon):
        con = data.contact[i]
        if {int(con.geom1), int(con.geom2)} == {ball_gid, paddle_gid}:
            return np.asarray(con.pos, dtype=np.float64).copy()
    return None


def _add_debug_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        IDENTITY_MAT,
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _update_hit_point_visuals(
    scene: mujoco.MjvScene | None,
    ball_pos: np.ndarray,
    hit_point: np.ndarray,
    gate_out: BallGateOutput,
) -> None:
    if scene is None:
        return
    scene.ngeom = 0
    valid_rgba = np.array([0.0, 0.9, 0.25, 0.95], dtype=np.float32)
    invalid_rgba = np.array([0.55, 0.55, 0.55, 0.45], dtype=np.float32)
    ball_rgba = np.array([1.0, 0.2, 0.05, 0.75], dtype=np.float32)
    rgba = valid_rgba if gate_out.engaged else invalid_rgba
    _add_debug_sphere(scene, hit_point, 0.035 if gate_out.engaged else 0.025, rgba)
    _add_debug_sphere(scene, ball_pos, 0.024, ball_rgba)


def _apply_ball_drag(model: mujoco.MjModel, data: mujoco.MjData, ball_vadr: int) -> None:
    v = data.qvel[ball_vadr : ball_vadr + 3]
    speed = float(np.linalg.norm(v))
    if speed <= 1e-9:
        return
    force = (-0.5 * 1.225 * (np.pi * BALL_RADIUS ** 2) * 0.4378 * speed) * v
    data.qfrc_applied[ball_vadr : ball_vadr + 3] += force


def _maybe_manual_hit(io: A1PolicyIO, ball_pos: np.ndarray, ball_vel: np.ndarray) -> bool:
    """Fallback contact model for fast balls if mesh contact misses.

    It is intentionally conservative: only flips a ball that is near the training
    paddle touch point and still traveling toward the robot.
    """

    if ball_vel[0] >= -0.05:
        return False
    paddle_pos = io.paddle_touch_point()
    dist = np.linalg.norm(ball_pos - paddle_pos) - BALL_RADIUS
    if dist > 0.045:
        return False
    ball_vel[0] = max(2.0, abs(ball_vel[0]) * 0.75)
    ball_vel[1] *= 0.3
    ball_vel[2] = max(0.8, ball_vel[2] * 0.3 + 1.0)
    return True


def _dead_ball(ball_pos: np.ndarray, ball_vel: np.ndarray) -> bool:
    return bool(
        not np.isfinite(ball_pos).all()
        or ball_pos[2] < 0.08
        or ball_pos[0] < HIT_PLANE_X - 0.20
        or ball_pos[0] > 1.65
        or abs(float(ball_pos[1])) > 1.2
    )


def run(args: argparse.Namespace) -> dict:
    if args.dynamic_pd and args.actuator_mode == "isaac_approx":
        args.actuator_mode = "torque_chain"
    model, data, xml_path = load_scene()
    io = A1PolicyIO(model, data)
    policy = None if args.no_policy else OnnxPolicy(args.policy)
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    serve = Serve(rng=rng, interval_steps=args.serve_interval)
    gate = BallValidityGate(
        BallGateConfig(
            confirm_frames=args.gate_confirm_frames,
            coast_frames=args.gate_coast_frames,
        )
    )
    ball_qadr, ball_vadr = ball_addresses(model)
    ball_bid = body_id(model, "ball")

    ball_active = False
    io.update_action(np.zeros(7, dtype=np.float32))

    stats = {
        "steps": 0,
        "control_steps": 0,
        "hits": 0,
        "serves": 0,
        "max_abs_action": 0.0,
        "max_abs_qvel": 0.0,
        "max_abs_tau": 0.0,
        "valid_frames": 0,
        "xml": str(xml_path),
        "last_hit_step": -1,
        "last_hit_source": "",
        "last_hit_ball_x": float("nan"),
        "last_hit_paddle_x": float("nan"),
        "last_hit_contact_x": float("nan"),
    }
    trace_rows: list[dict[str, float | str]] = []

    def effort_limit() -> np.ndarray:
        return DAMIAO_EFFORT if args.actuator_mode == "real_deploy_preview" else EFFORT

    def velocity_limit() -> np.ndarray | None:
        if args.no_qvel_clip:
            return None
        if args.actuator_mode == "real_deploy_preview":
            return DAMIAO_DQ_LIMIT
        return VEL_LIMIT

    def start_serve() -> None:
        nonlocal ball_active
        p, v = serve.sample()
        set_ball_state(model, data, p, v)
        io.reset_ball_history()
        gate.reset()
        ball_active = True
        stats["serves"] += 1

    def park_ball() -> None:
        nonlocal ball_active
        set_ball_state(model, data, PARKED_BALL_POS, PARKED_BALL_VEL)
        io.reset_ball_history()
        gate.reset()
        ball_active = False

    start_serve()

    def control_tick() -> BallGateOutput:
        stats["control_steps"] += 1
        if not ball_active and serve.due(stats["control_steps"]):
            start_serve()
        ball_pos = data.xpos[ball_bid].copy()
        ball_vel = data.qvel[ball_vadr : ball_vadr + 3].copy()
        gate_out = gate.update(ball_pos, ball_vel)
        if gate_out.engaged:
            stats["valid_frames"] += 1
        obs = io.observe(ball_pos, ball_vel, valid_ball=gate_out.engaged)
        action = np.zeros(7, dtype=np.float32) if policy is None else policy(obs)
        io.update_action(action)
        stats["max_abs_action"] = max(stats["max_abs_action"], float(np.max(np.abs(action))))
        if args.diag_every > 0 and stats["control_steps"] % args.diag_every == 0:
            paddle = io.paddle_touch_point()
            print(
                "[a1_sim2sim_diag] "
                f"step={stats['control_steps']} "
                f"action={np.round(action, 3).tolist()} "
                f"q={np.round(io.right_q(), 3).tolist()} "
                f"q_des={np.round(io.q_des, 3).tolist()} "
                f"dq={np.round(io.right_dq(), 3).tolist()} "
                f"ball={np.round(ball_pos, 3).tolist()} "
                f"hit={np.round(io.last_ball_pred, 3).tolist()} "
                f"paddle={np.round(paddle, 3).tolist()} "
                f"dx_paddle_hit={float(paddle[0] - io.last_ball_pred[0]):.3f} "
                f"last_hit={stats['last_hit_source']}@{stats['last_hit_step']} "
                f"ball_x={float(stats['last_hit_ball_x']):.3f} "
                f"paddle_x={float(stats['last_hit_paddle_x']):.3f} "
                f"contact_x={float(stats['last_hit_contact_x']):.3f} "
                f"gate={int(gate_out.engaged)}:{gate_out.reason}"
            )
        if args.trace_csv is not None:
            tau = io.estimate_mit_tau(effort_limit())
            row: dict[str, float | str] = {
                "step": float(stats["control_steps"]),
                "source": f"mujoco_{args.actuator_mode}",
                "obs_max_abs": float(np.max(np.abs(obs))),
                "valid_ball": float(gate_out.engaged),
                "gate_live": float(gate_out.live),
                "gate_reason": gate_out.reason,
                "gate_first_bounce_x": float(gate_out.first_bounce_x),
                "gate_own_bounces": float(gate_out.own_bounces),
                "gate_hit_seen": float(gate_out.hit_seen),
                "hits": float(stats["hits"]),
                "serves": float(stats["serves"]),
            }
            for i, value in enumerate(action, 1):
                row[f"action_{i}"] = float(value)
            for i, value in enumerate(io.right_q(), 1):
                row[f"q_{i}"] = float(value)
                row[f"q_rel_{i}"] = float(value - DEFAULT_RIGHT_Q[i - 1])
            for i, value in enumerate(io.q_des, 1):
                row[f"q_des_{i}"] = float(value)
            for i, value in enumerate(io.right_dq(), 1):
                row[f"dq_{i}"] = float(value)
            for i, value in enumerate(tau, 1):
                row[f"applied_tau_{i}"] = float(value)
                row[f"computed_tau_{i}"] = float(value)
                row[f"effort_limit_{i}"] = float(effort_limit()[i - 1])
                row[f"vel_limit_{i}"] = float(np.nan if velocity_limit() is None else velocity_limit()[i - 1])
            for i, value in enumerate(ball_pos, 1):
                row[f"ball_{i}"] = float(value)
            for i, value in enumerate(io.last_ball_pred, 1):
                row[f"ball_pred_{i}"] = float(value)
            trace_rows.append(row)
        return gate_out

    def physics_step() -> None:
        data.qfrc_applied[:] = 0.0
        if not ball_active:
            set_ball_state(model, data, PARKED_BALL_POS, PARKED_BALL_VEL)
        io.enforce_hold_joints()
        io.enforce_motor_limits(velocity_limit())
        if args.actuator_mode in ("torque_chain", "real_deploy_preview"):
            tau = io.apply_mit_pd(effort_limit())
        else:
            tau = io.estimate_mit_tau(effort_limit())
            servo_vel = VEL_LIMIT if args.fast_servo else ISAAC_PLAY_SERVO_VEL * args.servo_vel_scale
            io.apply_position_servo(PHYSICS_DT, servo_vel, args.servo_tau)
            mujoco.mj_forward(model, data)
        stats["max_abs_tau"] = max(stats["max_abs_tau"], float(np.max(np.abs(tau))))
        stats["max_abs_qvel"] = max(stats["max_abs_qvel"], float(np.max(np.abs(io.right_dq()))))
        if ball_active:
            _apply_ball_drag(model, data, ball_vadr)
            manual_hit = _maybe_manual_hit(io, data.qpos[ball_qadr : ball_qadr + 3], data.qvel[ball_vadr : ball_vadr + 3])
            if manual_hit:
                gate.register_paddle_hit()
                stats["hits"] += 1
                stats["last_hit_step"] = stats["control_steps"]
                stats["last_hit_source"] = "manual"
                stats["last_hit_ball_x"] = float(data.qpos[ball_qadr])
                stats["last_hit_paddle_x"] = float(io.paddle_touch_point()[0])
                stats["last_hit_contact_x"] = float("nan")
        mujoco.mj_step(model, data)
        contact_pos = _ball_paddle_contact_pos(model, data) if ball_active else None
        if contact_pos is not None:
            gate.register_paddle_hit()
            stats["last_hit_step"] = stats["control_steps"]
            stats["last_hit_source"] = "contact"
            stats["last_hit_ball_x"] = float(data.xpos[ball_bid][0])
            stats["last_hit_paddle_x"] = float(io.paddle_touch_point()[0])
            stats["last_hit_contact_x"] = float(contact_pos[0])
        if ball_active and _dead_ball(data.xpos[ball_bid], data.qvel[ball_vadr : ball_vadr + 3]):
            park_ball()
        if args.actuator_mode == "isaac_approx":
            io.apply_position_servo(0.0)
            mujoco.mj_forward(model, data)
        io.enforce_motor_limits(velocity_limit())
        io.enforce_hold_joints()
        stats["steps"] += 1

    if args.headless_steps > 0:
        for k in range(args.headless_steps * DECIMATION):
            physics_step()
            if (k + 1) % DECIMATION == 0:
                control_tick()
        print(
            "[a1_sim2sim] headless "
            f"control_steps={stats['control_steps']} serves={stats['serves']} hits={stats['hits']} "
            f"valid_frames={stats['valid_frames']} "
            f"max_abs_action={stats['max_abs_action']:.2f} "
            f"max_abs_qvel={stats['max_abs_qvel']:.2f} "
            f"max_abs_tau={stats['max_abs_tau']:.2f}/{float(np.max(EFFORT)):.1f} "
            f"ball={np.round(data.xpos[ball_bid], 3).tolist()}"
        )
        if args.trace_csv is not None:
            args.trace_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.trace_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
                writer.writeheader()
                writer.writerows(trace_rows)
            print(f"[a1_sim2sim] wrote trace {args.trace_csv}")
        return stats

    import mujoco.viewer as mj_viewer

    print("[a1_sim2sim] viewer running. Ctrl-C or close the window to stop.")
    print(f"[a1_sim2sim] actuator_mode={args.actuator_mode}")
    print(f"[a1_sim2sim] scene: {xml_path}")
    with mj_viewer.launch_passive(model, data) as viewer:
        next_time = time.perf_counter()
        k = 0
        while viewer.is_running():
            step_start = time.perf_counter()
            physics_step()
            k += 1
            if k % DECIMATION == 0:
                gate_out = control_tick()
                if not args.no_hit_viz:
                    with viewer.lock():
                        _update_hit_point_visuals(
                            viewer.user_scn,
                            data.xpos[ball_bid].copy(),
                            io.last_ball_pred.copy(),
                            gate_out,
                        )
                viewer.sync()
            next_time += PHYSICS_DT
            sleep_s = next_time - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif time.perf_counter() - step_start < PHYSICS_DT * 0.2:
                time.sleep(PHYSICS_DT * 0.2)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--headless-steps", type=int, default=0, help="Run N 50Hz control steps without viewer.")
    ap.add_argument("--serve-interval", type=int, default=250, help="Serve period in 50Hz control steps.")
    ap.add_argument("--diag-every", type=int, default=0, help="Print policy/control diagnostics every N control steps.")
    ap.add_argument("--trace-csv", type=Path, default=None, help="Write per-control-step diagnostics to CSV.")
    ap.add_argument("--seed", type=int, default=None, help="Seed for the MuJoCo serve sampler.")
    ap.add_argument("--servo-vel-scale", type=float, default=1.0, help="Scale the Isaac-play-calibrated visual servo velocity cap.")
    ap.add_argument("--servo-tau", type=float, default=0.25, help="First-order visual servo time constant in seconds.")
    ap.add_argument("--fast-servo", action="store_true", help="Use the training hard velocity limits for the visual servo.")
    ap.add_argument(
        "--actuator-mode",
        choices=["isaac_approx", "torque_chain", "real_deploy_preview"],
        default="isaac_approx",
        help="Actuator model: visual Isaac approximation, training torque chain, or DAMIAO real-deploy preview.",
    )
    ap.add_argument("--dynamic-pd", action="store_true", help="Use explicit MIT-PD torques instead of the stable position-servo sim2sim actuator.")
    ap.add_argument("--real-deploy-preview", dest="actuator_mode", action="store_const", const="real_deploy_preview", help="Alias for --actuator-mode real_deploy_preview.")
    ap.add_argument("--no-qvel-clip", action="store_true", help="Disable MuJoCo qvel clipping in torque modes. Diagnostic only.")
    ap.add_argument("--no-policy", action="store_true", help="Use zero raw action; useful for scene smoke tests.")
    ap.add_argument("--gate-confirm-frames", type=int, default=1, help="Consecutive live 50Hz frames required before ball is valid.")
    ap.add_argument("--gate-coast-frames", type=int, default=1, help="Consecutive invalid 50Hz frames before valid gate drops.")
    ap.add_argument("--no-hit-viz", action="store_true", help="Disable viewer debug marker for the predicted hit point.")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
