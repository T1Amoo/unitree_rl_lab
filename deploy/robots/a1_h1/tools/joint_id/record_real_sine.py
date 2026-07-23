#!/usr/bin/python3
"""Publish a single-joint sine target to the real A1 arm and log feedback.

Run from the local machine after sourcing ROS 2 Humble and the local deploy
workspace. The robot-side inference_arm_control_node remains on the robot.
"""

from __future__ import annotations

import argparse
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String

from joint_id_common import (
    JOINT_NAMES,
    TrialSpec,
    joint_state_to_ordered,
    open_csv_writer,
    parse_list7,
    sample_fieldnames,
    sine_target,
    write_trial_manifest,
)


class RealSineRecorder(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("a1_joint_id_real_sine")
        self.args = args
        self.last_joint_msg: JointState | None = None
        self.last_q = [math.nan] * 7
        self.last_dq = [math.nan] * 7
        self.last_tau = [math.nan] * 7
        self.create_subscription(JointState, args.joint_topic, self.joint_cb, 20)
        self.action_pub = self.create_publisher(Float64MultiArray, args.action_topic, 10)
        self.enable_pub = self.create_publisher(Bool, args.enable_topic, 10)
        self.damping_pub = self.create_publisher(Bool, args.damping_topic, 10)
        self.policy_enable_pub = self.create_publisher(Bool, args.policy_enable_topic, 10)
        self.fsm_pub = self.create_publisher(String, args.fsm_command_topic, 10)
        self.target_pub = self.create_publisher(Float64MultiArray, args.target_topic, 10)

    def joint_cb(self, msg: JointState) -> None:
        self.last_joint_msg = msg
        self.last_q = joint_state_to_ordered(list(msg.name), list(msg.position), self.last_q)
        self.last_dq = joint_state_to_ordered(list(msg.name), list(msg.velocity), self.last_dq)
        self.last_tau = joint_state_to_ordered(list(msg.name), list(msg.effort), self.last_tau)

    def publish_bool(self, publisher, value: bool, repeats: int = 5) -> None:
        msg = Bool()
        msg.data = bool(value)
        for _ in range(repeats):
            publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)

    def publish_passive_guard(self) -> None:
        msg = String()
        msg.data = "passive"
        for _ in range(3):
            self.fsm_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)
        self.publish_bool(self.policy_enable_pub, False, repeats=3)
        self.publish_bool(self.damping_pub, False, repeats=3)

    def publish_action(self, q: list[float], dq: list[float]) -> None:
        msg = Float64MultiArray()
        if self.args.action_format == "position_velocity":
            msg.data = [float(x) for x in q + dq]
        else:
            msg.data = [float(x) for x in q]
        self.action_pub.publish(msg)

        target_msg = Float64MultiArray()
        target_msg.data = [float(x) for x in q + dq]
        self.target_pub.publish(target_msg)


def check_wired_route(robot_ip: str, expected_dev: str) -> None:
    if not robot_ip:
        return
    proc = subprocess.run(
        ["ip", "route", "get", robot_ip],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    route = proc.stdout.strip()
    if proc.returncode != 0:
        raise RuntimeError(f"cannot check route to {robot_ip}: {route}")
    if expected_dev and f" dev {expected_dev} " not in f" {route} ":
        raise RuntimeError(
            f"route to {robot_ip} is not on {expected_dev}: {route}. "
            "Fix the wired route before enabling the arm."
        )


def wait_for_joint_state(node: RealSineRecorder, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.last_joint_msg is not None and all(math.isfinite(x) for x in node.last_q):
            return
    raise TimeoutError(f"no usable joint state on {node.args.joint_topic}")


def ensure_no_conflicting_publishers(node: RealSineRecorder) -> None:
    if node.args.allow_existing_action_publishers:
        return
    action_publishers = node.count_publishers(node.args.action_topic)
    # Count includes this node's publisher after construction.
    external_action_publishers = max(0, action_publishers - 1)
    fsm_publishers = node.count_publishers(node.args.fsm_state_topic)
    if external_action_publishers > 0 or fsm_publishers > 0:
        raise RuntimeError(
            "conflicting deployment publishers are active "
            f"(external {node.args.action_topic} publishers={external_action_publishers}, "
            f"{node.args.fsm_state_topic} publishers={fsm_publishers}). "
            "Stop a1_policy_bridge_cpp/a1_tt_fsm_supervisor before joint ID, "
            "or rerun with --allow-existing-action-publishers only if you know the node is inert."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joint", type=int, required=True, choices=range(1, 8), help="1-based right-arm joint index.")
    ap.add_argument("--freq", type=float, required=True, help="Sine frequency in Hz.")
    ap.add_argument("--amplitude", type=float, default=0.08, help="Sine amplitude in rad.")
    ap.add_argument("--cycles", type=float, default=8.0, help="Number of full sine cycles to record.")
    ap.add_argument("--duration-s", type=float, default=0.0, help="Override sine duration in seconds.")
    ap.add_argument("--warmup-s", type=float, default=2.0, help="Hold center before sine.")
    ap.add_argument("--post-hold-s", type=float, default=1.0, help="Hold center after sine.")
    ap.add_argument("--ramp-s", type=float, default=0.5, help="Smooth sine envelope ramp in/out.")
    ap.add_argument("--rate-hz", type=float, default=50.0, help="Command/logging rate.")
    ap.add_argument("--center", default="", help="7-value center q. Omit to use current joint state.")
    ap.add_argument("--action-format", choices=["position", "position_velocity"], default="position_velocity")
    ap.add_argument("--output", type=Path, default=Path("joint_id_real.csv"))
    ap.add_argument("--run-id", default="", help="Optional run id. Defaults to timestamp.")
    ap.add_argument("--param-tag", default="", help="Human-readable controller parameter label.")
    ap.add_argument("--real-kp", default="", help="7-value kp metadata for this run. Does not change robot params.")
    ap.add_argument("--real-kd", default="", help="7-value kd metadata for this run. Does not change robot params.")
    ap.add_argument("--enable-servo", action="store_true", help="Actually publish /model_control/enable=true.")
    ap.add_argument("--leave-enabled", action="store_true", help="Do not disable servo at the end.")
    ap.add_argument("--allow-existing-action-publishers", action="store_true")
    ap.add_argument("--check-route-ip", default="10.1.1.220")
    ap.add_argument("--expected-route-dev", default="enp8s0")
    ap.add_argument("--joint-topic", default="/right_joint_states")
    ap.add_argument("--action-topic", default="/model_action")
    ap.add_argument("--enable-topic", default="/model_control/enable")
    ap.add_argument("--damping-topic", default="/model_control/damping")
    ap.add_argument("--policy-enable-topic", default="/a1_tt/policy_enable")
    ap.add_argument("--fsm-command-topic", default="/a1_tt/fsm_command")
    ap.add_argument("--fsm-state-topic", default="/a1_tt/fsm_state")
    ap.add_argument("--target-topic", default="/joint_id/target")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.freq <= 0.0:
        raise SystemExit("--freq must be positive")
    if args.rate_hz <= 0.0:
        raise SystemExit("--rate-hz must be positive")
    if args.amplitude <= 0.0:
        raise SystemExit("--amplitude must be positive")
    sine_s = args.duration_s if args.duration_s > 0.0 else args.cycles / args.freq
    if sine_s <= 0.0:
        raise SystemExit("sine duration must be positive")

    check_wired_route(args.check_route_ip, args.expected_route_dev)

    rclpy.init()
    node = RealSineRecorder(args)
    stop_requested = False

    def handle_signal(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        time.sleep(0.3)
        ensure_no_conflicting_publishers(node)
        wait_for_joint_state(node, timeout_s=5.0)
        center = parse_list7(args.center, default=node.last_q)
        joint_zero = args.joint - 1
        run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

        spec = TrialSpec(
            joint_index=args.joint,
            joint_name=JOINT_NAMES[joint_zero],
            freq_hz=args.freq,
            amplitude_rad=args.amplitude,
            rate_hz=args.rate_hz,
            warmup_s=args.warmup_s,
            sine_s=sine_s,
            post_hold_s=args.post_hold_s,
            ramp_s=args.ramp_s,
            action_format=args.action_format,
            center=center,
        )

        fields = sample_fieldnames(["target_q", "target_dq", "actual_q", "actual_dq", "actual_tau"])
        writer, handle = open_csv_writer(args.output, fields)
        write_trial_manifest(
            args.output.with_suffix(".manifest.json"),
            spec,
            {
                "run_id": run_id,
                "output": str(args.output),
                "param_tag": args.param_tag,
                "real_kp": parse_list7(args.real_kp, default=[]) if args.real_kp else [],
                "real_kd": parse_list7(args.real_kd, default=[]) if args.real_kd else [],
                "enable_servo": args.enable_servo,
                "action_topic": args.action_topic,
                "joint_topic": args.joint_topic,
            },
        )

        node.publish_passive_guard()
        node.publish_action(center, [0.0] * 7)
        if args.enable_servo:
            node.publish_bool(node.enable_pub, True, repeats=10)
        else:
            node.get_logger().warning("dry run: --enable-servo not set, arm servo remains disabled")

        total_s = args.warmup_s + sine_s + args.post_hold_s
        dt = 1.0 / args.rate_hz
        start = time.monotonic()
        next_tick = start
        sample_index = 0
        last_q_target = center

        while not stop_requested:
            now = time.monotonic()
            t = now - start
            if t > total_s:
                break
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
                continue
            rclpy.spin_once(node, timeout_sec=0.0)

            if t < args.warmup_s:
                phase = "warmup"
                q_target = center
                dq_target = [0.0] * 7
            elif t < args.warmup_s + sine_s:
                phase = "sine"
                q_target, dq_target = sine_target(
                    center,
                    joint_zero,
                    args.amplitude,
                    args.freq,
                    t - args.warmup_s,
                    sine_s,
                    args.ramp_s,
                )
            else:
                phase = "post_hold"
                q_target = center
                dq_target = [0.0] * 7

            # Keep dq exactly consistent with the emitted target when the node
            # uses position_velocity. This avoids derivative-envelope mistakes.
            if args.action_format == "position_velocity" and sample_index > 0:
                dq_target = [(q_target[i] - last_q_target[i]) * args.rate_hz for i in range(7)]
            last_q_target = list(q_target)

            node.publish_action(q_target, dq_target)
            row: dict[str, float | int | str] = {
                "source": "real",
                "run_id": run_id,
                "stamp": time.time(),
                "t": t,
                "phase": phase,
                "joint_index": args.joint,
                "joint_name": JOINT_NAMES[joint_zero],
                "freq_hz": args.freq,
                "amplitude_rad": args.amplitude,
                "sample_index": sample_index,
            }
            for i in range(7):
                row[f"target_q{i + 1}"] = q_target[i]
                row[f"target_dq{i + 1}"] = dq_target[i]
                row[f"actual_q{i + 1}"] = node.last_q[i]
                row[f"actual_dq{i + 1}"] = node.last_dq[i]
                row[f"actual_tau{i + 1}"] = node.last_tau[i]
            writer.writerow(row)
            sample_index += 1
            next_tick += dt

        for _ in range(max(1, int(args.rate_hz * 0.3))):
            node.publish_action(center, [0.0] * 7)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(dt)

        if args.enable_servo and not args.leave_enabled:
            node.publish_bool(node.enable_pub, False, repeats=10)
        handle.close()
        node.get_logger().info("wrote %s", args.output)
        return 0
    finally:
        if "handle" in locals() and not handle.closed:
            handle.close()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
