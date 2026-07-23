#!/usr/bin/python3
"""Record FSM-driven q_des-only joint-ID tests.

This node does not publish arm commands. It waits for a1_tt_fsm_supervisor to
enter TEST, records /model_action and /right_joint_states, then stops when TEST
finishes or the timeout is reached.
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
from std_msgs.msg import Float64MultiArray, String

from joint_id_common import (
    JOINT_NAMES,
    TrialSpec,
    joint_state_to_ordered,
    open_csv_writer,
    parse_list7,
    sample_fieldnames,
    write_trial_manifest,
)


class FsmTestRecorder(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("a1_joint_id_fsm_test_recorder")
        self.args = args
        self.last_q = [math.nan] * 7
        self.last_dq = [math.nan] * 7
        self.last_tau = [math.nan] * 7
        self.last_target_q = [math.nan] * 7
        self.last_joint_stamp = 0.0
        self.last_action_stamp = 0.0
        self.last_state = ""
        self.joint_rx_count = 0
        self.action_rx_count = 0
        self.create_subscription(JointState, args.joint_topic, self.joint_cb, 50)
        self.create_subscription(Float64MultiArray, args.action_topic, self.action_cb, 50)
        self.create_subscription(String, args.fsm_state_topic, self.state_cb, 20)

    def joint_cb(self, msg: JointState) -> None:
        self.last_q = joint_state_to_ordered(list(msg.name), list(msg.position), self.last_q)
        self.last_dq = joint_state_to_ordered(list(msg.name), list(msg.velocity), self.last_dq)
        self.last_tau = joint_state_to_ordered(list(msg.name), list(msg.effort), self.last_tau)
        self.last_joint_stamp = time.monotonic()
        self.joint_rx_count += 1

    def action_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 7:
            self.last_target_q = [float(x) for x in msg.data[:7]]
            self.last_action_stamp = time.monotonic()
            self.action_rx_count += 1

    def state_cb(self, msg: String) -> None:
        self.last_state = msg.data

    def in_test(self) -> bool:
        return "state=TEST" in self.last_state


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


def phase_for_t(t: float, args: argparse.Namespace) -> str:
    active_s = args.duration_s if args.duration_s > 0.0 else args.cycles / args.freq
    if t < args.warmup_s:
        return "warmup"
    if t < args.warmup_s + active_s:
        return args.signal_type
    return "post_hold"


def freq_for_t(t: float, args: argparse.Namespace) -> float:
    active_s = args.duration_s if args.duration_s > 0.0 else args.cycles / args.freq
    if args.signal_type != "chirp" or active_s <= 0.0:
        return args.freq
    t_active = min(max(t - args.warmup_s, 0.0), active_s)
    alpha = t_active / active_s
    return args.chirp_start_hz + (args.chirp_end_hz - args.chirp_start_hz) * alpha


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joint", type=int, required=True, choices=range(1, 8))
    ap.add_argument("--signal-type", choices=["sine", "chirp"], default="sine")
    ap.add_argument("--freq", type=float, default=0.5)
    ap.add_argument("--chirp-start-hz", type=float, default=0.1)
    ap.add_argument("--chirp-end-hz", type=float, default=3.0)
    ap.add_argument("--amplitude", type=float, default=0.12)
    ap.add_argument("--cycles", type=float, default=8.0)
    ap.add_argument("--duration-s", type=float, default=0.0)
    ap.add_argument("--warmup-s", type=float, default=2.0)
    ap.add_argument("--post-hold-s", type=float, default=1.0)
    ap.add_argument("--ramp-s", type=float, default=0.5)
    ap.add_argument("--rate-hz", type=float, default=50.0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--param-tag", default="")
    ap.add_argument("--real-kp", default="")
    ap.add_argument("--real-kd", default="")
    ap.add_argument("--wait-timeout-s", type=float, default=120.0)
    ap.add_argument("--stop-grace-s", type=float, default=0.3)
    ap.add_argument("--spin-drain", type=int, default=16, help="Callbacks to drain at each logging tick.")
    ap.add_argument("--check-route-ip", default="10.1.1.220")
    ap.add_argument("--expected-route-dev", default="enp8s0")
    ap.add_argument("--joint-topic", default="/right_joint_states")
    ap.add_argument("--action-topic", default="/model_action")
    ap.add_argument("--fsm-state-topic", default="/a1_tt/fsm_state")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.freq <= 0.0:
        raise SystemExit("--freq must be positive")
    if args.chirp_start_hz <= 0.0 or args.chirp_end_hz <= 0.0:
        raise SystemExit("--chirp-start-hz and --chirp-end-hz must be positive")
    if args.rate_hz <= 0.0:
        raise SystemExit("--rate-hz must be positive")
    active_s = args.duration_s if args.duration_s > 0.0 else args.cycles / args.freq
    total_s = args.warmup_s + active_s + args.post_hold_s
    if total_s <= 0.0:
        raise SystemExit("test duration must be positive")

    check_wired_route(args.check_route_ip, args.expected_route_dev)

    rclpy.init()
    node = FsmTestRecorder(args)
    stop_requested = False

    def handle_signal(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    joint_zero = args.joint - 1
    fields = sample_fieldnames(["target_q", "target_dq", "actual_q", "actual_dq", "actual_tau"])
    writer, handle = open_csv_writer(args.output, fields)
    try:
        write_trial_manifest(
            args.output.with_suffix(".manifest.json"),
            TrialSpec(
                joint_index=args.joint,
                joint_name=JOINT_NAMES[joint_zero],
                freq_hz=args.chirp_start_hz if args.signal_type == "chirp" else args.freq,
                amplitude_rad=args.amplitude,
                rate_hz=args.rate_hz,
                warmup_s=args.warmup_s,
                sine_s=active_s,
                post_hold_s=args.post_hold_s,
                ramp_s=args.ramp_s,
                action_format="position",
                center=[],
            ),
            {
                "run_id": run_id,
                "output": str(args.output),
                "param_tag": args.param_tag,
                "signal_type": args.signal_type,
                "chirp_start_hz": args.chirp_start_hz,
                "chirp_end_hz": args.chirp_end_hz,
                "duration_s": active_s,
                "real_kp": parse_list7(args.real_kp, default=[]) if args.real_kp else [],
                "real_kd": parse_list7(args.real_kd, default=[]) if args.real_kd else [],
                "command_source": "a1_tt_fsm_supervisor TEST",
                "target_dq_semantics": "q_des_only_test_so_target_dq_is_zero",
                "action_topic": args.action_topic,
                "joint_topic": args.joint_topic,
                "fsm_state_topic": args.fsm_state_topic,
            },
        )

        node.get_logger().info(f"waiting for FSM TEST on {args.fsm_state_topic}")
        wait_deadline = time.monotonic() + args.wait_timeout_s
        while not stop_requested and time.monotonic() < wait_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.in_test():
                break
        if not node.in_test():
            raise TimeoutError("FSM did not enter TEST before wait timeout")

        start = time.monotonic()
        next_tick = start
        stop_since = 0.0
        sample_index = 0
        dt = 1.0 / args.rate_hz
        while not stop_requested:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
                continue
            for _ in range(max(1, args.spin_drain)):
                rclpy.spin_once(node, timeout_sec=0.0)
            now = time.monotonic()
            t = now - start

            if not node.in_test():
                if stop_since <= 0.0:
                    stop_since = now
                if now - stop_since >= args.stop_grace_s:
                    break
            else:
                stop_since = 0.0

            if t > total_s + max(args.stop_grace_s, 1.0):
                node.get_logger().warning("record timeout after expected test duration")
                break

            if not all(math.isfinite(x) for x in node.last_target_q + node.last_q):
                next_tick += dt
                continue

            row: dict[str, float | int | str] = {
                "source": "real",
                "run_id": run_id,
                "stamp": time.time(),
                "t": t,
                "phase": phase_for_t(t, args),
                "joint_index": args.joint,
                "joint_name": JOINT_NAMES[joint_zero],
                "signal_type": args.signal_type,
                "freq_hz": freq_for_t(t, args),
                "chirp_start_hz": args.chirp_start_hz if args.signal_type == "chirp" else "",
                "chirp_end_hz": args.chirp_end_hz if args.signal_type == "chirp" else "",
                "amplitude_rad": args.amplitude,
                "sample_index": sample_index,
            }
            for i in range(7):
                row[f"target_q{i + 1}"] = node.last_target_q[i]
                row[f"target_dq{i + 1}"] = 0.0
                row[f"actual_q{i + 1}"] = node.last_q[i]
                row[f"actual_dq{i + 1}"] = node.last_dq[i]
                row[f"actual_tau{i + 1}"] = node.last_tau[i]
            writer.writerow(row)
            sample_index += 1
            next_tick += dt

        node.get_logger().info(f"wrote {args.output} ({sample_index} samples)")
        return 0
    finally:
        handle.close()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
