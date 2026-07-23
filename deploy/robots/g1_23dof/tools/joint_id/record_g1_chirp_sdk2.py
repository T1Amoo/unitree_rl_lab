#!/usr/bin/env python3
"""Run a single-joint chirp on a real Unitree G1 23-DoF and log LowState feedback.

The default mode is dry-run. Real robot control requires --enable-control.
"""

from __future__ import annotations

import argparse
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

from g1_joint_id_common import (
    DEFAULT_JOINT_POS_23,
    JOINT_IDS_MAP,
    POLICY_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_ARM_POLICY_INDICES,
    RIGHT_ARM_SDK_INDICES,
    SDK_MOTOR_COUNT,
    VALID_G1_MOTOR_COUNT,
    add_vector,
    chirp_delta,
    clip_q23,
    default_output_path,
    gain_pair,
    open_csv_writer,
    ramp_interpolate,
    sample_fieldnames,
    sdk_state_to_policy23,
    write_json,
)


class Mode:
    PR = 0


def run_command(argv: list[str]) -> str:
    proc = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} failed: {proc.stdout.strip()}")
    return proc.stdout.strip()


def check_network(interface: str, route_ip: str) -> None:
    link = run_command(["ip", "link", "show", "dev", interface])
    if "state UP" not in link and "LOWER_UP" not in link:
        raise RuntimeError(f"network interface {interface!r} is not UP: {link}")
    if route_ip:
        route = run_command(["ip", "route", "get", route_ip])
        if f" dev {interface} " not in f" {route} ":
            raise RuntimeError(f"route to {route_ip} is not on {interface}: {route}")


class G1LowLevelSession:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.low_state = None
        self.mode_machine = 0
        self.received_low_state = False
        self.crc = None
        self.low_cmd = None
        self.publisher = None
        self.subscriber = None
        self._sdk = {}

    def init(self) -> None:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self._sdk = {
            "MotionSwitcherClient": MotionSwitcherClient,
            "ChannelPublisher": ChannelPublisher,
            "ChannelSubscriber": ChannelSubscriber,
            "LowCmd_": LowCmd_,
            "LowState_": LowState_,
        }

        ChannelFactoryInitialize(self.args.domain_id, self.args.network_interface)
        if not self.args.skip_release_mode:
            self.release_motion_modes()

        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self.low_state_handler, 10)

    def release_motion_modes(self) -> None:
        client = self._sdk["MotionSwitcherClient"]()
        client.SetTimeout(5.0)
        client.Init()
        for _ in range(8):
            status, result = client.CheckMode()
            name = ""
            if isinstance(result, dict):
                name = str(result.get("name", ""))
            if not name:
                return
            print(f"[g1_joint_id] releasing active motion mode: {name}", flush=True)
            client.ReleaseMode()
            time.sleep(0.5)
        raise RuntimeError("motion mode is still active after repeated ReleaseMode calls")

    def low_state_handler(self, msg) -> None:
        self.low_state = msg
        if not self.received_low_state:
            self.mode_machine = int(msg.mode_machine)
            self.received_low_state = True

    def wait_low_state(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.low_state is not None:
                return
            time.sleep(0.02)
        raise TimeoutError("no rt/lowstate received")

    def publish(self, q23: list[float], dq23: list[float], kp23: list[float], kd23: list[float]) -> None:
        assert self.low_cmd is not None
        assert self.publisher is not None
        assert self.crc is not None

        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine
        for motor_id in range(SDK_MOTOR_COUNT):
            cmd = self.low_cmd.motor_cmd[motor_id]
            cmd.mode = 0
            cmd.tau = 0.0
            cmd.q = 0.0
            cmd.dq = 0.0
            cmd.kp = 0.0
            cmd.kd = 0.0

        for policy_i, motor_id in enumerate(JOINT_IDS_MAP):
            cmd = self.low_cmd.motor_cmd[motor_id]
            cmd.mode = 1
            cmd.tau = 0.0
            cmd.q = float(q23[policy_i])
            cmd.dq = float(dq23[policy_i])
            cmd.kp = float(kp23[policy_i])
            cmd.kd = float(kd23[policy_i])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def publish_damping(self, kd23: list[float], duration_s: float, rate_hz: float) -> None:
        if duration_s <= 0.0 or self.low_state is None:
            return
        q23, _, _ = sdk_state_to_policy23(self.low_state)
        kp23 = [0.0] * 23
        dq23 = [0.0] * 23
        end = time.monotonic() + duration_s
        dt = 1.0 / rate_hz
        next_tick = time.monotonic()
        while time.monotonic() < end:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
                continue
            if self.low_state is not None:
                q23, _, _ = sdk_state_to_policy23(self.low_state)
            self.publish(q23, dq23, kp23, kd23)
            next_tick += dt


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enable-control", action="store_true", help="Actually initialize DDS and publish rt/lowcmd.")
    ap.add_argument("--network-interface", default="enx6c1ff76cb7d7", help="Unitree DDS network interface.")
    ap.add_argument("--domain-id", type=int, default=0, help="Unitree DDS domain id. Real robot default is 0.")
    ap.add_argument("--check-route-ip", default="192.168.123.161", help="Robot IP used for route validation.")
    ap.add_argument("--skip-route-check", action="store_true", help="Do not verify ip link/route before control.")
    ap.add_argument("--skip-release-mode", action="store_true", help="Do not call MotionSwitcher ReleaseMode.")
    ap.add_argument("--joint", type=int, required=True, choices=range(1, 6), help="1-based right-arm joint index.")
    ap.add_argument("--amplitude", type=float, default=0.08, help="Chirp amplitude in rad.")
    ap.add_argument("--chirp-start-hz", type=float, default=0.1)
    ap.add_argument("--chirp-end-hz", type=float, default=2.0)
    ap.add_argument("--duration-s", type=float, default=30.0, help="Active chirp duration.")
    ap.add_argument("--move-to-default-s", type=float, default=3.0, help="Ramp from current pose to TT default pose.")
    ap.add_argument("--warmup-s", type=float, default=2.0, help="Hold TT default before chirp.")
    ap.add_argument("--settle-before-chirp", dest="settle_before_chirp", action="store_true", default=True)
    ap.add_argument("--no-settle-before-chirp", dest="settle_before_chirp", action="store_false")
    ap.add_argument(
        "--settle-mode",
        choices=["stable", "target"],
        default="stable",
        help="stable gates only on actual velocity; target also requires small position error to the default pose.",
    )
    ap.add_argument("--settle-position-tolerance", type=float, default=0.01, help="Position tolerance used by --settle-mode target.")
    ap.add_argument("--settle-velocity-tolerance", type=float, default=0.05, help="Tested-joint velocity needed before chirp.")
    ap.add_argument("--settle-min-s", type=float, default=0.5, help="Continuous settled time required before chirp.")
    ap.add_argument("--settle-timeout-s", type=float, default=8.0, help="Maximum extra settle hold. Use 0 for no timeout.")
    ap.add_argument("--settle-timeout-action", choices=["abort", "start"], default="abort")
    ap.add_argument("--ramp-s", type=float, default=1.0, help="Smooth chirp envelope ramp in/out.")
    ap.add_argument("--post-hold-s", type=float, default=1.0, help="Hold TT default after chirp.")
    ap.add_argument("--finish-damping-s", type=float, default=0.5, help="Publish damping before exit. Use 0 to skip.")
    ap.add_argument("--rate-hz", type=float, default=500.0, help="LowCmd publish and log rate.")
    ap.add_argument("--gain-preset", choices=sorted(["table_tennis", "fixstand"]), default="table_tennis")
    ap.add_argument("--kp23", default="", help="Override 23-value kp list in policy order.")
    ap.add_argument("--kd23", default="", help="Override 23-value kd list in policy order.")
    ap.add_argument("--no-joint-limit-clip", action="store_true", help="Disable clipping to deploy joint limits.")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--date-tag", default="", help="Output date directory under 系统辨识/unitree. Defaults to YYYYMMDD.")
    ap.add_argument("--run-id", default="", help="Defaults to YYYYMMDD_HHMMSS.")
    ap.add_argument("--print-every-s", type=float, default=1.0)
    return ap


def validate_args(args: argparse.Namespace) -> None:
    if args.rate_hz <= 0.0:
        raise SystemExit("--rate-hz must be positive")
    if args.amplitude <= 0.0:
        raise SystemExit("--amplitude must be positive")
    if args.duration_s <= 0.0:
        raise SystemExit("--duration-s must be positive")
    if args.chirp_start_hz <= 0.0 or args.chirp_end_hz <= 0.0:
        raise SystemExit("chirp frequencies must be positive")
    if args.move_to_default_s < 0.0 or args.warmup_s < 0.0 or args.post_hold_s < 0.0:
        raise SystemExit("stage durations must be non-negative")
    if args.finish_damping_s < 0.0:
        raise SystemExit("--finish-damping-s must be non-negative")
    if args.settle_position_tolerance < 0.0 or args.settle_velocity_tolerance < 0.0:
        raise SystemExit("settle tolerances must be non-negative")
    if args.settle_min_s < 0.0 or args.settle_timeout_s < 0.0:
        raise SystemExit("settle durations must be non-negative")


def target_for_time(
    t: float,
    *,
    args: argparse.Namespace,
    start_q23: list[float],
    joint_policy_index: int,
) -> tuple[str, list[float], list[float], float, float]:
    move_end = args.move_to_default_s
    warmup_end = move_end + args.warmup_s
    chirp_end = warmup_end + args.duration_s
    post_end = chirp_end + args.post_hold_s

    if t < move_end:
        q, dq = ramp_interpolate(start_q23, DEFAULT_JOINT_POS_23, t, args.move_to_default_s)
        return "move_to_default", q, dq, 0.0, math.nan
    if t < warmup_end:
        return "warmup", list(DEFAULT_JOINT_POS_23), [0.0] * 23, 0.0, math.nan
    if t < chirp_end:
        chirp_t = t - warmup_end
        delta, d_delta, freq = chirp_delta(
            chirp_t,
            active_s=args.duration_s,
            amplitude=args.amplitude,
            start_hz=args.chirp_start_hz,
            end_hz=args.chirp_end_hz,
            ramp_s=args.ramp_s,
        )
        q = list(DEFAULT_JOINT_POS_23)
        dq = [0.0] * 23
        q[joint_policy_index] += delta
        dq[joint_policy_index] = d_delta
        return "chirp", q, dq, chirp_t, freq
    if t < post_end:
        return "post_hold", list(DEFAULT_JOINT_POS_23), [0.0] * 23, args.duration_s, math.nan
    return "done", list(DEFAULT_JOINT_POS_23), [0.0] * 23, args.duration_s, math.nan


def build_row(
    *,
    args: argparse.Namespace,
    source: str,
    run_id: str,
    t: float,
    stage: str,
    sample_index: int,
    joint_policy_index: int,
    joint_sdk_index: int,
    q_target23: list[float],
    dq_target23: list[float],
    q_cmd23: list[float],
    dq_cmd23: list[float],
    q_actual23: list[float],
    dq_actual23: list[float],
    tau_actual23: list[float],
    low_state,
    chirp_t: float,
    chirp_frequency: float,
    clipped: bool,
    settle_error: float,
    settle_velocity: float,
    settle_ready: bool,
    settle_elapsed: float,
) -> dict:
    row: dict[str, float | int | str] = {
        "source": source,
        "run_id": run_id,
        "wall_time_ns": time.time_ns(),
        "t": t,
        "stage": stage,
        "sample_index": sample_index,
        "joint_index": args.joint,
        "joint_name": POLICY_JOINT_NAMES[joint_policy_index],
        "policy_index": joint_policy_index,
        "sdk_motor_id": joint_sdk_index,
        "control_enabled": int(args.enable_control),
        "lowstate_tick": int(low_state.tick) if low_state is not None else -1,
        "mode_pr": int(low_state.mode_pr) if low_state is not None else -1,
        "mode_machine": int(low_state.mode_machine) if low_state is not None else -1,
        "chirp_t": chirp_t,
        "chirp_frequency_hz": chirp_frequency,
        "chirp_start_hz": args.chirp_start_hz,
        "chirp_end_hz": args.chirp_end_hz,
        "amplitude_rad": args.amplitude,
        "target_clipped": int(clipped),
        "settle_error_rad": settle_error,
        "settle_velocity_rad_s": settle_velocity,
        "settle_ready": int(settle_ready),
        "settle_elapsed_s": settle_elapsed,
    }
    add_vector(row, "target_q", q_target23, 23)
    add_vector(row, "target_dq", dq_target23, 23)
    add_vector(row, "cmd_q", q_cmd23, 23)
    add_vector(row, "cmd_dq", dq_cmd23, 23)
    add_vector(row, "actual_q", q_actual23, 23)
    add_vector(row, "actual_dq", dq_actual23, 23)
    add_vector(row, "actual_tau", tau_actual23, 23)
    add_vector(row, "right_actual_q", [q_actual23[i] for i in RIGHT_ARM_POLICY_INDICES], 5)
    add_vector(row, "right_actual_dq", [dq_actual23[i] for i in RIGHT_ARM_POLICY_INDICES], 5)
    add_vector(row, "right_actual_tau", [tau_actual23[i] for i in RIGHT_ARM_POLICY_INDICES], 5)
    return row


def write_manifest(
    args: argparse.Namespace,
    *,
    output: Path,
    run_id: str,
    kp23: list[float],
    kd23: list[float],
) -> None:
    write_json(
        output.with_suffix(".manifest.json"),
        {
            "run_id": run_id,
            "output": str(output),
            "date_tag": args.date_tag,
            "layout": "系统辨识/unitree/<date>/jointN/{real,sim,plots,merged,summary}",
            "control_enabled": bool(args.enable_control),
            "network_interface": args.network_interface,
            "domain_id": args.domain_id,
            "check_route_ip": args.check_route_ip,
            "joint": args.joint,
            "right_arm_joint_names": RIGHT_ARM_JOINT_NAMES,
            "right_arm_sdk_indices": RIGHT_ARM_SDK_INDICES,
            "tested_joint_name": RIGHT_ARM_JOINT_NAMES[args.joint - 1],
            "tested_policy_index": RIGHT_ARM_POLICY_INDICES[args.joint - 1],
            "tested_sdk_motor_id": RIGHT_ARM_SDK_INDICES[args.joint - 1],
            "policy_joint_names": POLICY_JOINT_NAMES,
            "joint_ids_map": JOINT_IDS_MAP,
            "default_joint_pos_23": DEFAULT_JOINT_POS_23,
            "kp23": kp23,
            "kd23": kd23,
            "gain_preset": args.gain_preset,
            "chirp_start_hz": args.chirp_start_hz,
            "chirp_end_hz": args.chirp_end_hz,
            "amplitude_rad": args.amplitude,
            "duration_s": args.duration_s,
            "move_to_default_s": args.move_to_default_s,
            "warmup_s": args.warmup_s,
            "settle_before_chirp": bool(args.settle_before_chirp),
            "settle_mode": args.settle_mode,
            "settle_position_tolerance": args.settle_position_tolerance,
            "settle_velocity_tolerance": args.settle_velocity_tolerance,
            "settle_min_s": args.settle_min_s,
            "settle_timeout_s": args.settle_timeout_s,
            "settle_timeout_action": args.settle_timeout_action,
            "ramp_s": args.ramp_s,
            "post_hold_s": args.post_hold_s,
            "finish_damping_s": args.finish_damping_s,
            "rate_hz": args.rate_hz,
            "valid_g1_motor_count": VALID_G1_MOTOR_COUNT,
            "sdk_motor_count": SDK_MOTOR_COUNT,
        },
    )


def prepare_output_dirs(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.name == "real":
        joint_root = output.parent.parent
        for name in ("sim", "plots", "merged", "summary"):
            (joint_root / name).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)
    kp23, kd23 = gain_pair(args.gain_preset, args.kp23, args.kd23)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    date_tag = args.date_tag or time.strftime("%Y%m%d")
    args.date_tag = date_tag
    output = args.output or default_output_path(args.joint, run_id, date_tag)
    output = Path(output)
    prepare_output_dirs(output)
    joint_policy_index = RIGHT_ARM_POLICY_INDICES[args.joint - 1]
    joint_sdk_index = RIGHT_ARM_SDK_INDICES[args.joint - 1]

    if args.enable_control and not args.skip_route_check:
        check_network(args.network_interface, args.check_route_ip)

    session = None
    if args.enable_control:
        session = G1LowLevelSession(args)
        session.init()
        session.wait_low_state(timeout_s=5.0)
        start_q23, _, _ = sdk_state_to_policy23(session.low_state)
    else:
        start_q23 = list(DEFAULT_JOINT_POS_23)
        print("[g1_joint_id] dry-run: not initializing DDS and not publishing rt/lowcmd", flush=True)

    writer, handle = open_csv_writer(output, sample_fieldnames())
    write_manifest(args, output=output, run_id=run_id, kp23=kp23, kd23=kd23)

    stop_requested = False

    def handle_signal(signum, frame):  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    dt = 1.0 / args.rate_hz
    start = time.monotonic()
    next_tick = start
    sample_index = 0
    last_print = start
    warmup_end = args.move_to_default_s + args.warmup_s
    settle_enabled = bool(args.settle_before_chirp and session is not None)
    settle_start_t: float | None = None
    settle_ready_since_t: float | None = None
    settle_elapsed_final = 0.0
    settle_gate_ready = not settle_enabled
    chirp_start_t: float | None = None

    try:
        while not stop_requested:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
                continue
            t = now - start
            low_state = session.low_state if session is not None else None
            if low_state is not None:
                q_actual23, dq_actual23, tau_actual23 = sdk_state_to_policy23(low_state)
            else:
                q_actual23 = [math.nan] * 23
                dq_actual23 = [math.nan] * 23
                tau_actual23 = [math.nan] * 23

            settle_error = math.nan
            settle_velocity = math.nan
            settle_ready = False
            settle_elapsed = settle_elapsed_final
            abort_after_write = False

            if t < args.move_to_default_s:
                stage, q_target23, dq_target23, chirp_t, chirp_freq = target_for_time(
                    t,
                    args=args,
                    start_q23=start_q23,
                    joint_policy_index=joint_policy_index,
                )
            elif t < warmup_end:
                stage = "warmup"
                q_target23 = list(DEFAULT_JOINT_POS_23)
                dq_target23 = [0.0] * 23
                chirp_t = 0.0
                chirp_freq = math.nan
            else:
                if chirp_start_t is None:
                    if settle_enabled:
                        if settle_start_t is None:
                            settle_start_t = t
                            settle_ready_since_t = None
                        settle_elapsed = t - settle_start_t
                        settle_error = abs(q_actual23[joint_policy_index] - DEFAULT_JOINT_POS_23[joint_policy_index])
                        settle_velocity = abs(dq_actual23[joint_policy_index])
                        velocity_ok = (
                            math.isfinite(settle_velocity)
                            and settle_velocity <= args.settle_velocity_tolerance
                        )
                        position_ok = (
                            math.isfinite(settle_error)
                            and settle_error <= args.settle_position_tolerance
                        )
                        within_tolerance = velocity_ok and (
                            args.settle_mode == "stable" or position_ok
                        )
                        if within_tolerance:
                            if settle_ready_since_t is None:
                                settle_ready_since_t = t
                            settle_ready = (t - settle_ready_since_t) >= args.settle_min_s
                        else:
                            settle_ready_since_t = None
                            settle_ready = False

                        timed_out = args.settle_timeout_s > 0.0 and settle_elapsed >= args.settle_timeout_s
                        if settle_ready or (timed_out and args.settle_timeout_action == "start"):
                            chirp_start_t = t
                            settle_elapsed_final = settle_elapsed
                            settle_gate_ready = settle_ready
                        elif timed_out:
                            stage = "settle_timeout"
                            q_target23 = list(DEFAULT_JOINT_POS_23)
                            dq_target23 = [0.0] * 23
                            chirp_t = 0.0
                            chirp_freq = math.nan
                            abort_after_write = True
                        else:
                            stage = "settle"
                            q_target23 = list(DEFAULT_JOINT_POS_23)
                            dq_target23 = [0.0] * 23
                            chirp_t = 0.0
                            chirp_freq = math.nan
                    else:
                        chirp_start_t = t
                        settle_elapsed_final = 0.0
                        settle_gate_ready = True

                if chirp_start_t is not None:
                    settle_ready = settle_gate_ready
                    settle_elapsed = settle_elapsed_final
                    chirp_t = t - chirp_start_t
                    if chirp_t < args.duration_s:
                        delta, d_delta, chirp_freq = chirp_delta(
                            chirp_t,
                            active_s=args.duration_s,
                            amplitude=args.amplitude,
                            start_hz=args.chirp_start_hz,
                            end_hz=args.chirp_end_hz,
                            ramp_s=args.ramp_s,
                        )
                        stage = "chirp"
                        q_target23 = list(DEFAULT_JOINT_POS_23)
                        dq_target23 = [0.0] * 23
                        q_target23[joint_policy_index] += delta
                        dq_target23[joint_policy_index] = d_delta
                    elif chirp_t < args.duration_s + args.post_hold_s:
                        stage = "post_hold"
                        q_target23 = list(DEFAULT_JOINT_POS_23)
                        dq_target23 = [0.0] * 23
                        chirp_t = args.duration_s
                        chirp_freq = math.nan
                    else:
                        break

            if args.no_joint_limit_clip:
                q_cmd23 = list(q_target23)
                clipped = False
            else:
                q_cmd23, clipped = clip_q23(q_target23)
            dq_cmd23 = list(dq_target23)

            if session is not None:
                session.publish(q_cmd23, dq_cmd23, kp23, kd23)

            writer.writerow(
                build_row(
                    args=args,
                    source="g1_sdk2_lowcmd" if args.enable_control else "dry_run",
                    run_id=run_id,
                    t=t,
                    stage=stage,
                    sample_index=sample_index,
                    joint_policy_index=joint_policy_index,
                    joint_sdk_index=joint_sdk_index,
                    q_target23=q_target23,
                    dq_target23=dq_target23,
                    q_cmd23=q_cmd23,
                    dq_cmd23=dq_cmd23,
                    q_actual23=q_actual23,
                    dq_actual23=dq_actual23,
                    tau_actual23=tau_actual23,
                    low_state=low_state,
                    chirp_t=chirp_t,
                    chirp_frequency=chirp_freq,
                    clipped=clipped,
                    settle_error=settle_error,
                    settle_velocity=settle_velocity,
                    settle_ready=settle_ready,
                    settle_elapsed=settle_elapsed,
                )
            )
            sample_index += 1
            next_tick += dt

            if args.print_every_s > 0.0 and now - last_print >= args.print_every_s:
                last_print = now
                print(
                    f"[g1_joint_id] t={t:7.3f}s stage={stage:>15s} "
                    f"joint={args.joint}:{RIGHT_ARM_JOINT_NAMES[args.joint - 1]} "
                    f"freq={chirp_freq if math.isfinite(chirp_freq) else 0.0:.3f}Hz "
                    f"clipped={int(clipped)} "
                    f"settle_err={settle_error if math.isfinite(settle_error) else 0.0:.4f} "
                    f"settle_dq={settle_velocity if math.isfinite(settle_velocity) else 0.0:.4f} "
                    f"settle_ready={int(settle_ready)}",
                    flush=True,
                )
            if abort_after_write:
                handle.flush()
                raise RuntimeError(
                    "settle timeout before chirp: "
                    f"joint={args.joint} error={settle_error:.4f}rad "
                    f"velocity={settle_velocity:.4f}rad/s elapsed={settle_elapsed:.3f}s"
                )
        handle.flush()
    finally:
        try:
            handle.close()
        finally:
            if session is not None and args.finish_damping_s > 0.0:
                print(f"[g1_joint_id] publishing damping for {args.finish_damping_s:.2f}s", flush=True)
                session.publish_damping(kd23, args.finish_damping_s, args.rate_hz)

    print(f"[g1_joint_id] wrote {output}", flush=True)
    print(f"[g1_joint_id] wrote {output.with_suffix('.manifest.json')}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError) as exc:
        print(f"[g1_joint_id] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
    except KeyboardInterrupt:
        raise SystemExit(130)
