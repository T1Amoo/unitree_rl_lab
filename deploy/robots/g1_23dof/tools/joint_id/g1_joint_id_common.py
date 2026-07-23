#!/usr/bin/env python3
"""Shared helpers for G1 23-DoF joint-identification chirp tests."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Iterable

SDK_MOTOR_COUNT = 35
VALID_G1_MOTOR_COUNT = 29

# Policy/deploy order: legs12, waist_yaw, left_arm5, right_arm5.
POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

# Policy/deploy order -> sparse Unitree G1 SDK motor indices.
JOINT_IDS_MAP = [
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12,
    15, 16, 17, 18, 19,
    22, 23, 24, 25, 26,
]

RIGHT_ARM_POLICY_INDICES = [18, 19, 20, 21, 22]
RIGHT_ARM_SDK_INDICES = [JOINT_IDS_MAP[i] for i in RIGHT_ARM_POLICY_INDICES]
RIGHT_ARM_JOINT_NAMES = [POLICY_JOINT_NAMES[i] for i in RIGHT_ARM_POLICY_INDICES]

DEFAULT_JOINT_POS_23 = [
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0,
    0.2, 0.2, 0.0, 0.6, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0,
]

# From config/policy/table_tennis/v16_35000/params/deploy.yaml.
TABLE_TENNIS_KP_23 = [
    40.18, 99.09, 40.18, 99.09, 28.50, 28.50,
    40.18, 99.09, 40.18, 99.09, 28.50, 28.50,
    99.09,
    14.25, 14.25, 14.25, 14.25, 14.25,
    14.25, 14.25, 14.25, 14.25, 14.25,
]
TABLE_TENNIS_KD_23 = [
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    6.309,
    0.907, 0.907, 0.907, 0.907, 0.907,
    0.907, 0.907, 0.907, 0.907, 0.907,
]

# From config/config.yaml FixStand. Useful for stiffer setup tests.
FIXSTAND_KP_23 = [
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
    200.0,
    40.0, 40.0, 40.0, 40.0, 40.0,
    40.0, 40.0, 40.0, 40.0, 40.0,
]
FIXSTAND_KD_23 = [
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
    5.0,
    2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0,
]

JOINT_LIMITS_23 = [
    (-2.5307, 2.8798),
    (-0.5236, 2.9671),
    (-2.7576, 2.7576),
    (-0.0873, 2.8798),
    (-0.8727, 0.5236),
    (-0.2618, 0.2618),
    (-2.5307, 2.8798),
    (-2.9671, 0.5236),
    (-2.7576, 2.7576),
    (-0.0873, 2.8798),
    (-0.8727, 0.5236),
    (-0.2618, 0.2618),
    (-2.618, 2.618),
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.9722, 1.9722),
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.9722, 1.9722),
]

GAIN_PRESETS = {
    "table_tennis": (TABLE_TENNIS_KP_23, TABLE_TENNIS_KD_23),
    "fixstand": (FIXSTAND_KP_23, FIXSTAND_KD_23),
}


def parse_float_list(
    text: str | None,
    *,
    expected_len: int | None = None,
    default: Iterable[float] | None = None,
) -> list[float]:
    if text is None or text == "":
        if default is None:
            raise ValueError("expected a numeric list")
        values = [float(x) for x in default]
    else:
        cleaned = text.strip().strip("[]()")
        values = [float(x) for x in cleaned.replace(",", " ").split()]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"expected {expected_len} values, got {len(values)}: {text!r}")
    return values


def gain_pair(preset: str, kp23: str | None = None, kd23: str | None = None) -> tuple[list[float], list[float]]:
    if preset not in GAIN_PRESETS:
        raise ValueError(f"unknown gain preset {preset!r}; choices={sorted(GAIN_PRESETS)}")
    kp_default, kd_default = GAIN_PRESETS[preset]
    return (
        parse_float_list(kp23, expected_len=23, default=kp_default),
        parse_float_list(kd23, expected_len=23, default=kd_default),
    )


def smoothstep(x: float) -> tuple[float, float]:
    x = min(max(float(x), 0.0), 1.0)
    y = x * x * (3.0 - 2.0 * x)
    dy_dx = 6.0 * x * (1.0 - x)
    return y, dy_dx


def ramp_interpolate(start: list[float], end: list[float], t: float, duration: float) -> tuple[list[float], list[float]]:
    if duration <= 1e-9:
        return list(end), [0.0] * len(end)
    ratio, ratio_dot_dx = smoothstep(t / duration)
    ratio_dot = ratio_dot_dx / duration
    q = [(1.0 - ratio) * a + ratio * b for a, b in zip(start, end)]
    dq = [ratio_dot * (b - a) for a, b in zip(start, end)]
    return q, dq


def envelope(t: float, active_s: float, ramp_s: float) -> tuple[float, float]:
    if active_s <= 0.0:
        return 0.0, 0.0
    ramp = min(max(ramp_s, 0.0), active_s * 0.5)
    if ramp <= 1e-9:
        return 1.0, 0.0
    if t < ramp:
        y, dy_dx = smoothstep(t / ramp)
        return y, dy_dx / ramp
    if t > active_s - ramp:
        y, dy_dx = smoothstep((active_s - t) / ramp)
        return y, -dy_dx / ramp
    return 1.0, 0.0


def chirp_delta(
    t: float,
    *,
    active_s: float,
    amplitude: float,
    start_hz: float,
    end_hz: float,
    ramp_s: float,
) -> tuple[float, float, float]:
    """Return q delta, dq delta, and instantaneous frequency for a linear chirp."""
    if active_s <= 0.0:
        return 0.0, 0.0, start_hz
    t = min(max(t, 0.0), active_s)
    k = (end_hz - start_hz) / active_s
    freq = start_hz + k * t
    phase = 2.0 * math.pi * (start_hz * t + 0.5 * k * t * t)
    env, env_dot = envelope(t, active_s, ramp_s)
    q_delta = amplitude * env * math.sin(phase)
    dq_delta = amplitude * (env_dot * math.sin(phase) + env * 2.0 * math.pi * freq * math.cos(phase))
    return q_delta, dq_delta, freq


def clip_q23(q: list[float]) -> tuple[list[float], bool]:
    out: list[float] = []
    clipped = False
    for value, (lo, hi) in zip(q, JOINT_LIMITS_23):
        clipped_value = min(max(value, lo), hi)
        clipped = clipped or clipped_value != value
        out.append(clipped_value)
    return out, clipped


def sdk_state_to_policy23(low_state) -> tuple[list[float], list[float], list[float]]:
    q: list[float] = []
    dq: list[float] = []
    tau: list[float] = []
    for motor_id in JOINT_IDS_MAP:
        state = low_state.motor_state[motor_id]
        q.append(float(state.q))
        dq.append(float(state.dq))
        tau.append(float(state.tau_est))
    return q, dq, tau


def unitree_trial_dirs(joint: int, date_tag: str | None = None) -> dict[str, Path]:
    date = date_tag or time.strftime("%Y%m%d")
    root = Path("系统辨识") / "unitree" / date / f"joint{joint}"
    return {
        "root": root,
        "real": root / "real",
        "sim": root / "sim",
        "plots": root / "plots",
        "merged": root / "merged",
        "summary": root / "summary",
    }


def ensure_unitree_trial_dirs(joint: int, date_tag: str | None = None) -> dict[str, Path]:
    dirs = unitree_trial_dirs(joint, date_tag)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def default_output_path(joint: int, run_id: str | None = None, date_tag: str | None = None) -> Path:
    dirs = ensure_unitree_trial_dirs(joint, date_tag)
    stem = run_id or time.strftime("%H%M%S")
    return dirs["real"] / f"j{joint}_chirp_{stem}.csv"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def open_csv_writer(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return writer, handle


def sample_fieldnames() -> list[str]:
    fields = [
        "source",
        "run_id",
        "wall_time_ns",
        "t",
        "stage",
        "sample_index",
        "joint_index",
        "joint_name",
        "policy_index",
        "sdk_motor_id",
        "control_enabled",
        "lowstate_tick",
        "mode_pr",
        "mode_machine",
        "chirp_t",
        "chirp_frequency_hz",
        "chirp_start_hz",
        "chirp_end_hz",
        "amplitude_rad",
        "target_clipped",
        "settle_error_rad",
        "settle_velocity_rad_s",
        "settle_ready",
        "settle_elapsed_s",
    ]
    for prefix in ("target_q", "target_dq", "cmd_q", "cmd_dq", "actual_q", "actual_dq", "actual_tau"):
        for i in range(1, 24):
            fields.append(f"{prefix}{i}")
    for prefix in ("right_actual_q", "right_actual_dq", "right_actual_tau"):
        for i in range(1, 6):
            fields.append(f"{prefix}{i}")
    return fields


def add_vector(row: dict[str, float | int | str], prefix: str, values: Iterable[float], count: int) -> None:
    vals = list(values)
    for i in range(count):
        row[f"{prefix}{i + 1}"] = float(vals[i]) if i < len(vals) else math.nan
