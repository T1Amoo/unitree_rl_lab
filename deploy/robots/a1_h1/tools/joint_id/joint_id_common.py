#!/usr/bin/python3
"""Shared helpers for A1 right-arm joint identification."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

JOINT_NAMES = [f"joint{i}-a1_r" for i in range(1, 8)]

# Current A1 right-arm training/deploy priors. These are starting values for
# experiments, not identified motor constants.
DEFAULT_KP = [200.0, 200.0, 200.0, 120.0, 120.0, 120.0, 120.0]
DEFAULT_KD = [3.5, 3.5, 3.5, 1.0, 1.0, 1.0, 1.0]
DEFAULT_EFFORT_LIMIT = [28.0, 28.0, 28.0, 8.0, 8.0, 8.0, 8.0]
DEFAULT_VELOCITY_LIMIT = [8.0, 8.0, 8.0, 20.0, 20.0, 20.0, 20.0]
DEFAULT_INERTIA = [0.032, 0.032, 0.032, 0.0018, 0.0018, 0.0018, 0.0018]
DEFAULT_VISCOUS = [0.02, 0.02, 0.02, 0.005, 0.005, 0.005, 0.005]
DEFAULT_COULOMB = [0.0] * 7


@dataclass
class TrialSpec:
    joint_index: int
    joint_name: str
    freq_hz: float
    amplitude_rad: float
    rate_hz: float
    warmup_s: float
    sine_s: float
    post_hold_s: float
    ramp_s: float
    action_format: str
    center: list[float]


def parse_list7(text: str | None, default: Iterable[float] | None = None) -> list[float]:
    if text is None or text == "":
        if default is None:
            raise ValueError("expected a 7-value list")
        return [float(x) for x in default]
    cleaned = text.strip().strip("[]()")
    values = [float(x) for x in cleaned.replace(",", " ").split()]
    if len(values) != 7:
        raise ValueError(f"expected 7 values, got {len(values)}: {text!r}")
    return values


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_trial_manifest(path: Path, spec: TrialSpec, extra: dict | None = None) -> None:
    data = asdict(spec)
    if extra:
        data.update(extra)
    write_json(path, data)


def joint_state_to_ordered(
    names: list[str],
    values: list[float],
    fallback: Iterable[float] | None = None,
) -> list[float]:
    fallback_values = list(fallback) if fallback is not None else [math.nan] * 7
    out = list(fallback_values)
    if not names:
        for i, value in enumerate(values[:7]):
            out[i] = float(value)
        return out
    index = {name: i for i, name in enumerate(names)}
    for i, joint_name in enumerate(JOINT_NAMES):
        if joint_name in index and index[joint_name] < len(values):
            out[i] = float(values[index[joint_name]])
        else:
            # Robot-side variants sometimes publish r1..r7 instead.
            short = f"r{i + 1}"
            if short in index and index[short] < len(values):
                out[i] = float(values[index[short]])
    return out


def smoothstep_envelope(t: float, active_s: float, ramp_s: float) -> tuple[float, float]:
    """Return envelope and envelope derivative for a sine segment."""
    if active_s <= 0.0:
        return 0.0, 0.0
    ramp = min(max(ramp_s, 0.0), active_s * 0.5)
    if ramp <= 1e-9:
        return 1.0, 0.0

    def smooth(x: float) -> tuple[float, float]:
        x = min(max(x, 0.0), 1.0)
        y = x * x * (3.0 - 2.0 * x)
        dy_dx = 6.0 * x * (1.0 - x)
        return y, dy_dx

    if t < ramp:
        y, dy_dx = smooth(t / ramp)
        return y, dy_dx / ramp
    if t > active_s - ramp:
        y, dy_dx = smooth((active_s - t) / ramp)
        return y, -dy_dx / ramp
    return 1.0, 0.0


def sine_target(
    center: list[float],
    joint_zero_based: int,
    amplitude: float,
    freq_hz: float,
    t_active: float,
    active_s: float,
    ramp_s: float,
) -> tuple[list[float], list[float]]:
    q = list(center)
    dq = [0.0] * 7
    env, env_dot = smoothstep_envelope(t_active, active_s, ramp_s)
    omega = 2.0 * math.pi * freq_hz
    phase = omega * t_active
    q_delta = amplitude * env * math.sin(phase)
    dq_delta = amplitude * (env_dot * math.sin(phase) + env * omega * math.cos(phase))
    q[joint_zero_based] = center[joint_zero_based] + q_delta
    dq[joint_zero_based] = dq_delta
    return q, dq


def open_csv_writer(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return writer, handle


def sample_fieldnames(prefixes: Iterable[str]) -> list[str]:
    fields = [
        "source",
        "run_id",
        "stamp",
        "t",
        "phase",
        "joint_index",
        "joint_name",
        "signal_type",
        "freq_hz",
        "chirp_start_hz",
        "chirp_end_hz",
        "amplitude_rad",
        "sample_index",
    ]
    for prefix in prefixes:
        for i in range(1, 8):
            fields.append(f"{prefix}{i}")
    return fields


def get_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi
