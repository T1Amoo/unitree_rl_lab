#!/usr/bin/env python3
"""Replay a recorded Isaac q_des sequence through the MuJoCo damiao_mit executor.

Feed the q_des_1..7 columns from an Isaac play-trace CSV into the mujoco
damiao_mit actuator model at DECIMATION physics steps per 50 Hz control tick,
and record the resulting q_1..7 (actual joint angles).

Also supports --mode direct_response which simply writes q_des as q (ideal
tracking, no torque dynamics) to produce a zero-latency baseline.

Usage
-----
    python replay_qdes_damiao_mit.py \\
        --isaac-trace /tmp/a1_v7_isaac_trace.csv \\
        --out /tmp/a1_v7_mujoco_damiao_mit.csv

    python replay_qdes_damiao_mit.py \\
        --isaac-trace /tmp/a1_v7_isaac_trace.csv \\
        --out /tmp/a1_v7_mujoco_direct.csv \\
        --mode direct_response
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

# By default use the sibling sim2sim implementation. Frozen deployment bundles
# can point this at their own copied implementation so scene/default/response
# constants come from the same snapshot as the policy being diagnosed.
MODULE_DIR = Path(os.environ.get("A1_SIM2SIM_MODULE_DIR", Path(__file__).parent)).resolve()
sys.path.insert(0, str(MODULE_DIR))

from a1_scene import PHYSICS_DT, build_scene_xml, load_scene, reset_robot_qpos  # noqa: E402
from policy_io import A1PolicyIO, FittedSecondOrderActionResponse  # noqa: E402

# Must match run_a1_tt_sim2sim.py's DECIMATION constant.
DECIMATION = 10  # physics steps per 50 Hz control tick


def _load_q_des(trace_csv: Path) -> np.ndarray:
    """Load q_des_1..7 from trace CSV, return (N, 7) float64 array."""
    cols = [f"q_des_{i}" for i in range(1, 8)]
    with trace_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{trace_csv} is empty")
    missing = set(cols) - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing columns in {trace_csv}: {sorted(missing)}")
    return np.array([[float(row[col]) for col in cols] for row in rows], dtype=np.float64)


def _load_initial_q(trace_csv: Path) -> np.ndarray | None:
    """Return q_1..7 from the first row when the real trace contains them."""
    cols = [f"q_{i}" for i in range(1, 8)]
    with trace_csv.open(newline="") as f:
        row = next(csv.DictReader(f), None)
    if row is None or any(not row.get(col, "") for col in cols):
        return None
    return np.array([float(row[col]) for col in cols], dtype=np.float64)


def _replay_damiao_mit(
    q_des_seq: np.ndarray,
    io: A1PolicyIO,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    real_response_model: bool = False,
) -> np.ndarray:
    """Replay q_des through damiao_mit: DECIMATION physics steps per tick.

    Returns (N, 7) array of actual joint angles recorded after each control tick.
    """
    n_ticks = len(q_des_seq)
    q_out = np.empty((n_ticks, 7), dtype=np.float64)
    response = (
        FittedSecondOrderActionResponse(PHYSICS_DT, io.right_q())
        if real_response_model
        else None
    )
    for tick in range(n_ticks):
        io.q_des = q_des_seq[tick].copy()
        for _ in range(DECIMATION):
            data.qfrc_applied[:] = 0.0
            io.enforce_hold_joints()
            io.enforce_motor_limits(None)  # damiao_mit has no external vel clip
            io.set_motor_q_des(
                response.step(io.q_des, PHYSICS_DT) if response is not None else io.q_des
            )
            io.apply_damiao_mit(PHYSICS_DT)
            mujoco.mj_step(model, data)
            io.enforce_motor_limits(None)
            io.enforce_hold_joints()
        q_out[tick] = io.right_q()
    return q_out


def _replay_direct_response(q_des_seq: np.ndarray, io: A1PolicyIO) -> np.ndarray:
    """Direct-response mode: q = q_des (clipped to soft joint limits).

    This is the zero-latency ideal-tracking baseline. RMSE between this and
    damiao_mit gives the torque-chain tracking error on the same input sequence.
    """
    q_min, q_max = io.q_min, io.q_max
    return np.clip(q_des_seq, q_min, q_max)


def replay(
    isaac_trace: Path,
    out_csv: Path,
    mode: str,
    *,
    real_response_model: bool = False,
    initial_from_trace: bool = False,
) -> None:
    model, data, xml_path = load_scene(build_scene_xml())
    reset_robot_qpos(model, data)
    mujoco.mj_forward(model, data)
    io = A1PolicyIO(model, data, use_predictor=False)
    print(f"[replay_qdes] scene: {xml_path}", flush=True)

    q_des_seq = _load_q_des(isaac_trace)
    initial_q = _load_initial_q(isaac_trace) if initial_from_trace else None
    if initial_q is not None:
        io.set_right_state(initial_q, np.zeros(7, dtype=np.float64), reset_targets=True)
        print(f"[replay_qdes] initial q from trace: {np.round(initial_q, 4).tolist()}", flush=True)
    n = len(q_des_seq)
    print(
        f"[replay_qdes] loaded {n} control ticks from {isaac_trace} | "
        f"mode={mode} real_response_model={real_response_model}",
        flush=True,
    )

    if mode == "damiao_mit":
        q_out = _replay_damiao_mit(
            q_des_seq,
            io,
            model,
            data,
            real_response_model=real_response_model,
        )
    elif mode == "direct_response":
        q_out = _replay_direct_response(q_des_seq, io)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [f"q_{i}" for i in range(1, 8)]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for q_row in q_out:
            writer.writerow({f"q_{i}": float(q_row[i - 1]) for i in range(1, 8)})
    print(f"[replay_qdes] wrote {out_csv} ({n} ticks)", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--isaac-trace", type=Path, required=True, help="Isaac trace CSV with q_des_1..7 columns.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/a1_v7_mujoco_damiao_mit.csv"),
        help="Output CSV with q_1..7 (actual joint angles).",
    )
    ap.add_argument(
        "--mode",
        choices=["damiao_mit", "direct_response"],
        default="damiao_mit",
        help="Actuator model: damiao_mit runs the full torque chain; direct_response passes q_des as q directly.",
    )
    ap.add_argument(
        "--real-response-model",
        action="store_true",
        help="Apply the frozen fitted second-order motor-target response before damiao_mit.",
    )
    ap.add_argument(
        "--initial-from-trace",
        action="store_true",
        help="Initialize q from q_1..7 in the first input row when available.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replay(
        args.isaac_trace,
        args.out,
        args.mode,
        real_response_model=args.real_response_model,
        initial_from_trace=args.initial_from_trace,
    )
