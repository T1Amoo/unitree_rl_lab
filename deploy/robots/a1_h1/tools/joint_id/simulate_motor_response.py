#!/usr/bin/python3
"""Replay a recorded target through a simple DC-motor PD response model."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path

from joint_id_common import (
    DEFAULT_COULOMB,
    DEFAULT_EFFORT_LIMIT,
    DEFAULT_INERTIA,
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_VELOCITY_LIMIT,
    DEFAULT_VISCOUS,
    get_float,
    open_csv_writer,
    parse_list7,
    sample_fieldnames,
    write_json,
)


def interp_scalar(times: list[float], values: list[float], t: float) -> float:
    if not times:
        return math.nan
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    hi = bisect.bisect_left(times, t)
    lo = max(0, hi - 1)
    t0 = times[lo]
    t1 = times[hi]
    if t1 <= t0:
        return values[lo]
    a = (t - t0) / (t1 - t0)
    return values[lo] * (1.0 - a) + values[hi] * a


def torque_limit_dc(effort_limit: float, velocity_limit: float, dq: float) -> float:
    if velocity_limit <= 1e-9:
        return abs(effort_limit)
    scale = max(0.0, 1.0 - abs(dq) / abs(velocity_limit))
    return abs(effort_limit) * scale


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="CSV produced by record_real_sine.py")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--motor-mode", choices=["constant", "dc"], default="dc")
    ap.add_argument("--kp", default="", help="7-value PD kp list.")
    ap.add_argument("--kd", default="", help="7-value PD kd list.")
    ap.add_argument("--inertia", default="", help="7-value joint inertia list.")
    ap.add_argument("--viscous", default="", help="7-value viscous damping list.")
    ap.add_argument("--coulomb", default="", help="7-value Coulomb friction list.")
    ap.add_argument("--effort-limit", default="", help="7-value zero-speed torque limit list.")
    ap.add_argument("--velocity-limit", default="", help="7-value no-load speed / velocity limit list.")
    ap.add_argument("--command-delay-s", type=float, default=0.0, help="Delay applied to q/dq command before PD.")
    ap.add_argument("--initial", choices=["real", "target"], default="real")
    ap.add_argument("--substeps", type=int, default=8)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    kp = parse_list7(args.kp, DEFAULT_KP)
    kd = parse_list7(args.kd, DEFAULT_KD)
    inertia = parse_list7(args.inertia, DEFAULT_INERTIA)
    viscous = parse_list7(args.viscous, DEFAULT_VISCOUS)
    coulomb = parse_list7(args.coulomb, DEFAULT_COULOMB)
    effort_limit = parse_list7(args.effort_limit, DEFAULT_EFFORT_LIMIT)
    velocity_limit = parse_list7(args.velocity_limit, DEFAULT_VELOCITY_LIMIT)
    substeps = max(1, int(args.substeps))

    with args.input.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty input: {args.input}")
    times = [get_float(r, "t", 0.0) for r in rows]
    target_q_series = [[get_float(r, f"target_q{i}", 0.0) for r in rows] for i in range(1, 8)]
    target_dq_series = [[get_float(r, f"target_dq{i}", 0.0) for r in rows] for i in range(1, 8)]

    first = rows[0]
    q = []
    dq = []
    for i in range(1, 8):
        if args.initial == "real":
            q.append(get_float(first, f"actual_q{i}", get_float(first, f"target_q{i}", 0.0)))
            dq.append(get_float(first, f"actual_dq{i}", 0.0))
        else:
            q.append(get_float(first, f"target_q{i}", 0.0))
            dq.append(get_float(first, f"target_dq{i}", 0.0))

    fields = sample_fieldnames(["target_q", "target_dq", "sim_q", "sim_dq", "sim_tau", "sim_tau_limit"])
    writer, handle = open_csv_writer(args.output, fields)
    try:
        last_t = get_float(first, "t", 0.0)
        for sample_index, row_in in enumerate(rows):
            t = get_float(row_in, "t", last_t)
            dt = max(0.0, t - last_t) if sample_index > 0 else 0.0
            t_cmd = t - max(0.0, args.command_delay_s)
            q_des = [interp_scalar(times, target_q_series[i], t_cmd) for i in range(7)]
            dq_des = [interp_scalar(times, target_dq_series[i], t_cmd) for i in range(7)]
            tau_last = [0.0] * 7
            tau_limit_last = [0.0] * 7
            h = dt / substeps if dt > 0.0 else 0.0
            for _ in range(substeps if dt > 0.0 else 1):
                for j in range(7):
                    tau_pd = kp[j] * (q_des[j] - q[j]) + kd[j] * (dq_des[j] - dq[j])
                    if args.motor_mode == "dc":
                        limit = torque_limit_dc(effort_limit[j], velocity_limit[j], dq[j])
                    else:
                        limit = abs(effort_limit[j])
                    tau = max(-limit, min(limit, tau_pd))
                    tau -= viscous[j] * dq[j]
                    if coulomb[j] > 0.0:
                        tau -= coulomb[j] * math.tanh(dq[j] / 0.02)
                    qdd = tau / max(inertia[j], 1e-9)
                    if h > 0.0:
                        dq[j] += qdd * h
                        q[j] += dq[j] * h
                    tau_last[j] = tau
                    tau_limit_last[j] = limit
            out: dict[str, float | int | str] = {
                "source": f"sim_{args.motor_mode}",
                "run_id": row_in.get("run_id", ""),
                "stamp": row_in.get("stamp", ""),
                "t": t,
                "phase": row_in.get("phase", ""),
                "joint_index": row_in.get("joint_index", ""),
                "joint_name": row_in.get("joint_name", ""),
                "freq_hz": row_in.get("freq_hz", ""),
                "amplitude_rad": row_in.get("amplitude_rad", ""),
                "sample_index": sample_index,
            }
            for i in range(7):
                out[f"target_q{i + 1}"] = q_des[i]
                out[f"target_dq{i + 1}"] = dq_des[i]
                out[f"sim_q{i + 1}"] = q[i]
                out[f"sim_dq{i + 1}"] = dq[i]
                out[f"sim_tau{i + 1}"] = tau_last[i]
                out[f"sim_tau_limit{i + 1}"] = tau_limit_last[i]
            writer.writerow(out)
            last_t = t
    finally:
        handle.close()

    write_json(
        args.output.with_suffix(".params.json"),
        {
            "input": str(args.input),
            "output": str(args.output),
            "motor_mode": args.motor_mode,
            "kp": kp,
            "kd": kd,
            "inertia": inertia,
            "viscous": viscous,
            "coulomb": coulomb,
            "effort_limit": effort_limit,
            "velocity_limit": velocity_limit,
            "command_delay_s": args.command_delay_s,
            "substeps": substeps,
        },
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
