#!/usr/bin/python3
"""Replay a fitted closed-loop joint response inside IsaacLab."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--plot", type=Path, required=True)
    ap.add_argument("--joint", type=int, default=1, choices=range(1, 8))
    ap.add_argument("--model", default="second_order")
    ap.add_argument("--input-column", default="cmd_q")
    ap.add_argument("--target-column", default="target_raw_q")
    ap.add_argument("--real-column", default="real_actual_q")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--legged-lab-root", type=Path, default=Path("Pingpong_TTRL/legged_lab"))
    AppLauncher.add_app_launcher_args(ap)
    return ap


args_cli = build_arg_parser().parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app = AppLauncher(args_cli).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402


def get_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def interp(times: list[float], values: list[float], t: float) -> float:
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    import bisect

    hi = bisect.bisect_left(times, t)
    lo = hi - 1
    t0 = times[lo]
    t1 = times[hi]
    if t1 <= t0:
        return values[lo]
    alpha = (t - t0) / (t1 - t0)
    return values[lo] * (1.0 - alpha) + values[hi] * alpha


def load_params(path: Path, model: str) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for item in data:
            if item.get("model") == model:
                return item
        raise SystemExit(f"model {model!r} not found in {path}")
    if data.get("model") not in (None, model):
        raise SystemExit(f"params model mismatch: {data.get('model')} != {model}")
    return data


def main() -> int:
    with args_cli.merged.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty merged CSV: {args_cli.merged}")
    params = load_params(args_cli.params, args_cli.model)

    rel_t = [get_float(r, "relative_s") for r in rows]
    target = [get_float(r, args_cli.target_column) for r in rows]
    cmd = [get_float(r, args_cli.input_column) for r in rows]
    real = [get_float(r, args_cli.real_column) for r in rows]

    legged_lab_root = args_cli.legged_lab_root.resolve()
    if not legged_lab_root.exists():
        raise SystemExit(f"missing legged_lab root: {legged_lab_root}")
    sys.path.insert(0, str(legged_lab_root))
    from assets.a1.a1 import A1_TT_CFG, A1_RIGHT_ARM_JOINTS  # noqa: PLC0415

    cfg = A1_TT_CFG.replace(prim_path="/World/Robot")
    cfg.spawn.articulation_props.fix_root_link = True
    init_pos = dict(cfg.init_state.joint_pos)
    for i in range(1, 8):
        init_pos[f"r{i}"] = get_float(rows[0], args_cli.input_column, init_pos.get(f"r{i}", 0.0)) if i == args_cli.joint else init_pos.get(f"r{i}", 0.0)
    cfg.init_state.joint_pos = init_pos

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    robot = Articulation(cfg)
    sim.reset()
    right_ids = [robot.joint_names.index(name) for name in A1_RIGHT_ARM_JOINTS]
    joint_local = args_cli.joint - 1

    fn_hz = float(params["fn_hz"])
    zeta = float(params["zeta"])
    delay_s = float(params["delay_s"])
    tau_zero_s = float(params.get("tau_zero_s", 0.0))
    linear_gain = float(params["linear_gain"])
    intercept = float(params["intercept"])
    u_mean = float(params["u_mean"])
    wn = 2.0 * math.pi * fn_hz

    x = interp(rel_t, cmd, rel_t[0] - delay_s) - u_mean
    v = 0.0
    sim_t = rel_t[0]

    def step_model(t_now: float, h: float) -> tuple[float, float, float]:
        nonlocal x, v
        u_delayed = interp(rel_t, cmd, t_now - delay_s) - u_mean
        acc = wn * wn * (u_delayed - x) - 2.0 * zeta * wn * v
        v += acc * h
        x += v * h
        q_pred = intercept + linear_gain * (x + tau_zero_s * v)
        return q_pred, x, v

    output_rows: list[dict[str, float | int | str]] = []
    default_right_q = [float(init_pos.get(f"r{i}", 0.0)) for i in range(1, 8)]
    q_state = list(default_right_q)
    q_state[joint_local] = intercept + linear_gain * x
    dq_state = [0.0] * 7

    for sample_index, row in enumerate(rows):
        target_t = rel_t[sample_index]
        while sim_t + 0.5 * args_cli.dt < target_t:
            q_pred, _, v_state = step_model(sim_t, args_cli.dt)
            q_state[joint_local] = q_pred
            dq_state[joint_local] = linear_gain * v_state
            for j in range(7):
                if j != joint_local:
                    q_state[j] = default_right_q[j]
                    dq_state[j] = 0.0
            robot.write_joint_state_to_sim(
                torch.tensor([q_state], dtype=torch.float32, device=robot.device),
                torch.tensor([dq_state], dtype=torch.float32, device=robot.device),
                joint_ids=right_ids,
            )
            sim.step(render=False)
            robot.update(args_cli.dt)
            sim_t += args_cli.dt

        q_pred, x_state, v_state = step_model(target_t, max(0.0, target_t - sim_t))
        sim_t = target_t
        q_state[joint_local] = q_pred
        dq_state[joint_local] = linear_gain * v_state
        robot.write_joint_state_to_sim(
            torch.tensor([q_state], dtype=torch.float32, device=robot.device),
            torch.tensor([dq_state], dtype=torch.float32, device=robot.device),
            joint_ids=right_ids,
        )
        robot.update(0.0)
        q_out = robot.data.joint_pos[0, right_ids[joint_local]].detach().cpu().item()
        output_rows.append(
            {
                "relative_s": target_t,
                "target_q": target[sample_index],
                "cmd_q": cmd[sample_index],
                "real_q": real[sample_index],
                "isaac_fitted_q": q_out,
                "model_x": x_state,
                "model_v": v_state,
                "sample_index": sample_index,
            }
        )

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0].keys())
    with args_cli.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    import numpy as np
    import matplotlib.pyplot as plt

    t_np = np.array([float(r["relative_s"]) for r in output_rows])
    target_np = np.array([float(r["target_q"]) for r in output_rows])
    cmd_np = np.array([float(r["cmd_q"]) for r in output_rows])
    real_np = np.array([float(r["real_q"]) for r in output_rows])
    fit_np = np.array([float(r["isaac_fitted_q"]) for r in output_rows])
    rmse = float(np.sqrt(np.mean((fit_np - real_np) ** 2)))
    maxabs = float(np.max(np.abs(fit_np - real_np)))

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    axes[0].plot(t_np, target_np, color="black", linewidth=1.0, label="target_raw_q")
    axes[0].plot(t_np, cmd_np, color="tab:gray", linewidth=1.0, label="cmd_q")
    axes[0].plot(t_np, real_np, color="tab:green", linewidth=1.2, label="real_q")
    axes[0].plot(t_np, fit_np, color="tab:orange", linewidth=1.0, label="isaac_fitted_q")
    axes[0].set_title(f"IsaacLab fitted response replay, RMSE={rmse:.5f} rad, maxabs={maxabs:.5f} rad")
    axes[0].set_ylabel("j1 q (rad)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=4)

    high = t_np >= max(t_np[0], t_np[-1] - 10.0)
    axes[1].plot(t_np[high], target_np[high], color="black", linewidth=1.0, label="target_raw_q")
    axes[1].plot(t_np[high], cmd_np[high], color="tab:gray", linewidth=1.0, label="cmd_q")
    axes[1].plot(t_np[high], real_np[high], color="tab:green", linewidth=1.2, label="real_q")
    axes[1].plot(t_np[high], fit_np[high], color="tab:orange", linewidth=1.0, label="isaac_fitted_q")
    axes[1].set_title("last 10 seconds")
    axes[1].set_ylabel("j1 q (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=4)

    axes[2].plot(t_np, fit_np - real_np, color="tab:red", linewidth=1.0, label="fit - real")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("relative time (s)")
    axes[2].set_ylabel("residual (rad)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    args_cli.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args_cli.plot, dpi=160)

    print(f"wrote {args_cli.output}", flush=True)
    print(f"wrote {args_cli.plot}", flush=True)
    print(f"rmse={rmse:.8f} maxabs={maxabs:.8f}", flush=True)
    if os.environ.get("A1_JOINT_ID_SKIP_APP_CLOSE") == "1":
        os._exit(0)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
