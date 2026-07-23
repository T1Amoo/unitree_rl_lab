#!/usr/bin/python3
"""Estimate theoretical right-arm PD gains from the IsaacLab A1 model.

The calculation is local to one posture.  It reports:

* joint-space mass-matrix diagonal at the posture as the single-joint effective inertia;
* gravity-compensation torque as the static load;
* dynamic gains for a target second-order natural frequency/damping ratio;
* static-load stiffness required to keep gravity sag below several error budgets.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--legged-lab-root", type=Path, default=Path("Pingpong_TTRL/legged_lab"))
    ap.add_argument(
        "--right-arm-q",
        default="",
        help="Optional seven-value right-arm posture. Defaults to A1_TT_CFG init_state r1..r7.",
    )
    ap.add_argument("--target-fn-hz", type=float, default=5.0)
    ap.add_argument("--target-zeta", type=float, default=0.5)
    ap.add_argument(
        "--error-budgets-rad",
        default="0.005 0.01 0.015 0.02 0.03",
        help="Static error budgets used for kp_load = abs(gravity_torque) / error.",
    )
    ap.add_argument(
        "--design-error-rad",
        type=float,
        default=0.015,
        help="Static error budget used for the recommended local kp/kd columns.",
    )
    ap.add_argument(
        "--amplitude-rad",
        type=float,
        default=0.08,
        help="Amplitude used for the conservative no-saturation stiffness bound effort/amplitude.",
    )
    ap.add_argument("--effort-limit", default="28 28 28 8 8 8 8")
    ap.add_argument("--velocity-limit", default="8 8 8 20 20 20 20")
    AppLauncher.add_app_launcher_args(ap)
    return ap


args_cli = build_arg_parser().parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app = AppLauncher(args_cli).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402


def parse_float_list(text: str, expected: int | None = None) -> list[float]:
    values = [float(x) for x in text.strip().strip("[]()").replace(",", " ").split()]
    if expected is not None and len(values) != expected:
        raise ValueError(f"expected {expected} values, got {len(values)}: {text!r}")
    return values


def tensor_row(values: list[float], device: str) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.float32, device=device)


def main() -> int:
    legged_lab_root = args_cli.legged_lab_root.resolve()
    if not legged_lab_root.exists():
        raise SystemExit(f"missing legged_lab root: {legged_lab_root}")
    sys.path.insert(0, str(legged_lab_root))

    from assets.a1.a1 import A1_RIGHT_ARM_JOINTS, A1_TT_CFG  # noqa: PLC0415

    cfg = copy.deepcopy(A1_TT_CFG).replace(prim_path="/World/Robot")
    cfg.spawn.articulation_props.fix_root_link = True

    if args_cli.right_arm_q:
        q_right = parse_float_list(args_cli.right_arm_q, 7)
    else:
        q_right = [float(cfg.init_state.joint_pos[name]) for name in A1_RIGHT_ARM_JOINTS]

    init_pos = dict(cfg.init_state.joint_pos)
    for name, value in zip(A1_RIGHT_ARM_JOINTS, q_right, strict=True):
        init_pos[name] = value
    cfg.init_state.joint_pos = init_pos

    effort_limit = parse_float_list(args_cli.effort_limit, 7)
    velocity_limit = parse_float_list(args_cli.velocity_limit, 7)
    error_budgets = parse_float_list(args_cli.error_budgets_rad)
    omega_n = 2.0 * math.pi * args_cli.target_fn_hz

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    robot = Articulation(cfg)
    sim.reset()

    right_ids = [robot.joint_names.index(name) for name in A1_RIGHT_ARM_JOINTS]
    q = tensor_row(q_right, robot.device)
    dq = torch.zeros_like(q)
    robot.write_joint_state_to_sim(q, dq, joint_ids=right_ids)
    robot.update(0.0)
    sim.step(render=False)
    robot.update(0.005)
    robot.write_joint_state_to_sim(q, dq, joint_ids=right_ids)
    robot.update(0.0)

    mass = robot.root_physx_view.get_generalized_mass_matrices()[0].detach().cpu()
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[0].detach().cpu()
    mass_right = mass[right_ids, :][:, right_ids]
    gravity_right = gravity[right_ids]

    rows = []
    for i, name in enumerate(A1_RIGHT_ARM_JOINTS):
        inertia = float(mass_right[i, i])
        gravity_tau = float(gravity_right[i])
        kp_dynamic = inertia * omega_n * omega_n
        kd_dynamic = 2.0 * args_cli.target_zeta * inertia * omega_n
        kp_load_design = abs(gravity_tau) / max(abs(args_cli.design_error_rad), 1e-9)
        kp_recommended = max(kp_dynamic, kp_load_design)
        kd_recommended = 2.0 * args_cli.target_zeta * math.sqrt(max(kp_recommended * inertia, 0.0))
        kp_no_sat_amp = abs(effort_limit[i]) / max(args_cli.amplitude_rad, 1e-9)
        row: dict[str, float | str | int] = {
            "joint": i + 1,
            "joint_name": name,
            "q_rad": q_right[i],
            "mass_diag_kgm2": inertia,
            "gravity_comp_tau_nm": gravity_tau,
            "abs_gravity_tau_nm": abs(gravity_tau),
            "target_fn_hz": args_cli.target_fn_hz,
            "target_zeta": args_cli.target_zeta,
            "kp_dynamic_nm_per_rad": kp_dynamic,
            "kd_dynamic_nms_per_rad": kd_dynamic,
            "design_error_rad": args_cli.design_error_rad,
            "kp_load_design_nm_per_rad": kp_load_design,
            "kp_recommended_local_nm_per_rad": kp_recommended,
            "kd_recommended_local_nms_per_rad": kd_recommended,
            "effort_limit_nm": effort_limit[i],
            "velocity_limit_rad_s": velocity_limit[i],
            "kp_no_sat_for_amp_nm_per_rad": kp_no_sat_amp,
            "gravity_tau_over_effort": abs(gravity_tau) / max(abs(effort_limit[i]), 1e-9),
        }
        for budget in error_budgets:
            key = f"kp_load_for_err_{budget:g}_rad"
            row[key] = abs(gravity_tau) / max(abs(budget), 1e-9)
        rows.append(row)

    args_cli.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args_cli.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "pose": "A1_TT_CFG init_state / FixStand" if not args_cli.right_arm_q else "custom",
        "right_arm_q": q_right,
        "right_arm_joints": A1_RIGHT_ARM_JOINTS,
        "target_fn_hz": args_cli.target_fn_hz,
        "target_zeta": args_cli.target_zeta,
        "design_error_rad": args_cli.design_error_rad,
        "amplitude_rad": args_cli.amplitude_rad,
        "effort_limit": effort_limit,
        "velocity_limit": velocity_limit,
        "error_budgets_rad": error_budgets,
        "mass_matrix_right_arm": mass_right.tolist(),
        "rows": rows,
    }
    args_cli.output_json.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args_cli.output_csv}")
    print(f"wrote {args_cli.output_json}")

    if os.environ.get("A1_JOINT_ID_SKIP_APP_CLOSE") != "1":
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
