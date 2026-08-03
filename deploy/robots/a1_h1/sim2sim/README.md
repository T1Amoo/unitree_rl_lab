# A1/H1 Table-Tennis Sim2sim（默认反手 v1）

当前无参数默认值统一到反手 v1 最终策略：base/table
`[-1.8,0,0.0282]`、ready
`[1.450,-0.762,-2.050,1.445,0.206,-0.827,1.043]`、hit plane
`x=-1.243`。动作默认走逐关节一阶 `tau_s`，随后走最新自然挥拍辨识的
二阶响应并以 `direct_response` 写入。旧正手模型只作为显式历史版本保留。

This folder is the sim2sim half of the custom A1/H1 ping-pong deployment path.
It uses the same DAMIAO MIT-control shape that `h1_pingpong/src/armcontrol`
uses on hardware:

```cpp
control_mit(motor, kp, kd, q_des, dq_des, tau_ff)
```

The default MuJoCo runner applies the training/deployment per-joint action-target
low-pass, the latest fitted second-order q_des-to-q response, and then writes
that response through `direct_response`. `isaac_approx`, explicit MIT-PD and
other actuator modes remain diagnostic-only options. The equivalent MIT torque
is still estimated and logged:

```text
tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
```

Use `--dynamic-pd` to test the explicit MIT-PD torque path. The runner generates
a temporary MJCF from the current URDF on every start, so fixed-joint calibration
changes cannot be hidden by a stale checked-in XML. The builder keeps the URDF
geometry/inertias and adds MuJoCo-only metadata that URDF did not carry: joint
armature, damping, frictionloss, DAMIAO actuator force ranges, motor actuators,
sensors, and table-tennis contact pairs. Those values are still initial estimates
until hardware logs are used to calibrate them.

## Run

Headless smoke test:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda run -n g1tt_sim2sim python sim2sim/build_a1_mjcf.py
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py --headless-steps 200
```

Viewer:

```bash
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py
```

The default is the fixed generalized range: all samples clear the net and
first-bounce on the robot side, with wider depth/y/height/speed coverage than
the frozen v1 training set:

```bash
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py
```

Use `--serve-profile trained_v1` only for historical narrow-range replay.

Explicit torque diagnostic:

```bash
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py --dynamic-pd --diag-every 20
```

Fast servo diagnostic using the training hard velocity limits:

```bash
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py --fast-servo --diag-every 20
```

Zero-action scene test:

```bash
conda run -n g1tt_sim2sim python sim2sim/run_a1_tt_sim2sim.py --headless-steps 50 --no-policy
```

## Defaults

- Scene MJCF: generated from the current URDF at `/tmp/a1_h1_tt_scene.xml` on every run
- Source URDF: `/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/legged_lab/assets/a1/X1_URDF_V1_3/urdf/X1_URDF_V1_3.urdf`
- Policy ONNX: `/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_backhand_real_v1_y055h105/2026-08-01_11-05-58_scratch_backhand_camera_age35_tau_delay_dr_servey055h105_10k10k10k/exported/policy.onnx`
- Control: 50 Hz policy, 500 Hz MuJoCo physics
- Action: `q_des = default_q + clip(raw_action, +/-10) * 0.25`
- Training torque limits: r1-r3 `28 Nm`, r4-r7 `8 Nm`
- MJCF DAMIAO actuator ranges: r1-r3 `28 Nm`, r4-r7 `8 Nm`

If `predictor.onnx` exists next to `policy.onnx`, the runner uses it for the
5-frame ball prediction path and falls back to the analytic hit-plane predictor
when the predictor output is outside the training gate.

## Ball Gate And Hit-Point Visuals

The runner uses a hard ball-validity gate before building the policy
observation. When the gate is invalid, the actor's ball slot and hit prediction
are both replaced with the same fixed home sentinel used by the A1 training
environment:

```text
(hit_plane_x, home_y + paddle_y_offset, hit_body_height + 0.2)
```

The gate rejects balls that are behind the hit plane, moving away, below table
height, double-bounced, already hit, outside the play volume, too fast, or
predicted to miss the robot half on first table contact. `--gate-confirm-frames`
and `--gate-coast-frames` can add hysteresis for noisy perception. MuJoCo uses
the deterministic gate; the real camera launch uses `confirm=1/coast=5`.

In viewer mode the predicted hit point is drawn as a marker. Green means the
gate is engaged and the policy sees the live ball; gray means invalid and the
marker is the home sentinel. Disable it with `--no-hit-viz`.

## Sim2real boundary

The reusable boundary is:

```text
policy obs -> policy.onnx -> raw action -> q_des/kp/kd/dq_des/tau_ff
```

Sim2sim consumes this command by writing MuJoCo joint torques. Sim2real should
consume the same command by sending DAMIAO MIT commands:

```cpp
control_mit(motor, kp, kd, q_des, dq_des, tau_ff)
```
