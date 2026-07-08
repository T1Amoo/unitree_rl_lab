# A1/H1 Table-Tennis Sim2sim

This folder is the sim2sim half of the custom A1/H1 ping-pong deployment path.
It uses the same DAMIAO MIT-control shape that `h1_pingpong/src/armcontrol`
uses on hardware:

```cpp
control_mit(motor, kp, kd, q_des, dq_des, tau_ff)
```

The default MuJoCo runner uses a stable position-servo actuator for visual
sim2sim, with the same action target and joint ranges as the training config.
Its velocity cap is calibrated from an Isaac play trace under the real 28/8 Nm
effort limits; the training velocity limits are hard caps, not the speed the
Isaac implicit actuator actually reaches. The servo also uses a first-order
tracking time constant (`--servo-tau`, default `0.25 s`) so it does not move at
the cap whenever the target changes. The equivalent MIT torque is still
estimated and logged:

```text
tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
```

Use `--dynamic-pd` to test the explicit MIT-PD torque path. The runner now
loads a generated MJCF scene (`sim2sim/scene/a1_tt_scene.xml`) instead of using
the raw URDF directly. The MJCF keeps the URDF geometry/inertias, and adds
MuJoCo-only metadata that URDF did not carry: joint armature, damping,
frictionloss, DAMIAO actuator force ranges, motor actuators, sensors, and
table-tennis contact pairs. Those values are still initial estimates until
hardware logs are used to calibrate them.

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

- Scene MJCF: `sim2sim/scene/a1_tt_scene.xml`
- Source URDF: `/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/legged_lab/assets/a1/X1_URDF_V1_1/urdf/X1_URDF_V1_1.urdf`
- Policy ONNX: `/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_v9/2026-07-07_02-50-16/exported/policy.onnx`
- Control: 50 Hz policy, 500 Hz MuJoCo physics
- Action: `q_des = default_q + clip(raw_action, +/-10) * 0.25`
- Training torque limits: r1-r3 `28 Nm`, r4-r7 `8 Nm`
- MJCF DAMIAO actuator ranges: r1-r4 `28 Nm`, r5-r7 `10 Nm`

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
and `--gate-coast-frames` can add hysteresis for noisy perception; defaults are
`1/1` to match the current deterministic sim/training mask.

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
