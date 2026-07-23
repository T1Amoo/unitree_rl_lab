# A1 Joint Identification Tools

These scripts run on the local workstation. The default workflow lets
`a1_tt_fsm_supervisor` command the robot-side `inference_arm_control_node`; the
recorder only subscribes and writes data.

## Directory Layout

Use this layout for all real/sim/plot artifacts:

```text
系统辨识/
  joint1/
    20260713/
      kp120_kd3.5/
        real/
        sim/
        merged/
        summary/
        plots/
        logs/
  joint2/
  ...
  joint7/
```

The experiment folder name records the tested joint's controller gains. Full
7D kp/kd metadata is also stored in each real-run manifest when `--real-kp` and
`--real-kd` are provided.

## Real Robot Recording Through FSM

Start the FSM with `test_enabled:=true`. In this mode the normal R2/play button
enters TEST instead of TABLE_TENNIS. The TEST command is q_des-only: `/model_action`
contains exactly 7 positions.

```bash
source /opt/ros/humble/setup.bash
source unitree_rl_lab/deploy/robots/a1_h1/install/setup.bash

/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/record_fsm_test.py \
  --joint 1 \
  --freq 0.5 \
  --amplitude 0.12 \
  --cycles 8 \
  --output 系统辨识/joint1/20260713/kp120_kd3.5/real/j1_f0.5hz.csv \
  --param-tag kp120_kd3.5
```

Start the recorder first, then press FixStand, then press R2/test. The recorder
starts writing when `/a1_tt/fsm_state` becomes `state=TEST` and stops when TEST
finishes. The CSV contains target position, actual joint position/velocity, and
reported effort for all seven joints. `target_dq*` is intentionally zero because
this workflow sends only q_des.

Use `--param-tag`, `--real-kp`, and `--real-kd` to record which robot-side
controller gains were active. These flags are metadata only; changing real
`kps/kds` still requires restarting `inference_arm_control_node` with new
parameters.

`record_real_sine.py` is still available for direct publishing tests, but it is
not the default path for the FSM-driven real-robot experiment.

## Sim Replay

Replay the exact target through the local parameterized DC-motor model:

```bash
/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/simulate_motor_response.py \
  --input 系统辨识/joint1/20260713/kp120_kd3.5/real/j1_f0.5hz.csv \
  --output 系统辨识/joint1/20260713/kp120_kd3.5/sim/j1_f0.5hz_sim.csv \
  --motor-mode dc
```

Useful knobs:

- `--kp`, `--kd`: seven-value MIT PD gains.
- `--inertia`: seven-value effective joint inertia.
- `--viscous`, `--coulomb`: joint damping/friction.
- `--effort-limit`: zero-speed torque limit.
- `--velocity-limit`: no-load velocity used by the torque-speed envelope.

## Merge And Metrics

```bash
/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/merge_joint_id_tables.py \
  --real 系统辨识/joint1/20260713/kp120_kd3.5/real/j1_f0.5hz.csv \
  --sim 系统辨识/joint1/20260713/kp120_kd3.5/sim/j1_f0.5hz_sim.csv \
  --output 系统辨识/joint1/20260713/kp120_kd3.5/merged/j1_f0.5hz_merged.csv \
  --summary 系统辨识/joint1/20260713/kp120_kd3.5/summary/j1_f0.5hz_summary.csv
```

The merged CSV is the requested table:

```text
t, target_q, real_q, sim_q, real_minus_target, sim_minus_target, real_minus_sim
```

The summary CSV reports amplitude gain, phase lag, and RMSE.

## Matrix

Generate the first-pass command list for seven joints and three frequencies:

```bash
/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/make_test_matrix.py
```

For the current joint-1 kp=120 test:

```bash
/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/make_test_matrix.py \
  --joints 1 \
  --kp 120 \
  --kd 3.5 \
  --amplitude 0.12 \
  --mkdirs
```

After merging the three frequency CSVs, generate one vertical plot:

```bash
/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/plot_joint_id_triplet.py \
  --joint 1 \
  --inputs \
    系统辨识/joint1/20260713/kp120_kd3.5/merged/j1_f0.5hz_merged.csv \
    系统辨识/joint1/20260713/kp120_kd3.5/merged/j1_f1hz_merged.csv \
    系统辨识/joint1/20260713/kp120_kd3.5/merged/j1_f1.5hz_merged.csv \
  --labels "0.5 Hz" "1.0 Hz" "1.5 Hz" \
  --title "joint1 kp120 kd3.5" \
  --output 系统辨识/joint1/20260713/kp120_kd3.5/plots/j1_kp120_kd3.5_target_real_sim.png
```
