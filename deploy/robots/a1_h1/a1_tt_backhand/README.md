# A1 TT Backhand Base

This directory isolates the mentor-provided `model_9700` backhand deployment
stack from the existing forehand sim2sim path.

The default launcher selects the exact required profile and control path:

- profile: `v1_3_backhand_low_arm_hitplane020`
- actuator: `damiao_mit`
- fitted real-response model: enabled
- serve pause: 100 control steps (about 2 seconds)
- policy: `Pingpong_TTRL/pretrained/a1_tt_backhand/base_9700_hitplane020/policy/policy.onnx`

The frozen coordinate, joint, hit-plane, timing, and direct-SDK actuator contract is in
`Pingpong_TTRL/pretrained/a1_tt_backhand/base_9700_hitplane020/DEPLOY_CONTRACT.md`.
It also records the checks required before this baseline is connected to the
real arm with zero desired velocity and zero feedforward torque.

Run the viewer:

```bash
deploy/robots/a1_h1/a1_tt_backhand/run_sim2sim.sh
```

After building `sim2real_bridge_cpp`, run the real camera pipeline with:

```bash
ros2 launch sim2real_bridge_cpp a1_tt_backhand_9700_camera.launch.py
```

The real launch starts in `PASSIVE`, consumes `/pingpong_location`, uses the
base-9700 table frame without the legacy `y+0.76` offset, and uses the exact
9700 training action-target limiter: at 50 Hz, `max_delta_per_tick` is
`[0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.10]`. The first-order `servo_tau_s`
branch is explicitly disabled for this frozen policy.

Run a deterministic headless check:

```bash
deploy/robots/a1_h1/a1_tt_backhand/run_sim2sim.sh \
  --headless-steps 600 --seed 42
```

The original runtime files and the matching joint-response fitting utilities
are kept under `sim2sim/` and `tools/joint_id/`.

For a real-versus-MuJoCo timing capture, start the wide trace recorder before
entering `FIXSTAND`:

```bash
python3 deploy/robots/a1_h1/tools/record_sim2real_trace.py \
  --output-dir /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/系统辨识/sim2real/20260801 \
  --label a1_backhand_9700 --rate-hz 100
```

The CSV stores local epoch/monotonic clocks, source and receive timestamps,
camera and filtered ball state, measured `q/dq/effort`, FSM/gate state, raw
policy output, raw and limited targets, observation/frame vectors, and bridge
compute/age timing. Keep the robot-side SDK CSV from the same trial so the two
clock domains can be aligned afterward.
