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

Run a deterministic headless check:

```bash
deploy/robots/a1_h1/a1_tt_backhand/run_sim2sim.sh \
  --headless-steps 600 --seed 42
```

The original runtime files and the matching joint-response fitting utilities
are kept under `sim2sim/` and `tools/joint_id/`.
