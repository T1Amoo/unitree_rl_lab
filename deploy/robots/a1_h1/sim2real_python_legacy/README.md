# A1 Sim2real Bridge

Legacy Python prototype. The active deployment path for this robot folder is
the C++ bridge in `../src/a1_policy_bridge_cpp.cpp`, matching the G1 deploy
style and avoiding Python `onnxruntime` as a runtime dependency.

This package connects the IsaacLab/MuJoCo A1 table-tennis policy to the existing
`h1_pingpong` motor SDK path.

The motor boundary is not reimplemented here. The bridge publishes ROS messages
that `armcontrol/inference_arm_control_node` already consumes:

```text
policy.onnx -> q_des[7] -> /model_action -> inference_arm_control_node -> DAMIAO control_mit(...)
```

## ROS Interfaces

Inputs:

- `/right_joint_states` (`sensor_msgs/msg/JointState`): actual right arm state
  from `inference_arm_control_node`.
- `/ball/state` (`std_msgs/msg/Float64MultiArray`): ball state in the training
  table frame. Use `data=[x, y, z, vx, vy, vz]`. If only `[x, y, z]` is sent,
  velocity is finite-differenced.

Outputs:

- `/model_action` (`std_msgs/msg/Float64MultiArray`): 7 joint targets in the
  same order as `joint1-a1_r ... joint7-a1_r`.
- `/model_control/enable` (`std_msgs/msg/Bool`): enable switch for the SDK node.
  The bridge sends `false` on startup unless `enable_on_start:=true`.
- `/sim2real/raw_action`, `/sim2real/q_des`, `/sim2real/gate`: diagnostics.

## Run

Prerequisites:

- Use the ROS Python, not the current conda Python. On this machine ROS Humble
  is Python 3.10; running ROS nodes from conda Python 3.13 will fail to import
  `rclpy`.
- `colcon` must be installed in the ROS environment.
- `onnxruntime` must be installed for `/usr/bin/python3`, because policy
  inference runs inside the ROS2 Python node.

Build:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/h1_pingpong
conda deactivate
source /opt/ros/humble/setup.bash
python3 -c "import onnxruntime"
colcon build --packages-select armcontrol sim2real_bridge
source install/setup.bash
```

Dry-run the bridge first. This does not publish `/model_action`; it only checks
joint feedback, ball gate, policy output, and diagnostics:

```bash
ros2 launch sim2real_bridge a1_policy_bridge.launch.py \
  start_arm_control:=true \
  publish_actions:=false \
  policy_path:=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_v13/2026-07-08_12-40-15/exported/policy.onnx
```

Watch diagnostics:

```bash
ros2 topic echo /sim2real/q_des
ros2 topic echo /sim2real/gate
```

When diagnostics look sane, restart with action publishing enabled but keep arm
control disabled:

```bash
ros2 launch sim2real_bridge a1_policy_bridge.launch.py \
  start_arm_control:=true \
  publish_actions:=true \
  enable_on_start:=false
```

Enable only after joint feedback, ball messages, and `/model_action` look sane:

```bash
ros2 topic pub --once /model_control/enable std_msgs/msg/Bool "{data: true}"
```

Send a test ball state:

```bash
ros2 topic pub --rate 50 /ball/state std_msgs/msg/Float64MultiArray \
  "{data: [1.0, 0.12, 1.05, -3.0, 0.0, 1.8]}"
```

## Notes

- The bridge runs the same observation contract as `sim2sim/policy_io.py`:
  5-frame history, 50 Hz control, fixed invalid-ball sentinel, fixed
  `hit_plane_x=-1.60`.
- Ball coordinates must already be transformed into the training table frame:
  robot/table origin conventions are the same as sim2sim.
- If ball state is stale, the bridge holds current joint positions by default
  instead of moving on a stale observation.
- The launch file defaults `enable_motors_on_start:=false`; the motor SDK node
  should not enter control until `/model_control/enable` is set true.
- The lower node still enforces joint limits, interpolation, timeouts, and MIT
  command logging. Tune those limits in `inference_arm_control_node` before
  raising deployment speed.
