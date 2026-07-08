# A1 Sim2real Bridge C++

For the full command-line deployment manual, see `MANUAL.md`.

This is the G1-style deployment path for the A1 table-tennis policy. It does
not use Python `onnxruntime`. The node is a C++ ROS2 executable that links the
vendored ONNX Runtime used by `unitree_rl_lab`:

```text
policy.onnx -> C++ OrtRunner -> q_des[7] -> /model_action -> inference_arm_control_node -> DAMIAO control_mit(...)
```

## Interfaces

Inputs:

- `/right_joint_states` (`sensor_msgs/msg/JointState`): actual right arm state
  from `armcontrol/inference_arm_control_node`.
- `/ball/state` (`std_msgs/msg/Float64MultiArray`): `data=[x,y,z,vx,vy,vz]`
  in the training table frame. If only `[x,y,z]` is sent, velocity is finite
  differenced.

Outputs:

- `/model_action` (`std_msgs/msg/Float64MultiArray`): 7 joint targets ordered
  as `joint1-a1_r ... joint7-a1_r`. If `publish_position_velocity:=true`, it
  publishes 14 values: `q_des[7] + dq_des[7]`.
- `/model_control/enable` (`std_msgs/msg/Bool`): bridge sends `false` on
  startup unless `enable_on_start:=true`.
- `/sim2real/raw_action`, `/sim2real/q_des`, `/sim2real/gate`: diagnostics.

## Build

Use system ROS, not an active conda environment. This matches the G1 deploy
scripts and avoids conda libraries shadowing system ROS libraries.

Build the low-level `armcontrol` package in `h1_pingpong`:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/h1_pingpong
conda deactivate
source /opt/ros/humble/setup.bash
colcon build --packages-select armcontrol \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

Build the C++ bridge in `unitree_rl_lab`:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
bash sim2sim/build_deploy.sh
```

The default ONNX Runtime root is:

```text
/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0
```

Override it if needed:

```bash
source /opt/ros/humble/setup.bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/opt/ros/humble \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.22.0
cmake --build build -j4
```

## Run

Start the SDK node from the restored `h1_pingpong` checkout:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/h1_pingpong
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run armcontrol inference_arm_control_node --ros-args \
  -p controlled_arms:=right \
  -p action_topic:=/model_action \
  -p enable_topic:=/model_control/enable \
  -p servo_enabled_on_start:=false \
  -p enable_motors_on_start:=false \
  -p right_arm_device:=/dev/ttyCANR
```

Dry-run the bridge from `unitree_rl_lab`. This does not publish `/model_action`:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
PUBLISH_ACTIONS=false bash sim2sim/run_deploy.sh
```

Watch diagnostics in another terminal:

```bash
ros2 topic echo /sim2real/q_des
ros2 topic echo /sim2real/gate
```

Then enable action publishing while keeping motor control disabled:

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
PUBLISH_ACTIONS=true bash sim2sim/run_deploy.sh
```

Enable motor control only after `/right_joint_states`, `/ball/state`,
`/sim2real/q_des`, and `/model_action` are sane:

```bash
ros2 topic pub --once /model_control/enable std_msgs/msg/Bool "{data: true}"
```

## Notes

- Observation/action semantics match `sim2sim/policy_io.py`: 5-frame history,
  195-dim actor input, fixed invalid-ball sentinel, fixed `hit_plane_x=-1.55`,
  and `q_des = default_q + clip(raw_action, +/-10) * 0.25`.
- If the ball is stale, the bridge holds the current measured joint pose and
  resets policy/predictor history.
- The launch file defaults `enable_motors_on_start:=false`.
