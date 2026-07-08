# A1/H1 乒乓 Sim2Real C++ Bridge — 完整命令行手册

覆盖 **策略导出 / C++ ONNX bridge / ROS dry-run / sim2real 真机接入** 全流程命令。
最后更新 2026-07-08。

> 路径约定
> - 训练仓库：`/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/`
> - 部署仓库：`/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1/`
> - 底层电机 SDK 仓库：`/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/h1_pingpong/`
> - 底层控制包：`h1_pingpong/src/armcontrol/`，只负责 DAMIAO/ROS 硬件接口。
> - ONNX Runtime：`/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/`
> - ROS：系统 `/opt/ros/humble`。部署编译和运行时不要激活 conda。

---

## 0. 当前结构速查

这版仿照 G1 的部署方式，不用 Python `onnxruntime`：

```text
policy.onnx / predictor.onnx
  -> sim2real_bridge_cpp::a1_policy_bridge_cpp  (C++ Ort::Session)
  -> /model_action  (q_des[7] 或 q_des[7]+dq_des[7])
  -> armcontrol::inference_arm_control_node
  -> DAMIAO control_mit(...)
```

关键 topic：

| Topic | 类型 | 作用 |
|---|---|---|
| `/right_joint_states` | `sensor_msgs/msg/JointState` | `inference_arm_control_node` 发布的右臂实际关节 |
| `/ball/state` | `std_msgs/msg/Float64MultiArray` | 球状态，`[x,y,z,vx,vy,vz]`，训练桌面坐标系 |
| `/model_action` | `std_msgs/msg/Float64MultiArray` | bridge 发给底层控制器的关节目标 |
| `/model_control/enable` | `std_msgs/msg/Bool` | 底层 MIT servo 开关 |
| `/sim2real/q_des` | `std_msgs/msg/Float64MultiArray` | bridge 诊断输出的目标关节 |
| `/sim2real/gate` | `std_msgs/msg/String` | 球有效门控/预测点诊断 |

---

## 1. 策略版本速查

当前默认策略写在 launch 和 C++ 节点里：

```text
Pingpong_TTRL/logs/a1_tt_v11/2026-07-07_10-48-31/exported/policy.onnx
```

本地可用 A1 policy：

```bash
find /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs \
  -path '*a1_tt_v*/exported/policy.onnx' | sort
```

本地可用 predictor：

```bash
find /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs \
  -path '*a1_tt_v*/exported/predictor.onnx' | sort
```

目前已知：

| 版本 | policy | predictor | 备注 |
|---|---|---|---|
| `a1_tt_v11/2026-07-07_10-48-31` | 有 | 默认无 | bridge 会用 analytic prediction fallback |
| `a1_tt_v10/2026-07-07_07-51-54` | 有 | 有 | 可显式传 `predictor_path:=.../predictor.onnx` |
| `a1_tt_v9/2026-07-07_02-50-16` | 有 | 有 | 可作对照 |

---

## 2. ONNX 导出

### 2a. policy.onnx

`eval/play` 导出后应落在：

```text
<run>/exported/policy.onnx
```

部署前先确认模型输入维度是 actor 需要的 195：

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/h1_pingpong
source /opt/ros/humble/setup.bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
./build/a1_policy_bridge_cpp \
  --ros-args \
  -p publish_actions:=false \
  -p policy_path:=/path/to/policy.onnx
```

如果输入维度不对，节点会直接报：

```text
policy input must be 195
```

### 2b. predictor.onnx

predictor 不是必须项。没有 predictor 时，bridge 会按当前球位/速度解析预测击球平面。

显式启用 predictor：

```bash
ros2 launch sim2real_bridge_cpp a1_policy_bridge_cpp.launch.py \
  predictor_path:=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_v10/2026-07-07_07-51-54/exported/predictor.onnx
```

---

## 3. 编译 C++ Bridge

### 3a. 编译底层 armcontrol

`h1_pingpong` 恢复为底层 SDK 仓库。只在这里编译 `armcontrol`，不再放 sim2sim/sim2real 逻辑。

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/h1_pingpong
conda deactivate
source /opt/ros/humble/setup.bash

colcon build --packages-select armcontrol \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

source install/setup.bash
```

### 3b. 编译 A1/H1 bridge

和 G1 deploy 一样，部署编译时先退出 conda，避免 conda 的 Python/C++ 库污染系统 ROS。

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
bash sim2sim/build_deploy.sh
```

如果 ONNX Runtime 不在默认路径：

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
source /opt/ros/humble/setup.bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/opt/ros/humble \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.22.0
cmake --build build -j4
```

### 3c. 手动 CMake 编译

`sim2sim/build_deploy.sh` 本质就是以下命令：

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
source /opt/ros/humble/setup.bash

rm -rf build
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/opt/ros/humble \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

cmake --build build -j4
```

### 3d. 编译常见错误

如果看到：

```text
ModuleNotFoundError: No module named 'catkin_pkg'
```

通常是 CMake/ament 找到了 conda 的 Python。重跑：

```bash
conda deactivate
source /opt/ros/humble/setup.bash
cmake ... -DPython3_EXECUTABLE=/usr/bin/python3
```

---

## 4. 离线 smoke test

这个测试不连电机，不发 `/model_action`，只验证：

1. C++ ONNX Runtime 能加载 policy。
2. bridge 能订阅假关节和假球。
3. 门控变 valid。
4. 策略能吐 `q_des`。

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/h1_pingpong
source /opt/ros/humble/setup.bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1

./build/a1_policy_bridge_cpp \
  --ros-args -p publish_actions:=false -p diag_every:=20
```

另开终端发假关节：

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --rate 20 /right_joint_states sensor_msgs/msg/JointState \
  "{name: ['joint1-a1_r','joint2-a1_r','joint3-a1_r','joint4-a1_r','joint5-a1_r','joint6-a1_r','joint7-a1_r'], position: [0.569,-0.692,0.717,1.13,-1.24,0.0314,0.772], velocity: [0.0,0.0,0.0,0.0,0.0,0.0,0.0]}"
```

再开终端发假球：

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --rate 20 /ball/state std_msgs/msg/Float64MultiArray \
  "{data: [1.0, 0.1, 1.05, -3.0, 0.0, 1.8]}"
```

期望 bridge 日志类似：

```text
gate=1/valid
q_des=[...]
pred=[-1.550, ...]
```

---

## 5. ROS dry-run

### 5a. 只看策略，不发动作

这一步建议先在无球/假球状态下做，确认所有 topic 和坐标都对。

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/h1_pingpong
conda deactivate
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

另开终端起 bridge dry-run：

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
PUBLISH_ACTIONS=false bash sim2sim/run_deploy.sh
```

看诊断：

```bash
ros2 topic echo /right_joint_states
ros2 topic echo /sim2real/q_des
ros2 topic echo /sim2real/gate
```

### 5b. 发动作但不 enable 底层 servo

这一步让 `/model_action` 出来，但 `inference_arm_control_node` 仍不进入 MIT servo。

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
PUBLISH_ACTIONS=true bash sim2sim/run_deploy.sh
```

检查：

```bash
ros2 topic echo /model_action
ros2 topic echo /sim2real/gate
```

---

## 6. sim2real 真机流程

### 6a. 前置条件

1. CAN/串口设备存在，默认右臂设备是 `/dev/ttyCANR`。
2. `armcontrol/inference_arm_control_node` 能发布 `/right_joint_states`。
3. 视觉/动捕节点能发布 `/ball/state`，数据必须已经转到训练桌面坐标系。
4. bridge dry-run 的 `/sim2real/q_des` 不发散。
5. 急停/断电/软停流程明确。

### 6b. 启动 ROS 和底层节点

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/h1_pingpong
conda deactivate
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

另开终端起 bridge：

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
PUBLISH_ACTIONS=true bash sim2sim/run_deploy.sh
```

### 6c. 进入控制

只在以下项都确认后执行：

```bash
ros2 topic echo --once /right_joint_states
ros2 topic echo --once /ball/state
ros2 topic echo --once /model_action
ros2 topic echo --once /sim2real/gate
```

打开底层 servo：

```bash
ros2 topic pub --once /model_control/enable std_msgs/msg/Bool "{data: true}"
```

软停：

```bash
ros2 topic pub --once /model_control/enable std_msgs/msg/Bool "{data: false}"
```

### 6d. 停止进程

```bash
ros2 topic pub --once /model_control/enable std_msgs/msg/Bool "{data: false}"
pkill -f a1_policy_bridge_cpp
pkill -f inference_arm_control_node
```

---

## 7. 视觉/球坐标约定

`/ball/state` 必须是训练桌面坐标系：

```text
data = [x, y, z, vx, vy, vz]
```

约定和 sim2sim/policy 一致：

- `hit_plane_x = -1.55`
- 桌面/机器人位置用训练中的 table frame
- 无效球时 actor 看到固定 sentinel，不应喂随机假球

如果视觉只能给 `[x,y,z]`，bridge 会差分速度，但实机建议视觉侧直接给稳定滤波后的 `[vx,vy,vz]`。

---

## 8. 参数速查

bridge 参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `policy_path` | v11 policy | `policy.onnx` 路径 |
| `predictor_path` | 空 | 空则找 policy 同目录 `predictor.onnx`，不存在则禁用 |
| `use_predictor` | `true` | 是否尝试加载 predictor |
| `control_hz` | `50.0` | bridge 推理频率 |
| `publish_actions` | `true` | 是否发布 `/model_action` |
| `publish_position_velocity` | `false` | `false` 发 7 维 q，`true` 发 14 维 q+dq |
| `enable_on_start` | `false` | bridge 启动时是否发 `/model_control/enable=true` |
| `hold_when_ball_stale` | `true` | 球超时时保持当前关节并 reset history |
| `joint_timeout_s` | `0.25` | 关节反馈超时 |
| `ball_timeout_s` | `0.20` | 球状态超时 |

底层 `inference_arm_control_node` 关键参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `controlled_arms` | `right` | 当前只控右臂 |
| `control_rate_hz` | `100.0` | 底层 MIT 控制循环 |
| `action_topic` | `/model_action` | bridge 动作输入 |
| `enable_topic` | `/model_control/enable` | servo enable |
| `right_arm_device` | `/dev/ttyCANR` | 右臂设备 |
| `servo_enabled_on_start` | `false` | 启动时不进 servo |
| `enable_motors_on_start` | launch 中设 `false` | 启动时不主动 enable 电机 |
| `max_vel` | `[8,8,8,30,30,30,30]` | 底层速度限幅 |
| `max_delta_per_cycle` | `[0.05,0.05,0.05,0.05,0.12,0.12,0.12]` | 每控制周期目标变化限幅 |

---

## 9. 常见坑速查

- **不要在 conda 环境里编译部署包**。G1 deploy 也是这么要求的。必要时显式加 `-DPython3_EXECUTABLE=/usr/bin/python3`。
- **C++ bridge 不需要 Python onnxruntime**。它链接 vendored `libonnxruntime.so.1.22.0`。
- **`publish_actions:=false` 是 dry-run**，不会发 `/model_action`。
- **`enable_on_start:=false` 只是不打开 servo**，如果你手动发了 `/model_control/enable=true`，底层就会开始执行最新 `/model_action`。
- **`predictor=(disabled)` 不一定是错误**。说明没有找到 predictor，bridge 会用解析预测 fallback。
- **球坐标系错会直接让策略乱打**。先看 `/sim2real/gate` 的 `reason` 和 `pred`，再考虑 enable。
- **关节名必须匹配**：`joint1-a1_r ... joint7-a1_r`。bridge 也兼容 `joint1-r` 和 `r1` 这种别名，但实机建议统一用 A1 名称。
- **一次只跑一个 bridge / 一个底层控制节点**。重复节点会抢 topic 和硬件设备。
- **先软停再杀进程**：先发 `/model_control/enable=false`，再 `pkill`。
