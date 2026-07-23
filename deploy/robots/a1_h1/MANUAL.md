# A1/H1 乒乓 Sim2Real C++ Bridge — 完整命令行手册

覆盖 **策略导出 / C++ ONNX bridge / ROS dry-run / sim2real 真机接入** 全流程命令。
最后更新 2026-07-09。

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
  -> sim2real_bridge_cpp::a1_tt_fsm_supervisor  (Passive/FixStand/Ready/TableTennis)
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
| `/a1_tt/policy_enable` | `std_msgs/msg/Bool` | FSM 放行策略动作，默认 false |
| `/a1_tt/fsm_command` | `std_msgs/msg/String` | 无手柄时手动切状态：`fixstand/table_tennis/passive` |
| `/a1_tt/fsm_state` | `std_msgs/msg/String` | FSM 当前状态和 ready 误差 |
| `/sim2real/q_des` | `std_msgs/msg/Float64MultiArray` | bridge 诊断输出的目标关节 |
| `/sim2real/gate` | `std_msgs/msg/String` | 球有效门控/预测点诊断 |

FSM 状态顺序：

```text
PASSIVE --(27 或 fixstand)--> FIXSTAND --> READY --(28 或 table_tennis)--> TABLE_TENNIS
任意状态 --(21 或 passive)--> PASSIVE
```

`PASSIVE` 会持续发布 `/a1_tt/policy_enable=false` 和 `/model_control/enable=false`。`FIXSTAND/READY` 只发布 default pose，不放行策略。只有 `READY -> TABLE_TENNIS` 后才会放行策略动作。

---

## 1. 策略版本速查

当前默认策略写在 launch 和 C++ 节点里：

```text
Pingpong_TTRL/logs/a1_tt_v13/2026-07-08_12-40-15/exported/policy.onnx
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
| `a1_tt_v13/2026-07-08_12-40-15` | 有 | 有 | 当前默认；v13 `model_29999.pt` 导出 |
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
./install/sim2real_bridge_cpp/lib/sim2real_bridge_cpp/a1_policy_bridge_cpp \
  --ros-args \
  -p publish_actions:=false \
  -p policy_enabled_on_start:=false \
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
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch sim2real_bridge_cpp a1_policy_bridge_cpp.launch.py \
  predictor_path:=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_v13/2026-07-08_12-40-15/exported/predictor.onnx
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
source /opt/ros/humble/setup.bash

colcon build --packages-select sim2real_bridge_cpp --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

如果 ONNX Runtime 不在默认路径：

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
source /opt/ros/humble/setup.bash
colcon build --packages-select sim2real_bridge_cpp --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.22.0
```

### 3c. 旧本地 CMake 编译

`sim2sim/build_deploy.sh` 会生成 `build/a1_policy_bridge_cpp` 等旧本地二进制，只用于开发临时对照；当前 sim2real 启动流程使用 `colcon build` 后的 `install/sim2real_bridge_cpp/lib/sim2real_bridge_cpp/*`。

旧脚本本质是：

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

./install/sim2real_bridge_cpp/lib/sim2real_bridge_cpp/a1_policy_bridge_cpp \
  --ros-args \
  -p publish_actions:=false \
  -p policy_enabled_on_start:=true \
  -p diag_every:=20
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
pred=[-1.600, ...]
```

---

## 5. FSM dry-run

这个测试只测状态机，不需要电机节点。先启动 supervisor：

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
source /opt/ros/humble/setup.bash
./install/sim2real_bridge_cpp/lib/sim2real_bridge_cpp/a1_tt_fsm_supervisor
```

另开终端发假关节：

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --rate 50 /right_joint_states sensor_msgs/msg/JointState \
  "{name: ['joint1-a1_r','joint2-a1_r','joint3-a1_r','joint4-a1_r','joint5-a1_r','joint6-a1_r','joint7-a1_r'], position: [0.569,-0.692,0.717,1.13,-1.24,0.0314,0.772], velocity: [0.0,0.0,0.0,0.0,0.0,0.0,0.0]}"
```

切状态：

```bash
ros2 topic pub --once /a1_tt/fsm_command std_msgs/msg/String "{data: fixstand}"
ros2 topic echo --once /a1_tt/fsm_state
ros2 topic pub --once /a1_tt/fsm_command std_msgs/msg/String "{data: table_tennis}"
ros2 topic echo --once /a1_tt/policy_enable
ros2 topic pub --once /a1_tt/fsm_command std_msgs/msg/String "{data: passive}"
ros2 topic echo --once /model_control/enable
```

手柄映射不使用 G1 的 bit-packed 组合键，而是使用 h1_pingpong 现有 `/joystick_info` 动作码，避免改 SDK：

已实测手柄是 BETOP JZ-V4 XINPUT / Xbox360 映射：

| 实体按键 | Linux button | `/joystick_info` code | FSM 动作 |
|---|---:|---:|---|
| Back / Select | `6` | `21` | 任意状态 -> Passive |
| 左摇杆按下 L3 | `9` | `27` | FixStand -> Ready |
| 右摇杆按下 R3 | `10` | `28` | Ready -> TableTennis |

如果按键无反应，先确认手柄节点是否在发布 `/joystick_info`：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /joystick_info
```

本机直接读手柄可用：

```bash
jstest --event /dev/input/js0
```

## 6. ROS dry-run

### 6a. VRPN 动捕接入

G1 项目里真实动捕默认球刚体 topic 是：

```text
/vrpn_mocap/U_Tracker0/pose
```

A1 策略 bridge 不直接吃 `PoseStamped`，而是吃 `/ball/state`：

```text
std_msgs/msg/Float64MultiArray data=[x,y,z,vx,vy,vz]
```

所以先启动 VRPN，再启动 A1 的转换节点：

```bash
conda deactivate
source /opt/ros/humble/setup.bash
ros2 launch vrpn_mocap client.launch.yaml server:=10.1.1.198 port:=3883
```

另开终端：

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
source /opt/ros/humble/setup.bash
./install/sim2real_bridge_cpp/lib/sim2real_bridge_cpp/a1_vrpn_ball_state_bridge --ros-args \
  -p input_topic:=/vrpn_mocap/U_Tracker0/pose \
  -p output_topic:=/ball/state \
  -p origin_in_training_world:="[0.0, 0.0, 0.76]" \
  -p rotation_wxyz_to_training:="[1.0, 0.0, 0.0, 0.0]"
```

检查：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo --once /vrpn_mocap/U_Tracker0/pose
ros2 topic echo --once /ball/state
ros2 topic hz /ball/state
```

### 6b. 只看策略，不发动作

这一步建议先在无球/假球状态下做，确认所有 topic 和坐标都对。这里不启动 FSM，用 `policy_enabled_on_start:=true` 只放行策略计算；`publish_actions:=false` 保证不发 `/model_action`。

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
  -p right_arm_device:=/dev/ttyACM1
```

另开终端起 bridge dry-run：

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
source /opt/ros/humble/setup.bash

./install/sim2real_bridge_cpp/lib/sim2real_bridge_cpp/a1_policy_bridge_cpp --ros-args \
  -p policy_path:=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_v13/2026-07-08_12-40-15/exported/policy.onnx \
  -p joint_state_topic:=/right_joint_states \
  -p ball_state_topic:=/ball/state \
  -p publish_actions:=false \
  -p policy_enabled_on_start:=true \
  -p enable_on_start:=false \
  -p hold_when_ball_stale:=false \
  -p diag_every:=1
```

看诊断：

```bash
ros2 topic echo /right_joint_states
ros2 topic echo /sim2real/q_des
ros2 topic echo /sim2real/gate
```

### 6c. 发动作但不 enable 底层 servo

这一步让 `/model_action` 出来，但 `inference_arm_control_node` 仍不进入 MIT servo。

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
source /opt/ros/humble/setup.bash

./install/sim2real_bridge_cpp/lib/sim2real_bridge_cpp/a1_policy_bridge_cpp --ros-args \
  -p policy_path:=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_v13/2026-07-08_12-40-15/exported/policy.onnx \
  -p joint_state_topic:=/right_joint_states \
  -p ball_state_topic:=/ball/state \
  -p action_topic:=/model_action \
  -p publish_actions:=true \
  -p policy_enabled_on_start:=true \
  -p enable_on_start:=false \
  -p hold_when_ball_stale:=false \
  -p max_delta_per_tick:="[0.020, 0.024, 0.036, 0.032, 0.080, 0.064, 0.160]" \
  -p diag_every:=1
```

检查：

```bash
ros2 topic echo /model_action
ros2 topic echo /sim2real/gate
```

---

## 7. sim2real 真机流程

### 7a. 前置条件

1. 机器人端底层 armcontrol 可运行，并能发布 `/right_joint_states`。
2. 本机能连接机器人 ROS 网络，能看到 `/right_joint_states`、`/model_action`、`/model_control/enable`。
3. 本机动捕能启动 VRPN，并通过 bridge 发布 `/ball/state`。
4. 手柄插在机器人或本机时，对应机器存在 `/dev/input/js0`，且 joystick 节点能发布 `/joystick_info`。
5. 急停/断电/软停流程明确。

旧的强化学习流程是：

```text
启动 inference_arm_control_node
手动四段 /movej_right_angle 到击球准备位
手动 /model_control/enable=true
启动/播放策略
```

现在 FSM 正常流程是：

```text
启动 inference_arm_control_node
启动 VRPN + /ball/state bridge
启动 joystick_node
启动 a1_tt_fsm_supervisor
启动 a1_policy_bridge_cpp
L3 -> FixStand/Ready
R3 -> TableTennis
Back/Select -> Passive
```

因此正常使用 FSM 时 **不要手动执行四段 `/movej_right_angle`，也不要手动发 `/model_control/enable=true`**。FSM 会统一控制 `/model_control/enable` 和 `/a1_tt/policy_enable`。

### 7b. 启动 Skill：完整重启并准备实机测试

这个 skill 用于“右臂硬件确认后，从零重启 sim2real 全链路并准备按 L3/R3 测试”。它只负责把进程和日志准备好，默认不会进入策略模式。

安全边界：

- 不手动发布 `/model_control/enable=true`。
- 不手动发布 `/a1_tt/policy_enable=true`。
- 不绕过 FSM 直接发策略动作。
- 如果 `/right_joint_states` 是全 0、stale、关节名不对，先排查硬件/底层节点，不要按 R3。

#### 7b-1. 可选：改过 C++ bridge 后重新编译

只在改了 `src/*.cpp` 或 launch 后执行：

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
source /opt/ros/humble/setup.bash

colcon build --packages-select sim2real_bridge_cpp --cmake-args -DCMAKE_BUILD_TYPE=Release
```

确认安装后的 launch 里带有限幅：

```bash
grep -RIn "max_delta_per_tick" \
  install/sim2real_bridge_cpp/share/sim2real_bridge_cpp/launch/a1_policy_bridge_cpp.launch.py \
  src/a1_policy_bridge_cpp.cpp \
  MANUAL.md
```

期望看到：

```text
[0.020, 0.024, 0.036, 0.032, 0.080, 0.064, 0.160]
```

#### 7b-2. 本机：检查并停止旧进程

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy
source /opt/ros/humble/setup.bash

pgrep -af "a1_policy_bridge_cpp|a1_tt_fsm_supervisor|a1_vrpn_ball_state_bridge|ros2 bag record|vrpn_mocap|client_node" || true
```

按 pid 文件软停旧进程，再兜底 kill：

```bash
LOG=/tmp/a1_sim2real_logs
mkdir -p "$LOG"

for f in "$LOG"/*.pid; do
  [ -f "$f" ] || continue
  p=$(cat "$f" 2>/dev/null || true)
  [ -n "$p" ] && kill "$p" 2>/dev/null || true
done

sleep 1

for f in "$LOG"/*.pid; do
  [ -f "$f" ] || continue
  p=$(cat "$f" 2>/dev/null || true)
  [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true
done

pgrep -af "a1_policy_bridge_cpp|a1_tt_fsm_supervisor|a1_vrpn_ball_state_bridge|ros2 bag record|vrpn_mocap|client_node" || true
```

#### 7b-3. 机器人端：安全重启 armcontrol

机器人：

```text
wlab@10.1.1.220
password: wlab
```

先确认网络：

```bash
ping -c 1 -W 1 10.1.1.220
```

进入机器人并启动底层节点：

```bash
ssh wlab@10.1.1.220

source /home/wlab/pingpong/install/setup.bash

if [ -f /tmp/a1_armcontrol.pid ]; then
  old=$(cat /tmp/a1_armcontrol.pid)
  kill "$old" 2>/dev/null || true
  sleep 1
  kill -9 "$old" 2>/dev/null || true
fi

nohup ros2 run armcontrol inference_arm_control_node --ros-args \
  --params-file /home/wlab/pingpong/install/armcontrol/share/armcontrol/config/inference_arm_control_node.yaml \
  -p servo_enabled_on_start:=false \
  -p enable_motors_on_start:=false > /tmp/a1_armcontrol.log 2>&1 &

echo $! > /tmp/a1_armcontrol.pid
sleep 2

echo PID=$(cat /tmp/a1_armcontrol.pid)
pgrep -af inference_arm_control_node || true
tail -n 80 /tmp/a1_armcontrol.log || true
```

期望看到：

```text
inference_arm_control_node ready
action_rate=50.0 Hz
interpolation=linear
```

注意：底层日志可能显示 `Motors enabled`，这是底层节点初始化电机通信的日志；FSM 仍应保持 `/model_control/enable=false`，不要把这个日志当成策略已经放行。

#### 7b-4. 本机：启动 VRPN、球 bridge、FSM、policy bridge、rosbag

下面这段是当前推荐的一次性启动命令。它使用 install 目录里的新二进制，不使用旧 `build/` 路径。

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy
conda deactivate
source /opt/ros/humble/setup.bash

LOG=/tmp/a1_sim2real_logs
ROOT=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
BIN=$ROOT/install/sim2real_bridge_cpp/lib/sim2real_bridge_cpp
POLICY=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_v13/2026-07-08_12-40-15/exported/policy.onnx

mkdir -p "$LOG/bags"

setsid ros2 launch vrpn_mocap client.launch.yaml server:=10.1.1.198 port:=3883 \
  > "$LOG/vrpn_mocap_restart.log" 2>&1 < /dev/null &
echo $! > "$LOG/vrpn_mocap_restart.pid"

sleep 1

setsid "$BIN/a1_vrpn_ball_state_bridge" --ros-args \
  -p input_topic:=/vrpn_mocap/U_Tracker0/pose \
  -p output_topic:=/ball/state \
  -p origin_in_training_world:="[0.0, 0.0, 0.76]" \
  -p rotation_wxyz_to_training:="[1.0, 0.0, 0.0, 0.0]" \
  -p diag_every:=10 > "$LOG/a1_vrpn_ball_state_bridge.log" 2>&1 < /dev/null &
echo $! > "$LOG/a1_vrpn_ball_state_bridge.pid"

setsid "$BIN/a1_tt_fsm_supervisor" --ros-args \
  -p joint_state_topic:=/right_joint_states \
  -p action_topic:=/model_action \
  -p right_movej_topic:=/movej_right_angle \
  -p enable_topic:=/model_control/enable \
  -p policy_enable_topic:=/a1_tt/policy_enable \
  -p joystick_topic:=/joystick_info \
  -p command_topic:=/a1_tt/fsm_command \
  -p state_topic:=/a1_tt/fsm_state \
  -p control_hz:=50.0 \
  -p joint_timeout_s:=2.0 \
  -p joystick_fixstand_code:=27 \
  -p joystick_table_tennis_code:=28 \
  -p joystick_passive_code:=21 \
  -p fixstand_use_movej:=true \
  -p enable_republish_ticks:=0 \
  -p diag_every:=5 > "$LOG/a1_tt_fsm_supervisor.log" 2>&1 < /dev/null &
echo $! > "$LOG/a1_tt_fsm_supervisor.pid"

setsid "$BIN/a1_policy_bridge_cpp" --ros-args \
  -p policy_path:="$POLICY" \
  -p joint_state_topic:=/right_joint_states \
  -p ball_state_topic:=/ball/state \
  -p action_topic:=/model_action \
  -p enable_topic:=/model_control/enable \
  -p policy_enable_topic:=/a1_tt/policy_enable \
  -p control_hz:=50.0 \
  -p joint_timeout_s:=2.0 \
  -p diag_every:=1 \
  -p publish_actions:=true \
  -p publish_position_velocity:=false \
  -p policy_enabled_on_start:=false \
  -p enable_on_start:=false \
  -p hold_when_ball_stale:=false \
  -p max_delta_per_tick:="[0.020, 0.024, 0.036, 0.032, 0.080, 0.064, 0.160]" \
  > "$LOG/a1_policy_bridge_cpp.log" 2>&1 < /dev/null &
echo $! > "$LOG/a1_policy_bridge_cpp.pid"

BAG="$LOG/bags/a1_debug_$(date +%Y%m%d_%H%M%S)"
echo "$BAG" > "$LOG/a1_debug_bag.path"

setsid ros2 bag record -o "$BAG" \
  /right_joint_states /ball/state /vrpn_mocap/U_Tracker0/pose /joystick_info \
  /a1_tt/fsm_state /a1_tt/policy_enable /model_control/enable /model_action \
  /sim2real/gate /sim2real/raw_action /sim2real/q_des /sim2real/obs /sim2real/frame \
  > "$LOG/a1_debug_bag.log" 2>&1 < /dev/null &
echo $! > "$LOG/a1_debug_bag.pid"

sleep 3

cat "$LOG"/*.pid
pgrep -af "a1_policy_bridge_cpp|a1_tt_fsm_supervisor|a1_vrpn_ball_state_bridge|ros2 bag record|vrpn_mocap|client_node"
```

#### 7b-5. 本机：启动后确认

确认 policy bridge 是新二进制/新参数：

```bash
head -n 20 /tmp/a1_sim2real_logs/a1_policy_bridge_cpp.log
```

必须看到：

```text
max_delta_per_tick=[0.020, 0.024, 0.036, 0.032, 0.080, 0.064, 0.160]
policy runtime gate: topic=/a1_tt/policy_enable enabled=false
```

确认 FSM 安全态：

```bash
tail -n 50 /tmp/a1_sim2real_logs/a1_tt_fsm_supervisor.log

source /opt/ros/humble/setup.bash
ros2 topic echo --once /a1_tt/fsm_state
```

期望初始状态：

```text
state=PASSIVE
policy_enable=false
```

确认关键 topic 存在：

```bash
source /opt/ros/humble/setup.bash
ros2 topic list | sort | grep -E 'right_joint_states|ball/state|joystick_info|a1_tt|sim2real|model_control|model_action|vrpn_mocap/U_Tracker0'
```

确认右臂反馈。**如果 position 全 0，先排查右臂硬件/编码器/底层节点，不要进策略：**

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo --once /right_joint_states
```

确认动捕。没有球时 `echo --once` 超时是正常的，但 topic 必须存在；有球时必须有 `/ball/state`：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo --once /vrpn_mocap/U_Tracker0/pose
ros2 topic echo --once /ball/state
```

确认手柄。没有按键时 `echo --once /joystick_info` 可能一直等；按 L3/R3/Back 时应该打印 `27/28/21`：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /joystick_info
```

确认 bag 正在录：

```bash
cat /tmp/a1_sim2real_logs/a1_debug_bag.path
tail -n 30 /tmp/a1_sim2real_logs/a1_debug_bag.log
```

#### 7b-6. 测试时的操作顺序

硬件反馈正常后再开始：

```text
1. 按 L3。
2. 观察 /a1_tt/fsm_state，等 state=READY。
3. 只在 READY 后按 R3。
4. 任何异常按 Back/Select。
```

对应 ROS 命令：

```bash
source /opt/ros/humble/setup.bash

ros2 topic echo /a1_tt/fsm_state
ros2 topic echo /sim2real/q_des
ros2 topic echo /sim2real/gate
tail -f /tmp/a1_sim2real_logs/a1_policy_bridge_cpp.log
tail -f /tmp/a1_sim2real_logs/a1_tt_fsm_supervisor.log
```

无手柄时的等价命令：

```bash
source /opt/ros/humble/setup.bash

ros2 topic pub --once /a1_tt/fsm_command std_msgs/msg/String "{data: fixstand}"
ros2 topic echo /a1_tt/fsm_state

# 只在看到 state=READY 后执行：
ros2 topic pub --once /a1_tt/fsm_command std_msgs/msg/String "{data: table_tennis}"
```

#### 7b-7. 停止 skill

先软停：

```bash
source /opt/ros/humble/setup.bash

ros2 topic pub --once /a1_tt/fsm_command std_msgs/msg/String "{data: passive}"
ros2 topic pub --once /model_control/enable std_msgs/msg/Bool "{data: false}"
```

停止本机进程：

```bash
LOG=/tmp/a1_sim2real_logs

for f in "$LOG"/*.pid; do
  [ -f "$f" ] || continue
  p=$(cat "$f" 2>/dev/null || true)
  [ -n "$p" ] && kill "$p" 2>/dev/null || true
done

sleep 1

for f in "$LOG"/*.pid; do
  [ -f "$f" ] || continue
  p=$(cat "$f" 2>/dev/null || true)
  [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true
done
```

停止机器人底层节点：

```bash
ssh wlab@10.1.1.220

if [ -f /tmp/a1_armcontrol.pid ]; then
  old=$(cat /tmp/a1_armcontrol.pid)
  kill "$old" 2>/dev/null || true
  sleep 1
  kill -9 "$old" 2>/dev/null || true
fi
```

### 7c. 分终端参考：机器人端启动底层 armcontrol

在机器人 `wlab@10.1.1.220` 上运行。机器人端使用 `/home/wlab/pingpong/install/setup.bash`，不是本机的 `/opt/ros/humble`：

```bash
ssh wlab@10.1.1.220
source /home/wlab/pingpong/install/setup.bash
```

先确认没有重复底层节点：

```bash
ps -eo pid,ppid,cmd | grep -E '[i]nference_arm_control_node|[a]rmcontrol' || true
ros2 node list | grep inference_arm_control_node || true
```

推荐使用显式安全参数启动，确保 topic 和 FSM/bridge 一致，并且启动时不进入 servo 跟随：

```bash
ros2 run armcontrol inference_arm_control_node --ros-args \
  -p controlled_arms:=right \
  -p action_topic:=/model_action \
  -p enable_topic:=/model_control/enable \
  -p servo_enabled_on_start:=false \
  -p enable_motors_on_start:=false \
  -p right_arm_device:=/dev/ttyACM1 \
  -p right_arm_can_name:=can1
```

不要在同一个 ROS graph 里同时启动第二个 `inference_arm_control_node`。如果误启动了第二个，先软停：

```bash
ros2 topic pub --once /model_control/enable std_msgs/msg/Bool "{data: false}"
```

然后只保留一个底层节点。

`--params-file $(ros2 pkg prefix armcontrol)/share/armcontrol/config/inference_arm_control_node.yaml` 这条旧启动方式可以用于参考配置，但当前文件里 `enable_motors_on_start: true`，不作为推荐实机启动命令，除非你明确覆盖为 false。

### 7d. 分终端参考：本机终端 1 启动 VRPN 动捕

本机每个 ROS 终端都先退出 conda，并 source Humble：

```bash
conda deactivate
source /opt/ros/humble/setup.bash
ros2 launch vrpn_mocap client.launch.yaml server:=10.1.1.198 port:=3883
```

### 7e. 分终端参考：本机终端 2 VRPN 球 -> `/ball/state`

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
source /opt/ros/humble/setup.bash

./install/sim2real_bridge_cpp/lib/sim2real_bridge_cpp/a1_vrpn_ball_state_bridge --ros-args \
  -p input_topic:=/vrpn_mocap/U_Tracker0/pose \
  -p output_topic:=/ball/state \
  -p origin_in_training_world:="[0.0, 0.0, 0.76]" \
  -p rotation_wxyz_to_training:="[1.0, 0.0, 0.0, 0.0]"
```

检查：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo --once /vrpn_mocap/U_Tracker0/pose
ros2 topic echo --once /ball/state
ros2 topic hz /ball/state
```

### 7f. 分终端参考：机器人端或本机端启动手柄 `/joystick_info`

G1 那套 FSM 走 Unitree SDK/DDS 的遥控器输入，所以不需要在本机接手柄。A1/h1_pingpong 这套底层没有直接暴露同样的 Unitree WirelessController 输入，当前 FSM 只订阅 ROS topic `/joystick_info`。因此手柄应该接在哪里，取决于哪个机器负责发布 `/joystick_info`：

- 手柄插在机器人上：在机器人端启动 `joystick_node`，这是更接近实机的推荐路径。
- 手柄插在本机上：在本机启动 `joystick_node`，只作为本机调试/替代路径。

如果不用手柄、只用 `ros2 topic pub /a1_tt/fsm_command ...` 手动切状态，这一步可以跳过。

#### 方案 A：手柄接在机器人上

```bash
ssh wlab@10.1.1.220
source /home/wlab/pingpong/install/setup.bash

ls -l /dev/input/js0
jstest --event /dev/input/js0
```

启动机器人端 h1_pingpong 的 joystick 节点：

```bash
source /home/wlab/pingpong/install/setup.bash
ros2 run joystick joystick_node --ros-args \
  -p joystick_device:=/dev/input/js0 \
  -p joystick_topic:=/joystick_info
```

如果机器人端找不到 `joystick` 包，需要先在机器人端确认 `/home/wlab/pingpong/install` 里是否包含该包，或把 h1_pingpong 的 joystick 包编译/部署到机器人。

#### 方案 B：手柄接在本机上

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/h1_pingpong
conda deactivate
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run joystick joystick_node --ros-args \
  -p joystick_device:=/dev/input/js0 \
  -p joystick_topic:=/joystick_info
```

如果本机 `ros2 run joystick joystick_node` 找不到包，先编译 joystick：

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/h1_pingpong
conda deactivate
source /opt/ros/humble/setup.bash
colcon build --packages-select joystick --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

手柄实测映射：

```text
Back / Select -> button 6  -> /joystick_info 21 -> Passive
L3            -> button 9  -> /joystick_info 27 -> FixStand/Ready
R3            -> button 10 -> /joystick_info 28 -> TableTennis
```

检查 topic：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /joystick_info
```

### 7g. 分终端参考：本机终端 4 启动 FSM supervisor

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
source /opt/ros/humble/setup.bash
./install/sim2real_bridge_cpp/lib/sim2real_bridge_cpp/a1_tt_fsm_supervisor
```

FSM 启动后默认是 `PASSIVE`，会持续发布：

```text
/model_control/enable = false
/a1_tt/policy_enable = false
```

### 7h. 分终端参考：本机终端 5 启动策略 bridge

注意这里策略动作必须由 FSM 放行，所以 `policy_enabled_on_start:=false`：

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1
conda deactivate
source /opt/ros/humble/setup.bash

./install/sim2real_bridge_cpp/lib/sim2real_bridge_cpp/a1_policy_bridge_cpp --ros-args \
  -p policy_path:=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_v13/2026-07-08_12-40-15/exported/policy.onnx \
  -p joint_state_topic:=/right_joint_states \
  -p ball_state_topic:=/ball/state \
  -p action_topic:=/model_action \
  -p enable_topic:=/model_control/enable \
  -p policy_enable_topic:=/a1_tt/policy_enable \
  -p control_hz:=50.0 \
  -p joint_timeout_s:=2.0 \
  -p diag_every:=1 \
  -p publish_actions:=true \
  -p publish_position_velocity:=false \
  -p policy_enabled_on_start:=false \
  -p enable_on_start:=false \
  -p hold_when_ball_stale:=false \
  -p max_delta_per_tick:="[0.020, 0.024, 0.036, 0.032, 0.080, 0.064, 0.160]"
```

### 7i. 分终端参考：本机检查

只在以下项都确认后进入模式：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo --once /right_joint_states
ros2 topic echo --once /ball/state
ros2 topic echo --once /joystick_info
ros2 topic echo --once /a1_tt/fsm_state
ros2 topic echo --once /sim2real/gate
```

也可以检查订阅关系：

```bash
source /opt/ros/humble/setup.bash
ros2 topic info -v /model_action
ros2 topic info -v /model_control/enable
ros2 topic info -v /a1_tt/policy_enable
```

### 7j. 分终端参考：进入模式

手柄流程：

```text
按 L3            -> 进入 FixStand，FSM 平滑移动到 default q，达到后 state=READY
看到 READY 后按 R3 -> 进入 TableTennis，策略开始发布 /model_action
任意时刻按 Back    -> 回 Passive，关闭策略和底层 enable
```

注意：FSM 流程里不再手动执行下面这条旧命令：

```bash
ros2 topic pub --once /model_control/enable std_msgs/msg/Bool "{data: true}"
```

原因是 `/model_control/enable` 现在由 FSM 统一控制：

```text
PASSIVE       -> /model_control/enable=false, /a1_tt/policy_enable=false
FIXSTAND/READY -> /model_control/enable=true,  /a1_tt/policy_enable=false
TABLE_TENNIS  -> /model_control/enable=true,  /a1_tt/policy_enable=true
```

也就是说，按 L3 或发送 `fixstand` 后，FSM 会自动打开底层跟随并发布 default q；按 R3 或发送 `table_tennis` 后，FSM 才放行策略 bridge 发布真实策略动作。手动发 `enable=true` 会绕过状态保护，而且在 `PASSIVE` 下会被 FSM 立即改回 `false`。

无手柄时用 ROS 命令等价操作：

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --once /a1_tt/fsm_command std_msgs/msg/String "{data: fixstand}"
ros2 topic echo /a1_tt/fsm_state

# 看到 state=READY 后再执行：
ros2 topic pub --once /a1_tt/fsm_command std_msgs/msg/String "{data: table_tennis}"
ros2 topic echo --once /a1_tt/policy_enable
ros2 topic echo --once /model_action
```

软停/回退阻尼：

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --once /a1_tt/fsm_command std_msgs/msg/String "{data: passive}"
```

### 7k. 分终端参考：停止进程

先软停，再杀本机 bridge/FSM/joystick。机器人端底层节点最后停：

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --once /a1_tt/fsm_command std_msgs/msg/String "{data: passive}"
ros2 topic pub --once /model_control/enable std_msgs/msg/Bool "{data: false}"

pkill -f a1_tt_fsm_supervisor
pkill -f a1_policy_bridge_cpp
pkill -f a1_vrpn_ball_state_bridge
pkill -f joystick_node
```

机器人端停止底层节点：

```bash
ssh wlab@10.1.1.220
pkill -f inference_arm_control_node
```

---

## 8. 视觉/球坐标约定

`/ball/state` 必须是训练桌面坐标系：

```text
data = [x, y, z, vx, vy, vz]
```

约定和 sim2sim/policy 一致：

- `hit_plane_x = -1.60`
- 桌面/机器人位置用训练中的 table frame
- 当前实机/下一版 sim2sim 对齐使用 `r1` 关节中线离地最大高度 `1.15 m`
- 无效球时 actor 看到固定 sentinel，不应喂随机假球

如果视觉只能给 `[x,y,z]`，bridge 会差分速度，但实机建议视觉侧直接给稳定滤波后的 `[vx,vy,vz]`。

---

## 9. 参数速查

bridge 参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `policy_path` | v13/model_29999 policy | `policy.onnx` 路径 |
| `predictor_path` | 空 | 空则找 policy 同目录 `predictor.onnx`，不存在则禁用 |
| `use_predictor` | `true` | 是否尝试加载 predictor |
| `control_hz` | `50.0` | bridge 推理频率 |
| `publish_actions` | `true` | 是否发布 `/model_action` |
| `publish_position_velocity` | `false` | `false` 发 7 维 q，`true` 发 14 维 q+dq |
| `enable_on_start` | `false` | bridge 启动时是否发 `/model_control/enable=true` |
| `hold_when_ball_stale` | `false` | 球超时时按 invalid/sentinel 观测继续走策略，由策略输出默认动作 |
| `max_delta_per_tick` | `[0.020,0.024,0.036,0.032,0.080,0.064,0.160]` | 每个 50Hz policy tick 发布给 `/model_action` 的最大关节目标变化量，对齐 sim2sim 默认速度上限 |
| `joint_timeout_s` | `2.0` | 关节反馈超时 |
| `ball_timeout_s` | `0.20` | 球状态超时 |

底层 `inference_arm_control_node` 关键参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `controlled_arms` | `right` | 当前只控右臂 |
| `control_rate_hz` | `100.0` | 底层 MIT 控制循环 |
| `action_topic` | `/model_action` | bridge 动作输入 |
| `enable_topic` | `/model_control/enable` | servo enable |
| `right_arm_device` | `/dev/ttyACM1` | 当前机器人右臂设备 |
| `servo_enabled_on_start` | `false` | 启动时不进 servo |
| `enable_motors_on_start` | launch 中设 `false` | 启动时不主动 enable 电机 |
| `kps` | `[200,200,200,90,90,90,90]` | 与 v13 训练 stiffness 对齐 |
| `kds` | `[3.5,3.5,3.5,0.5,0.5,0.5,0.5]` | 与 v13 训练 damping 对齐 |
| `interpolation_mode` | `linear` | 与机器人端当前 yaml 对齐 |
| `torque_ff_scale` | `[0,0,0,0,0,0,0]` | 当前真机先不加 RBDL 前馈 |
| `enable_mit_velocity` | `false` | 当前真机 MIT 速度项关闭 |
| `max_vel` | `[8,8,8,30,30,30,30]` | 当前机器人 yaml；实际还受下方单周期限幅约束 |
| `max_acc` | `[80,80,80,160,160,160,160]` | 当前机器人 yaml |
| `max_delta_per_cycle` | `[0.08,0.08,0.08,0.16,0.16,0.16,0.16]` | 100Hz 下等效单周期速度上限约 `[8,8,8,16,16,16,16]` |

---

## 10. 常见坑速查

- **不要在 conda 环境里编译部署包**。G1 deploy 也是这么要求的。必要时显式加 `-DPython3_EXECUTABLE=/usr/bin/python3`。
- **C++ bridge 不需要 Python onnxruntime**。它链接 vendored `libonnxruntime.so.1.22.0`。
- **`publish_actions:=false` 是 dry-run**，不会发 `/model_action`。
- **`enable_on_start:=false` 只是不打开 servo**，如果你手动发了 `/model_control/enable=true`，底层就会开始执行最新 `/model_action`。
- **`predictor=(disabled)` 不一定是错误**。说明没有找到 predictor，bridge 会用解析预测 fallback。
- **球坐标系错会直接让策略乱打**。先看 `/sim2real/gate` 的 `reason` 和 `pred`，再考虑 enable。
- **关节名必须匹配**：`joint1-a1_r ... joint7-a1_r`。bridge 也兼容 `joint1-r` 和 `r1` 这种别名，但实机建议统一用 A1 名称。
- **一次只跑一个 bridge / 一个底层控制节点**。重复节点会抢 topic 和硬件设备。
- **先软停再杀进程**：先发 `/model_control/enable=false`，再 `pkill`。
