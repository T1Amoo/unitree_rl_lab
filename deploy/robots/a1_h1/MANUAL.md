# A1/H1 乒乓 Sim2Real C++ Bridge — 完整命令行手册

覆盖 **策略导出 / C++ ONNX bridge / ROS dry-run / sim2real 真机接入** 全流程命令。
最后更新 2026-07-27（新增第 11 节：相机版 CycloneDDS 三机实机流程 + 限幅机制澄清）。

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
Pingpong_TTRL/logs/a1_tt_real_v7/2026-07-24_14-18-11_resume10000_range10k_hold20k/exported/policy.onnx
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
| `a1_tt_real_v7/2026-07-24_14-18-11_resume10000_range10k_hold20k` | 有 | 有 | 当前默认；v7 `model_25300.pt` 导出 |
| `a1_tt_v13/2026-07-08_12-40-15` | 有 | 有 | 旧正手基线；v13 `model_29999.pt` 导出 |
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
  predictor_path:=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_real_v7/2026-07-24_14-18-11_resume10000_range10k_hold20k/exported/predictor.onnx
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
  -p policy_path:=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_real_v7/2026-07-24_14-18-11_resume10000_range10k_hold20k/exported/policy.onnx \
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
  -p policy_path:=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_real_v7/2026-07-24_14-18-11_resume10000_range10k_hold20k/exported/policy.onnx \
  -p joint_state_topic:=/right_joint_states \
  -p ball_state_topic:=/ball/state \
  -p action_topic:=/model_action \
  -p publish_actions:=true \
  -p policy_enabled_on_start:=true \
  -p enable_on_start:=false \
  -p hold_when_ball_stale:=false \
  -p max_delta_per_tick:="[0.050, 0.050, 0.050, 0.100, 0.100, 0.100, 0.100]" \
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
[0.050, 0.050, 0.050, 0.100, 0.100, 0.100, 0.100]
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
POLICY=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_real_v7/2026-07-24_14-18-11_resume10000_range10k_hold20k/exported/policy.onnx

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
  -p max_delta_per_tick:="[0.050, 0.050, 0.050, 0.100, 0.100, 0.100, 0.100]" \
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
max_delta_per_tick=[0.050, 0.050, 0.050, 0.100, 0.100, 0.100, 0.100]
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
  -p policy_path:=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_real_v7/2026-07-24_14-18-11_resume10000_range10k_hold20k/exported/policy.onnx \
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
  -p max_delta_per_tick:="[0.050, 0.050, 0.050, 0.100, 0.100, 0.100, 0.100]"
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
| `policy_path` | v7/model_25300 policy | `policy.onnx` 路径 |
| `predictor_path` | 空 | 空则找 policy 同目录 `predictor.onnx`，不存在则禁用 |
| `use_predictor` | `true` | 是否尝试加载 predictor |
| `control_hz` | `50.0` | bridge 推理频率 |
| `publish_actions` | `true` | 是否发布 `/model_action` |
| `publish_position_velocity` | `false` | `false` 发 7 维 q，`true` 发 14 维 q+dq |
| `enable_on_start` | `false` | bridge 启动时是否发 `/model_control/enable=true` |
| `hold_when_ball_stale` | `false` | 球超时时按 invalid/sentinel 观测继续走策略，由策略输出默认动作 |
| `servo_filter_enabled` | `true` | **默认限幅器**：开则走一阶滤波，`max_delta_per_tick` 此时不生效（详见 11.7）|
| `servo_tau_s` | `0.25` | 一阶滤波时间常数（秒），实机 07-25 实际用的就是它 |
| `servo_velocity_limit` | `[1.0,1.2,1.8,1.6,4.0,3.2,8.0]` | 一阶滤波支路每关节速度上限 rad/s |
| `qdes_slew_enabled` | `false` | 仅当 `servo_filter_enabled=false` 时启用硬限幅 |
| `max_delta_per_tick` | `[0.050,...]` | 每 tick 最大关节变化量；**只在 `servo_filter=false && qdes_slew=true` 时生效**（默认死参数，见 11.7）|
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

---

## 11. 相机版 sim2real 实机流程（2026-07-25 实测，CycloneDDS 三机）

> 这一节记录 **2026-07-25 真机跑通的实际流程**：球位置输入从 VRPN 动捕换成 **本体相机（Jetson ZED-X）**，三台机器统一走 **CycloneDDS unicast**。第 6/7 节的 VRPN 流程仍然有效，本节是相机变体，且 kp/kd、限幅、球坐标偏移都和第 7 节的旧默认不同，实机以本节为准。

### 11.0 三机架构

```text
Jetson 10.1.1.231 (jetson/yahboom, humble)
  pingpong-detect.service -> /pingpong_location  (geometry_msgs/PoseStamped, 相机系, ~56Hz)

本机 10.1.1.150 (humble, enp8s0)
  a1_vrpn_ball_state_bridge  /pingpong_location -> /ball/state  (y+0.76 偏移)
  a1_tt_fsm_supervisor       FSM
  a1_policy_bridge_cpp       policy.onnx (+predictor) -> /model_action
  ros2 bag record

机器人 10.1.1.220 (wlab/wlab, jazzy, enp8s0)
  inference_arm_control_node  /model_action -> DAMIAO 右臂 /dev/ttyACM1 (v7 kp/kd)
  发布 /right_joint_states /joystick_info
```

- 有线网卡 IP：本机 `10.1.1.150`、Jetson `10.1.1.231`、机器人 `10.1.1.220`。
- **三方 RMW 必须统一 `rmw_cyclonedds_cpp`**。本机若缺 `librmw_cyclonedds_cpp.so`，`export RMW=cyclonedds` 会**静默 fallback 到 fastrtps**，和 cyclonedds 的相机/机器人不通：`sudo apt install ros-humble-rmw-cyclonedds-cpp`。
- 这网段多播不稳，靠 **unicast peers**，三方 `cyclonedds.xml` 的 Peers 要互列对方 IP。

### 11.1 网络前置（每台机一次）

**本机 CycloneDDS 配置** `/tmp/cyclonedds_enp8s0.xml`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain>
    <General>
      <NetworkInterfaceAddress>10.1.1.150</NetworkInterfaceAddress>
    </General>
    <Discovery>
      <Peers>
        <Peer address="10.1.1.231"/>
        <Peer address="10.1.1.220"/>
      </Peers>
      <ParticipantIndex>auto</ParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
```

**本机每个 ROS 终端的环境**（conda 污染会让 rmw/rclpy 崩，务必先清）：

```bash
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -vE 'anaconda3|miniconda|/conda' | paste -sd:)
unset PYTHONPATH PYTHONHOME
source /opt/ros/humble/setup.bash
source /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/a1_h1/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///tmp/cyclonedds_enp8s0.xml
```

**本机防火墙**：ufw active 会挡 DDS UDP。测试期间 `sudo ufw disable`，**测完务必 `sudo ufw enable`**。

**机器人端** `/home/wlab/cyclonedds.xml` 的 Peers 要含本机 `10.1.1.150` 和 Jetson `10.1.1.231`；**Jetson 端** `/home/jetson/cyclonedds_eth.xml` 同理。

> `ros2 param set` 在 cyclonedds 下会报 `empty node name from RMW`，**动态调参不可用，一律用启动参数 + 重启**。

### 11.2 机器人端：启动底层 inference（v7 kp/kd）

机器人 `wlab@10.1.1.220`（jazzy，用 `/home/wlab/pingpong/install`）。SDK 用 `ssh nohup` detach 不住，**在机器人终端前台/tmux 跑**：

```bash
source /home/wlab/pingpong/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/wlab/cyclonedds.xml

ros2 run armcontrol inference_arm_control_node --ros-args \
  --params-file /home/wlab/pingpong/install/armcontrol/share/armcontrol/config/inference_arm_control_node.yaml \
  -p servo_enabled_on_start:=false \
  -p enable_motors_on_start:=false \
  -p right_arm_device:=/dev/ttyACM1 \
  -p kps:='[300.0,300.0,300.0,120.0,120.0,120.0,120.0]' \
  -p kds:='[3.5,3.5,3.5,1.0,1.0,1.0,1.0]'
```

- **kp/kd = v7 值 `[300,300,300,120,120,120,120]` / `[3.5,3.5,3.5,1.0,1.0,1.0,1.0]`**（不是第 9 节旧的 v13 `[200/90]`、`[3.5/0.5]`）。
- 右臂 DAMIAO 是 **`/dev/ttyACM1`（HDSC 芯片）**；`ttyACM0` 是 ESP32 调试口，不是臂。USB 重枚举后 ACM 号会浮动，用 `udevadm info /dev/ttyACM* | grep -i hdsc` 认。
- CAN 断 → `/right_joint_states` 全 0 → FSM FixStand 超时（看着像“没反应”，其实是 CAN 掉了，重启 CAN）。

### 11.3 Jetson 相机端：pingpong-detect 服务（发 /pingpong_location）

相机节点由 systemd `pingpong-detect.service` 托管，需带 cyclonedds drop-in：

`/home/jetson/cyclonedds_eth.xml`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces><NetworkInterface name="enP8p1s0"/></Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
```

`/etc/systemd/system/pingpong-detect.service.d/rmw.conf`：

```ini
[Service]
Environment="RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
Environment="CYCLONEDDS_URI=file:///home/jetson/cyclonedds_eth.xml"
```

改完 `sudo systemctl daemon-reload && sudo systemctl restart pingpong-detect.service`。

- ZED-X 走 **GMSL 非 USB**，`lsusb` 看不到是正常的；相机 `open failed` 通常是被占用/需要重启 Jetson。
- 相机**检测到球才发**，桌面没球时 `/pingpong_location` 可能无数据，不是故障。

本机侧确认相机在发：

```bash
ros2 topic hz /pingpong_location            # 有球时 ~56Hz
ros2 topic echo /pingpong_location --field pose.position --once
```

> `ros2 topic echo /pingpong_location` 偶发 `serdata.cpp:384 string data is not null-terminated / invalid data size` 是**间歇噪声**（`hz` 能正常计数、`--field` 能取到值即证明数据在流），不影响主数据。

### 11.4 本机：球 bridge（相机 → /ball/state，y+0.76 偏移）

相机原点在桌中心，机器人物理站 `[-1.8, 0]`，训练系机器人在 `[-1.8, 0.76]`（差 y 0.76）。**不移机器人**，让球桥把每个球 `y+0.76`（等价“虚拟机器人在 y=0.76”），robot_pos 写死训练值不变。

```bash
"$BIN/a1_vrpn_ball_state_bridge" --ros-args \
  -p input_topic:=/pingpong_location \
  -p output_topic:=/ball/state \
  -p origin_in_training_world:="[0.0, 0.76, 0.76]" \
  -p rotation_wxyz_to_training:="[1.0, 0.0, 0.0, 0.0]" \
  -p diag_every:=50
```

（`$BIN` = `.../install/sim2real_bridge_cpp/lib/sim2real_bridge_cpp`）验证：桌中心球 → `/ball/state` ≈ `[-0.05, 0.74, 0.78, ...]`（y 已 +0.76）。**注意与第 7 节 VRPN 版的 `origin_in_training_world=[0,0,0.76]` 不同（多了 y=0.76）。**

### 11.5 本机：FSM + policy bridge

FSM（正手用硬编码 default，无需 `-p default_q`；反手变体见 11.8）：

```bash
"$BIN/a1_tt_fsm_supervisor" --ros-args \
  -p joint_state_topic:=/right_joint_states -p action_topic:=/model_action \
  -p right_movej_topic:=/movej_right_angle -p enable_topic:=/model_control/enable \
  -p policy_enable_topic:=/a1_tt/policy_enable -p joystick_topic:=/joystick_info \
  -p command_topic:=/a1_tt/fsm_command -p state_topic:=/a1_tt/fsm_state \
  -p control_hz:=50.0 -p joint_timeout_s:=2.0 \
  -p joystick_fixstand_code:=27 -p joystick_table_tennis_code:=28 -p joystick_passive_code:=2
```

policy bridge（正手 v7 policy + 同源 predictor）：

```bash
POLICY=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_real_v7/2026-07-24_14-18-11_resume10000_range10k_hold20k/exported/policy.onnx
PRED=/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL/logs/a1_tt_real_v7/2026-07-24_14-18-11_resume10000_range10k_hold20k/exported/predictor.onnx

"$BIN/a1_policy_bridge_cpp" --ros-args \
  -p policy_path:="$POLICY" \
  -p use_predictor:=true -p predictor_path:="$PRED" \
  -p joint_state_topic:=/right_joint_states \
  -p ball_state_topic:=/ball/state \
  -p action_topic:=/model_action \
  -p enable_topic:=/model_control/enable \
  -p policy_enable_topic:=/a1_tt/policy_enable \
  -p control_hz:=50.0 -p joint_timeout_s:=2.0 -p diag_every:=1 \
  -p publish_actions:=true -p publish_position_velocity:=false \
  -p policy_enabled_on_start:=false -p enable_on_start:=false \
  -p hold_when_ball_stale:=false
```

启动后 `head -20 .../a1_policy_bridge_cpp.log` 必须看到 ready 行，其中限幅段是：

```text
servo_filter=true servo_tau=0.250 servo_vel=[1.000, 1.200, 1.800, 1.600, 4.000, 3.200, 8.000] qdes_slew=false max_delta_per_tick=[...]
```

### 11.6 本机：rosbag

```bash
BAG=/tmp/a1_sim2real_logs/bags/a1_run_$(date +%H%M%S)
ros2 bag record -o "$BAG" \
  /right_joint_states /ball/state /pingpong_location /joystick_info \
  /a1_tt/fsm_state /a1_tt/policy_enable /model_control/enable /model_action \
  /sim2real/gate /sim2real/raw_action /sim2real/q_des /sim2real/obs
```

> bag 在 `/tmp` 重启会丢，测完立刻 `cp -r` 到持久盘（如 `系统辨识/sim2real/<日期>/`），再 `ros2 bag reindex` 补 `metadata.yaml`。

操作顺序同第 7 节：`L3`(27)→等 `state=READY`→`R3`(28)→TableTennis；异常 `Back`(21/2)→Passive。

### 11.7 限幅机制澄清（一阶滤波 vs 硬限幅）

**2026-07-25 实机用的是一阶滤波（`servo_tau_s`），不是硬限幅（`max_delta_per_tick`）。** bridge 里两套限幅**互斥**：

| 参数 | 默认 | 作用 |
|---|---:|---|
| `servo_filter_enabled` | **`true`** | 开则走一阶滤波（**此分支下 `max_delta_per_tick` 完全不生效**）|
| `servo_tau_s` | **`0.25`** | 一阶滤波时间常数（秒）：`dq = (raw_q_des - q_cmd)/tau`，再按 `servo_velocity_limit` 限速 |
| `servo_velocity_limit` | `[1.0,1.2,1.8,1.6,4.0,3.2,8.0]` | 一阶滤波支路的每关节速度上限（rad/s，= `kIsaacServoVelocityLimit`）|
| `qdes_slew_enabled` | `false` | **仅当 `servo_filter_enabled=false`** 时才启用硬限幅 `clampDelta(max_delta_per_tick)` |
| `max_delta_per_tick` | `[0.02,0.024,0.036,0.032,0.08,0.064,0.16]`* | 每 tick 目标最大变化量；**只有 `servo_filter=false && qdes_slew=true` 时才起作用** |

\* 07-25 传入的 `max_delta_per_tick` 值虽在 ready 行打印，但因 `qdes_slew=false` 是**死参数**。

代码逻辑（`a1_policy_bridge_cpp.cpp` ~869-905）：

```text
if (!servo_filter_enabled_)      # 默认不进这支
    cmd = qdes_slew_enabled_ ? clampDelta(raw_q_des) : raw_q_des;
else                              # 默认走这支：一阶滤波
    dq = (raw_q_des - servo_q_cmd) / servo_tau_s;
    dq = clamp(dq, ±servo_velocity_limit);
    servo_q_cmd += dq * dt;
```

要改成硬限幅需显式传 `-p servo_filter_enabled:=false -p qdes_slew_enabled:=true -p max_delta_per_tick:="[...]"`。

> 第 7/9 节里把 `max_delta_per_tick` 当唯一限幅器的描述，对默认配置（`servo_filter=true`）是**误导的**——默认下真正的限幅是一阶滤波 `servo_tau_s=0.25` + `servo_velocity_limit`。

### 11.8 反手策略变体（2026-07-25 晚）

反手策略（`a1_backhand_deploy_model_10999`）除 FSM FixStand 外，**policy bridge 有三个正手硬编码常量必须改并重编译**（否则观测基准/击球面/预测都错、甩臂危险）：

| 常量（`a1_policy_bridge_cpp.cpp`） | 正手 | 反手 |
|---|---|---|
| `kDefaultRightQ`（观测基准 + `q_des=action*scale+default`）| `{0.569,-0.692,0.717,1.13,-1.24,0.0314,0.772}` | `{1.769,-0.762,-1.863,1.445,0.206,-0.827,1.043}` |
| `kHitPlaneX` | `-1.60` | `-1.43` |
| `kPaddleYOffset` | `-0.66` | `-0.03` |

FSM 侧用参数覆盖即可（不必改源码）：`-p default_q:="[1.769,-0.762,-1.863,1.445,0.206,-0.827,1.043]"`。
反手包**没有 predictor.onnx**，用 `-p use_predictor:=true -p predictor_path:=<正手 v7 predictor>`（球飞行物理与正反手无关，正手 predictor 通用，比 analytic 抛物线 fallback 更准）。
