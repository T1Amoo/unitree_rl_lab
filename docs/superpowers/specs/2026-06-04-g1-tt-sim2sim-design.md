# G1 23DoF 乒乓 sim2sim 管线 — 设计 (项目 B)

- 日期: 2026-06-04
- 状态: 设计已与用户逐节确认,待 spec 审阅 → writing-plans
- 目标: 用**真实 C++ deploy 栈**驱动 G1 在 **unitree_mujoco(Python sim)**里打乒乓,球/基体数据走 **ROS2 topic(与真机 Nokov 同接口)**,为 **sim2real** 铺路降险。
- 前置: 项目 A(deploy 管线数值等价)已完成——obs 项 / predictor.onnx / policy.onnx / deploy.yaml / TTBallSource 抽象均可复用。

## 1. 背景与动机
用户下一步是 sim2real,故 sim2sim 要走**真机将用的那套 deploy 栈**(C++ + unitree_sdk2 DDS + ROS2 球源),在 mujoco 里先验证全链路。unitree_mujoco 经 DDS 发 LowState/收 LowCmd,外部控制器像连真机一样连它;它只仿机器人本体,球桌/球要自己加进 MJCF 场景。球+基体位姿在真机由 Nokov 动捕经 **ROS2 `geometry_msgs/PoseStamped`(位姿)+ `TwistStamped`(速度)** 提供;sim 侧用同样 topic 发布 → deploy 同一份订阅代码 sim/real 通用。

## 2. 架构
```
[Python unitree_mujoco sim]                         [C++ deploy (g1_23dof)]
  TT 场景(G1+拍+桌+球)                                FSM: Passive→FixStand→TableTennis
  发球控制器(复刻 reset_ball)                          TableTennis 状态:
  每步:                                                 - 关节+IMU ← LowState (unitree_sdk2 DDS)
   - 步进 mujoco                                         - 球+基体位姿 ← ROS2 订阅(RosBallSource)
   - 发 LowState / 收 LowCmd  ──DDS(domain 1)──►        - predictor.onnx 实时跑
   - 读 mjData 球+躯干位姿/速度                          - obs 拼装(项目A的 obs项/deploy.yaml)
   - rclpy 发布 PoseStamped/TwistStamped ──ROS2──►       - policy.onnx → LowCmd ──DDS──►
```
- **数据分工**:关节/IMU 走 unitree_sdk2 DDS(LowState);球+基体位姿走 ROS2 mocap topic。与真机一致。
- **seam**:deploy 的 `TTBallSource` → sim 用 `RosBallSource`(订阅 ROS2);真机也用 `RosBallSource`(Nokov 发同样 topic)——**deploy 一行不改**即可上真机。

## 3. 环境(新建 conda env `g1tt_sim2sim`)
ROS2 Humble(robostack)+ mujoco + unitree_sdk2_python + cyclonedds + pygame + onnxruntime + numpy。C++ deploy 构建时 source 此 env 取 `rclcpp`+`geometry_msgs`(CMake find_package)+ 原 unitree_sdk2。
- ⚠️ 风险:ROS2(robostack)的 CycloneDDS 与 unitree_sdk2_python 的 cyclonedds 共存(CYCLONEDDS_HOME)。安装时处理;若冲突需记录方案。

## 4. ROS2 接口(占位 topic,真机定后改)
| 数据 | topic(占位) | 类型 | 说明 |
|---|---|---|---|
| 球位姿 | `/mocap/ball/pose` | PoseStamped | 球 3D 位置(球桌坐标系) |
| 球速度 | `/mocap/ball/twist` | TwistStamped | 球线速度 |
| 基体位姿 | `/mocap/base/pose` | PoseStamped | 躯干位置+朝向 |
| 基体速度 | `/mocap/base/twist` | TwistStamped | 躯干速度 |
frame_id 用球桌坐标系(与训练 obs 一致:ball_pos=ball-table、robot_pos=trunk-table)。obs 取用:ball_pos←ball/pose;robot_pos←base/pose.position;heading←base/pose.orientation→yaw。topic 名/frame 真机确定后统一改。

## 5. 三个里程碑

### B1 — 打通 deploy ↔ unitree_mujoco(无球)
clone unitree_mujoco,用 `simulate_python/`,配 G1 场景(先 stock g1_23dof MJCF),domain id=1。C++ deploy 用现有 Passive/FixStand 状态经 DDS 连上,确认 mujoco 里 G1 站立/响应手柄。**验证 DDS 桥 + 控制环。**

### B2 — TT 场景 + 发球 + ROS2 发布
- **场景 MJCF**:基于 `g1_23dof_rev_1_0.xml` 加:球拍(右腕加适配杆+拍面 geom,几何沿用 URDF 那套,restitution 材质)、球桌(MJCF 板块+网,尺寸对齐 IsaacSim:台高 0.76、自/对台 x∈∓1.37、网 x=0)、球(sphere r0.02、3.4g、restitution 0.9)。物理对齐 IsaacSim(拍 0.8/球 0.9,combine)。
- **发球控制器**(Python):按训练 serve range(`ball_speed_x/y/z`、`ball_pos_y_range`)定期重置球 qpos/qvel 朝机器人发,复刻 `reset_ball` 节奏。
- **ROS2 发布**(rclpy,在 sim 循环内):每步读 mjData 球+躯干 → 发 PoseStamped/TwistStamped。
- 验证:rostopic echo 看到球/基体位姿;发球轨迹合理。

### B3 — 实时 TableTennis 状态 + RosBallSource → 打球
- C++ deploy 加 **TableTennis FSM 状态**(复用项目 A:tt_observations 4 项、tt_predictor、deploy.yaml;predictor 实时跑)。
- **RosBallSource**:rclcpp 订阅 4 个 topic,提供 `get()→{ball_pos, robot_pos, heading}`(替代 ReplayBallSource)。deploy 主循环加 ROS2 spin(单独线程)。
- deploy CMake 加 `find_package(rclcpp geometry_msgs)`,链接;main 起 rclcpp::init + 订阅节点。
- 运行:起 Python sim(场景+发球+发布,domain1)→ 起 deploy(TableTennis,domain1)→ mujoco viewer 看 G1 接发球。

## 6. 验证(sim2sim)
- B1:G1 在 mujoco 站稳、响应。
- B2:`ros2 topic echo` 看到球/基体位姿;发球进入机器人可达区。
- B3:G1 跑位拦截+回球;统计 N 个发球的命中/回球率,与 IsaacSim eval **定性**对比。
- **预期 sim2sim 有差距**(mujoco 物理≠IsaacSim:接触/restitution/电机模型),不追求复现 97%;**动作合理 + 能打回若干球**即管线通,差距大小本身是 sim2real 的有用信息。

## 7. 复用 / 改动面
- 复用(项目 A):`tt_observations.h`、`tt_predictor.h`、`deploy.yaml`、`policy.onnx`、`predictor.onnx`、TTBallSource 抽象。
- 新增:unitree_mujoco(clone)、TT 场景 MJCF + 发球 + ROS2 发布脚本(Python)、`RosBallSource`(C++ rclcpp)、TableTennis FSM 状态、deploy CMake 加 ROS2、g1_23dof config 加 TableTennis 状态。
- 球拍质量/几何沿用项目 A 的标定(0.337m 触点等);MJCF 物理参数对齐 IsaacSim。

## 8. 风险
- **CycloneDDS 双份**(ROS2 vs unitree_sdk2py)共存 —— 安装期处理。
- **deploy 同时用 unitree_sdk2 DDS + ROS2** —— 两套 DDS 中间件在一个进程,domain/rmw 需隔离(unitree 用其 CycloneDDS,ROS2 用 rmw_cyclonedds;可能要分 domain 或确认不串扰)。这是最大集成风险,B3 重点验证。
- **mujoco G1 电机/增益**:LowCmd 的 kp/kd 要在 mujoco 里产生合理力矩(unitree_mujoco 用 PD on torque);与 IsaacSim implicit actuator 不同,可能要调。
- **sim2sim 物理差距**:接触/restitution 调到接近 IsaacSim,但本质不同,差距预期内。
- 真机 topic 名/frame 未定 → 用占位,后续统一。
