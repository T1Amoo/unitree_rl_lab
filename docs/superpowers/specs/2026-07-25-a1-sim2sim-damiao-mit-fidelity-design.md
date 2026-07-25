# A1 sim2sim `damiao_mit` 保真模式设计（对齐训练 Isaac）

- 日期：2026-07-25（v2，按代码核实结果重写；v1 曾错误假设执行器开了二阶响应）
- 分支：`unitree_rl_lab @ g1-tt-sim2sim`（sim2sim 改动）、`Pingpong_TTRL @ a1-tt-migration`（验证脚本、参数源）
- 相关记忆：`a1-tt-real-v7-and-sysid-2026-07-24`、`a1-tt-sim2sim-default-mode`

## 背景与问题

v7 训练用自定义显式执行器 `DamiaoMITActuator`（`Pingpong_TTRL/legged_lab/actuators/damiao_mit.py`），任务映射链：`--task=a1_tt_real` → `A1TableTennisTorqueOnlyEnvCfg` → `A1_TT_REAL_TORQUE_ONLY_CFG`（`legged_lab/assets/a1/a1.py`）。

**代码核实后的真实执行器配置**（`A1_TT_REAL_TORQUE_ONLY_CFG` 传给 `DamiaoMITActuatorCfg` 的字段）：
- `response_model_enable=False`（**二阶响应没开**——`A1_REAL_FITTED_*` 那套二阶参数在本 cfg 未被使用）
- `command_delay_s=0`、`command_lead_s=0`、`command_bias_rad=0`
- `use_command_velocity=False`（MIT 阻尼项目标速度 = 0，纯阻尼）
- `viscous_friction=0`、`coulomb_friction=0`、`torque_time_constant=0`（无摩擦、无力矩低通）
- `torque_speed_limit_enable=True`、`brake_effort_limit=_EFFORT`（**有 torque-speed 包络**）
- `stiffness=_REAL_FITTED_NODE_KP`、`damping=_REAL_FITTED_NODE_KD`、`control_dt=0.002`

所以 v7 执行器的实际链路很简单：
```
q_cmd → slew限速(_VEL) → MIT-PD力矩[ kp·(q_slew−q) + kd·(0−dq) ] → torque-speed包络clip → PhysX积分 → q
```
关节响应主要由 **MIT-PD 纯阻尼跟踪滞后 + torque-speed 包络限力矩 + 惯量** 塑造，不是由目标指令直接塑造。

而 sim2sim 默认的 `direct_response`（二阶响应算出 q 直接写 qpos）与此不符；`torque_chain`/`real_deploy_preview` 虽然跑 MIT-PD，但（a）力矩用了 sim2sim 的二阶响应 `motor_q_des`（训练没开二阶）、（b）`estimate_mit_tau_raw` 里加了 `qfrc_bias` 重力补偿（训练不加，靠 MIT-PD 自扛重力）、（c）只有静态 effort clip 没有 torque-speed 包络、（d）常量与训练不一致。

**目标（对齐训练 Isaac）**：新增一个 sim2sim 模式，忠实复刻 v7 DamiaoMIT 的力矩链路，使 mujoco 关节响应与 IsaacLab 训练里的关节响应一致，用于部署前预览"训练策略在训练执行器下的行为"。

## 关键：sim2sim 现有常量与训练 v7 不一致（复刻必须用训练值）

| 量 | 训练 v7（`a1.py`，关节 r1..r7） | sim2sim 现有（`policy_io.py`） |
|---|---|---|
| `effort_limit` | `[28,28,28,8,8,8,8]`（idx<3 为 28，其余 8） | `EFFORT=[28,28,28,28,8,8,8]`（r4=28） |
| `velocity_limit` | `_VEL=[8,8,8,20,20,20,20]` | `VEL_LIMIT/DAMIAO_DQ=[10,10,10,30,30,30,30]` |
| `stiffness/kp` | `[300,300,300,120,120,120,120]` | `KP=[300,300,300,120,120,120,120]`（一致）|
| `damping/kd` | `[3.5,3.5,3.5,1,1,1,1]` | `KD=[3.5,3.5,3.5,1,1,1,1]`（一致）|
| `armature` | `[0.032×3, 0.0018×4]` | （MJCF 内） |
| `control_dt` | `0.002` | `PHYSICS_DT=0.002`（一致，DECIMATION=10 → 50Hz control）|

新模式为自身固化一组**从训练 cfg 精确抄写**的常量（`DAMIAO_MIT_*`），不复用 sim2sim 现有的 `EFFORT/VEL_LIMIT`。

## 范围

**做**：
- 在 `run_a1_tt_sim2sim.py` 新增 `--actuator-mode damiao_mit`。
- 每物理步（dt=0.002）执行：slew 限速（`_VEL`）→ MIT-PD 力矩（kp/kd，desired_vel=0，**不加重力补偿**）→ torque-speed 包络 clip（复刻 `_clip_effort`）→ 写 `qfrc_applied`，由 mujoco `mj_step` 积分。
- 复刻的 torque-speed 包络与 slew 抽成可单测的纯 numpy 函数（`policy_io.py`），注释标明"镜像 `DamiaoMIT._clip_effort` / `_apply_command_slew`"。
- 验证：真实轨迹 + 扫频/阶跃两类输入；附带量化"直接写 q（direct_response）vs 力矩积分（damiao_mit）"各自与 Isaac 的差距。

**不做（本次明确排除）**：
- 不复刻二阶响应（训练没开）、摩擦、力矩低通、command lead/delay（训练全为 0/关）。
- 不改动 `direct_response`（仍是默认模式）；不改 `torque_chain`/`real_deploy_preview`。
- 不做端到端（policy+obs）对齐——obs/gate/predictor 时序差异（07-14 观察到 Isaac "提前 0.2s"）会污染执行器对齐判断，故隔离喂同一 q_des。
- 不跨仓库共享代码：sim2sim 侧独立 numpy 实现，注释标明镜像来源与抄写日期。

## 架构与数据流

每物理步（500Hz，dt=0.002）：
```
q_des（policy 50Hz 更新后 hold；或验证信号）
  → slew:  cmd += clamp(q_des − cmd, ±_VEL·dt)      # 镜像 _apply_command_slew（response/lead/delay 均为恒等）
  → tau  = kp·(cmd − q) + kd·(0 − dq)                # 镜像 MIT-PD，desired_vel=0，无重力补偿
  → tau  = clip_effort_torque_speed(tau, dq)         # 镜像 _clip_effort（torque_speed_limit_enable=True）
  → data.qfrc_applied[dof] = tau
  → mujoco.mj_step  → q, dq 演化
```
`clip_effort_torque_speed`：`speed_scale=clamp(1−|dq|/velocity_limit,0,1)`；`accel_limit=effort_limit·speed_scale`；`brake_limit=max(brake_effort_limit,effort_limit)`；`dq>0` 时 `max=accel_limit,min=−brake_limit`，`dq<0` 时 `max=brake_limit,min=−accel_limit`，`dq==0` 两侧 `brake_limit`。

## 组件

1. **`policy_io.py` 常量**：`DAMIAO_MIT_KP/KD/EFFORT/VEL/BRAKE_EFFORT`，值从训练 cfg 精确抄写，并列注释训练来源。
2. **`policy_io.py` 纯函数**：
   - `damiao_slew(cmd, q_des, vel_limit, dt) -> cmd_next`
   - `damiao_clip_effort(tau, dq, vel_limit, effort_limit, brake_effort_limit) -> tau_clipped`
   - `PolicyIO.apply_damiao_mit(dt)`：组合 slew + MIT-PD（无重力补偿）+ clip，写 `qfrc_applied`。
3. **`run_a1_tt_sim2sim.py` 接线**：`--actuator-mode` choices 加 `damiao_mit`；`physics_step` 加分支调用 `io.apply_damiao_mit(PHYSICS_DT)`（走力矩路径，类似 `torque_chain`，但不叠加二阶 `motor_q_des`、用新常量、不加 `qfrc_bias`）。
4. **验证**：`record_a1_play_trace.py` 录 Isaac play `(q_des 序列, 实际 q 序列)`；mujoco 用同一 q_des 跑 `damiao_mit`；`compare_sim2real_trace.py` 出 7 关节角对比图 + 每关节 RMSE。扫频/阶跃信号生成器补一段标准输入。输出 `系统辨识/sim2real/<日期>/damiao_mit_isaac_align/`。

## 验收标准

- **主判据**（真实轨迹自由段，per-joint RMSE，mujoco `damiao_mit` vs Isaac 实际 q）：J1-3 < **0.03 rad**，J4-7 < **0.05 rad**（跑力矩会有积分/接触差异，阈值比"直接写目标 q"宽）。
- **扫频**：相位/幅值随频率的趋势与 Isaac 一致（torque-speed 包络在高速段起作用的证据）。
- **附带结论（回答方案选择）**：同一输入下并列出 `direct_response`（直接写 q）与 `damiao_mit`（力矩积分）各自对 Isaac 的 RMSE。若两者接近 → 记录"直接写 q 的近似在 v7 高刚度下也够用"；若 `damiao_mit` 明显更接近 → 确认力矩层不可省。

## 实现前置核实（实现第一步）

- 核实 `Pingpong_TTRL/legged_lab/scripts/record_a1_play_trace.py` 是否记录喂给执行器的 `q_des`（policy 输出经 action_scale 后的 joint_position_target）与实际 `q` 两列。若缺，先补记录再验证。

## 风险与已知取舍

- **重力补偿**：新模式**不加** `qfrc_bias`，让 mujoco 自算重力、MIT-PD 通过 kp 位置误差扛重力，以复现训练里的稳态位置误差；与现有 `torque_chain` 的关键区别之一。
- **mujoco vs PhysX 动力学差异**：两者接触/求解器不同，力矩路径下会引入非执行器来源的差异，计入验收阈值余量。
- **参数漂移**：sim2sim 独立抄常量，训练改 cfg 会漂移；缓解=注释标源+抄写日期，值与训练并列。
