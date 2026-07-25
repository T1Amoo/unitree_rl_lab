# A1 sim2sim `damiao_mit` 保真模式设计（对齐训练 Isaac）

- 日期：2026-07-25
- 分支：`unitree_rl_lab @ g1-tt-sim2sim`（sim2sim 改动）、`Pingpong_TTRL @ a1-tt-migration`（验证脚本、参数源）
- 相关记忆：`a1-tt-real-v7-and-sysid-2026-07-24`、`a1-tt-sim2sim-default-mode`

## 背景与问题

v7 训练用的是自定义显式执行器 `DamiaoMITActuatorCfg`（`Pingpong_TTRL/legged_lab/actuators/damiao_mit.py`），它把「二阶阻尼响应 + MIT-PD 力矩 + 28Nm 力矩限制 + 摩擦 + 力矩低通 + torque-speed 包络」统一进一个执行器。

而 sim2sim（`unitree_rl_lab/deploy/robots/a1_h1/sim2sim/run_a1_tt_sim2sim.py`）默认的 `direct_response` 模式，是「二阶响应算出关节角直接写 qpos/qvel」的理想化链路，与训练执行器在力矩饱和区并不一致；现有 `torque_chain`/`real_deploy_preview` 走的是简化 MIT-PD（纯阻尼 desired_vel=0、无摩擦、无力矩低通、静态 effort clip），也不是训练执行器的复刻。

**目标（对齐训练 Isaac）**：新增一个 sim2sim 模式，使其关节响应与 IsaacLab 训练里 DamiaoMIT 执行器产生的关节响应一致，用于部署前在 mujoco 里预览「训练策略在训练执行器下的行为」。

## 范围

**做**：
- 在 sim2sim 新增 `--actuator-mode damiao_mit`，与现有 4 模式并列。
- 用 numpy **逐行镜像**训练 `DamiaoMIT.compute` 中「得到 `delayed_command` 之前」的前处理链路。
- 把前处理算出的目标关节角**直接写 mujoco `qpos/qvel`**（沿用 `apply_direct_response` 写法），不施加力矩、不跑物理积分执行器。
- 隔离层验证：真实轨迹 + 扫频/阶跃两类输入。

**不做（本次明确排除）**：
- 不复刻力矩层：MIT-PD 力矩、viscous/coulomb 摩擦、力矩低通（`torque_time_constant`）、torque-speed 包络（`_clip_effort`）、effort clip 都不做。接受「高刚度完美跟踪」近似。
- 不改动 `direct_response`（仍是默认模式）。
- 不做端到端（policy+obs）对齐——那是 policy_io 对齐的另一话题，obs/gate/predictor 时序差异（07-14 观察到 Isaac「提前 0.2s」）会污染执行器对齐判断，故隔离。
- 不跨仓库共享代码：前处理在 sim2sim 侧独立 numpy 实现，注释标明镜像来源。

## 架构与数据流

```
raw_command (policy q_cmd, 或验证信号)
  → _apply_response_model   # 二阶: fn/zeta/delay/gain/bias/u_mean → response_command
  → _apply_command_lead     # command_lead_s 前瞻
  → _apply_command_slew     # command_velocity_limit · dt 限速
  → _delayed_command        # command_delay_s 延迟
  = delayed_command
  → 直接写 mujoco qpos；qvel 用相邻步差分
```

与现有 `SecondOrderResponse` 的关键差异（要修正的错序）：
- 现状：先 `max_delta` 限速 → 再二阶响应（delay 埋在 `step()` 内）。
- 训练顺序：先二阶响应 → 再 lead → 再 slew → 再 delay。

## 组件

### 1. 前处理函数（`policy_io.py`）
- 新增函数/类，严格按训练顺序实现上述 4 步，每步注释标明「镜像 `DamiaoMIT.compute` 第 N 行」。
- 二阶响应参数复用现有 `REAL_RESPONSE_*`（已核对与训练 `A1_REAL_FITTED_*` 逐位相同）。
- `command_lead_s` / `command_velocity_limit` / `command_delay_s` 的数值从训练 cfg `A1_TT_REAL_TORQUE_ONLY_CFG`（DamiaoMITActuatorCfg 字段）抄过来固化为常量，注释标明来源与抄写日期。

### 2. sim2sim 接线（`run_a1_tt_sim2sim.py`）
- `--actuator-mode` 的 `choices` 加入 `damiao_mit`。
- `physics_step` 与收尾（现 `isaac_approx`/`direct_response` 分支处）加 `damiao_mit` 分支：调用新前处理 → 写 qpos/qvel（复用 `apply_direct_response` 或平行新方法）。
- 诊断打印（trace/headless 统计）沿用现有字段。

### 3. 验证
- **真实轨迹**：`Pingpong_TTRL/legged_lab/scripts/record_a1_play_trace.py` 录 Isaac play 的 `(raw_command 序列, 实际 q 序列)`；mujoco 用同一 `raw_command` 跑 `damiao_mit`；`compare_sim2real_trace.py` 出 7 关节角对比图 + 每关节 RMSE。
- **扫频/阶跃**：生成标准 q_des（chirp 0.1–3Hz + 阶跃）喂两边，同样出图 + RMSE。
- 输出目录：`系统辨识/sim2real/<日期>/damiao_mit_isaac_align/`（沿用既有结构）。

## 验收标准

- **主判据**（真实轨迹自由段，per-joint RMSE）：J1-3 < **0.02 rad**，J4-7 < **0.03 rad**（参考既有 direct/真机 ~0.03 量级，直接写 q 应更小）。
- **扫频**：无明显相位错位（lead/slew/delay 顺序正确的证据）。
- **附带结论**：本验证同时检验「直接写 q 的高刚度近似」是否成立。若 RMSE 明显偏大且集中在力矩饱和/高速段 → 说明力矩层不可忽略，需回头评估补 torque-speed 包络（**已知后续分支，不在本次范围**）。

## 实现前置核实（TODO，实现第一步）

- 核实 `record_a1_play_trace.py` 是否记录了喂给执行器的 `raw_command`（policy 输出经 action_scale 后的 joint_position_target）与实际 `q` 两列。若缺，先补这两列记录，再做验证。

## 风险与已知取舍

- **近似风险**：直接写 q 忽略力矩动力学，力矩饱和/接触冲击时会与 Isaac 分叉。因对齐目标是 Isaac（非真机）且 v7 已降饱和，通常可接受；由验收「附带结论」兜底暴露。
- **参数漂移风险**：sim2sim 独立抄参数，未来训练侧改 cfg 会漂移。缓解：注释标明来源+抄写日期，抄写值与训练 cfg 并排列出便于核对。
