# G1 23DoF 乒乓策略部署 — 设计 (项目 A: deploy 管线 + 数值等价验证)

- 日期: 2026-06-03
- 状态: 设计已与用户逐节确认,待 spec 审阅 → writing-plans
- 范围: **项目 A** = 在 unitree_rl_lab 里把 Pingpong_TTRL 训练导出的 G1 23DoF 乒乓 ONNX 策略接成可运行的 deploy 管线,并以**两层数值等价**验证其正确复现训练策略。**不依赖机器人/仿真。**
- 延后(项目 B): unitree_mujoco 可视化(装 mujoco + 场景加球桌/球 + 接 Nokov 球追踪),用户在此做视觉评价。

## 1. 背景

`Pingpong_TTRL`(IsaacLab 2.1.1, 定制 legged_lab env)训练出 G1 23DoF 乒乓策略,eval 时导出 `policy.onnx`(输入 actor_obs 435 = 单帧 87 ×5 历史,输出 23 关节)+ `predictor.pt`。`unitree_rl_lab/deploy` 是生产级 **C++** 真机栈(Unitree DDS SDK + FSM + onnxruntime C++ + yaml 驱动 obs 拼装),**已含 g1_23dof 目标**。目标:复用该框架部署我们的乒乓策略。

关键事实(已核实):
- C++ `ObservationManager` 有 `use_gym_history` 模式(外层历史/内层项 = **frame-major**),与训练 `buffer.reshape([N,H,frame])` **一致**;`actor_obs_history_length=5`。
- obs 项是可扩展注册表(`REGISTER_OBSERVATION`);标准项(base_ang_vel/projected_gravity/joint_pos_rel/joint_vel_rel/last_action)已实现。
- deploy 无 mujoco/sim,走 DDS 连真机或 unitree_mujoco(项目 B)。
- 训练单帧 obs(`compute_current_observations_perception`)= ang_vel3 + projected_gravity3 + joint_pos23 + joint_vel23 + action23 + **ball_perception6 + ball_prediction3 + rel_target_xy2 + heading1** = 87。

## 2. 关键决策(已确认)

- **A. 用框架通用 obs 管理器** + 4 个自定义 `REGISTER_OBSERVATION` 项(不自写拼装器)。
- **两层数值等价**作为"跑通"判据(不需硬件/仿真)。
- 项目 A 的球数据 = **文件回放**(ReplayBallSource);Nokov 实时源属项目 B。
- scale/项顺序/history **由录制脚本从训练 env 自动导出,不手抄**(防转写错误)。

## 3. 设计

### ① 代码落位(都在 `unitree_rl_lab/deploy`,复用 g1_23dof,最小改共享码)
新增:
- `robots/g1_23dof/config/policy/table_tennis/v0/params/deploy.yaml`
- `.../table_tennis/v0/exported/{policy.onnx, predictor.onnx}`(从 Pingpong 拷入)
- `robots/g1_23dof/include/tt_observations.h` — 4 个自定义 obs 项(robot-local 注册,不污染共享 `observations.h`)
- `robots/g1_23dof/include/tt_ball_source.h` — `TTBallSource` 接口 + `ReplayBallSource`(A)
- `robots/g1_23dof/include/tt_predictor.h` — 5 帧球位置环形缓冲 + predictor.onnx 的 OrtRunner
修改(最小):
- `robots/g1_23dof/config/config.yaml` — FSM 加 `TableTennis` 状态(type RLBase, policy_dir 指向 table_tennis/v0)+ 转换键
- `robots/g1_23dof/main.cpp` / `CMakeLists.txt` — include 新头、第二个 ort session 链接
- 原 Velocity 状态不动。

### ② 4 个自定义 obs 项
| 项 | 维度 | 来源 |
|---|---|---|
| `tt_ball_perception` | 6 | ball_source 的 [ball_pos3, robot_pos3](A: 回放含已对齐延迟值) |
| `tt_ball_prediction` | 3 | predictor.onnx(最近5帧球位置→未来球点) |
| `tt_rel_target_xy` | 2 | `[pred_x-0.1, pred_y+0.6] - robot_pos_xy`(照训练 tt_env:502-506) |
| `tt_heading` | 1 | 机器人 heading |
标准段用框架已有项。顺序/scale/history=5/use_gym_history=true 由 deploy.yaml 指定,数值由 ⑥ 录制脚本导出。

### ③ predictor → ONNX
predictor = 64-64 MLP,输入 15(最近5帧球位置 `[p_{t-5}…p_{t-1}]` flatten,顺序照 runner `hist.permute(1,0,2).reshape`)→ 输出 3(未来球点 env-local)。从 ckpt `pred_state_dict` 重建 `_MLPPredictor` → `torch.onnx.export` 成 `predictor.onnx`([1,15]→[1,3])。C++ 第二个 OrtRunner 加载;`tt_predictor.h` 维护缓冲每步推理。

### ④ deploy.yaml(table_tennis/v0)
- `step_dt: 0.02`,`use_gym_history: true`
- `observations`(严格照训练单帧顺序): base_ang_vel(1.0) / projected_gravity(1.0) / joint_pos_rel(1.0,23) / joint_vel_rel(1.0) / last_action(1.0) / tt_ball_perception(0.007,6) / tt_ball_prediction(ball_pos scale,3) / tt_rel_target_xy(robot_pos scale,2) / tt_heading(1.0) — 每项 history_length:5
- `actions.JointPositionAction`: scale 0.25, offset=default_pos
- `joint_ids_map / stiffness / damping / default_joint_pos`: 填全(**只在项目 B 执行用;A 的数值等价在策略关节序、不经此**)
- **所有 scale 与项顺序由 ⑥ 录制脚本从训练 env 导出填入,不手抄。**

### ⑤ 球数据接口(A=回放)
抽象 `TTBallSource: get(step)->{ball_pos3, robot_pos3, heading}`。
- A: `ReplayBallSource` 读录制文件逐步回放(含已对齐延迟)。
- B(后续): `NokovBallSource` 实时动捕。
obs 项只依赖接口,A→B 换实现不改 obs。

### ⑥ 录制 + 等价校验
- 录制脚本(Python/Pingpong):IsaacLab eval(1 env、确定性、关 perception 随机延迟),逐步 dump 原始输入(球/躯干位姿、heading、joint_pos/vel、last_action、ang_vel、grav、predictor 5 帧球缓冲)+ 拼好 `actor_obs(435)` + `action(23)` + obs_scales/顺序/history 元数据 → `.npz`。
- predictor 校验:torch predictor vs predictor.onnx 同输入 <1e-5。
- Tier-1(onnx):录制 actor_obs(435)→C++ OrtRunner(policy.onnx)→action,比录制 action <1e-4。
- Tier-2(拼装):录制原始输入→C++ obs 项+predictor.onnx→拼出 obs,比录制 actor_obs <1e-4。历史顺序(oldest/newest)若反 Tier-2 抓到,改 C++ 填充序。
- deploy 二进制加 replay/test 模式:吃 ReplaySource+录制文件,逐步 dump obs+action,Python 比对。

### ⑦ 验证判据("跑通"=全过)
1. deploy 二进制编译链接通过(onnxruntime + unitree_sdk2)
2. predictor.onnx ≡ torch predictor(<1e-5)
3. Tier-2: C++ 拼装 obs ≡ IsaacLab actor_obs,max|Δ|<1e-4(N 步)
4. Tier-1: C++ policy.onnx action ≡ IsaacLab action,max|Δ|<1e-4
5. replay 端到端跑完不崩,输出 23 个合理关节目标

## 4. 延后(项目 B,另立 spec)
装 unitree_mujoco;往 mujoco 场景加球桌+球;`NokovBallSource` 接动捕;真机 joint_ids_map/kp-kd 实测校验;用户在 mujoco 做视觉评价。

## 5. 风险/注意
- **obs 顺序/scale/历史堆叠**是头号风险——靠"录制导出 + 两层等价"消解(不手抄、机器比对)。
- perception 训练有随机延迟——录制时关闭/对齐,保证确定性等价。
- IsaacLab 版本差(训练 2.1.1 / 框架 2.3.0):deploy 经 ONNX,版本无关。
- joint_ids_map/kp-kd 正确性留到项目 B 真机/仿真验证。
