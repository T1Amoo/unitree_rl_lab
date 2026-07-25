# A1 sim2sim `damiao_mit` 保真模式 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 A1 sim2sim 里新增 `--actuator-mode damiao_mit`，忠实复刻训练 v7 `DamiaoMIT` 执行器的力矩链路（slew → MIT-PD 纯阻尼 → torque-speed 包络 → mujoco 积分），使 mujoco 关节响应对齐 IsaacLab 训练。

**Architecture:** 把训练 `DamiaoMIT` 的 torque-speed 包络与 slew 复刻成 numpy 纯函数，组合进 `A1PolicyIO.apply_damiao_mit`，写 `qfrc_applied` 由 `mj_step` 积分；`run_a1_tt_sim2sim.py` 加一个 actuator 分支。参数从训练 cfg 精确抄写，不复用 sim2sim 现有常量（两处不一致）。

**Tech Stack:** Python、numpy、mujoco、pytest。

## Global Constraints

- 关节顺序固定 `r1..r7`（`RIGHT_ARM_JOINTS`）。所有 7 维数组按此顺序。
- 训练 v7 执行器参数（从 `Pingpong_TTRL/legged_lab/assets/a1/a1.py` + `DamiaoMITActuatorCfg` 精确抄写，关节 r1..r7）：
  - `kp = [300,300,300,120,120,120,120]`
  - `kd = [3.5,3.5,3.5,1.0,1.0,1.0,1.0]`
  - `effort_limit = [28,28,28,8,8,8,8]`（r1-3=28，r4-7=8）
  - `velocity_limit = [8,8,8,20,20,20,20]`
  - `brake_effort_limit = effort_limit`
  - `control_dt = 0.002`；执行器 `response_model_enable=False`、`friction=0`、`torque_time_constant=0`、`command_delay_s=command_lead_s=command_bias_rad=0`、`use_command_velocity=False`、`torque_speed_limit_enable=True`。
- **不加重力补偿**（不叠加 `qfrc_bias`）：让 mujoco 自算重力、MIT-PD 通过 kp 位置误差扛重力，以复现稳态位置误差。这是与现有 `torque_chain` 的关键区别。
- 不改 `direct_response`/`torque_chain`/`real_deploy_preview`。
- sim2sim 时步：`PHYSICS_DT=0.002`，`DECIMATION=10`（50Hz control）。执行器每物理步更新。
- 代码根目录：`unitree_rl_lab/deploy/robots/a1_h1/sim2sim/`（本文件路径均相对此，除非标注 `Pingpong_TTRL/`）。
- 测试运行环境：能 `import mujoco` 且能 import `policy_io` 的 Python。纯函数测试（Task 1/2）只需 numpy；集成 smoke（Task 4）需 mujoco + onnxruntime（沿用现有 sim2sim 运行环境）。测试命令统一用 `python -m pytest`；实现第一步用 `python -c "import numpy, mujoco"` 确认所选解释器。

---

### Task 1: torque-speed 包络纯函数 `damiao_clip_effort`

**Files:**
- Modify: `policy_io.py`（新增顶层函数）
- Test: `test_a1_sim2sim.py`（新增测试函数）

**Interfaces:**
- Produces: `damiao_clip_effort(tau, dq, vel_limit, effort_limit, brake_effort_limit) -> np.ndarray`，全部入参为 shape (7,) 的 `np.ndarray`（float64），返回 clip 后的力矩 (7,)。镜像 `DamiaoMIT._clip_effort`（`torque_speed_limit_enable=True` 分支）。

- [ ] **Step 1: Write the failing test**

在 `test_a1_sim2sim.py` 末尾添加：
```python
def test_damiao_clip_effort_torque_speed_envelope():
    from policy_io import damiao_clip_effort
    vel_limit = np.array([8, 8, 8, 20, 20, 20, 20], dtype=np.float64)
    effort_limit = np.array([28, 28, 28, 8, 8, 8, 8], dtype=np.float64)
    brake = effort_limit.copy()

    # 静止(dq=0): 两侧都用 brake_limit=effort_limit
    tau = np.array([100, -100, 0, 100, -100, 0, 5], dtype=np.float64)
    dq0 = np.zeros(7, dtype=np.float64)
    out = damiao_clip_effort(tau, dq0, vel_limit, effort_limit, brake)
    np.testing.assert_allclose(out, np.array([28, -28, 0, 8, -8, 0, 5], dtype=np.float64))

    # 正速度且加速方向(tau>0): 受 speed_scale 限制; 反向(刹车)用 brake
    # r1: dq=4, vel_limit=8 -> speed_scale=0.5 -> accel_limit=14; 加速 tau=100 -> 14
    # r2: dq=4 -> 反向 tau=-100 -> -brake=-28
    dq = np.array([4, 4, 0, 0, 0, 0, 0], dtype=np.float64)
    tau2 = np.array([100, -100, 0, 0, 0, 0, 0], dtype=np.float64)
    out2 = damiao_clip_effort(tau2, dq, vel_limit, effort_limit, brake)
    assert out2[0] == 14.0
    assert out2[1] == -28.0

    # 负速度: 加速方向是负, 正向是刹车
    # r1: dq=-4 -> speed_scale=0.5 -> accel_limit=14; tau=-100(加速) -> -14; 
    dq3 = np.array([-4, -4, 0, 0, 0, 0, 0], dtype=np.float64)
    tau3 = np.array([-100, 100, 0, 0, 0, 0, 0], dtype=np.float64)
    out3 = damiao_clip_effort(tau3, dq3, vel_limit, effort_limit, brake)
    assert out3[0] == -14.0   # 加速方向受限
    assert out3[1] == 28.0    # 正向刹车用 brake
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_a1_sim2sim.py::test_damiao_clip_effort_torque_speed_envelope -v`
Expected: FAIL with `ImportError: cannot import name 'damiao_clip_effort'`

- [ ] **Step 3: Write minimal implementation**

在 `policy_io.py` 顶层（常量区之后、类定义之前）添加：
```python
def damiao_clip_effort(tau, dq, vel_limit, effort_limit, brake_effort_limit):
    """Mirror DamiaoMIT._clip_effort (torque_speed_limit_enable=True)."""
    vl = np.maximum(vel_limit, 1.0e-6)
    speed_scale = np.clip(1.0 - np.abs(dq) / vl, 0.0, 1.0)
    accel_limit = effort_limit * speed_scale
    brake_limit = np.maximum(brake_effort_limit, effort_limit)
    tau_max = np.where(dq > 0.0, accel_limit, brake_limit)
    neg_abs_limit = np.where(dq < 0.0, accel_limit, brake_limit)
    tau_min = -neg_abs_limit
    return np.clip(tau, tau_min, tau_max)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_a1_sim2sim.py::test_damiao_clip_effort_torque_speed_envelope -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deploy/robots/a1_h1/sim2sim/policy_io.py deploy/robots/a1_h1/sim2sim/test_a1_sim2sim.py
git commit -m "feat(a1-sim2sim): add damiao_clip_effort torque-speed envelope"
```

---

### Task 2: slew 纯函数 `damiao_slew`

**Files:**
- Modify: `policy_io.py`
- Test: `test_a1_sim2sim.py`

**Interfaces:**
- Produces: `damiao_slew(cmd, q_des, vel_limit, dt) -> np.ndarray`，`cmd`/`q_des`/`vel_limit` 为 (7,) float64，`dt` float，返回本步 slew 后的命令 (7,)。镜像 `DamiaoMIT._apply_command_slew`（response/lead/delay 在 v7 均为恒等，故 slew 输入即 raw q_des）。

- [ ] **Step 1: Write the failing test**

```python
def test_damiao_slew_limits_per_step_delta():
    from policy_io import damiao_slew
    vel_limit = np.array([8, 8, 8, 20, 20, 20, 20], dtype=np.float64)
    dt = 0.002
    cmd = np.zeros(7, dtype=np.float64)
    # 目标远大于 max_delta=vel_limit*dt=[0.016,...,0.04,...]
    q_des = np.ones(7, dtype=np.float64)
    out = damiao_slew(cmd, q_des, vel_limit, dt)
    np.testing.assert_allclose(out[:3], 0.016)   # 8*0.002
    np.testing.assert_allclose(out[3:], 0.040)   # 20*0.002
    # 目标在步长内则直达
    cmd2 = np.zeros(7, dtype=np.float64)
    q_des2 = np.full(7, 0.001, dtype=np.float64)
    out2 = damiao_slew(cmd2, q_des2, vel_limit, dt)
    np.testing.assert_allclose(out2, 0.001)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_a1_sim2sim.py::test_damiao_slew_limits_per_step_delta -v`
Expected: FAIL with `ImportError: cannot import name 'damiao_slew'`

- [ ] **Step 3: Write minimal implementation**

在 `policy_io.py`（`damiao_clip_effort` 旁）添加：
```python
def damiao_slew(cmd, q_des, vel_limit, dt):
    """Mirror DamiaoMIT._apply_command_slew (response/lead/delay are identity in v7)."""
    max_delta = vel_limit * dt
    delta = q_des - cmd
    return cmd + np.clip(delta, -max_delta, max_delta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_a1_sim2sim.py::test_damiao_slew_limits_per_step_delta -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deploy/robots/a1_h1/sim2sim/policy_io.py deploy/robots/a1_h1/sim2sim/test_a1_sim2sim.py
git commit -m "feat(a1-sim2sim): add damiao_slew per-step command limiter"
```

---

### Task 3: `A1PolicyIO.apply_damiao_mit` + 常量 + slew 状态

**Files:**
- Modify: `policy_io.py`（常量、`__init__`、`set_initial_state` 或等效重置方法、新增方法）
- Test: `test_a1_sim2sim.py`

**Interfaces:**
- Consumes: `damiao_clip_effort`, `damiao_slew`（Task 1/2）；`self.q_des`（policy 目标）、`self.qpos_addr`/`self.dof_addr`/`self.data`。
- Produces:
  - 常量 `DAMIAO_MIT_KP, DAMIAO_MIT_KD, DAMIAO_MIT_EFFORT, DAMIAO_MIT_VEL, DAMIAO_MIT_BRAKE_EFFORT`（均 (7,) np.ndarray）。
  - `A1PolicyIO.apply_damiao_mit(dt: float) -> np.ndarray`：执行 slew→MIT-PD(无重力补偿)→torque-speed clip，`self.data.qfrc_applied[self.dof_addr] += tau`，返回 tau (7,)。
  - `A1PolicyIO.damiao_cmd`：slew 命令状态，`__init__` 与状态重置时置为 `self.right_q()`。

- [ ] **Step 1: Write the failing test**

```python
def test_apply_damiao_mit_torque_signs_and_limit(tmp_path):
    # 用现有 scene 构建一个 io，验证力矩方向与受限
    import mujoco
    from a1_scene import build_scene_xml, load_scene
    from policy_io import A1PolicyIO, DAMIAO_MIT_EFFORT
    model, data = load_scene(build_scene_xml())
    io = A1PolicyIO(model, data, policy=None)
    q = io.right_q()
    # 目标设在当前位置 +0.5rad(远超步长)，期望力矩为正且不超 effort_limit
    io.q_des = (q + 0.5).astype(np.float64)
    io.damiao_cmd = q.copy()
    tau = io.apply_damiao_mit(0.002)
    assert tau.shape == (7,)
    assert np.all(tau >= -DAMIAO_MIT_EFFORT - 1e-9)
    assert np.all(tau <= DAMIAO_MIT_EFFORT + 1e-9)
    assert tau[0] > 0.0   # 目标在正方向 -> 正力矩
    # qfrc_applied 已写入
    assert np.allclose(data.qfrc_applied[io.dof_addr], tau)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_a1_sim2sim.py::test_apply_damiao_mit_torque_signs_and_limit -v`
Expected: FAIL（`AttributeError: 'A1PolicyIO' object has no attribute 'apply_damiao_mit'` 或 import 常量失败）

- [ ] **Step 3: Write minimal implementation**

在 `policy_io.py` 常量区添加：
```python
DAMIAO_MIT_KP = np.array([300.0, 300.0, 300.0, 120.0, 120.0, 120.0, 120.0], dtype=np.float64)
DAMIAO_MIT_KD = np.array([3.5, 3.5, 3.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
# effort r1-3=28, r4-7=8 (训练 _EFFORT: idx<3 -> 28 else 8). 注意与旧 EFFORT r4 不同。
DAMIAO_MIT_EFFORT = np.array([28.0, 28.0, 28.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float64)
# velocity_limit 训练 _VEL: idx<3 -> 8 else 20. 注意与旧 VEL_LIMIT 不同。
DAMIAO_MIT_VEL = np.array([8.0, 8.0, 8.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float64)
DAMIAO_MIT_BRAKE_EFFORT = DAMIAO_MIT_EFFORT.copy()
```

在 `A1PolicyIO.__init__` 中 `self.servo_q = self.right_q()` 附近添加：
```python
        self.damiao_cmd = self.right_q()
```

在状态重置方法（`policy_io.py` 中把 `self.servo_q = q.copy()` 那段，即设置初始机器人状态的方法）里，紧随 `self.servo_q = q.copy()` 添加：
```python
            self.damiao_cmd = q.copy()
```

新增方法（放在 `apply_mit_pd` 附近）：
```python
    def apply_damiao_mit(self, dt: float) -> np.ndarray:
        """Faithful mujoco replica of the training DamiaoMIT torque chain.

        Mirrors DamiaoMIT.compute for A1_TT_REAL_TORQUE_ONLY_CFG:
        slew(_VEL) -> MIT-PD (desired_vel=0, no gravity ff) -> torque-speed clip.
        """
        q = self.data.qpos[self.qpos_addr]
        dq = self.data.qvel[self.dof_addr]
        self.damiao_cmd = damiao_slew(self.damiao_cmd, self.q_des, DAMIAO_MIT_VEL, dt)
        tau = DAMIAO_MIT_KP * (self.damiao_cmd - q) + DAMIAO_MIT_KD * (0.0 - dq)
        tau = damiao_clip_effort(tau, dq, DAMIAO_MIT_VEL, DAMIAO_MIT_EFFORT, DAMIAO_MIT_BRAKE_EFFORT)
        self.data.qfrc_applied[self.dof_addr] += tau
        return tau
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_a1_sim2sim.py::test_apply_damiao_mit_torque_signs_and_limit -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deploy/robots/a1_h1/sim2sim/policy_io.py deploy/robots/a1_h1/sim2sim/test_a1_sim2sim.py
git commit -m "feat(a1-sim2sim): add A1PolicyIO.apply_damiao_mit torque chain + constants"
```

---

### Task 4: `run_a1_tt_sim2sim.py` 接线 + headless smoke

**Files:**
- Modify: `run_a1_tt_sim2sim.py`（`--actuator-mode` choices、`physics_step` 分支、`velocity_limit()`）
- Test: 手动 headless smoke（无单测；本任务交付物是"能跑通且力矩受限"）

**Interfaces:**
- Consumes: `A1PolicyIO.apply_damiao_mit`（Task 3）。
- Produces: `--actuator-mode damiao_mit` 可用；headless 运行打印统计。

- [ ] **Step 1: choices 加 `damiao_mit`**

修改 `--actuator-mode` 的 `add_argument`（当前 `choices=["isaac_approx", "direct_response", "torque_chain", "real_deploy_preview"]`）：
```python
        choices=["isaac_approx", "direct_response", "torque_chain", "real_deploy_preview", "damiao_mit"],
```

- [ ] **Step 2: physics_step 加分支**

在 `physics_step` 的 actuator 分派处（当前 `if args.actuator_mode in ("torque_chain", "real_deploy_preview"): tau = io.apply_mit_pd(effort_limit())`）改为在其前面加一个独立分支：
```python
        if args.actuator_mode == "damiao_mit":
            tau = io.apply_damiao_mit(PHYSICS_DT)
        elif args.actuator_mode in ("torque_chain", "real_deploy_preview"):
            tau = io.apply_mit_pd(effort_limit())
        elif args.actuator_mode == "direct_response":
            tau = io.estimate_mit_tau(effort_limit())
            io.apply_direct_response(PHYSICS_DT)
            mujoco.mj_forward(model, data)
        else:
            tau = io.estimate_mit_tau(effort_limit())
            servo_vel = VEL_LIMIT if args.fast_servo else ISAAC_PLAY_SERVO_VEL * args.servo_vel_scale
            io.apply_position_servo(PHYSICS_DT, servo_vel, args.servo_tau)
            mujoco.mj_forward(model, data)
```

- [ ] **Step 3: velocity_limit() 对 damiao_mit 用其自身包络（不额外裁速）**

`damiao_mit` 的力矩已含 torque-speed 包络，收尾 `enforce_motor_limits` 只保留位置 clip。修改 `velocity_limit()`：
```python
    def velocity_limit() -> np.ndarray | None:
        if args.no_qvel_clip or args.actuator_mode == "damiao_mit":
            return None
        if args.actuator_mode == "real_deploy_preview":
            return DAMIAO_DQ_LIMIT
        return VEL_LIMIT
```
（`enforce_motor_limits(None)` 仍做 q 的 `q_min/q_max` clip，只是不裁速度——确认 `enforce_motor_limits` 在 `velocity_limit is None` 时跳过速度 clip、仍 clip 位置；若不是，实现时按此语义调整。）

- [ ] **Step 4: Run headless smoke**

Run（用现有 sim2sim 运行环境，替换 `<PY>` 为能 import mujoco+onnxruntime 的解释器；`<POLICY>` 为 v7 导出的 `policy.onnx`）：
```bash
<PY> run_a1_tt_sim2sim.py --policy <POLICY> --actuator-mode damiao_mit --headless-steps 500 --seed 0
```
Expected: 打印 `[a1_sim2sim] headless ... max_abs_tau=<=28.00/28.0`，进程正常退出（不崩、不 NaN）；`max_abs_qvel` 有限。

- [ ] **Step 5: Commit**

```bash
git add deploy/robots/a1_h1/sim2sim/run_a1_tt_sim2sim.py
git commit -m "feat(a1-sim2sim): wire --actuator-mode damiao_mit"
```

---

### Task 5: 隔离验证（真实轨迹 + 扫频）与对齐报告

**Files:**
- Verify/Modify: `Pingpong_TTRL/legged_lab/scripts/record_a1_play_trace.py`（确认/补记 `q_des` 与实际 `q`）
- Create: `deploy/robots/a1_h1/sim2sim/gen_chirp_qdes.py`（扫频/阶跃 q_des 生成器）
- Output: `系统辨识/sim2real/2026-07-25/damiao_mit_isaac_align/`（图 + RMSE json）

**Interfaces:**
- Consumes: Task 4 的 `--actuator-mode damiao_mit`；现有 `compare_sim2real_trace.py`。

- [ ] **Step 1: 核实 trace 列**

Run: `grep -n "q_des\|q_actual\|joint_pos\|writerow\|fieldnames" Pingpong_TTRL/legged_lab/scripts/record_a1_play_trace.py`
Expected: 确认是否已记录喂执行器的 `q_des`（action 经 scale 后的 joint_position_target）与实际关节 `q`。
- 若两列都在 → 进入 Step 2。
- 若缺 → 在 record 脚本里补记这两列（读取 `env` 的 `actions`/`processed action` 与 `robot.data.joint_pos[:, arm_ids]`），重跑一小段确认列存在，再进入 Step 2。

- [ ] **Step 2: 录 Isaac play 真实轨迹**

Run（Pingpong_TTRL 环境）:
```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL
OMNI_KIT_ACCEPT_EULA=YES <PINGPONG_PY> -m legged_lab.scripts.record_a1_play_trace \
  --task=a1_tt_real --predictor --num_envs=1 \
  --load_run <V7_RUN> --checkpoint model_<ITER>.pt \
  --out /tmp/a1_v7_isaac_trace.csv --steps 1500
```
Expected: 生成 `/tmp/a1_v7_isaac_trace.csv`，含 `q_des_1..7` 与 `q_1..7`（实际角）列。

- [ ] **Step 3: mujoco 用同一 q_des 跑 damiao_mit**

用 trace 里的 `q_des` 序列驱动 sim2sim（隔离执行器：喂 q_des，不重跑 policy）。若 `run_a1_tt_sim2sim.py` 无"外部 q_des 序列回放"入口，则在本 step 用一段最小脚本：加载 scene、构建 `A1PolicyIO`、逐 tick `io.q_des = trace_qdes[k]` 后按 DECIMATION 调 `io.apply_damiao_mit(PHYSICS_DT)` + `mj_step`，记录 `io.right_q()`。
```bash
<PY> - <<'PYEOF'
# 读 /tmp/a1_v7_isaac_trace.csv 的 q_des 列, 逐物理步 slew+apply_damiao_mit+mj_step,
# 写 /tmp/a1_v7_mujoco_damiao_mit.csv 的 q 列 (与 Isaac q 对齐比较用).
PYEOF
```
Expected: 生成 `/tmp/a1_v7_mujoco_damiao_mit.csv`。

- [ ] **Step 4: 出对比图 + RMSE**

Run:
```bash
<PY> compare_sim2real_trace.py \
  --isaac /tmp/a1_v7_isaac_trace.csv \
  --mujoco /tmp/a1_v7_mujoco_damiao_mit.csv \
  --out-dir "/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/系统辨识/sim2real/2026-07-25/damiao_mit_isaac_align"
```
（若 `compare_sim2real_trace.py` 现有入参不同，按其实际签名适配；产出为 7 关节角对比图 + per-joint RMSE json。）
Expected: 自由段 per-joint RMSE：J1-3 < 0.03 rad，J4-7 < 0.05 rad。

- [ ] **Step 5: 扫频/阶跃对照 + 附带 direct 对比**

- 新建 `gen_chirp_qdes.py` 生成 chirp 0.1–3Hz + 阶跃的 q_des CSV（fixstand 起点、amp 0.08）。
- 用 Step 3 的方式分别跑 `damiao_mit` 与 `direct_response`（同一 q_des）；再录一段 Isaac 对同一 q_des 的响应（隔离，若可行）。
- 出图对比 `Isaac / damiao_mit / direct_response` 三条，记 RMSE，作为"力矩层是否可省"的附带结论。
Expected: `damiao_mit` 的 RMSE ≤ `direct_response`；扫频高速段可见 torque-speed 包络效应。

- [ ] **Step 6: Commit**

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab
git add deploy/robots/a1_h1/sim2sim/gen_chirp_qdes.py
git commit -m "feat(a1-sim2sim): add chirp q_des generator for damiao_mit alignment"
# 若改了 record 脚本:
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL
git add legged_lab/scripts/record_a1_play_trace.py
git commit -m "feat(a1-sim2sim): record q_des and actual q for executor isolation"
```

---

## Self-Review

- **Spec 覆盖**：范围（新增 damiao_mit、力矩链路复刻、不改默认、隔离验证、附带 direct 对比）→ Task 1-5 全覆盖；两处常量不一致 → Task 3 常量注释显式处理；不加重力补偿 → Task 3 `apply_damiao_mit` 无 `qfrc_bias`；torque-speed 包络 → Task 1。
- **占位符**：Task 5 的 `<PY>/<PINGPONG_PY>/<V7_RUN>/<ITER>/<POLICY>` 是运行时环境/路径变量，非代码占位符，实现首步确认；Step 3/5 的最小回放脚本给了明确职责（读 q_des→逐步 apply→写 q），因需依赖 trace 实际列名，留给实现按列名落地。
- **类型一致**：`damiao_clip_effort`/`damiao_slew` 签名与 Task 3 调用一致；常量名 `DAMIAO_MIT_*` 全程一致；`apply_damiao_mit(dt)` 返回 (7,) 与 physics_step 用法一致。
