# A1 右臂关节辨识测试计划

## 目标

用和 sim2real 部署一致的命令链路测试 7 个右臂关节的频率响应：

```text
本机 a1_tt_fsm_supervisor(TEST) -> /model_action(7维q_des) -> 机器人 inference_arm_control_node
本机 record_fsm_test.py 只订阅 /model_action 和 /right_joint_states，不直接发命令
```

每个关节、每个频率保存一组对齐后的数据表：

```text
target_q, real_q, sim_q
```

第一轮先测 7 个电机在 `0.5 Hz`、`1.0 Hz`、`1.5 Hz` 下的响应，用来估计低频到中频的增益、相位滞后和 RMSE。

## amplitude 是什么

`amplitude` 是正弦目标相对当前中心位姿的峰值偏移，单位是 rad，不是总摆幅。

脚本给第 `i` 个关节发的目标是：

```text
q_target_i(t) = q_center_i + amplitude * envelope(t) * sin(2*pi*freq*t)
```

所以：

- `amplitude=0.12 rad` 表示目标最多偏离中心 `+/-0.12 rad`。
- 峰峰值是 `0.24 rad`，约 `13.8 deg`。
- 单侧最大偏移 `0.12 rad`，约 `6.9 deg`。

对应正弦目标的最大速度大约是：

| 频率 | amplitude=0.12 rad 时目标最大速度 |
|---:|---:|
| 0.5 Hz | 0.38 rad/s |
| 1.0 Hz | 0.75 rad/s |
| 1.5 Hz | 1.13 rad/s |

这明显低于当前部署链路里的速度安全限制，所以第一轮测的是相对线性的跟踪响应，不是电机极限。

## 为什么第一轮正式测试用 0.12 rad

`0.12 rad` 是第一轮辨识的折中值：

1. 足够大，能明显超过编码器噪声、微小静摩擦和关节松动带来的小误差。
2. 足够小，不会让机械臂大范围扫动，降低桌面、拍子、线缆和限位风险。
3. 在 `0.5/1.0/1.5 Hz` 下目标速度不高，便于先看闭环带宽、相位滞后和是否欠阻尼。
4. 这轮目标不是马上打到扭矩-速度饱和，而是先得到稳定的线性区频响。

正式测试前仍然先用 `0.03 rad` 做安全探测。如果 `0.12 rad` 下某个关节明显抖动、撞限位或耦合太强，就退回 `0.08 rad`、`0.05 rad` 或 `0.03 rad`。如果后续要辨识扭矩-速度饱和，再单独设计更大幅值或更高频率的测试，不和第一轮混在一起。

## 安全前置

真机使能前必须确认：

1. 启动机器人侧 `inference_arm_control_node`，真机 kp/kd 使用当前实验参数。
2. 本机启动 `a1_tt_fsm_supervisor`，并设置 `test_enabled:=true`。
3. `test_enabled=true` 时，原来的 R2/play 进入 TEST，不进入 TABLE_TENNIS。
4. TEST 只发 7 维 `q_des`，不发 `dq_des`。
5. 用户先按 FixStand，到 READY 后再按 R2/test。阻尼/急停仍然可以抢占 TEST。
6. 确认机器人网段走有线：

```bash
ip route get 10.1.1.220
# 期望看到: dev enp8s0 src 10.1.1.150
```

真实采集脚本 `record_fsm_test.py` 只记录，不发布 `/model_action`、`/model_control/enable` 或阻尼命令，避免多个节点抢控制权。旧的 `record_real_sine.py` 仍可用于直发对照，但今晚默认不用。

## 数据目录结构

所有系统辨识数据统一放在工作区根目录的 `系统辨识/` 下面：

```text
系统辨识/
  joint1/
    20260713/
      kp120_kd3.5/
        real/
        sim/
        merged/
        summary/
        plots/
        logs/
  joint2/
  ...
  joint7/
```

规则：

1. `joint1` 到 `joint7` 分别对应 7 个右臂关节。
2. 日期目录用 `YYYYMMDD`，例如 `20260713`。
3. 每个实验目录用被测关节的实际增益命名，例如 `kp120_kd3.5`。
4. `real/` 放真机采集 CSV，`sim/` 放仿真复放 CSV，`merged/` 放 target-real-sim 对齐表，`summary/` 放指标汇总，`plots/` 放图。

## 真机测试矩阵

第一轮矩阵：

| 关节 | 频率 | 幅值 | 周期数 |
|---|---:|---:|---:|
| 1-7 | 0.5, 1.0, 1.5 Hz | 0.12 rad | 8 |

安全探测顺序：

1. 先测关节 7：`0.5 Hz`、`0.03 rad`、`3 cycles`。
2. 如果正常，再测关节 1：`0.5 Hz`、`0.03 rad`、`3 cycles`。
3. 如果两者都正常，再跑完整 `0.12 rad`、`8 cycles` 矩阵。

生成完整命令：

```bash
/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/make_test_matrix.py \
  --mkdirs
```

当前先做 joint1、kp=120、kd=3.5 的三频测试，生成命令：

```bash
/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/make_test_matrix.py \
  --joints 1 \
  --kp 120 \
  --kd 3.5 \
  --amplitude 0.12 \
  --mkdirs
```

单次真机测试示例：

```bash
source /opt/ros/humble/setup.bash
source unitree_rl_lab/deploy/robots/a1_h1/install/setup.bash

ros2 launch sim2real_bridge_cpp a1_policy_bridge_cpp.launch.py \
  start_policy_bridge:=false \
  start_fsm:=true \
  start_arm_control:=false \
  start_vrpn_ball_bridge:=false \
  fixstand_use_movej:=false \
  test_enabled:=true \
  test_joint_index:=1 \
  test_freq_hz:=0.5 \
  test_amplitude_rad:=0.12 \
  test_cycles:=8.0

/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/record_fsm_test.py \
  --joint 1 \
  --freq 0.5 \
  --amplitude 0.12 \
  --cycles 8 \
  --output 系统辨识/joint1/20260713/kp120_kd3.5/real/j1_f0.5hz.csv \
  --param-tag kp120_kd3.5
```

执行顺序：

1. 先启动 `record_fsm_test.py`，它会等待 `state=TEST`。
2. 用户按 FixStand，FSM 到 READY 后持续 hold fixstand 位姿。
3. 用户按 R2/test，FSM 以 fixstand 的 `default_q` 为中心发单关节正弦 `q_des`。
4. TEST 完成后自动回 READY，记录器检测到 TEST 结束后落盘。

## 需要辨识或扫描的参数

真实机器人底层 MIT 参数：

- `kps`
- `kds`

当前 `inference_arm_control_node` 是启动时读取 `kps/kds`，所以改真机 kp/kd 需要重启机器人侧节点。采集脚本的 `--real-kp`、`--real-kd`、`--param-tag` 只用于记录元数据，不会动态修改机器人参数。

仿真侧参数：

- `kp`, `kd`：MIT PD 增益。
- `inertia`：等效关节惯量，也对应 Isaac 里的 armature 先验。
- `viscous`：关节粘性阻尼。
- `coulomb`：库仑摩擦。
- `effort-limit`：零速最大扭矩。
- `velocity-limit`：DC motor 扭矩-速度包络里的空载速度。
- `command-delay-s`：命令/通信/控制延迟，用来把相位滞后和阻尼效应分开。

第一轮仿真初始先验：

```text
kp             = [200, 200, 200, 90, 90, 90, 90]
kd             = [3.5, 3.5, 3.5, 0.5, 0.5, 0.5, 0.5]
inertia        = [0.032, 0.032, 0.032, 0.0018, 0.0018, 0.0018, 0.0018]
effort-limit   = [28, 28, 28, 8, 8, 8, 8]
velocity-limit = [8, 8, 8, 20, 20, 20, 20]
```

## 仿真复放和表格合并

对每个真实 CSV，先用同一条目标轨迹复放仿真：

```bash
/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/simulate_motor_response.py \
  --input 系统辨识/joint1/20260713/kp120_kd3.5/real/j1_f0.5hz.csv \
  --output 系统辨识/joint1/20260713/kp120_kd3.5/sim/j1_f0.5hz_sim.csv \
  --motor-mode dc
```

然后合并成一张 target/real/sim 对齐表：

```bash
/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/merge_joint_id_tables.py \
  --real 系统辨识/joint1/20260713/kp120_kd3.5/real/j1_f0.5hz.csv \
  --sim 系统辨识/joint1/20260713/kp120_kd3.5/sim/j1_f0.5hz_sim.csv \
  --output 系统辨识/joint1/20260713/kp120_kd3.5/merged/j1_f0.5hz_merged.csv \
  --summary 系统辨识/joint1/20260713/kp120_kd3.5/summary/j1_f0.5hz_summary.csv
```

三频测试都完成并 merge 后，生成一张从上到下排列的 target-real-sim 图：

```bash
/usr/bin/python3 unitree_rl_lab/deploy/robots/a1_h1/tools/joint_id/plot_joint_id_triplet.py \
  --joint 1 \
  --inputs \
    系统辨识/joint1/20260713/kp120_kd3.5/merged/j1_f0.5hz_merged.csv \
    系统辨识/joint1/20260713/kp120_kd3.5/merged/j1_f1hz_merged.csv \
    系统辨识/joint1/20260713/kp120_kd3.5/merged/j1_f1.5hz_merged.csv \
  --labels "0.5 Hz" "1.0 Hz" "1.5 Hz" \
  --title "joint1 kp120 kd3.5" \
  --output 系统辨识/joint1/20260713/kp120_kd3.5/plots/j1_kp120_kd3.5_target_real_sim.png
```

summary 里重点看：

- `real_gain`, `sim_gain`
- `real_phase_lag_rad`, `sim_phase_lag_rad`
- `real_rmse_to_target`
- `sim_rmse_to_target`
- `real_sim_rmse`

## 拟合顺序

建议按这个顺序排查，不要一开始同时调所有参数：

1. 先用所有频率共有的相位偏移估计 `command-delay-s`。
2. 再用频率升高时的增益衰减拟合 `inertia` 和 `velocity-limit`。
3. 用低频稳态误差、小幅响应死区拟合 `viscous` 和 `coulomb`。
4. 只有在目标速度/幅值足够大、能明显看到饱和时，再拟合 `effort-limit`。
5. 最后再决定是否需要改真实 `kp/kd`，不要把模型误差直接用真机增益硬补。

## 第一轮验收标准

候选仿真参数至少要满足：

1. sim 和 real 的增益曲线在 `0.5/1.0/1.5 Hz` 上趋势一致。
2. 每个关节的 sim 相位滞后和 real 相位滞后大致同量级，优先目标是误差小于约 `20%`。
3. `real_sim_rmse` 相比当前先验在至少 5 个关节上变小，才考虑拿这组参数进入训练。
