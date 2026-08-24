# WOAN4310 四足行走训练初版

任务 ID：`Unitree-WOAN4310-Velocity`

## 初版边界

- 仿真框架：当前仓库的 Isaac Lab `ManagerBasedRLEnv` + RSL-RL PPO。
- 资产：WOAN_GYM `dog_V2` 的 1 个 URDF 和 21 个必要 STL，使用仓库内相对位置，不依赖外部 `unitree_ros`。
- 控制：12 关节位置目标，`action_scale=0.25`，20 ms policy step（5 ms physics step、decimation 4）。
- 电机基线：12.5 Nm、30 rad/s、Kp 12.5、Kd 0.25，来自 WOAN4310 原任务配置。
- 初始姿态：髋侧摆 0、髋俯仰 0.8、膝关节 -1.5 rad，base 初始高度 0.30 m。该高度由足端几何和带界面落地检查确定，避免源任务 0.42 m 带来的约 13 cm reset 自由落体。
- PPO：沿用当前 Unitree locomotion 的 512/256/128 ELU 网络与 PPO 参数，仅使用独立日志目录 `woan4310_velocity`。

## 奖励基线

初版采用仓库中 `Unitree-Go2-Velocity` 的奖励结构，避免直接移植旧 IsaacGym 任务中重复定义、绑定旧索引的奖励实现：

- 速度跟踪：平面线速度 1.5，偏航角速度 0.75；
- 稳定性：竖直速度、横滚/俯仰角速度、平姿态惩罚；
- 动作质量：关节速度/加速度、力矩、功耗、动作变化率和关节限位惩罚；
- 步态质量：足端腾空时间、四足腾空/接触时间方差、足端滑动；
- 安全接触：`Link_(ZQ|ZH|YQ|YH)[34]` 非足端接触惩罚；base、`Link_*3` 接触或姿态倾覆终止。

线速度和偏航命令都从 ±0.1 的低速范围开始，分别逐步扩展到前后 ±1.0 m/s、横向 ±0.4 m/s、偏航 ±1.0 rad/s。

首版刻意不加入 base-height 和 foot-clearance 奖励；先用已验证的 Unitree Go2 奖励骨架得到可解释基线，再根据训练行为增补。

## 运行

先安装/更新 editable package：

```bash
./unitree_rl_lab.sh -i
```

确认任务注册：

```bash
./unitree_rl_lab.sh -l
```

正式训练前先做带界面、确定性的零动作落地检查：

```bash
DISPLAY=:1 PYTHONPATH="$PWD/source/unitree_rl_lab" \
python scripts/inspect_woan4310.py \
  --mode stance \
  --steps 500 \
  --real-time \
  --device cuda:0
```

再用固定基座的纯运动学脉冲核对三个关节族的正方向：

```bash
DISPLAY=:1 PYTHONPATH="$PWD/source/unitree_rl_lab" \
python scripts/inspect_woan4310.py \
  --mode joint-directions \
  --pulse 0.2 \
  --device cuda:0
```

检查结果和截图写到被 Git 忽略的 `logs/inspection/woan4310/`。

小规模 smoke（验证环境能创建和迭代，不代表正式训练）：

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-WOAN4310-Velocity \
  --num_envs 16 \
  --max_iterations 1
```

正式基线训练：

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-WOAN4310-Velocity \
  --num_envs 4096 \
  --seed 42 \
  --experiment_name woan4310_velocity_baseline_v1 \
  --run_name env4096_seed42
```

查看策略：

```bash
python scripts/rsl_rl/play.py --task Unitree-WOAN4310-Velocity
```

## 已完成验证

2026-08-24 在 Isaac Sim 5.1.0 / Isaac Lab v2.3.2 环境完成了带界面资产检查：

- 原始 0.42 m reset 高度会造成约 13 cm 自由落体；零动作试验最低 base z 为 0.177 m，最大横滚/俯仰为 0.288 rad；
- 修正为 0.30 m 后，10 秒零动作试验无 termination，最低 base z 提升到 0.221 m，最大横滚/俯仰降到 0.150 rad；
- 四足均稳定接触，最终 base pitch 约 -0.104 rad；该低刚度合规形变留给策略用非零 action 补偿，不修改 Kp/Kd 或默认关节角；
- 对髋侧摆、髋俯仰和膝关节分别施加 +0.05 rad 运动学脉冲，12 个关节均精确命中目标，四足位移对称且方向与 URDF FK 一致。

修正后也重新完成了 16 env、1 iteration 的 headless smoke：

- URDF 成功转换并创建 16 个并行环境；
- action 维度 12，actor observation 维度 45，critic observation 维度 60；
- 16 个奖励项、4 个终止项和 3 个 curriculum 项全部解析；
- 完成 384 timesteps 和一次 PPO 更新，无 NaN/Inf；
- 成功导出 `deploy.yaml`，step dt 为 0.02 s，12 个关节的 Kp/Kd 为 12.5/0.25。

这仍只是结构和数值通路验证，不代表策略已经学会站立或行走。

## 下一轮建议按实测曲线讨论

1. 若先学会站立但速度跟踪差，提高速度跟踪权重或放慢命令课程。
2. 若出现跳跃式前进，降低足端腾空奖励并增加动作变化率/竖直速度惩罚。
3. 若拖脚，调足端滑动权重并增加与该机腿长匹配的 clearance 项。
4. 若高频抖动，先检查 Kp/Kd、action scale 和关节速度，再调 PPO 熵或学习率。
5. 在 controller joint 顺序、关节正方向和实机扭矩边界复核前，不进入 sim2real。
