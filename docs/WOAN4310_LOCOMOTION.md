# WOAN4310 四足行走训练初版

任务 ID：`Unitree-WOAN4310-Velocity`

## 初版边界

- 仿真框架：当前仓库的 Isaac Lab `ManagerBasedRLEnv` + RSL-RL PPO。
- 资产：WOAN_GYM `dog_V2` 的 1 个 URDF 和 21 个必要 STL，使用仓库内相对位置，不依赖外部 `unitree_ros`。
- 控制：12 关节位置目标，`action_scale=0.25`，20 ms policy step（5 ms physics step、decimation 4）。
- 电机基线：12.5 Nm、30 rad/s、Kp 12.5、Kd 0.25，来自 WOAN4310 原任务配置。
- 初始姿态：髋侧摆 0、髋俯仰 0.8、膝关节 -1.5 rad，base 初始高度 0.42 m。
- PPO：沿用当前 Unitree locomotion 的 512/256/128 ELU 网络与 PPO 参数，仅使用独立日志目录 `woan4310_velocity`。

## 奖励基线

初版采用仓库中 `Unitree-Go2-Velocity` 的奖励结构，避免直接移植旧 IsaacGym 任务中重复定义、绑定旧索引的奖励实现：

- 速度跟踪：平面线速度 1.5，偏航角速度 0.75；
- 稳定性：竖直速度、横滚/俯仰角速度、平姿态惩罚；
- 动作质量：关节速度/加速度、力矩、功耗、动作变化率和关节限位惩罚；
- 步态质量：足端腾空时间、四足腾空/接触时间方差、足端滑动；
- 安全接触：`Link_(ZQ|ZH|YQ|YH)[34]` 非足端接触惩罚；base 接触或姿态倾覆终止。

命令课程从 ±0.1 m/s 的低速范围开始，逐步扩展到前后 ±1.0 m/s、横向 ±0.4 m/s、偏航 ±1.0 rad/s。

## 运行

先安装/更新 editable package：

```bash
./unitree_rl_lab.sh -i
```

确认任务注册：

```bash
./unitree_rl_lab.sh -l
```

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
  --num_envs 4096
```

查看策略：

```bash
python scripts/rsl_rl/play.py --task Unitree-WOAN4310-Velocity
```

## 已完成验证

2026-08-24 在 Isaac Sim 5.1.0 / Isaac Lab v2.3.2 环境完成了 16 env、1 iteration 的 headless smoke：

- URDF 成功转换并创建 16 个并行环境；
- action 维度 12，actor observation 维度 45，critic observation 维度 60；
- 16 个奖励项、3 个终止项和 2 个 curriculum 项全部解析；
- 完成 384 timesteps 和一次 PPO 更新，无 NaN/Inf；
- 成功导出 `deploy.yaml`，step dt 为 0.02 s，12 个关节的 Kp/Kd 为 12.5/0.25。

这仍只是结构和数值通路验证，不代表策略已经学会站立或行走。

## 下一轮建议按实测曲线讨论

1. 若先学会站立但速度跟踪差，提高速度跟踪权重或放慢命令课程。
2. 若出现跳跃式前进，降低足端腾空奖励并增加动作变化率/竖直速度惩罚。
3. 若拖脚，调足端滑动权重并增加与该机腿长匹配的 clearance 项。
4. 若高频抖动，先检查 Kp/Kd、action scale 和关节速度，再调 PPO 熵或学习率。
5. 在 controller joint 顺序、关节正方向和实机扭矩边界复核前，不进入 sim2real。
