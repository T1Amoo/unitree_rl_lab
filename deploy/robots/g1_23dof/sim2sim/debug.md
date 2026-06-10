# G1 23-DoF Table-Tennis — Debug & Data Log

> 滚动记录调试过程、关键数据、决策。最近更新:2026-06-08。

---

## 0. 系统总览(sim2sim 管线)

真实 C++ 部署栈(`unitree_rl_lab`)通过 DDS 驱动 Python `unitree_mujoco` 里的 G1;球 + 底座位姿走 ROS2 PoseStamped(模拟 mocap)。

- **训练端**:`Pingpong_TTRL`(IsaacLab 2.1.1,legged_lab TTEnv,RSL-RL PPO + OnPolicyPredictorRegressionRunner)。conda env=`pingpong`。
- **部署/仿真端**:`unitree_rl_lab/deploy/robots/g1_23dof`。conda env=`g1tt_sim2sim`。
- **频率**:physics 500Hz,LowState 500Hz,mocap 250Hz,policy 50Hz(step_dt 0.02),FSM/LowCmd 1000Hz。
- **关节序(23)**:0-5 左腿,6-11 右腿,12 waist_yaw,13-17 左臂,18-22 右臂。joint_ids_map=identity。
- **predictor**:50Hz 训练,输入 5×3=15 维(5 个历史球位,20ms 间隔),输出未来球点 3 维。

启动:
- 终端 A `sim2sim/run_sim.sh`(mujoco+bridge+发球+ROS2);终端 B `sim2sim/run_deploy.sh`(C++ g1_ctrl,从 config.yaml 的 policy_dir 加载 onnx)。
- MuJoCo 窗口按键:`f`=FixStand `g`=TableTennis `p`=Passive `7/8`=升降弹力带 `9`=收放 `q`=退出。摔了:`p`→`f`。

---

## 1. 周末长训 run `g1_tt_weekend`(2026-06-05 → 06-08)

- run 目录:`Pingpong_TTRL/logs/g1_tt_weekend/2026-06-05_19-16-19/`(全新开训,旧 `g1_table_tennis/2026-06-03`=model_9000 保留)。
- 4096 env,~5.4-5.7s/iter,扛过整个周末**无崩溃、笔记本没睡、一次 fresh start**,停在 **iter 45500**,456 个 ckpt(save_interval=100)。watchdog PID 存 `/tmp/g1_watchdog.pid`。

### 本次相对旧训练的改动(g1_tt_config.py / tt_env.py / tt_config.py)
| 项 | 内容 |
|---|---|
| **发球课程** | `serve_curriculum_steps=300000` 控制步(≈iter 12500)内,从配置易范围线性加宽到 `_wide`。c=sim_step_counter/300000。⚠️续训会让 sim_step_counter 归零→课程重头。|
| _wide 目标 | x(-7.0,-4.5) y(-1.1,0.7) z(1.3,2.3) pos_y(±0.3) **pos_z(-0.10,+0.13)新增发球高度** |
| 易起步(base) | x(-6.5,-5.0) y(-0.8,0.4) z(1.5,2.0) pos_y(±0.1) |
| **随机化** | perception_delay 2-15 步(4-30ms,原 2-5);启用 action_delay 1-3 步。(针对 sim2sim 观测到的感知延迟主因)|
| eval 变体 | `serve_curriculum_steps=0` 固定分布,公平比 ckpt |

---

## 2. Eval 方法与结果

- 命令:`python -m legged_lab.scripts.eval --task=g1_tt_eval --predictor --headless --seed N --load_run 2026-06-05_19-16-19 --checkpoint model_X.pt --num_envs 100`
- eval **是确定性的**(固定 seed→同发球序列→同结果)。换 seed 复测可排除「对单条序列过拟合」。
- ⚠️ **8GB 卡同时只能跑一个 eval/IsaacSim**,并发会 OOM/超时(本次就因并发污染过 2 个结果)。

### 全 ckpt 成功率(eval 固定易分布,~6300 发球/ckpt)
| ckpt | success(默认seed) | success(seed7777) | hit | mean_ep_len(训练全宽) |
|---|---|---|---|---|
| 13000 | 0.374 | — | 0.92 | 297 |
| 19000 | 0.596 | — | 0.97 | 308 |
| 22000 | 0.483 | — | 0.97 | ~327 |
| **22500** | ~0.50 | — | 0.97 | **329.6(最稳)** |
| 23000 | 0.544 | — | 0.98 | ~325 |
| 24000 | 0.337 | — | 0.95 | 327 |
| 30000 | 0.402 | — | 0.98 | 259 |
| 35000 | 0.707 | 0.726 | 0.98 | 223 |
| 40000 | 0.730 | 0.747 | 0.97 | 211 |
| 45000 | 0.582 | 0.604 | 0.97 | ~215 |
| **45500** | **0.782** | **0.788** | 0.97 | 217 |

---

## 3. 关键发现

### ⭐ 训练日志的 reward 是误导指标,挑 ckpt 只能看 eval 真成功率
- `reward_table_success` 等是 rsl_rl **塑形/归一化后的 episode 奖励均值**,被回合长度等混淆。
- 它显示「峰值22-24k → 退化到0.15」,但 eval 真成功率结论**相反**:后段 35k-45.5k 反而最高,**45500=0.785 是全场最佳**且跨 seed 稳。
- 后段是**剧烈震荡**(45000=0.59 vs 45500=0.78,500iter 差 20pp),非单调。

### ⭐ 总 Mean reward 从 ~iter13000 一路下滑 ≠ 策略变差(Q1 分析)
数据指向「后期摔得更多」:
| 信号 | iter 24000(reward最好) | iter 45500 | 变化 |
|---|---|---|---|
| 平均回合长度 | 311 | 217 | **-31%(提前终止=摔)** |
| termination_penalty | -0.06 | -0.60 | **×10** |
| action_rate_l2 | -1.00 | -0.68 | 一直很大(动作激烈)|
| 任务结果项(contact/landing/success) | 强 | 依旧强 | — |
| 任务塑形项(future_vel_base/dis_ro) | 0.56/0.35 | 0.39/0.24 | 掉~30% |

解读:课程拉满后策略为够刁钻球**越来越激进**——普通球回得更成功(eval↑),但极端球够过头摔倒(回合短、终止惩罚涨)。训练 reward 在全宽难分布(含摔)上算所以下滑;eval 在较易分布上算所以上升。

### ⭐ 稳定性 / 激进度 权衡(真机关切)
- **45500**:success 0.78,但回合最短(217)、动作最激进(action_rate 大)、摔得多 → sim2sim 稳,但**真机有风险**。
- **22500**:回合最长(330,最稳)、最不激进,success~0.50 → 保守稳健,适合真机首测打底。
- 后续计划:**调高 termination/稳定性权重 或 降 action_rate**,再(正确)续训,挑又稳又高的 ckpt。

---

## 4. 部署 policy 目录(`config/policy/table_tennis/`)

`config.yaml` FSM.TableTennis.policy_dir 指向其一。State_TableTennis 加载 `<dir>/params/deploy.yaml` + `<dir>/exported/{policy.onnx, predictor.onnx}`。**onnx/yaml 运行时从源码树读,改 policy_dir + 放好 onnx 即生效,无需重编译**(CMake 不 copy config)。

| dir | 模型 | 说明 |
|---|---|---|
| v0 | 旧 model_9000(2026-06-03)| 原始,保留不动 |
| v1 | **model_45500** | weekend 最强(凌空 eval 0.78),激进 |
| v2 | **model_22500** | weekend 最稳(回合 330),保守 |
| v3 | **model_36000(rally)** | ⭐**落台发球版,eval 0.74 跨seed稳,当前部署用这个** |

- 切换:改 `config.yaml:72` 的 `policy_dir`。
- deploy.yaml 在不同 ckpt 间通用(obs 结构/关节序/增益没变,只换权重)。
- 导出:`policy.onnx` 由 eval.py 启动时导到 run 目录 exported/(需跑一次 eval);`predictor.onnx` 由 `legged_lab/scripts/export_predictor_onnx.py --ckpt <pt> --out <onnx>`(纯 torch,不需 IsaacSim)。

---

## 5. sim2sim 故障与修复史

- **lowcmd 通道被占 / 段错误**:残留 g1_ctrl 进程占着 DDS lowcmd 通道。`run_deploy.sh` 原 `pkill -f "build/g1_ctrl"` 漏杀以 `./g1_ctrl` 启动的实例 → 已改 **`pkill -9 -x g1_ctrl`**(按二进制名精确匹配)。手动清:`kill -9 $(pgrep -x g1_ctrl)`。
- **bridge 发布全 0**:MJCF 缺 `<sensor>` 块(bridge 按位置读 sensordata);加 joint+IMU+frame_pos 传感器。IMU 还需名为 `frame_pos` 的传感器才开门。
- **抖动根因**:MJCF 缺 `armature`(转子惯量),显式积分器发散 → 加 IsaacLab armature 后 FixStand jvel 179→6.8;再加 joint `damping=1.0`+`frictionloss=0.1` 抑制滞后振荡。
- **libfmt.so.11 not found**:run_deploy.sh 导出 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`。
- **球拍 248m 红平面**:tt_paddle.stl 单位是 mm → scale 0.001。
- **球拍碰撞**:重建为 URDF 链 + 去掉右橡胶手;厚 puck(cyl 0.02)+ contact margin 防快球穿透;碰撞盘 quat 单位阵(法向=拍面)。
- **追地上的球摔倒**:训练 z<0.1 重发;部署加 `mask_invalid` 门(ball x<-1.9 / vx>0.3 / z<0.7 / (x<-1.35&vz<0) → hold ready 不追)。球停腿上不重置 → 加 settled(|vel|<0.3)/timeout 重发。
- **弹力带大风车**:改 6-DOF(力+角度恢复),锚点在机器人起点上方 z=1.5。

### 已知 sim2sim↔IsaacLab 差异(非 bug)
右臂动作 mujoco 比 IsaacLab 更「扣球」:(a) IsaacLab eval 同步零延迟慢放 vs mujoco 实时带延迟;(b) 快速击球臂对延迟敏感。腿+左臂一致(策略可迁移)。本次训练已把 perception_delay 加宽到 4-30ms 来缓解。

---

## 6. ⭐球拍是物理建模的,上台是真物理击球(纠正一个误解)

训练机器人资产 = `g1_description/g1_23dof_tt_paddle.urdf`(`legged_lab/assets/unitree/g1.py:23`),**球拍有完整碰撞体**:adapter mount cyl(0.012)→ 杆 cyl(r0.018 L0.16)→ holder box → **拍盘 = cylinder r0.075 L0.005 @末端 ~0.337m**(+ 拍柄 box)。全局 `restitution=0.8`(tt_env.py:169),球是真物理刚体,**真撞拍面弹**;击球后无脚本改球速。

- `paddle_offset=(0.337,0,0)` + `contact_threshold=0.05`(tt_env.py:875-902)只是 **proximity 整形奖励**(引导靠近球),**不是物理**。
- **回球是物理弹**,方向由拍面朝向(肩/肘/腕roll)物理决定;`reward_future_landing_dis / pass_net / table_success` 都依赖这条物理轨迹 → **拍面朝向是被这些项监督学到的**。所以 ~0.3-0.4 上台率 = 真物理击球,拍面朝向能调也学了。
- ⚠️ 细节:proximity 奖励的偏移点在 roll 轴(X)上 → proximity 对 wrist_roll 无梯度;roll 仅靠回球质量项(落点/过网/上台)监督,**信号比肩/肘稍弱**。要更利落的拍面控制可加朝向奖励项。
- 默认姿态 FK:腕在 pelvis 前0.04/右0.20/上0.03;小臂轴朝前下45°;拍面法向≈腕 -Y=**机器人右侧(外侧)略朝上**。

---

## 7. RALLY 续训(g1_tt_rally,从 22500 warm-start,2026-06-08)

**动机**:旧训练发球偏「远台 + 出界凌空」,中台薄、近台空(见 §3 落点表)。改成**正规比赛式落台球**:每球必落机器人半场(无凌空),覆盖中台+远台。横向**直接给满**(22500 已在周末版完整展开的分布上训过、有跑动能力,无需再课程展开;唯一新东西是「球先弹一下」,课程帮不上)。

**改动**(只改发球分布,不动奖励——隔离变量):
- `tt_env.py reset_ball`:新增**按落点采样**路径(`serve_bounce_enable`)。采样目标落点 (x_b, y_b) + vz,反推 vx/vy/vz(抛体,launch (1.35,0,1.03),弹跳 z=0.78,g=9.81),保证落台不出界、过网余量 0.14–0.28m。旧 speed-curriculum 路径保留作 fallback。
- `tt_config.py BallCfg`:`serve_bounce_enable / serve_bounce_x_range(-1.25,-0.65) / serve_bounce_vz_range(1.5,1.9) / serve_y_start(0.05) / serve_y_wide(0.65)`。
- `g1_tt_config.py`:启用 bounce 发球,**横向给满 ±0.60、`serve_curriculum_steps=0`(无课程)**,保留 perception/action delay。experiment_name → **g1_tt_rally**。
- warm-start:`logs/g1_tt_rally/seed_22500/model_22500.pt`(从 weekend 拷),train.py `--resume --load_run seed_22500 --checkpoint model_22500.pt` 从 iter 22500 续(curriculum sim_step_counter 从 0 重来 = 先居中)。
- watchdog:`legged_lab/scripts/watchdog_train_rally.sh`(EXP=g1_tt_rally,TARGET=45000,日志 train_rally_watchdog.log)。
- 已 5-iter 验证 warm-start 加载 + 落点采样无报错,验证 run 归档于 `logs/_smoke_verify/`。

**启动**:`nohup bash legged_lab/scripts/watchdog_train_rally.sh >/dev/null 2>&1 &`(PID 即 watchdog,kill 它停全部)。
**注意**:训完后部署前,要把 **sim2sim 的 serve.py 也改成同样的落点采样**(现在是临时的中部弹球),才与训练分布一致。

### phase-1 结果(iter 22500→~25700,2026-06-08)
落台球 fine-tune **效果好**:Mean reward 转正(+5~7,周末版是 -5~-8)、回合长 ~355(很稳、几乎不摔)、reward_contact ~0.71、reward_table_success ~0.37(满 ±0.60 横向)。warm-start 前 1-2k iter 就吃下了弹跳轨迹。

### phase-2 进阶课程(iter 25700→37500,过夜 ~14h)
phase-1 已收敛,改用剩余算力**逐渐加难**(从 25700 续):`reset_ball` 的 bounce 路径现在对 x/vz 范围也做课程 lerp(`_bclerp`,易→hard,c=sim_step_counter/serve_curriculum_steps)。
- 易起步(=phase-1):x(-1.25,-0.65) vz(1.5,1.9) y±0.60
- hard 目标:x(-1.35,-0.60) vz(1.3,2.1) y±0.72(更深+更短、更平快+更高慢、打边角),仍中远台无近台
- `serve_curriculum_steps=180000`(~7500 iter 到满难度),TARGET=37500。
- 早上 eval 在「易→难」整条 ckpt 谱系里挑最佳。

### rally phase-2 结果(2026-06-09 跑完 iter 37499,119 ckpt)
落台发球 eval(g1_tt_eval,c=0 易球档,~6300 发球)成功率:26000=0.66 / 28000=0.71 / 30000=0.68 / 32000=0.72 / 34000=0.70 / **36000=0.74** / 37499=0.71。换 seed 复测 top3:32000≈0.72、37499≈0.71、**36000≈0.74(默认0.737/seed7777 0.741)**。**最佳=model_36000**(满难度训练、跨seed稳)→ 导出到 **v3** 部署。命中率全程 0.95-0.97。
⚠️ watchdog off-by-one:max_iter=N → 只到 model_(N-1),TARGET=N 永远不满足会死循环重启。下次 TARGET 设 N-1 或改判据 `>= TARGET-1`。
sim2sim `serve.py` 已改成落台发球(BOUNCE_X(-1.25,-0.65)/BOUNCE_VZ(1.5,1.9)/BOUNCE_Y±0.55,反推 vx/vy/vz),与 rally 训练分布一致。
真拍硬件:9cm 连接件 → 击球点 ~30cm(轴心起)vs 训练 33.7cm,**误差 3.7cm**,在拍盘半径 7.5cm 内、平拍反弹方向不变 → **直接部署,不用重训/微调**。

### ⚠️ action_rate 尖峰 = 任务固有噪声,别误判为发散(2026-06-08)
phase-2 训练日志偶尔出现 `action_rate_l2` 蹦到 -400~-9979、`value_function loss` 上千万、`Mean reward` -18000+ 的吓人数值。**这不是发散**,是少数 env 在刁钻落点上扑救失衡乱抖一下随即恢复的固有噪声。验证:读 phase-1 tfevents(`2026-06-08_12-58-03`)发现**phase-1 从 iter 22500 就有同样尖峰,5% 尖峰率、中位 action_rate -1.07 健康**,而 phase-1 正是产出好用的 model_25700 的 run。判据:看 **中位数 action_rate(应 ~-1.0~-1.3)+ 成功率/回合长是否稳定**,别看含尖峰的 Mean reward/value_loss。fresh-optimizer(LOAD_OPTIMIZER=0,已加到 train.py 的 env 开关)不解决尖峰(证明非 optimizer 问题)。读 tfevents 命令见下。

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea=EventAccumulator("logs/g1_tt_rally/<run>"); ea.Reload()
[ (s.step,s.value) for s in ea.Scalars('Episode_Reward/action_rate_l2') ]
```

## 9. PerceptionTracker — mocap 感知鲁棒层(2026-06-09)

部署端策略原本直接吃 raw mocap + 单帧 vx/vz 门 → 球落地/击球/飞出桌时门逐帧抖动 → obs 跳变 → 策略抽风。新增 `include/perception_tracker.h`(纯 C++/Eigen,无 ROS,可独立单测)夹在 `RosBallSource` 和策略之间,产出干净连续的球/基体 + 带迟滞的 engage。

**防御层**(spec/plan 在 `docs/superpowers/{specs,plans}/2026-06-09-perception-tracker*`):
- 播放体积门 + 速度门;KF(常速+重力)平滑+短丢外推;track-continuity 选点拒反光;
- 死球检测(出界/飞走/桌上滚动/静止);双弹检测(己方半场弹≥2 未击中);凌空/出界拒(弹道预测首落点须在己方半场);
- 迟滞状态机(confirm K=3 / coast M=8);基体 track(低通+跳变拒+丢失退出);
- **engage = 球live AND 基体valid**;脱离时喂固定 ready 点(hold,不抖)+ engage 边沿 ramp;
- mocap 消息**陈旧检测**(`tt_ros_ball_source.h`,STALE=50ms → 真 has_ball/has_base);基体丢失→退 hold,机器人靠本体感知**不摔**。

**安全不变量**:平衡只用本体感知(IMU+编码器,LowState),不依赖 mocap → mocap 全丢也不倒,只 hold 等恢复。

单元测试:`test/test_perception_tracker.cpp`(11 个,assert),编译:
`g++ -std=c++17 -O2 -I include -I $CONDA(g1tt_sim2sim)/include/eigen3 test/test_perception_tracker.cpp -o /tmp/tpt && /tmp/tpt`。
sim2sim 故障注入:`TT_FAULT=drop/jitter/baseocc bash sim2sim/run_sim.sh`(ros_publish.py)。
未做:`set_paddle_hit` 未接(deploy env 无该信号,双弹靠计数仍 work);`false`(假点)注入待真多刚体 mocap。

## 11. IDLE 重训(无球站稳)— 见 spec `docs/superpowers/specs/2026-06-09-unified-validity-gate-idle.md`

根因=策略没学过无球(训练里球几乎一直在),无球时发散("要起飞")。修法 A:sentinel 从 self-ref robot_pos 改固定 home(-1.6,-0.55,body_height+0.2)给回正力;B:训练加无球间隙(姿态/平衡奖励驱动学 idle)。

### 第一次 g1_tt_idle(2026-06-09)失败 — PPO 发散
- no_ball_period_s=8/ball_active_s=5,**无课程**(gap 从 step0 就满 3s)、clip_actions=100。
- 结果:Mean reward -10M、ep_len ~40(持续)、action_rate_l2 -837k。**warm-start 策略一上来就撞满幅无球 OOD → 输出爆(clip=100 太松)→ PPO value/advantage 炸 → 全局发散**。
- run 已归档 `logs/_idle_failed_diverged/g1_tt_idle_DIVERGED`(不删)。

### B-retry:g1_tt_idle2(2026-06-10,进行中)
两处修复:
1. **无球课程** `no_ball_curriculum_steps=100000`:gap 从 0 线性涨到 3s(`_tt_no_ball_now` 里 active=period−c·(period−active_target),c=sim_step/cstep)。早期≈无无球间隙→先稳成 rally,再慢慢喂无球,避免一上来满幅 OOD。
2. **收紧 clip** `normalization.clip_actions 100→20`:仍远超正常击球动作,但封住爆炸幅度。
- warm-start `logs/g1_tt_idle2/seed_36000/model_36000.pt`(从 rally 拷),TARGET=44000(8000 iter)。watchdog PID `/tmp/idle2_watchdog.pid`,日志 `train_idle2_watchdog.log`。
- watchdog **off-by-one 已修**(`>= TARGET-1`,到 43999 即停,不再死循环)。
- ⚠️ **教训:64-env 短 verify 判不出发散**。idle2 的 5-iter/64-env verify 显示 Mean reward -10M / action_rate -639k,看着像发散,实为 §8 那个固有噪声被小 env 数放大(-18000×4096/64≈-1.15M 对得上)。**真正判据看 ep_len**(健康~200+,发散=持续~40),不看尖峰污染的 mean reward。课程早期 gap≈0 也决定了短 verify 测不到无球行为。
- eval(下次):`TT_SERVE_PERIOD=10` 看无球间隙站不站得住;ckpt 选最佳(别假设 last=best)。
- phase2 待办:预测式门控(不等落地)+ reachability + train/deploy 门控数值一致 + 导 onnx + sim2sim。

### ⭐ idle2 也发散 → 诊断到真根因(2026-06-10)
两次 warm-start 微调(idle1 clip100、idle2 clip20)**都发散**:action_rate_l2 **每个 iter** -36万~-88万(健康 -1.0),持续非偶发 → 不是 §8 噪声。代码确认两条根因:
1. **clip_actions 用错了杠杆**。`tt_env.step` 里 clip 只裁"施加到 sim 的动作"(processed_actions);**obs(`action_buffer.buffer[:,-1]`)和 action_rate 惩罚用的都是网络裁前原始输出**。所以 clip=20/100 对发散机制根本没碰到,clip=20 还把输出和动力学脱钩(输出±1000 机器人只感受±20),有害。
2. **固定 home sentinel 改在了普通 rally 路径**(`compute_intermediate_values` 的 mask_invalid,每个回合间隙都触发),但 model_36000 是用旧自指 sentinel 训的 → warm-start value 函数失配 → advantage 爆 → 发散。(resume 确认加载了 critic+optimizer,排除"value 从零"。)
两个发散 run 归档 `logs/_idle_failed_diverged/`。

### g1_tt_idle3 — FROM-SCRATCH + 显式硬门控(2026-06-10,进行中)
放弃 warm-start(项目铁律:warm-start 是部署杀手)。**从零训**,带统一**硬验证门控**:
- **硬门控**(`compute_current_observations_perception`):invalid 时把 **actor 的 ball_prediction + rel_target 规则替换成精确 home sentinel**(= critic 的 ball_future_pose 在 invalid 时的同一组数),并把 **delayed_perception 的球槽[0:3]** 也置成同一 home 点(帧转换:`table + sentinel_rel − env_origins`)。→ actor/critic 在无球态**逐数对齐**,actor 永不见 predictor MLP 的 OOD 垃圾。有球时 actor=predictor、critic=ground-truth 的非对称是**故意保留**(privileged critic;部署无 ground-truth)。
- **predictor 之前是"软筛选"**(训练目标在 invalid 时=sentinel,靠 MLP 学,OOD 外推=垃圾)——这就是发散根源;硬门控把它变确定性规则。
- **无球=真实分布大头**:period 10s,ball_active 目标 3s → **最终 70% 无球(连续 7s 窗口)**;但用**反向无球课程**(`no_ball_curriculum_steps=300000`)从 0%(iter0 全球,先 bootstrap 命中)渐增到 70%(~12.5k iter),避免随机策略学成"只站不打"局部最优。
- **⚠️ 修了 unit bug**:`_tt_no_ball_now` 原来拿 `sim_step_counter`(物理步,decimation=10)直接 `*50` 当控制步 → 周期短 10×(8s 跑成 0.8s)。改 `cs = sim_step_counter // decimation` 后正确(已验证 6–8s/10s 无球窗口)。
- 发球:bounce 易→难课程(easy x(-0.95,-0.75)/vz(1.7,1.9)/y0.08 → hard x(-1.35,-0.60)/vz(1.3,2.1)/y0.72,`serve_curriculum_steps=300000`)。clip_actions 回 100。
- eval 档对齐:固定中等 bounce(x(-1.15,-0.80)/vz(1.5,1.9)/y±0.45)+ **关无球注入**(no_ball_period_s=0,测纯命中);idle 用 `TT_SERVE_PERIOD` env hook 单独测。
- warm 起:无(fresh)。TARGET=40000(~2.5天)。watchdog PID `/tmp/idle3_watchdog.pid`,日志 `train_idle3_watchdog.log`,off-by-one 已修。
- 判健康:从零 ep_len 早期低是正常(随机策略),看**趋势上升**;发散看 action_rate 持续 -1e5。

## 10. TODO

- [ ] **进行中:见 §11(g1_tt_idle3 from-scratch + 硬门控,2026-06-10 起训,TARGET40k)**。跑起来后看 ep_len 趋势上升 + ~12.5k iter 后 70% 无球下仍稳;eval 用固定档挑 ckpt + `TT_SERVE_PERIOD` 测无球站稳。
- [ ] **GATE(待你跑)**:sim2sim 验证 PerceptionTracker:正常 rally 不抽风;`TT_FAULT=jitter/drop/baseocc` 各跑一遍确认不抽风/不摔;发球飞出桌、桌上滚动、双弹 → 机器人 hold。
- [ ] **击球瞬间 FK**:抓 has_touch_paddle 翻 True 帧的手臂关节角,FK 出拍面法向,指导真拍椭圆孔朝向。
- [ ] **真拍 sim2real**:把 URDF 几何换成真拍(9cm 连接件→击球点~30cm,误差 3.7cm 在拍盘半径内,可直接部署;若改长度则改 paddle_offset 重训)。
- [ ] **Tier 3**:训练端感知域随机化(球/基体 丢帧/抖动/假点)fine-tune v3,让策略本身抗脏输入。
- [ ] 旧 model_9000 与新 run 硬数字对比(experiment_name 临时指回)。
