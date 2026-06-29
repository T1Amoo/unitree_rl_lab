# G1-23DoF 乒乓 — 完整命令行手册

覆盖 **训练 / Eval / sim2sim / sim2real** 全流程命令 + **手柄/键盘状态操作**。
最后更新 2026-06-17。

> 路径约定
> - 训练仓库 `Pingpong_TTRL/`(IsaacLab 2.1.1)。本地 python：`/home/woan/.conda/envs/pingpong/bin/python`;云端：`/root/miniconda3/envs/pingpong/bin/python`。
> - 部署仓库 `unitree_rl_lab/deploy/robots/g1_23dof/`。conda env：`g1tt_sim2sim`(含 ROS2 humble + onnxruntime)。
> - 云端 SSH：`ssh -p 1021 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@39.101.75.133`(exit 255 = 网络抖动,重试 1-2 次)。
> - 当前最佳策略 = `g1_tt_idle12/.../model_24500`(易球 95.8% / hard 90.5%),部署版本 = `config/policy/table_tennis/v5`。

---

## 0. 任务名 / 版本速查

| 任务名 | 用途 |
|---|---|
| `g1_tt` | 训练(性能门控难度课程) |
| `g1_tt_eval` | Eval：易球分布(饱和 ~90-95%,不区分高低) |
| `g1_tt_eval_hard` | Eval：训练满难度 serve_c=1.0 分布(区分 ckpt 强弱) |

环境变量 hook（训练+eval+sim 通用,按需 export）：
- `TT_SERVE_PERIOD=N`  每 N 颗球插一段无球期(测无球站稳)
- `TT_NO_SERVE=1`      完全不发球
- `TT_TRACE=1`         eval 时逐步写 `/tmp/tt_trace.csv`(env0 球/pred/base/action 诊断)
- `LOAD_OPTIMIZER=0`   warm-start 进入偏移分布时用全新优化器
- `TT_SIM_STEP_OFFSET=N` resume 时续上课程时钟(=已训 iter × 240)
- `TT_SUCC_WINDOW=x`   性能门控难度推进的成功率窗口(默认 0.20,压缩尺度)

---

## 1. 训练

### 1a. 从零训练
```bash
cd Pingpong_TTRL
/home/woan/.conda/envs/pingpong/bin/python -m legged_lab.scripts.train \
  --task=g1_tt --num_envs=4096 --headless \
  --logger=tensorboard --predictor --max_iterations=25000
```
- `--predictor` 必须带(训练辅助击球点预测器,部署 predictor.onnx 靠它)。
- 日志/ckpt 落 `logs/g1_tt/<时间戳>/`;experiment_name 在 `g1_tt_config.py: G1TableTennisAgentCfg.experiment_name`。

### 1b. Resume / warm-start
```bash
LOAD_OPTIMIZER=0 TT_SIM_STEP_OFFSET=$((10000*240)) \
/home/woan/.conda/envs/pingpong/bin/python -m legged_lab.scripts.train \
  --task=g1_tt --num_envs=4096 --headless --logger=tensorboard --predictor \
  --max_iterations=15000 \
  --resume true --load_run <run目录名> --checkpoint model_10000.pt
```

### 1c. Watchdog 托管(断了自动续,扛过夜)
```bash
cd Pingpong_TTRL
# 本地
TRAIN_PY=/home/woan/.conda/envs/pingpong/bin/python \
  nohup bash legged_lab/scripts/watchdog_train_idle12.sh > /dev/null 2>&1 &
# 云端
TRAIN_PY=/root/miniconda3/envs/pingpong/bin/python \
  nohup bash legged_lab/scripts/watchdog_train_idle12.sh > /dev/null 2>&1 &
```
脚本里改 `EXP / TARGET / NUM_ENVS`。监控：
```bash
tail -f train_idle12_watchdog.log
grep "\[curriculum\]" train_idle12_watchdog.log    # 看 serve_c / succ_ema 推进
```

### 1d. 停训(铁律:先杀 watchdog 再杀 trainer,按 PID 不用 pgrep -f)
```bash
pkill -f watchdog_train_idle12.sh      # 先停 watchdog(否则 30s 内自动重启抢 GPU)
ps -eo pid,cmd | grep "[s]cripts.train"   # 找 trainer PID
kill -9 <trainer_pid>
nvidia-smi    # 用 memory.used 判 GPU 是否真空
```

### 1e. TensorBoard
```bash
/home/woan/.conda/envs/pingpong/bin/tensorboard --logdir Pingpong_TTRL/logs/g1_tt_idle12
```

---

## 2. Eval(挑 ckpt,务必用真成功率,别信训练 reward)

### 2a. 易球 eval
```bash
cd Pingpong_TTRL
/home/woan/.conda/envs/pingpong/bin/python -m legged_lab.scripts.eval \
  --task=g1_tt_eval --num_envs=100 --seed=0 --predictor --headless \
  --load_run <run目录名> --checkpoint model_24500.pt
```
### 2b. Hard eval(区分强弱用这个)
```bash
... --task=g1_tt_eval_hard ...   # 其余同上
```
### 2c. 无球站稳测试
```bash
TT_SERVE_PERIOD=10 ... --task=g1_tt_eval ...
```
### 2d. 逐步诊断 trace
```bash
TT_TRACE=1 ... --task=g1_tt_eval ...   # 写 /tmp/tt_trace.csv
```
> ⚠️ GUI eval 本地会卡死(URDF importer GIL),一律 `--headless`;要看动作录视频。
> eval 启动时**自动导出** `policy.onnx`(+`predictor.pt`)到该 run 的 `exported/`。

---

## 3. ONNX 导出(部署用)

eval 自动导 `policy.onnx`。predictor 单独导(本地有 onnxruntime,云端没有→本地导):
```bash
cd Pingpong_TTRL
/home/woan/.conda/envs/pingpong/bin/python legged_lab/scripts/export_predictor_onnx.py \
  --ckpt logs/g1_tt_idle12/<run>/model_24500.pt \
  --out  <部署policy目录>/exported/predictor.onnx \
  --history_len 5
```
部署需要两个 onnx：`exported/policy.onnx` + `exported/predictor.onnx` + `params/deploy.yaml`。

---

## 4. 部署版本管理

部署策略目录：`unitree_rl_lab/deploy/robots/g1_23dof/config/policy/table_tennis/vN/`
内含 `exported/{policy,predictor}.onnx` + `params/deploy.yaml`。

切版本：改 `config/config.yaml` →
```yaml
  TableTennis:
    policy_dir: config/policy/table_tennis/v5   # 改这里即生效,不用重编
```
当前：**v5 = model_24500(最佳)**;v4 = model_10000。

---

## 5. sim2sim(MuJoCo,本机两进程)

### 5a. 编译 deploy(只在改了 C++ 时需要)
```bash
cd unitree_rl_lab/deploy/robots/g1_23dof
bash sim2sim/build_deploy.sh      # 生成 build/g1_ctrl
```

### 5b. 启动(开两个终端 / 或后台)
```bash
# 终端1:物理+发球+ROS2 mocap 发布
bash sim2sim/run_sim.sh
# 终端2:策略(等终端1的 viewer 出来再起)
bash sim2sim/run_deploy.sh
```
- `run_deploy.sh` 用 `--network lo` + DDS domain 1。
- 一次只能一个 g1_ctrl(脚本会先 `pkill -9 -x g1_ctrl`)。

### 5c. sim 端可调环境变量(放 run_sim.sh 前面)
```bash
TT_NOBALL_STEPS=0      # 每个发球周期末尾无球的控制步数(125=2.5s;0=一直有球)
TT_INVALID_FRAC=0      # 无效球(出界/下网等)比例
TT_BLACKOUT_START=20   # 感知失踪测试:发球后第几步开始断 mocap
TT_BLACKOUT_LEN=0      # 断多少步(50=1s;0=关)。球照常飞、只断感知发布
TT_SERVE_PERIOD / TT_NO_SERVE   # 同 §0
# 例:1s 感知失踪鲁棒性测试
TT_NOBALL_STEPS=0 TT_INVALID_FRAC=0 TT_BLACKOUT_START=20 TT_BLACKOUT_LEN=50 bash sim2sim/run_sim.sh
```

### 5d. sim2sim 控制(键盘,焦点在 MuJoCo 窗口)
| 键 | 动作 |
|---|---|
| `f` | 进 FixStand(站立) |
| `g` | 进 TableTennis(打球) |
| `p` | 进 Passive(**阻尼/软**) |
| `7` / `8` | 发球落点带 抬高 / 降低 |
| `9` | 放球(发一颗) |
| `q` | 退出 |

### 5e. 停 sim2sim
```bash
pkill -9 -f tt_sim_mujoco.py; pkill -9 -x g1_ctrl; pkill -9 -f run_deploy.sh
```

---

## 6. sim2real(真机)

### ✅ 6a. DDS domain(已自动判断,无需再改)
`main.cpp` 已按网卡自动选 domain:`--network lo` → domain 1(mujoco sim);其它网卡(`eth0`/`enpXs0`)→ domain 0(真机 G1)。**同一个 g1_ctrl 二进制 sim/real 通用,不用重编换 domain。** 真机直接 `--network <网卡>` 即可。

### 6b. config.yaml:接 mocap 真话题 + 帧标定
`config/config.yaml` 的 `TableTennis` 段:
```yaml
    ros:
      ball_topic: /vrpn_mocap/<球刚体名>/pose
      base_topic: /vrpn_mocap/<机器人刚体名>/pose
    input_frame:
      origin_in_training_world: [ox, oy, oz]      # 由标定反解(见 §6c)
      rotation_wxyz_to_training: [w, x, y, z]      # 轴对齐时 = [1,0,0,0]
```

### 6c. 帧标定(不知道 mocap 原点也能做)
训练世界系 W:**原点 = 球台中心正下方地面**,+x 朝对面、+y 机器人左、+z 上;台高 0.76,机器人 home=(-1.6,0,0.76)。
```bash
ros2 topic list                                   # 1. 找真话题名
ros2 topic echo --once /vrpn_mocap/<球>/pose       # 2. 把球放已知点逐个读数
```
把球放这几个**已知 W 坐标**的点,各读一次 mocap 值,交给我反解 R+origin(Kabsch):
| 物理位置 | W 坐标 |
|---|---|
| 台面中心 | (0, 0, 0.78) |
| 台心地面 | (0, 0, 0.02) |
| 一个台角(台面) | (1.37, 0.76, 0.78) |
| 对角(台面) | (-1.37, -0.76, 0.78) |
验证:FixStand 时 base 话题(变换后)应≈(-1.6,0,0.76);台心球应≈(0,0,0.78)。

### 6d. 启动真机
```bash
# 0) mocap 节点已在发 /vrpn_mocap/.../pose
# 1) 机器人上电、进低层控制(厂家流程)
# 2) 起控制器(--network = 连机器人的真实网卡名,如 eth0 / enpXs0;不是 lo)
cd unitree_rl_lab/deploy/robots/g1_23dof
bash sim2sim/build_deploy.sh                       # 若改过 domain
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/local/lib \
  ./build/g1_ctrl --network enx6c1ff76cb7d7


source /opt/ros/humble/setup.bash
ros2 launch vrpn_mocap client.launch.yaml server:=10.1.1.198 port:=3883

cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/g1_23dof
bash sim2sim/run_deploy_real.sh
```

### 6e. 真机控制(Unitree 手柄)

**状态流程(顺序,2026-06-29):`Passive → FixStand → Velocity → TableTennis`**
TableTennis 只能从 Velocity 进(不再有 FixStand 直达捷径)。三个运行态都能 `L2+B` 回阻尼。

| 操作 | 按键 |
|---|---|
| 开机默认 | **Passive(阻尼)** |
| Passive → FixStand(站立) | **L2 + ↑(方向键上)** |
| FixStand → Velocity(行走) | **R1 + X** |
| Velocity → TableTennis(打球) | **R1 + Y** |
| FixStand → 回阻尼 | **L2 + B** |
| Velocity → 回阻尼 | **L2 + B** |
| TableTennis → 回阻尼 | **L2 + B** |

**回阻尼/软急停 = `L2 + B`**(FixStand、Velocity、TableTennis 三态都能直接回 Passive)。Passive 本身就是阻尼模式(只给 kd、不锁位置)。
> 注:进 TableTennis 必须走 `L2+↑`(FixStand)→ `R1+X`(Velocity)→ `R1+Y`(TableTennis);不能跳级。进 TableTennis 时机器人要正对球桌(IMU heading 零点在进入瞬间标定)。

---

## 7. 真机剩余准备(代码无关)

1. **标定数值**(§6c)+ `main.cpp` domain(§6a)。
2. **PD 增益**:乒乓段的 kp/kd **不在 config.yaml**,而是训练导出的 `<policy>/params/deploy.yaml` 里的 `stiffness`/`damping`(= 训练 implicit actuator,NF=10·2π、ζ=2),`State_TableTennis::enter()` 逐关节写进 motor cmd → **deploy PD = 训练 PD,sim/real 一致**。config.yaml 的 kp/kd 只给 Passive/FixStand。真机上要核的是这套增益在**真实电机的扭矩上限/发热**下是否安全(不是"缺 PD")。
3. **球拍**:真实质量/惯量/回弹 vs 训练刚性 mesh 的差异。
4. **安全**:关节限位钳制、急停、摔倒保护、人工安全启动。
5. **⚠️ 感知丢帧鲁棒性**:sim 实测 0.4s mocap 丢帧就接不住会摔(§5c blackout 测出)。真机 mocap 必有遮挡/丢帧 → 真机前大概率要回去训一版抗丢帧的(训练加整段丢帧随机化 / predictor 历史外推)。

---

## 8. 常见坑速查
- 重启训练前先杀 watchdog,否则它 30s 内重启 trainer 撞 PhysX OOM。
- 挑 ckpt 用 eval 真成功率,**last ≠ best**(过训会退)。
- 云端无 onnxruntime → predictor.onnx 本地导。
- `<Eigen/Dense>` 别在 rclcpp 之后引入(宏冲突);`tt_ros_ball_source.h` 故意用纯 float。
- sim2sim 一次只能一个 g1_ctrl(脚本自动 pkill)。
- 代码同步走 git push/pull,不 scp。
```
