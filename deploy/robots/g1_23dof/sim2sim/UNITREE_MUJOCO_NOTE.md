# G1 23DoF Table-Tennis sim2sim — run notes

`lgy/unitree_mujoco` is cloned separately (not vendored). See plan B1.1.

## What was built (all on branch g1-tt-sim2sim)
- **Scene**: `scene/g1_23dof_tt_scene.xml` — G1 23DoF (23-motor MJCF, policy joint order) + paddle (right wrist, +X 0.337, +90°-X roll) + table (2.74×1.525, top z=0.76) + net + ball (r0.02, 3.4g). Bounce tuned (contact pairs) to ratio ≈0.90. Includes the unitree_mujoco `<sensor>` block (23 jointpos/vel/torque + IMU framequat `imu_quat` + gyro + accel + `frame_pos`/`frame_vel`) — REQUIRED or the bridge publishes no IMU.
- **Sim driver**: `tt_sim.py` (bridge auto-publishes LowState; serve + ROS2 publish at 50 Hz). `serve.py`, `ros_publish.py`.
- **Deploy**: `State_TableTennis` FSM state + `RosBallSource` (subscribes `/mocap/ball/pose`, `/mocap/base/pose`); `g1_ctrl` runs on DDS domain 1; predictor.onnx runs live. Real PD gains in `deploy.yaml`.

## unitree_mujoco config (already edited in the clone, NOT under git)
`simulate_python/config.py`: `ROBOT="g1"`, `ROBOT_SCENE=<abs path to scene/g1_23dof_tt_scene.xml>`, `DOMAIN_ID=1`, `INTERFACE="lo"`, `USE_JOYSTICK=1` (`JOYSTICK_TYPE="xbox"`, `JOYSTICK_DEVICE=0`).

**Gamepad required to drive the FSM.** Plug in the pad BEFORE launching the sim and confirm the PC sees it:
```bash
ls /dev/input/js*          # expect js0
```
If the pad uses the Switch layout, set `JOYSTICK_TYPE="switch"`. If it's not `js0`, set `JOYSTICK_DEVICE` accordingly. No `/dev/input/js*` ⇒ the PC does not recognize the pad (e.g. a controller that only pairs with the robot) ⇒ the sim can't read it; fall back to a keyboard WirelessController helper.

## Build the deploy (the documented `cmake .. && make` does NOT work here)
The conda gcc-14 toolchain conflicts with the /usr/local unitree SDK, and `fmt` must
resolve to conda's libfmt.so.11. Use the wrapper:
```bash
bash deploy/robots/g1_23dof/sim2sim/build_deploy.sh
```

## Run sequence (B3.5)
Terminal 1 — sim (needs a display for the viewer):
```bash
cd .../lgy/unitree_rl_lab/deploy/robots/g1_23dof/sim2sim
conda run -n g1tt_sim2sim python tt_sim.py
```
Terminal 2 — verify mocap topics:
```bash
conda run -n g1tt_sim2sim bash -c "ros2 topic echo --once /mocap/ball/pose; ros2 topic hz /mocap/ball/pose"
```
Terminal 3 — deploy:
```bash
cd .../lgy/unitree_rl_lab/deploy/robots/g1_23dof/build
conda run -n g1tt_sim2sim ./g1_ctrl --network lo
```
Then drive the FSM: `[L2 + Up]` → FixStand, then `[R1 + Y]` → TableTennis. (No gamepad? options below.)

## Known risks / fallbacks
- **DDS cross-process discovery on `lo`**: cyclonedds disables multicast on loopback (`"lo" is not multicast-capable`). Within one process it works; if the sim and deploy do NOT see each other's topics/LowState, export in BOTH shells:
  `export CYCLONEDDS_URI=file://<abs path>/sim2sim/cyclonedds_loopback.xml`
- **FSM input = gamepad** (`USE_JOYSTICK=1`). Plug the pad in before launching; verify `ls /dev/input/js*`. Drive: `[L2+Up]`→FixStand, `[R1+Y]`→TableTennis, `[L2+B]`→Passive. If the pad isn't recognized by the PC, fall back to a keyboard WirelessController publisher (sim-only, no FSM change).
- **Uncontrolled robot collapses** until you enter FixStand/TableTennis — expected.
- **sim2sim physics gap vs IsaacSim** is expected (mujoco contacts/restitution/motor model differ). Goal: sensible motion + returns some serves, not 97% reproduction.
