"""G1 table-tennis sim2sim driver — forked from unitree_mujoco/simulate_python/unitree_mujoco.py.

Keeps unitree_mujoco's PROVEN structure verbatim: SimulationThread (one mj_step per
loop, real-time paced to timestep) + PhysicsViewerThread (viewer.sync at VIEWER_DT) +
`locker` mutex + UnitreeSdk2Bridge for DDS. Adds ONLY the table-tennis extras:
  1. ball serve (write ball qpos/qvel on a cadence)
  2. ROS2 mocap publish (ball + pelvis pose) at the control rate
  3. keyboard FSM input via the viewer's glfw key_callback (no gamepad needed):
       f=FixStand(LT+up)  g=TableTennis(RB+Y)  p=Passive(LT+B)
       7/8 = raise/lower elastic band,  9 = release/grab band,  q = quit
  4. 6-DOF elastic band (force + angular restoring) to suspend the robot upright,
     and GR00T-style auto-reset when the pelvis falls below FALL_Z.

Run via run_sim.sh (conda run --no-capture-output). Scene/domain come from
unitree_mujoco's config.py (ROBOT=g1, ROBOT_SCENE=our TT scene, DOMAIN_ID=1).
"""
import sys
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer

SIM_PY = "/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_mujoco/simulate_python"
sys.path.insert(0, SIM_PY)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402
from unitree_sdk2py_bridge import UnitreeSdk2Bridge  # noqa: E402
import config  # noqa: E402  (unitree_mujoco config: ROBOT_SCENE, DOMAIN_ID, INTERFACE)

import rclpy  # noqa: E402
from serve import Serve  # noqa: E402
from ros_publish import MocapPublisher  # noqa: E402

glfw = mujoco.glfw.glfw

PHYS_DT = 0.002        # physics step (matches our TT scene tuning)
DECIMATION = 10        # 50 Hz control / serve
PUB_DECIM = 2          # 250 Hz mocap publish (fresher ball pose -> lower perception latency)
FALL_Z = 0.4           # pelvis z below this -> fallen -> auto-reset
BAND_Z0 = 1.5          # band anchor height: hangs feet ~0.6 m off the ground

# FSM transition chords -> (wireless_remote[2], wireless_remote[3]) bytes.
# byte2=[0,0,LT,RT,SELECT,START,LB,RB] msb..lsb ; byte3=[left,down,right,up,Y,X,B,A].
CHORDS = {
    glfw.KEY_F: (0x20, 0x10),  # LT + up -> FixStand
    glfw.KEY_G: (0x01, 0x08),  # RB + Y  -> TableTennis
    glfw.KEY_P: (0x20, 0x02),  # LT + B  -> Passive
}


def quat_to_rotvec(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    v = np.array([x, y, z]); n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros(3)
    a = 2.0 * np.arctan2(n, w)
    if a > np.pi:
        a -= 2.0 * np.pi
    return (a / n) * v


# ---- shared state (set by key_callback in the viewer thread, read in SimulationThread) ----
S = {
    "quit": False,
    "b2": 0, "b3": 0, "seq": -1,      # two-phase chord: modifier alone, then +edge, then release
    "band_enable": True, "band_z": BAND_Z0,
}
CHORD_PRE, CHORD_BOTH = 12, 30        # frames: modifier-only, then modifier+edge


def key_callback(keycode):
    if keycode == glfw.KEY_Q:
        S["quit"] = True
    elif keycode in CHORDS:
        S["b2"], S["b3"] = CHORDS[keycode]
        S["seq"] = 0
        print(f"[tt_sim] chord ({S['b2']:#04x},{S['b3']:#04x})", flush=True)
    elif keycode == glfw.KEY_8:
        S["band_z"] -= 0.1; print(f"[tt_sim] band lower z={S['band_z']:.2f}", flush=True)
    elif keycode == glfw.KEY_7:
        S["band_z"] += 0.1; print(f"[tt_sim] band raise z={S['band_z']:.2f}", flush=True)
    elif keycode == glfw.KEY_9:
        S["band_enable"] = not S["band_enable"]; print(f"[tt_sim] band enable={S['band_enable']}", flush=True)


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_model.opt.timestep = PHYS_DT
mj_data = mujoco.MjData(mj_model)
mujoco.mj_forward(mj_model, mj_data)

viewer = mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback)

# handles
band_link = mj_model.body("torso_link").id
ball_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
ball_qadr = mj_model.jnt_qposadr[ball_jid]
ball_vadr = mj_model.jnt_dofadr[ball_jid]
ball_bid = mj_model.body("ball").id
pelvis_bid = mj_model.body("pelvis").id
pelvis_qadr = mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")]
init_qpos = mj_data.qpos.copy()

# band gains (6-DOF: force + angular restoring), anchored above the robot start spot
BAND_XY = (float(mj_data.xpos[band_link][0]), float(mj_data.xpos[band_link][1]))
KP_POS, KD_POS, KP_ANG, KD_ANG = 3000.0, 300.0, 500.0, 30.0  # band gains (tested stable; higher damping NaNs the explicit integrator)
_bvel = np.zeros(6)


def apply_chord(low_state):
    # two-phase: modifier byte alone first (so it is established across several
    # published LowStates), then add the edge byte, then release.
    if S["seq"] < 0:
        low_state.wireless_remote[2] = 0; low_state.wireless_remote[3] = 0
    elif S["seq"] < CHORD_PRE:
        low_state.wireless_remote[2] = S["b2"]; low_state.wireless_remote[3] = 0
        S["seq"] += 1
    elif S["seq"] < CHORD_PRE + CHORD_BOTH:
        low_state.wireless_remote[2] = S["b2"]; low_state.wireless_remote[3] = S["b3"]
        S["seq"] += 1
    else:
        low_state.wireless_remote[2] = 0; low_state.wireless_remote[3] = 0
        S["seq"] = -1


def SimulationThread():
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    bridge = UnitreeSdk2Bridge(mj_model, mj_data)  # auto-publishes LowState, applies LowCmd PD
    rclpy.init()
    pub = MocapPublisher()
    serve = Serve(interval_steps=150)
    ctrl_step = 0
    cnt = 0
    ball_age = 0           # control ticks since last serve; reserve on land or timeout
    MAX_BALL_AGE = 150     # 3 s safety timeout (also reset on land/settle)
    last_reset = -100000
    RESET_COOLDOWN = 750   # >=1.5 s between auto-resets so recovery (p/f) isn't fought
    print("[tt_sim] focus the MuJoCo window, then: f=FixStand g=TableTennis p=Passive | 8=lower 7=raise 9=release | q=quit", flush=True)

    while viewer.is_running() and not S["quit"]:
        step_start = time.perf_counter()
        locker.acquire()

        # 6-DOF elastic band (suspend upright)
        if S["band_enable"]:
            mujoco.mj_objectVelocity(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY, band_link, _bvel, 0)
            pt = np.array([BAND_XY[0], BAND_XY[1], S["band_z"]])
            f = KP_POS * (pt - mj_data.xpos[band_link]) - KD_POS * _bvel[3:6]
            tau = -KP_ANG * quat_to_rotvec(mj_data.xquat[band_link]) - KD_ANG * _bvel[0:3]
            mj_data.xfrc_applied[band_link] = np.concatenate([f, tau])
        else:
            mj_data.xfrc_applied[band_link] = 0.0

        mujoco.mj_step(mj_model, mj_data)
        apply_chord(bridge.low_state)

        # auto-reset on fall, with a cooldown so it does not spam-reset (which fights
        # recovery). After a fall: press 'p' (Passive stops the policy, band holds it)
        # then 'f'. Free-standing on the ground is the policy sim2sim gap.
        if mj_data.qpos[pelvis_qadr + 2] < FALL_Z and (cnt - last_reset) > RESET_COOLDOWN:
            print(f"[tt_sim] FALL (pelvis z={mj_data.qpos[pelvis_qadr+2]:.2f}) -> reset + re-catch "
                  f"(press 'p' then 'f' to recover; policy can't free-stand in mujoco)", flush=True)
            mj_data.qpos[:] = init_qpos
            mj_data.qvel[:] = 0.0
            mj_data.xfrc_applied[:] = 0.0
            mujoco.mj_forward(mj_model, mj_data)
            S["band_z"] = BAND_Z0
            S["band_enable"] = True
            last_reset = cnt

        cnt += 1
        # serve at the control rate (50 Hz)
        if cnt % DECIMATION == 0:
            ctrl_step += 1
            ball_age += 1
            # Re-serve when the ball is done: landed (z<0.1), SETTLED anywhere
            # (|vel|<0.3 — e.g. resting on a leg / on the table), or timed out.
            # Mirrors training (reset_ball on floor/timeout) and also handles the
            # ball coming to rest on the robot's body (which never hits the floor).
            ball_speed = float(np.linalg.norm(mj_data.qvel[ball_vadr:ball_vadr + 3]))
            ball_on_floor = mj_data.xpos[ball_bid][2] < 0.1
            ball_settled = ball_speed < 0.3
            if ball_age > 10 and (ball_on_floor or ball_settled or ball_age > MAX_BALL_AGE):
                pos, vel = serve.sample()
                mj_data.qpos[ball_qadr:ball_qadr + 3] = pos
                mj_data.qpos[ball_qadr + 3:ball_qadr + 7] = [1, 0, 0, 0]
                mj_data.qvel[ball_vadr:ball_vadr + 3] = vel
                mj_data.qvel[ball_vadr + 3:ball_vadr + 6] = 0.0
                ball_age = 0
        # publish mocap FASTER than control (every PUB_DECIM steps) so the deploy
        # always reads a fresh ball pose -> low perception latency (training delay
        # was ~4-10 ms). The deploy's predictor still consumes at its own 50 Hz.
        if cnt % PUB_DECIM == 0:
            pub.publish(mj_data.xpos[ball_bid].copy(),
                        mj_data.xpos[pelvis_bid].copy(),
                        mj_data.xquat[pelvis_bid].copy())

        locker.release()

        dt_left = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if dt_left > 0:
            time.sleep(dt_left)

    pub.destroy_node()
    rclpy.shutdown()


def PhysicsViewerThread():
    while viewer.is_running() and not S["quit"]:
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    vt = threading.Thread(target=PhysicsViewerThread)
    st = threading.Thread(target=SimulationThread)
    vt.start(); st.start()
    vt.join(); st.join()
