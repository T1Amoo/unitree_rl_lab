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
import os
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
    glfw.KEY_V: (0x01, 0x04),  # RB + X  -> Velocity (stand/walk)
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
    "vx": 0.0, "vy": 0.0, "wz": 0.0,  # velocity command (WASD/QE drive the joystick sticks)
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
    # WASD velocity command (Velocity policy). W/S=fwd/back, A/D=left/right, Space=stop.
    elif keycode == glfw.KEY_W:
        S["vx"] = 0.5;  print(f"[tt_sim] cmd vx={S['vx']:.2f} vy={S['vy']:.2f}", flush=True)
    elif keycode == glfw.KEY_S:
        S["vx"] = -0.3; print(f"[tt_sim] cmd vx={S['vx']:.2f} vy={S['vy']:.2f}", flush=True)
    elif keycode == glfw.KEY_A:
        S["vy"] = 0.3;  print(f"[tt_sim] cmd vx={S['vx']:.2f} vy={S['vy']:.2f}", flush=True)
    elif keycode == glfw.KEY_D:
        S["vy"] = -0.3; print(f"[tt_sim] cmd vx={S['vx']:.2f} vy={S['vy']:.2f}", flush=True)
    elif keycode == glfw.KEY_SPACE:
        S["vx"] = S["vy"] = S["wz"] = 0.0; print("[tt_sim] cmd STOP (0,0,0)", flush=True)


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


def set_stick(low_state):
    # Map WASD command -> joystick sticks the C++ Velocity obs reads (observations.h:
    # vx=ly, vy=-lx, wz=-rx). wireless_remote float offsets: lx@4, rx@8, ly@20 (LE f32).
    lx = -S["vy"]; rx = -S["wz"]; ly = S["vx"]
    for i, b in enumerate(np.float32(lx).tobytes()):  low_state.wireless_remote[4 + i]  = b
    for i, b in enumerate(np.float32(rx).tobytes()):  low_state.wireless_remote[8 + i]  = b
    for i, b in enumerate(np.float32(ly).tobytes()):  low_state.wireless_remote[20 + i] = b


def SimulationThread():
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    bridge = UnitreeSdk2Bridge(mj_model, mj_data)  # auto-publishes LowState, applies LowCmd PD
    # TT_NO_MOCAP=1: skip the rclpy mocap publisher. It cannot coexist with the
    # Unitree SDK's in-process CycloneDDS (rclpy node creation fails: "rmw handle
    # is invalid"). Proprio (LowState/LowCmd) still flows via the Unitree DDS, so
    # the policy runs — only the ROS base/ball mocap is absent (robot_pos -> 0).
    NO_MOCAP = os.environ.get("TT_NO_MOCAP", "0") == "1"
    pub = None
    if not NO_MOCAP:
        rclpy.init()
        pub = MocapPublisher()
    serve = Serve(interval_steps=150)
    ctrl_step = 0
    cnt = 0
    ball_age = 0           # control ticks (50Hz) since last serve
    MAX_BALL_AGE = 150     # 3 s safety timeout (unused now that serve is fixed-cadence)
    SERVE_INTERVAL = 250   # fixed serve cadence: 250 ctrl steps @50Hz = 5 s per ball
    # No-ball window: the LAST NOBALL_STEPS of each serve cycle have NO ball -> the deploy's
    # invalid-ball gate drives a stable HOME-anchor idle (mirrors Isaac TT_SERVE_PERIOD no-ball
    # injection used to test idle). Default 125 ctrl steps = 2.5 s -> 50/50 ball/no-ball, matching
    # TT_SERVE_PERIOD=5. Set TT_NOBALL_STEPS=0 for the old always-ball behavior.
    NOBALL_STEPS = int(os.environ.get("TT_NOBALL_STEPS", "125"))
    # TT_NO_SERVE=1: never serve a ball at all -> the ball stays parked and the
    # publisher emits the (0,0,0) no-ball sentinel continuously (a "non-existent
    # ball trajectory"). Use to validate pure no-ball standing stability before
    # throwing any ball, in sim and (with the same config) on the real robot.
    NO_SERVE = os.environ.get("TT_NO_SERVE", "0") == "1"
    noball_now = NO_SERVE
    # TT_AUTO=1: headless test driver — auto-issue the FixStand chord, then the
    # TableTennis chord, on a timer (no keyboard needed). Lets sim2sim run from a
    # script so cmd/act can be checked for the same divergence seen on the robot.
    AUTO = os.environ.get("TT_AUTO", "0") == "1"
    # TT_AUTO_VEL=1: in AUTO mode, the second chord enters Velocity (stand/walk)
    # instead of TableTennis — lets sim2sim auto-test the velocity policy headless.
    AUTO_VEL = os.environ.get("TT_AUTO_VEL", "0") == "1"
    # Perception BLACKOUT test: the ball keeps FLYING (physics untouched), but its mocap is DROPPED
    # for TT_BLACKOUT_LEN ctrl steps starting TT_BLACKOUT_START after each serve -> the deploy loses
    # sight of it mid-flight and re-detects it later at its MOVED (flown-on) position (NOT where it
    # vanished). Tests perception-dropout robustness. Default off (TT_BLACKOUT_LEN=0); 50 steps = 1 s.
    BLACKOUT_START = int(os.environ.get("TT_BLACKOUT_START", "20"))
    BLACKOUT_LEN = int(os.environ.get("TT_BLACKOUT_LEN", "0"))
    last_reset = -100000
    RESET_COOLDOWN = 750   # >=1.5 s between auto-resets so recovery (p/f) isn't fought
    print("[tt_sim] focus the MuJoCo window, then: f=FixStand v=Velocity g=TableTennis p=Passive | 8=lower 7=raise 9=release | q=quit", flush=True)

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

        # Air drag on the ball to MATCH TRAINING (Isaac AeroForceField, cd=0.4378). mujoco models
        # zero aerodynamic drag by default, so hit returns fly too far and overshoot the table
        # (sim2sim gap: high hit-rate but balls don't land). F = -0.5*rho*A*cd*|v|*v (world frame).
        _v = mj_data.qvel[ball_vadr:ball_vadr + 3]
        _sp = float(np.linalg.norm(_v))
        mj_data.xfrc_applied[ball_bid, :3] = (-0.5 * 1.225 * (np.pi * 0.02 ** 2) * 0.4378 * _sp) * _v

        mujoco.mj_step(mj_model, mj_data)
        apply_chord(bridge.low_state)
        set_stick(bridge.low_state)
        # TT_JOINT_DIAG=1: per-50-step joint health log (ankle divergence watch). pos/vel per
        # joint + global max|qvel|. ankle_roll limit ~+/-0.262; runaway shows as big qvel/pinned pos.
        if os.environ.get("TT_JOINT_DIAG", "0") == "1" and cnt % 50 == 0:
            try:
                _qv = mj_data.qvel.copy(); _qv[ball_vadr:ball_vadr + 6] = 0.0  # exclude the flying ball
                _maxv = float(np.abs(_qv).max())
                def _jpv(nm):
                    _j = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, nm)
                    if _j < 0:
                        return (float("nan"), float("nan"))
                    return (float(mj_data.qpos[mj_model.jnt_qposadr[_j]]), float(mj_data.qvel[mj_model.jnt_dofadr[_j]]))
                _arl, _arr = _jpv("left_ankle_roll_joint"), _jpv("right_ankle_roll_joint")
                _apl, _apr = _jpv("left_ankle_pitch_joint"), _jpv("right_ankle_pitch_joint")
                print(f"[JDIAG] cnt={cnt} maxqvel={_maxv:.2f} pelvisZ={float(mj_data.qpos[pelvis_qadr+2]):.3f} | "
                      f"ankleRoll L={_arl[0]:+.3f}({_arl[1]:+.2f}) R={_arr[0]:+.3f}({_arr[1]:+.2f}) | "
                      f"anklePitch L={_apl[0]:+.3f}({_apl[1]:+.2f}) R={_apr[0]:+.3f}({_apr[1]:+.2f})", flush=True)
            except Exception:
                pass

        # headless auto-FSM: FixStand at ~3 s, then TableTennis at ~8 s (gives the
        # controller time to connect). Only fires when no chord is mid-sequence.
        if AUTO:
            # After FixStand, ramp the band DOWN so the feet plant on the ground
            # (BAND_Z0=1.5 hangs them ~0.6 m up -> torso anchor 0.90 grounds them).
            # Only enter TableTennis once grounded, so the policy sees a real
            # standing pose, not a dangling one.
            if S.get("did_fix") and S["band_z"] > 0.90:
                S["band_z"] = max(0.90, S["band_z"] - 0.001)
            if cnt % 500 == 0:
                print("[AUTO] cnt=%d band_z=%.2f pelvis_z=%.3f" % (
                    cnt, S["band_z"], float(mj_data.qpos[pelvis_qadr + 2])), flush=True)
            if S["seq"] < 0:
                if cnt >= 1500 and not S.get("did_fix"):
                    S["b2"], S["b3"] = CHORDS[glfw.KEY_F]; S["seq"] = 0; S["did_fix"] = True
                    print("[AUTO] -> FixStand", flush=True)
                elif S.get("did_fix") and cnt >= 6000 and S["band_z"] <= 0.91 and not S.get("did_tt"):
                    _k = glfw.KEY_V if AUTO_VEL else glfw.KEY_G
                    S["b2"], S["b3"] = CHORDS[_k]; S["seq"] = 0; S["did_tt"] = True
                    print("[AUTO] -> %s (grounded)" % ("Velocity" if AUTO_VEL else "TableTennis"), flush=True)
                elif AUTO_VEL and S.get("did_tt") and cnt >= 8000 and not S.get("did_release"):
                    # free-stand test: drop the band ~2 s after entering Velocity so the
                    # policy must hold the robot up on its own (no torso anchor).
                    S["band_enable"] = False; S["did_release"] = True
                    print("[AUTO] -> band RELEASED (free-stand test)", flush=True)

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
            # Fixed-cadence serve: one ball every SERVE_INTERVAL ctrl steps (5 s).
            # The ball flies+lands well before 5 s, then rests (robot holds via the
            # invalid-ball gate) until the next serve -> easy to watch one ball at a time.
            if NO_SERVE:
                # never serve: keep the ball parked underground so the mocap
                # publish sends (0,0,0) -> C++ have_ball=false -> idle, forever.
                mj_data.qpos[ball_qadr:ball_qadr + 3] = [0.0, 0.0, -50.0]
                mj_data.qvel[ball_vadr:ball_vadr + 6] = 0.0
                noball_now = True
            elif ball_age >= SERVE_INTERVAL:
                pos, vel = serve.sample()
                mj_data.qpos[ball_qadr:ball_qadr + 3] = pos
                mj_data.qpos[ball_qadr + 3:ball_qadr + 7] = [1, 0, 0, 0]
                mj_data.qvel[ball_vadr:ball_vadr + 3] = vel
                mj_data.qvel[ball_vadr + 3:ball_vadr + 6] = 0.0
                ball_age = 0
                noball_now = False
            elif NOBALL_STEPS > 0 and ball_age == (SERVE_INTERVAL - NOBALL_STEPS):
                # enter no-ball window: park the ball far underground (out of sight) and flag it
                # so the mocap publish below sends (0,0,0) -> C++ have_ball=false -> idle.
                mj_data.qpos[ball_qadr:ball_qadr + 3] = [0.0, 0.0, -50.0]
                mj_data.qvel[ball_vadr:ball_vadr + 6] = 0.0
                noball_now = True
        # publish mocap FASTER than control (every PUB_DECIM steps) so the deploy
        # always reads a fresh ball pose -> low perception latency (training delay
        # was ~4-10 ms). The deploy's predictor still consumes at its own 50 Hz.
        if pub is not None and cnt % PUB_DECIM == 0:
            blackout_now = (BLACKOUT_LEN > 0 and BLACKOUT_START <= ball_age < BLACKOUT_START + BLACKOUT_LEN)
            ball_xpos = np.zeros(3) if (noball_now or blackout_now) else mj_data.xpos[ball_bid].copy()
            pub.publish(ball_xpos,
                        mj_data.xpos[pelvis_bid].copy(),
                        mj_data.xquat[pelvis_bid].copy())

        locker.release()

        dt_left = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if dt_left > 0:
            time.sleep(dt_left)

    if pub is not None:
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
