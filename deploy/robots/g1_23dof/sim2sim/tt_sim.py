"""G1 table-tennis sim2sim driver.

- Loads the TT MJCF scene.
- Starts the unitree_mujoco DDS bridge (auto-publishes LowState, applies LowCmd PD).
- Serves the ball on a fixed cadence (Serve).
- Publishes ball + pelvis positions as ROS2 PoseStamped at the 50 Hz control rate.

Domain 1, loopback. Run with --headless N to run N control steps with no viewer
(for validation); otherwise opens the MuJoCo viewer.
"""
import sys
import os
import time
import argparse
import termios
import tty
import select
import threading
import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from serve import Serve
from ros_publish import MocapPublisher

# unitree_mujoco bridge
sys.path.insert(0, "/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_mujoco/simulate_python")
from unitree_sdk2py_bridge import UnitreeSdk2Bridge  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402

SCENE = "/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/g1_23dof/sim2sim/scene/g1_23dof_tt_scene.xml"
DOMAIN_ID = 1
INTERFACE = "lo"
PHYS_DT = 0.002
DECIMATION = 10          # 50 Hz control / mocap publish


def _ball_addrs(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    return model.jnt_qposadr[jid], model.jnt_dofadr[jid]


def _quat_to_rotvec(quat):
    """MuJoCo quat (w,x,y,z) -> rotation vector (axis*angle). Restoring this to
    zero keeps the body at its spawn orientation (upright)."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    v = np.array([x, y, z])
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(n, w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return (angle / n) * v


class _Band:
    """6-DOF virtual suspension (force + angular restoring), GR00T-WholeBodyControl
    style. Unlike unitree's force-only band, the angular term keeps the hanging
    robot upright instead of spinning like a tetherball. Applied as an external
    wrench on torso_link via xfrc_applied. Stiff so it holds torso ~at `point`."""
    def __init__(self, point):
        self.kp_pos = 3000.0
        self.kd_pos = 300.0
        self.kp_ang = 500.0
        self.kd_ang = 30.0
        self.point = np.array(point, dtype=float)
        self.enable = True

    def wrench(self, pos, quat, lin_vel, ang_vel):
        f = self.kp_pos * (self.point - pos) - self.kd_pos * lin_vel
        tau = -self.kp_ang * _quat_to_rotvec(quat) - self.kd_ang * ang_vel
        return np.concatenate([f, tau])


class _KeyFSM:
    """Reads single keypresses from the terminal and writes the matching FSM
    transition chord into LowState.wireless_remote, replicating the bridge's
    bit-packing (byte2=[0,0,LT,RT,SELECT,START,LB,RB], byte3=[left,down,right,up,Y,X,B,A]).
    The bridge leaves wireless_remote untouched when no real joystick is set up,
    so these writes are what the deploy parses as `lowstate->joystick`.
    Keys: f=FixStand(LT+Up), g=TableTennis(RB+Y), p=Passive(LT+B), q=quit.
    Elastic-band (suspension) keys, if a band is attached: 8=lower, 7=raise, 9=release/grab.
    """
    CHORDS = {
        "f": (0x20, 0x10),  # LT + up   -> FixStand
        "g": (0x01, 0x08),  # RB + Y    -> TableTennis
        "p": (0x20, 0x02),  # LT + B    -> Passive
    }

    def __init__(self, hold_steps=60, band=None):
        self.b2 = 0          # modifier byte (LT/RB) — held first
        self.b3 = 0          # edge byte (up/Y/B) — added after modifier is established
        self.seq = -1        # -1 = idle; >=0 = frames since keypress
        self.band = band
        self.quit = False
        self._lock = threading.Lock()
        # Phase lengths (in apply() calls): modifier-only, then modifier+edge, then release.
        self.PRE = 12
        self.BOTH = 30

    def start(self):
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            print("[tt_sim] WARNING: stdin is not a TTY -> keyboard FSM disabled. "
                  "Run via run_sim.sh (conda run --no-capture-output).", flush=True)
            return
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.quit:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    c = sys.stdin.read(1)
                    if c == "q":
                        self.quit = True
                    elif c in self.CHORDS:
                        with self._lock:
                            self.b2, self.b3 = self.CHORDS[c]
                            self.seq = 0
                        print(f"[tt_sim] key '{c}' -> chord ({self.b2:#04x},{self.b3:#04x})")
                    elif self.band is not None and c in ("7", "8", "9"):
                        if c == "8":
                            self.band.point[2] -= 0.1
                            print(f"[tt_sim] band lower, anchor_z={self.band.point[2]:.2f}")
                        elif c == "7":
                            self.band.point[2] += 0.1
                            print(f"[tt_sim] band raise, anchor_z={self.band.point[2]:.2f}")
                        else:
                            self.band.enable = not self.band.enable
                            print(f"[tt_sim] band enable={self.band.enable} (9=toggle release)")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def apply(self, low_state):
        # Two-phase chord so the deploy's "(modifier held) AND (edge .on_pressed)"
        # fires: set the modifier byte alone first (PRE frames) so it is established
        # across several published LowStates, THEN add the edge byte (BOTH frames),
        # then release. Avoids the non-atomic-write race where the edge button's
        # rising edge lands on a message before the modifier byte is set.
        with self._lock:
            if self.seq < 0:
                low_state.wireless_remote[2] = 0
                low_state.wireless_remote[3] = 0
            elif self.seq < self.PRE:
                low_state.wireless_remote[2] = self.b2
                low_state.wireless_remote[3] = 0
                self.seq += 1
            elif self.seq < self.PRE + self.BOTH:
                low_state.wireless_remote[2] = self.b2
                low_state.wireless_remote[3] = self.b3
                self.seq += 1
            else:
                low_state.wireless_remote[2] = 0
                low_state.wireless_remote[3] = 0
                self.seq = -1


def run(headless_steps=None):
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ChannelFactoryInitialize(DOMAIN_ID, INTERFACE)
    bridge = UnitreeSdk2Bridge(model, data)  # auto-publishes LowState; applies LowCmd

    rclpy.init()
    pub = MocapPublisher()

    qadr, vadr = _ball_addrs(model)
    pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    serve = Serve(interval_steps=150)
    phys_count = 0
    ctrl_step = 0

    def control_tick():
        nonlocal ctrl_step
        ctrl_step += 1
        if serve.due(ctrl_step):
            pos, vel = serve.sample()
            data.qpos[qadr:qadr+3] = pos
            data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]
            data.qvel[vadr:vadr+3] = vel
            data.qvel[vadr+3:vadr+6] = 0.0
        ball_pos = data.xpos[ball_bid].copy()
        base_pos = data.xpos[pelvis_bid].copy()
        base_quat = data.xquat[pelvis_bid].copy()  # (w,x,y,z)
        pub.publish(ball_pos, base_pos, base_quat)
        return ball_pos, base_pos

    if headless_steps is not None:
        last = None
        # seed a serve immediately so validation sees a moving ball
        serve_pos, serve_vel = serve.sample()
        data.qpos[qadr:qadr+3] = serve_pos
        data.qvel[vadr:vadr+3] = serve_vel
        start_x = float(serve_pos[0])
        # Track peak horizontal travel: the uncontrolled robot collapses and can
        # trigger a QACC NaN late in the run, which MuJoCo resets by snapping the
        # ball free-joint back to its model default. The final frame may therefore
        # land back near the serve start; the peak shows the ball actually flew.
        min_x = start_x
        for _ in range(headless_steps * DECIMATION):
            mujoco.mj_step(model, data)
            phys_count += 1
            if phys_count % DECIMATION == 0:
                last = control_tick()
                bx = last[0][0]
                if np.isfinite(bx):
                    min_x = min(min_x, float(bx))
        bp, rp = last
        print(f"[headless] ran {headless_steps} control steps; "
              f"ball_pos={np.round(bp,3).tolist()} base_pos={np.round(rp,3).tolist()} "
              f"finite={np.isfinite(bp).all() and np.isfinite(rp).all()} "
              f"ball_min_x={round(min_x,3)} (start_x={round(start_x,3)}, moved={min_x < start_x - 0.5})")
        pub.destroy_node()
        rclpy.shutdown()
        return

    # 6-DOF elastic suspension on torso_link: holds the robot upright in the air
    # (force + angular restoring) so it neither free-falls nor spins. Anchored
    # above the robot's start spot (x=-1.6, behind table); z=1.5 hangs the feet
    # ~0.6 m OFF the ground so there is no foot contact (no NaN/reset loop) while
    # suspended. Lower with key 8 until feet near floor, then release with 9.
    band_link = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    band = _Band(point=[float(data.xpos[band_link][0]), float(data.xpos[band_link][1]), 1.5])

    # Initial robot+ball state, for auto-reset on fall (GR00T-style check_fall).
    init_qpos = data.qpos.copy()
    pelvis_jadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")]
    FALL_Z = 0.4  # pelvis z below this -> fallen -> reset + re-catch with band

    keyfsm = _KeyFSM(band=band)
    keyfsm.start()
    print("[tt_sim] keys (type in THIS terminal): "
          "f=FixStand  g=TableTennis  p=Passive | band: 8=lower 7=raise 9=release | q=quit")
    _bvel = np.zeros(6)
    # Canonical sim pacing (unitree_mujoco / GR00T): one mj_step per loop paced to
    # sim_dt in REAL TIME; sync the viewer only every DECIMATION steps (~50 fps).
    # Syncing every physics step (the old code) throttled physics to ~50-100 Hz
    # while the bridge publishes LowState at 500 Hz and the deploy runs PD at
    # 1000 Hz -> rate mismatch -> control jitter.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not keyfsm.quit:
            step_start = time.monotonic()
            if band.enable:
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                         band_link, _bvel, 0)  # world frame: [ang(3), lin(3)]
                data.xfrc_applied[band_link] = band.wrench(
                    data.xpos[band_link], data.xquat[band_link], _bvel[3:6], _bvel[0:3])
            else:
                data.xfrc_applied[band_link] = 0.0
            mujoco.mj_step(model, data)
            # auto-reset on fall: restore robot+ball to start, zero velocity, re-catch
            # with the band so the robot never stays collapsed on the ground.
            if data.qpos[pelvis_jadr + 2] < FALL_Z:
                print(f"[tt_sim] FALL detected (pelvis z={data.qpos[pelvis_jadr+2]:.2f}) -> reset + re-catch")
                data.qpos[:] = init_qpos
                data.qvel[:] = 0.0
                data.xfrc_applied[:] = 0.0
                mujoco.mj_forward(model, data)
                band.point[2] = 1.5
                band.enable = True
            keyfsm.apply(bridge.low_state)
            phys_count += 1
            if phys_count % DECIMATION == 0:
                control_tick()
                viewer.sync()  # ~50 fps render, decoupled from the 500 Hz physics
            elapsed = time.monotonic() - step_start
            if (sleep_t := PHYS_DT - elapsed) > 0:
                time.sleep(sleep_t)

    pub.destroy_node()
    rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", type=int, default=None,
                    help="run N control steps with no viewer (validation)")
    args = ap.parse_args()
    run(headless_steps=args.headless)


if __name__ == "__main__":
    main()
