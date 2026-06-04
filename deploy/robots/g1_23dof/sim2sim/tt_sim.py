"""G1 table-tennis sim2sim driver.

- Loads the TT MJCF scene.
- Starts the unitree_mujoco DDS bridge (auto-publishes LowState, applies LowCmd PD).
- Serves the ball on a fixed cadence (Serve).
- Publishes ball + pelvis positions as ROS2 PoseStamped at the 50 Hz control rate.

Domain 1, loopback. Run with --headless N to run N control steps with no viewer
(for validation); otherwise opens the MuJoCo viewer.
"""
import sys
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


class _KeyFSM:
    """Reads single keypresses from the terminal and writes the matching FSM
    transition chord into LowState.wireless_remote, replicating the bridge's
    bit-packing (byte2=[0,0,LT,RT,SELECT,START,LB,RB], byte3=[left,down,right,up,Y,X,B,A]).
    The bridge leaves wireless_remote untouched when no real joystick is set up,
    so these writes are what the deploy parses as `lowstate->joystick`.
    Keys: f=FixStand(LT+Up), g=TableTennis(RB+Y), p=Passive(LT+B), q=quit.
    """
    CHORDS = {
        "f": (0x20, 0x10),  # LT + up   -> FixStand
        "g": (0x01, 0x08),  # RB + Y    -> TableTennis
        "p": (0x20, 0x02),  # LT + B    -> Passive
    }

    def __init__(self, hold_steps=60):
        self.b2 = 0
        self.b3 = 0
        self.ttl = 0
        self.hold = hold_steps
        self.quit = False
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        fd = sys.stdin.fileno()
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
                            self.ttl = self.hold
                        print(f"[tt_sim] key '{c}' -> chord ({self.b2:#04x},{self.b3:#04x})")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def apply(self, low_state):
        # Called every physics step. Holds the chord for `hold` steps then releases
        # (0,0) so the deploy's `.on_pressed` edge fires exactly once per keypress.
        with self._lock:
            if self.ttl > 0:
                low_state.wireless_remote[2] = self.b2
                low_state.wireless_remote[3] = self.b3
                self.ttl -= 1
            else:
                low_state.wireless_remote[2] = 0
                low_state.wireless_remote[3] = 0


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

    keyfsm = _KeyFSM()
    keyfsm.start()
    print("[tt_sim] keyboard FSM (type in THIS terminal): "
          "'f'=FixStand(L2+Up)  'g'=TableTennis(R1+Y)  'p'=Passive  'q'=quit")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not keyfsm.quit:
            mujoco.mj_step(model, data)
            keyfsm.apply(bridge.low_state)
            phys_count += 1
            if phys_count % DECIMATION == 0:
                control_tick()
            viewer.sync()
            time.sleep(PHYS_DT)

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
