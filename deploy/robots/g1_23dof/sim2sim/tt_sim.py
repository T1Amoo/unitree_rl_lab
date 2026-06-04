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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
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
