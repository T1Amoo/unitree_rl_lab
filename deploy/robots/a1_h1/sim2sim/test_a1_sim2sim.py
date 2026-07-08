import numpy as np
import mujoco
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from a1_scene import RIGHT_ARM_EFFORT_LIMITS, RIGHT_ARM_JOINTS, ball_addresses, build_scene_xml, load_scene
from ball_gate import BallValidityGate
from policy_io import PRED_SENTINEL
from policy_io import A1PolicyIO, DEFAULT_POLICY, EFFORT, OBS_SIZE, OnnxPolicy
from run_a1_tt_sim2sim import parse_args, run
from serve import BOUNCE_X, BOUNCE_VZ, SERVE_Y_CENTER, SERVE_Y_HALF, Serve


def test_scene_loads_with_robot_table_and_ball():
    xml_path = build_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)}
    assert {"a1_root", "Link_r7", "table", "ball"}.issubset(names)
    geom_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)}
    assert "floor" in geom_names
    assert model.nlight >= 2
    for joint in RIGHT_ARM_JOINTS:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) >= 0
    ball_addresses(model)


def test_generated_mjcf_has_motor_metadata():
    xml_path = build_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nu == len(RIGHT_ARM_JOINTS)
    for joint in RIGHT_ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        dof = int(model.jnt_dofadr[jid])
        effort = RIGHT_ARM_EFFORT_LIMITS[joint]
        assert np.allclose(model.jnt_actfrcrange[jid], [-effort, effort])
        assert model.dof_armature[dof] > 0.0
        assert model.dof_damping[dof] > 0.0


def test_policy_obs_and_action_shape():
    model, data, _ = load_scene()
    io = A1PolicyIO(model, data)
    obs = io.observe(np.array([1.0, 0.0, 1.0]), np.array([-3.0, 0.0, 1.0]))
    assert obs.shape == (OBS_SIZE,)
    policy = OnnxPolicy(DEFAULT_POLICY)
    action = policy(obs)
    assert action.shape == (7,)


def test_mit_pd_torque_is_clipped_to_motor_limits():
    model, data, _ = load_scene()
    io = A1PolicyIO(model, data)
    io.update_action(np.array([100, -100, 100, -100, 100, -100, 100], dtype=np.float32))
    data.qfrc_applied[:] = 0.0
    tau = io.apply_mit_pd()
    assert np.all(np.abs(tau) <= EFFORT + 1e-6)
    assert np.isclose(np.max(np.abs(tau[:3])), 28.0)
    assert np.isclose(np.max(np.abs(tau[3:])), 8.0)


def test_serve_bounce_sampler_ranges():
    s = Serve(rng=np.random.default_rng(0))
    for _ in range(200):
        pos, vel = s.sample()
        assert np.allclose(pos, [1.35, 0.0, 1.03])
        assert vel[0] < 0.0
        assert BOUNCE_VZ[0] <= vel[2] <= BOUNCE_VZ[1]
        # Reconstruct sampled bounce y from launch vy and time-to-bounce.
        t_b = (vel[2] + np.sqrt(vel[2] * vel[2] + 2.0 * 9.81 * (1.03 - 0.78))) / 9.81
        x_b = 1.35 + vel[0] * t_b
        y_b = vel[1] * t_b
        assert BOUNCE_X[0] <= x_b <= BOUNCE_X[1]
        assert SERVE_Y_CENTER - SERVE_Y_HALF <= y_b <= SERVE_Y_CENTER + SERVE_Y_HALF


def test_ball_gate_accepts_valid_serve_and_rejects_invalid_ball():
    gate = BallValidityGate()
    pos, vel = Serve(rng=np.random.default_rng(1)).sample()
    out = gate.update(pos, vel)
    assert out.engaged
    assert out.reason == "valid"
    assert -1.37 <= out.first_bounce_x <= 0.0

    out = gate.update(np.array([-1.65, 0.0, 1.0]), np.array([-2.0, 0.0, 0.0]))
    assert not out.engaged
    assert out.reason == "behind_hit_plane"


def test_invalid_ball_observation_uses_home_sentinel():
    model, data, _ = load_scene()
    io = A1PolicyIO(model, data)
    obs = io.observe(np.array([-1.65, 0.3, 0.6]), np.array([1.0, 0.0, 0.0]), valid_ball=False)
    newest = obs[-39:]
    assert np.allclose(newest[27:30], PRED_SENTINEL)
    assert np.allclose(newest[33:36], PRED_SENTINEL)


def test_short_headless_rollout_without_policy():
    args = parse_args([])
    args.headless_steps = 5
    args.no_policy = True
    stats = run(args)
    assert stats["control_steps"] == 5
    assert stats["steps"] == 50
    assert stats["max_abs_tau"] <= 28.0 + 1e-6
