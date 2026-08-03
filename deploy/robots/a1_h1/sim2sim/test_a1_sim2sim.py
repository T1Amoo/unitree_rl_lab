import numpy as np
import mujoco
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from a1_scene import (
    CONTACT_MARGIN,
    PADDLE_BALL_SOLREF,
    PHYSICS_DT,
    RIGHT_ARM_EFFORT_LIMITS,
    RIGHT_ARM_JOINTS,
    TABLE_BALL_SOLREF,
    ball_addresses,
    build_scene_xml,
    load_scene,
)
from ball_gate import BallValidityGate
from policy_io import PRED_SENTINEL
from policy_io import (
    A1PolicyIO,
    BRIDGE_MAX_DELTA_PER_TICK,
    DEFAULT_POLICY,
    DEFAULT_RIGHT_Q,
    EFFORT,
    FittedSecondOrderActionResponse,
    OBS_SIZE,
    OnnxPolicy,
    REAL_RESPONSE_MAX_DELTA_PER_TICK,
)
from run_a1_tt_sim2sim import parse_args, run
from serve import (
    BOUNCE_X,
    BOUNCE_VZ,
    GENERALIZED_BOUNCE_X,
    NET_CENTER_CLEARANCE_Z,
    SERVE_Y_CENTER,
    SERVE_Y_HALF,
    Serve,
    first_flight_metrics,
)


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


def test_contact_pairs_match_training_material_contract():
    model, _, _ = load_scene(None)
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    expected = {
        "table_top": TABLE_BALL_SOLREF,
        "table_support": TABLE_BALL_SOLREF,
        "net": TABLE_BALL_SOLREF,
        "paddle_blade": PADDLE_BALL_SOLREF,
    }
    found = {}
    for pair_id in range(model.npair):
        geom1 = int(model.pair_geom1[pair_id])
        geom2 = int(model.pair_geom2[pair_id])
        if ball not in (geom1, geom2):
            continue
        other = geom2 if geom1 == ball else geom1
        other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
        if other_name not in expected:
            continue
        np.testing.assert_allclose(model.pair_solref[pair_id], expected[other_name])
        np.testing.assert_allclose(model.pair_friction[pair_id], [0.1, 0.1, 0.005, 0.0001, 0.0001])
        assert np.isclose(model.pair_margin[pair_id], CONTACT_MARGIN)
        found[other_name] = True
    assert set(found) == set(expected)


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


def test_bridge_qdes_delta_limit_matches_deploy_envelope():
    model, data, _ = load_scene()
    io = A1PolicyIO(model, data)
    q_des = io.update_action(np.full(7, 100.0, dtype=np.float32), max_delta_per_tick=BRIDGE_MAX_DELTA_PER_TICK)
    assert np.allclose(q_des, DEFAULT_RIGHT_Q + BRIDGE_MAX_DELTA_PER_TICK)
    q_des = io.update_action(np.full(7, -100.0, dtype=np.float32), max_delta_per_tick=BRIDGE_MAX_DELTA_PER_TICK)
    assert np.allclose(q_des, DEFAULT_RIGHT_Q)


def test_real_response_model_smooths_limited_qdes_step():
    response = FittedSecondOrderActionResponse(PHYSICS_DT, DEFAULT_RIGHT_Q)
    command = DEFAULT_RIGHT_Q + REAL_RESPONSE_MAX_DELTA_PER_TICK
    prev = response.response.copy()
    max_step = 0.0
    for _ in range(10):
        current = response.step(command)
        max_step = max(max_step, float(np.max(np.abs(current - prev))))
        prev = current.copy()
    assert np.isfinite(response.response).all()
    assert max_step < float(np.max(REAL_RESPONSE_MAX_DELTA_PER_TICK))
    assert np.max(np.abs(response.response - DEFAULT_RIGHT_Q)) < float(np.max(REAL_RESPONSE_MAX_DELTA_PER_TICK))


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


def test_generalized_serve_clears_net_and_bounces_on_robot_side():
    serve = Serve(rng=np.random.default_rng(7), profile="generalized")
    for _ in range(256):
        _, vel = serve.sample()
        metrics = first_flight_metrics(vel)
        assert metrics is not None
        net_z, bounce_x, _ = metrics
        assert net_z >= NET_CENTER_CLEARANCE_Z
        assert GENERALIZED_BOUNCE_X[0] <= bounce_x <= GENERALIZED_BOUNCE_X[1]


def test_ball_gate_accepts_valid_serve_and_rejects_invalid_ball():
    gate = BallValidityGate()
    pos, vel = Serve(rng=np.random.default_rng(1)).sample()
    out = gate.update(pos, vel)
    assert out.engaged
    assert out.reason == "valid"
    assert -1.37 <= out.first_bounce_x <= 0.0

    out = gate.update(np.array([-1.66, 0.0, 1.0]), np.array([-2.0, 0.0, 0.0]))
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


def test_damiao_clip_effort_torque_speed_envelope():
    from policy_io import damiao_clip_effort
    vel_limit = np.array([8, 8, 8, 20, 20, 20, 20], dtype=np.float64)
    effort_limit = np.array([28, 28, 28, 8, 8, 8, 8], dtype=np.float64)
    brake = effort_limit.copy()

    # 静止(dq=0): 两侧都用 brake_limit=effort_limit
    tau = np.array([100, -100, 0, 100, -100, 0, 5], dtype=np.float64)
    dq0 = np.zeros(7, dtype=np.float64)
    out = damiao_clip_effort(tau, dq0, vel_limit, effort_limit, brake)
    np.testing.assert_allclose(out, np.array([28, -28, 0, 8, -8, 0, 5], dtype=np.float64))

    # 正速度且加速方向(tau>0): 受 speed_scale 限制; 反向(刹车)用 brake
    # r1: dq=4, vel_limit=8 -> speed_scale=0.5 -> accel_limit=14; 加速 tau=100 -> 14
    # r2: dq=4 -> 反向 tau=-100 -> -brake=-28
    dq = np.array([4, 4, 0, 0, 0, 0, 0], dtype=np.float64)
    tau2 = np.array([100, -100, 0, 0, 0, 0, 0], dtype=np.float64)
    out2 = damiao_clip_effort(tau2, dq, vel_limit, effort_limit, brake)
    assert out2[0] == 14.0
    assert out2[1] == -28.0

    # 负速度: 加速方向是负, 正向是刹车
    # r1: dq=-4 -> speed_scale=0.5 -> accel_limit=14; tau=-100(加速) -> -14;
    dq3 = np.array([-4, -4, 0, 0, 0, 0, 0], dtype=np.float64)
    tau3 = np.array([-100, 100, 0, 0, 0, 0, 0], dtype=np.float64)
    out3 = damiao_clip_effort(tau3, dq3, vel_limit, effort_limit, brake)
    assert out3[0] == -14.0   # 加速方向受限
    assert out3[1] == 28.0    # 正向刹车用 brake


def test_damiao_slew_limits_per_step_delta():
    from policy_io import damiao_slew
    vel_limit = np.array([8, 8, 8, 20, 20, 20, 20], dtype=np.float64)
    dt = 0.002
    cmd = np.zeros(7, dtype=np.float64)
    # 目标远大于 max_delta=vel_limit*dt=[0.016,...,0.04,...]
    q_des = np.ones(7, dtype=np.float64)
    out = damiao_slew(cmd, q_des, vel_limit, dt)
    np.testing.assert_allclose(out[:3], 0.016)   # 8*0.002
    np.testing.assert_allclose(out[3:], 0.040)   # 20*0.002
    # 目标在步长内则直达
    cmd2 = np.zeros(7, dtype=np.float64)
    q_des2 = np.full(7, 0.001, dtype=np.float64)
    out2 = damiao_slew(cmd2, q_des2, vel_limit, dt)
    np.testing.assert_allclose(out2, 0.001)


def test_apply_damiao_mit_torque_signs_and_limit(tmp_path):
    # 用现有 scene 构建一个 io，验证力矩方向与受限
    import mujoco
    from a1_scene import build_scene_xml, load_scene
    from policy_io import A1PolicyIO, DAMIAO_MIT_EFFORT
    # Adaptation: load_scene returns (model, data, path); use use_predictor=False
    # instead of non-existent policy= kwarg.
    model, data, _ = load_scene(build_scene_xml())
    io = A1PolicyIO(model, data, use_predictor=False)
    q = io.right_q()
    # 目标设在当前位置 +0.5rad(远超步长)，期望力矩为正且不超 effort_limit
    io.q_des = (q + 0.5).astype(np.float64)
    io.damiao_cmd = q.copy()
    tau = io.apply_damiao_mit(0.002)
    assert tau.shape == (7,)
    assert np.all(tau >= -DAMIAO_MIT_EFFORT - 1e-9)
    assert np.all(tau <= DAMIAO_MIT_EFFORT + 1e-9)
    assert tau[0] > 0.0   # 目标在正方向 -> 正力矩
    # qfrc_applied 已写入
    assert np.allclose(data.qfrc_applied[io.dof_addr], tau)
