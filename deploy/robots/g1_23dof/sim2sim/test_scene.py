import os
import mujoco

SCENE = os.path.join(os.path.dirname(__file__), "scene", "g1_23dof_tt_scene.xml")

EXPECTED_ACTUATORS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
]


def test_scene_loads_with_23_actuators_in_policy_order():
    model = mujoco.MjModel.from_xml_path(SCENE)
    assert model.nu == 23, f"expected 23 actuators, got {model.nu}"
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert names == EXPECTED_ACTUATORS, f"actuator order mismatch: {names}"


def test_pelvis_starts_at_table_frame():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    assert abs(data.xpos[bid][0] - (-1.6)) < 0.05


def test_sensor_block_matches_bridge_layout():
    model = mujoco.MjModel.from_xml_path(SCENE)
    # 69 motor sensors (3 x 23) + imu quat(4)+gyro(3)+accel(3) = 79 sensordata entries minimum
    # find imu_quat
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)]
    assert "imu_quat" in names, "bridge needs a sensor named imu_quat"
    imu_idx = names.index("imu_quat")
    # all 69 joint sensors must precede imu_quat
    assert imu_idx >= 69, f"imu_quat at sensor #{imu_idx}, expected >= 69 (after 3x23 joint sensors)"
    # first 23 sensors are jointpos for the 23 actuators in actuator order
    import mujoco as mj
    for i in range(23):
        act_joint = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        sens_name = names[i]
        assert sens_name.startswith(act_joint), f"sensor {i} ({sens_name}) should be jointpos of actuator {i} ({act_joint})"


def test_sensordata_length_covers_imu():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nsensordata >= 79, f"expected >=79 sensordata, got {model.nsensordata}"
