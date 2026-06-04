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
