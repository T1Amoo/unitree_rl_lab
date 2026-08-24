"""Configuration for the WOAN4310 12-DoF quadruped."""

from pathlib import Path

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg, UnitreeUrdfFileCfg

WOAN4310_DESCRIPTION_DIR = Path(__file__).resolve().parent / "woan4310_description"
WOAN4310_URDF_PATH = WOAN4310_DESCRIPTION_DIR / "urdf" / "dog_V2.urdf"

# This order comes from the controller_joint_names list shipped with the source
# robot description. It is also the order used when exporting deploy.yaml.
WOAN4310_JOINT_SDK_NAMES = [
    "joint_ZQ2",
    "joint_ZQ3",
    "joint_ZQ4",
    "joint_ZH2",
    "joint_ZH3",
    "joint_ZH4",
    "joint_YQ2",
    "joint_YQ3",
    "joint_YQ4",
    "joint_YH2",
    "joint_YH3",
    "joint_YH4",
]


WOAN4310_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=str(WOAN4310_URDF_PATH),
        root_link_name="base",
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=False,
        self_collision=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            "joint_.*2": 0.0,
            "joint_.*3": 0.8,
            "joint_.*4": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "WOAN4310": IdealPDActuatorCfg(
            joint_names_expr=["joint_(ZQ|ZH|YQ|YH)[234]"],
            effort_limit=12.5,
            velocity_limit=30.0,
            stiffness=12.5,
            damping=0.25,
            friction=0.01,
        ),
    },
    joint_sdk_names=WOAN4310_JOINT_SDK_NAMES,
)
