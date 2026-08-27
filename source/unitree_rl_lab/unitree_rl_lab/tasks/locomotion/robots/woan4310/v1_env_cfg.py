"""WOAN4310 v1 flat locomotion environment with symmetry-oriented foot rewards."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import (
    WOAN4310_FOOT_PATTERN,
    RewardsCfg,
    RobotEnvCfg,
    RobotPlayEnvCfg,
)


@configclass
class V1RewardsCfg(RewardsCfg):
    """Minimal v1 reward changes aimed at reducing contact-time foot drag."""

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WOAN4310_FOOT_PATTERN
            ),
            "command_name": "base_velocity",
            "threshold": 0.25,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=WOAN4310_FOOT_PATTERN),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=WOAN4310_FOOT_PATTERN
            ),
        },
    )


@configclass
class RobotV1EnvCfg(RobotEnvCfg):
    """Training configuration for the isolated WOAN4310 symmetry v1 task."""

    rewards: V1RewardsCfg = V1RewardsCfg()


@configclass
class RobotV1PlayEnvCfg(RobotPlayEnvCfg):
    """Deterministic flat viewer for WOAN4310 symmetry v1 checkpoints."""

    rewards: V1RewardsCfg = V1RewardsCfg()
