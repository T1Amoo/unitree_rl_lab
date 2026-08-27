"""Rough triangle-mesh terrain variants for WOAN4310 locomotion."""

import copy

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .velocity_env_cfg import RobotEnvCfg


# Adapted from LeggedLab's gravel terrain. Isaac Lab converts this sampled
# height field to a collision/render triangle mesh before spawning it.
WOAN4310_RANDOM_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    curriculum=False,
    size=(4.0, 4.0),
    border_width=10.0,
    num_rows=16,
    num_cols=16,
    color_scheme="none",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(-0.02, 0.04),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)


@configclass
class RobotRoughEnvCfg(RobotEnvCfg):
    """WOAN4310 velocity task on random rough triangle meshes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = copy.deepcopy(WOAN4310_RANDOM_ROUGH_TERRAINS_CFG)
        self.scene.terrain.max_init_terrain_level = None
        # Random-uniform height fields do not use the difficulty input.
        self.curriculum.terrain_levels = None


@configclass
class RobotRoughPlayEnvCfg(RobotRoughEnvCfg):
    """Deterministic 256-environment viewer for a flat-trained checkpoint."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 256
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_generator.num_rows = 16
        self.scene.terrain.terrain_generator.num_cols = 16
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.26, 0.29, 0.26), roughness=0.9
        )

        # Isolate terrain generalization from observation and physics randomization.
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params = {
            "pose_range": {},
            "velocity_range": {},
        }

        # Exercise the complete command envelope learned by the flat baseline.
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges = copy.deepcopy(self.commands.base_velocity.limit_ranges)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
