from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg

from ..symmetry import compute_symmetric_states


@configclass
class WOAN4310PPORunnerCfg(BasePPORunnerCfg):
    """Unitree locomotion PPO baseline with a dedicated WOAN4310 log namespace."""

    experiment_name = "woan4310_velocity"


@configclass
class WOAN4310RoughPPORunnerCfg(BasePPORunnerCfg):
    """WOAN4310 rough-terrain runs use a separate log namespace."""

    experiment_name = "woan4310_rough"


@configclass
class WOAN4310V1PPORunnerCfg(BasePPORunnerCfg):
    """WOAN4310 v1 with left-right PPO data augmentation."""

    experiment_name = "woan4310_velocity_symmetry_v1"

    def __post_init__(self):
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_symmetric_states,
            mirror_loss_coeff=0.0,
        )
