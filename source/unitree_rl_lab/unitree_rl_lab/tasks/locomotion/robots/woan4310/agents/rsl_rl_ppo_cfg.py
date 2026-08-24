from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class WOAN4310PPORunnerCfg(BasePPORunnerCfg):
    """Unitree locomotion PPO baseline with a dedicated WOAN4310 log namespace."""

    experiment_name = "woan4310_velocity"
