import gymnasium as gym

gym.register(
    id="Unitree-WOAN4310-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:WOAN4310PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-WOAN4310-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RobotRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rough_env_cfg:RobotRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:WOAN4310RoughPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-WOAN4310-Velocity-Symmetry-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v1_env_cfg:RobotV1EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.v1_env_cfg:RobotV1PlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:WOAN4310V1PPORunnerCfg",
    },
)
