"""Left-right symmetry augmentation for the WOAN4310 velocity task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "build_left_right_joint_map",
    "compute_symmetric_states",
    "reflect_critic_observation",
    "reflect_joint_data",
    "reflect_policy_observation",
]


_POLICY_OBS_DIM = 45
_CRITIC_OBS_DIM = 60
_ACTION_DIM = 12
_MIRRORED_LEG = {"ZQ": "YQ", "YQ": "ZQ", "ZH": "YH", "YH": "ZH"}
_EXPECTED_JOINT_NAMES = {
    f"joint_{leg}{joint_number}" for leg in _MIRRORED_LEG for joint_number in (2, 3, 4)
}
_EXPECTED_POLICY_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)
_EXPECTED_CRITIC_TERMS = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "joint_effort",
    "last_action",
)
_RUNTIME_CACHE_ATTR = "_woan4310_left_right_joint_names"


def _mirrored_joint_name(joint_name: str) -> str:
    """Return the joint name reflected across the robot's sagittal plane."""
    if not joint_name.startswith("joint_"):
        raise ValueError(f"Unexpected WOAN4310 joint name: {joint_name!r}")

    suffix = joint_name.removeprefix("joint_")
    leg_name, joint_number = suffix[:2], suffix[2:]
    if leg_name not in _MIRRORED_LEG or joint_number not in {"2", "3", "4"}:
        raise ValueError(f"Unexpected WOAN4310 joint name: {joint_name!r}")
    return f"joint_{_MIRRORED_LEG[leg_name]}{joint_number}"


def build_left_right_joint_map(
    joint_names: Sequence[str],
) -> tuple[list[int], list[float]]:
    """Build a name-based left-right permutation and sign vector.

    Joint 2 is the hip abduction/adduction axis and changes sign under a
    sagittal-plane reflection. Joints 3 and 4 only exchange legs.
    """
    names = tuple(joint_names)
    if (
        len(names) != _ACTION_DIM
        or len(set(names)) != _ACTION_DIM
        or set(names) != _EXPECTED_JOINT_NAMES
    ):
        raise ValueError(
            "WOAN4310 symmetry requires exactly the 12 active joints "
            f"{sorted(_EXPECTED_JOINT_NAMES)}, received {list(names)}"
        )

    index_by_name = {name: index for index, name in enumerate(names)}
    source_indices = [index_by_name[_mirrored_joint_name(name)] for name in names]
    signs = [-1.0 if name.endswith("2") else 1.0 for name in names]
    return source_indices, signs


def reflect_joint_data(
    joint_data: torch.Tensor, joint_names: Sequence[str]
) -> torch.Tensor:
    """Reflect joint-aligned data using the runtime joint-name ordering."""
    if joint_data.shape[-1] != _ACTION_DIM:
        raise ValueError(
            f"Expected {_ACTION_DIM} joint values, received shape {tuple(joint_data.shape)}"
        )

    source_indices, signs = build_left_right_joint_map(joint_names)
    source_indices_tensor = torch.tensor(
        source_indices, device=joint_data.device, dtype=torch.long
    )
    signs_tensor = torch.tensor(signs, device=joint_data.device, dtype=joint_data.dtype)
    return joint_data.index_select(-1, source_indices_tensor) * signs_tensor


def _multiply_vector(
    vector: torch.Tensor, signs: tuple[float, float, float]
) -> torch.Tensor:
    signs_tensor = torch.tensor(signs, device=vector.device, dtype=vector.dtype)
    return vector * signs_tensor


def reflect_policy_observation(
    policy_obs: torch.Tensor, joint_names: Sequence[str]
) -> torch.Tensor:
    """Reflect the 45-dimensional WOAN4310 actor observation."""
    if policy_obs.shape[-1] != _POLICY_OBS_DIM:
        raise ValueError(
            f"WOAN4310 v1 symmetry expects policy observations of size {_POLICY_OBS_DIM}, "
            f"received shape {tuple(policy_obs.shape)}"
        )

    reflected = policy_obs.clone()
    reflected[..., 0:3] = _multiply_vector(policy_obs[..., 0:3], (-1.0, 1.0, -1.0))
    reflected[..., 3:6] = _multiply_vector(policy_obs[..., 3:6], (1.0, -1.0, 1.0))
    reflected[..., 6:9] = _multiply_vector(policy_obs[..., 6:9], (1.0, -1.0, -1.0))
    reflected[..., 9:21] = reflect_joint_data(policy_obs[..., 9:21], joint_names)
    reflected[..., 21:33] = reflect_joint_data(policy_obs[..., 21:33], joint_names)
    reflected[..., 33:45] = reflect_joint_data(policy_obs[..., 33:45], joint_names)
    return reflected


def reflect_critic_observation(
    critic_obs: torch.Tensor, joint_names: Sequence[str]
) -> torch.Tensor:
    """Reflect the 60-dimensional WOAN4310 privileged critic observation."""
    if critic_obs.shape[-1] != _CRITIC_OBS_DIM:
        raise ValueError(
            f"WOAN4310 v1 symmetry expects critic observations of size {_CRITIC_OBS_DIM}, "
            f"received shape {tuple(critic_obs.shape)}"
        )

    reflected = critic_obs.clone()
    reflected[..., 0:3] = _multiply_vector(critic_obs[..., 0:3], (1.0, -1.0, 1.0))
    reflected[..., 3:6] = _multiply_vector(critic_obs[..., 3:6], (-1.0, 1.0, -1.0))
    reflected[..., 6:9] = _multiply_vector(critic_obs[..., 6:9], (1.0, -1.0, 1.0))
    reflected[..., 9:12] = _multiply_vector(critic_obs[..., 9:12], (1.0, -1.0, -1.0))
    reflected[..., 12:24] = reflect_joint_data(critic_obs[..., 12:24], joint_names)
    reflected[..., 24:36] = reflect_joint_data(critic_obs[..., 24:36], joint_names)
    reflected[..., 36:48] = reflect_joint_data(critic_obs[..., 36:48], joint_names)
    reflected[..., 48:60] = reflect_joint_data(critic_obs[..., 48:60], joint_names)
    return reflected


def _resolve_and_validate_runtime_joint_names(
    env: ManagerBasedRLEnv,
) -> tuple[str, ...]:
    """Resolve the runtime ABI once and fail fast if the action ordering changes."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    joint_names = tuple(robot.joint_names)

    cached_joint_names = getattr(unwrapped, _RUNTIME_CACHE_ATTR, None)
    if cached_joint_names is not None:
        if cached_joint_names != joint_names:
            raise RuntimeError(
                "WOAN4310 runtime joint ordering changed after symmetry initialization"
            )
        return joint_names

    build_left_right_joint_map(joint_names)

    active_observation_terms = unwrapped.observation_manager.active_terms
    policy_terms = tuple(active_observation_terms.get("policy", ()))
    critic_terms = tuple(active_observation_terms.get("critic", ()))
    if policy_terms != _EXPECTED_POLICY_TERMS or critic_terms != _EXPECTED_CRITIC_TERMS:
        raise RuntimeError(
            "WOAN4310 observation term ordering changed; update and re-test the symmetry slices: "
            f"policy={list(policy_terms)}, critic={list(critic_terms)}"
        )

    action_terms = unwrapped.action_manager.active_terms
    if len(action_terms) != 1:
        raise RuntimeError(
            f"WOAN4310 symmetry expects one action term, received {action_terms}"
        )
    action_term = unwrapped.action_manager.get_term(action_terms[0])
    action_joint_names = tuple(getattr(action_term, "_joint_names", ()))
    if action_joint_names != joint_names:
        raise RuntimeError(
            "WOAN4310 observation and action joint orderings differ: "
            f"observation={list(joint_names)}, action={list(action_joint_names)}"
        )

    default_joint_pos = robot.data.default_joint_pos[0]
    reflected_default_joint_pos = reflect_joint_data(default_joint_pos, joint_names)
    if not torch.allclose(
        default_joint_pos, reflected_default_joint_pos, atol=1.0e-6, rtol=0.0
    ):
        raise RuntimeError("WOAN4310 default joint pose is not left-right symmetric")

    setattr(unwrapped, _RUNTIME_CACHE_ATTR, joint_names)
    return joint_names


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Return original and left-right-reflected samples for RSL-RL PPO."""
    joint_names = _resolve_and_validate_runtime_joint_names(env)

    if obs is not None:
        if "policy" not in obs or "critic" not in obs:
            raise KeyError(
                "WOAN4310 v1 symmetry requires both 'policy' and 'critic' observation groups"
            )
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][batch_size:] = reflect_policy_observation(
            obs["policy"], joint_names
        )
        obs_aug["critic"][batch_size:] = reflect_critic_observation(
            obs["critic"], joint_names
        )
    else:
        obs_aug = None

    if actions is not None:
        if actions.shape[-1] != _ACTION_DIM:
            raise ValueError(
                f"WOAN4310 v1 symmetry expects actions of size {_ACTION_DIM}, received shape {tuple(actions.shape)}"
            )
        actions_aug = torch.cat(
            (actions, reflect_joint_data(actions, joint_names)), dim=0
        )
    else:
        actions_aug = None

    return obs_aug, actions_aug
