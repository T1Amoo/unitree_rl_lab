"""Pure-Torch tests for the WOAN4310 left-right symmetry contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


SYMMETRY_PATH = (
    Path(__file__).parents[1]
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/woan4310/symmetry.py"
)
SPEC = importlib.util.spec_from_file_location(
    "woan4310_symmetry_under_test", SYMMETRY_PATH
)
assert SPEC is not None and SPEC.loader is not None
symmetry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = symmetry
SPEC.loader.exec_module(symmetry)


RUNTIME_JOINT_NAMES = (
    "joint_YH2",
    "joint_YQ2",
    "joint_ZH2",
    "joint_ZQ2",
    "joint_YH3",
    "joint_YQ3",
    "joint_ZH3",
    "joint_ZQ3",
    "joint_YH4",
    "joint_YQ4",
    "joint_ZH4",
    "joint_ZQ4",
)
POLICY_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)
CRITIC_TERMS = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "joint_effort",
    "last_action",
)


class _FakeActionManager:
    active_terms = ["JointPositionAction"]

    def __init__(self, joint_names: tuple[str, ...]):
        self._term = SimpleNamespace(_joint_names=list(joint_names))

    def get_term(self, name: str):
        assert name == "JointPositionAction"
        return self._term


class _FakeEnv:
    def __init__(self):
        default_joint_pos = torch.tensor(
            [
                [
                    0.0 if name.endswith("2") else 0.8 if name.endswith("3") else -1.5
                    for name in RUNTIME_JOINT_NAMES
                ]
            ]
        )
        robot = SimpleNamespace(
            joint_names=list(RUNTIME_JOINT_NAMES),
            data=SimpleNamespace(default_joint_pos=default_joint_pos),
        )
        self.unwrapped = SimpleNamespace(
            scene={"robot": robot},
            action_manager=_FakeActionManager(RUNTIME_JOINT_NAMES),
            observation_manager=SimpleNamespace(
                active_terms={
                    "policy": list(POLICY_TERMS),
                    "critic": list(CRITIC_TERMS),
                }
            ),
        )


def test_runtime_joint_map_matches_verified_physx_order():
    indices, signs = symmetry.build_left_right_joint_map(RUNTIME_JOINT_NAMES)
    assert indices == [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]
    assert signs == [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_joint_reflection_sentinel_and_involution():
    values = torch.arange(1.0, 13.0).unsqueeze(0)
    reflected = symmetry.reflect_joint_data(values, RUNTIME_JOINT_NAMES)
    expected = torch.tensor(
        [[-3.0, -4.0, -1.0, -2.0, 7.0, 8.0, 5.0, 6.0, 11.0, 12.0, 9.0, 10.0]]
    )
    torch.testing.assert_close(reflected, expected)
    torch.testing.assert_close(
        symmetry.reflect_joint_data(reflected, RUNTIME_JOINT_NAMES), values
    )


@pytest.mark.parametrize(
    ("size", "reflect"),
    [
        (45, symmetry.reflect_policy_observation),
        (60, symmetry.reflect_critic_observation),
    ],
)
def test_observation_reflection_is_an_involution(size, reflect):
    torch.manual_seed(42)
    observation = torch.randn(7, size)
    reflected_twice = reflect(
        reflect(observation, RUNTIME_JOINT_NAMES), RUNTIME_JOINT_NAMES
    )
    torch.testing.assert_close(reflected_twice, observation)


def test_policy_and_critic_vector_signs():
    policy = torch.ones(1, 45)
    critic = torch.ones(1, 60)
    reflected_policy = symmetry.reflect_policy_observation(policy, RUNTIME_JOINT_NAMES)
    reflected_critic = symmetry.reflect_critic_observation(critic, RUNTIME_JOINT_NAMES)

    torch.testing.assert_close(
        reflected_policy[0, 0:3], torch.tensor([-1.0, 1.0, -1.0])
    )
    torch.testing.assert_close(reflected_policy[0, 3:6], torch.tensor([1.0, -1.0, 1.0]))
    torch.testing.assert_close(
        reflected_policy[0, 6:9], torch.tensor([1.0, -1.0, -1.0])
    )
    torch.testing.assert_close(reflected_critic[0, 0:3], torch.tensor([1.0, -1.0, 1.0]))
    torch.testing.assert_close(
        reflected_critic[0, 3:6], torch.tensor([-1.0, 1.0, -1.0])
    )
    torch.testing.assert_close(
        reflected_critic[0, 9:12], torch.tensor([1.0, -1.0, -1.0])
    )


def test_tensor_dict_augmentation_mirrors_actor_critic_and_actions():
    torch.manual_seed(7)
    batch_size = 5
    policy = torch.randn(batch_size, 45)
    critic = torch.randn(batch_size, 60)
    extra = torch.randn(batch_size, 2)
    actions = torch.randn(batch_size, 12)
    observations = TensorDict(
        {"policy": policy, "critic": critic, "extra": extra},
        batch_size=[batch_size],
    )

    observations_aug, actions_aug = symmetry.compute_symmetric_states(
        env=_FakeEnv(), obs=observations, actions=actions
    )
    assert observations_aug is not None and actions_aug is not None
    assert observations_aug.batch_size == torch.Size([2 * batch_size])
    torch.testing.assert_close(observations_aug["policy"][:batch_size], policy)
    torch.testing.assert_close(observations_aug["critic"][:batch_size], critic)
    torch.testing.assert_close(observations_aug["extra"], extra.repeat(2, 1))
    torch.testing.assert_close(
        observations_aug["policy"][batch_size:],
        symmetry.reflect_policy_observation(policy, RUNTIME_JOINT_NAMES),
    )
    torch.testing.assert_close(
        observations_aug["critic"][batch_size:],
        symmetry.reflect_critic_observation(critic, RUNTIME_JOINT_NAMES),
    )
    torch.testing.assert_close(actions_aug[:batch_size], actions)
    torch.testing.assert_close(
        actions_aug[batch_size:],
        symmetry.reflect_joint_data(actions, RUNTIME_JOINT_NAMES),
    )


def test_none_inputs_and_shape_changes_fail_explicitly():
    observations_aug, actions_aug = symmetry.compute_symmetric_states(
        env=_FakeEnv(), obs=None, actions=None
    )
    assert observations_aug is None and actions_aug is None

    with pytest.raises(ValueError, match="policy observations of size 45"):
        symmetry.reflect_policy_observation(torch.zeros(1, 225), RUNTIME_JOINT_NAMES)
    with pytest.raises(ValueError, match="critic observations of size 60"):
        symmetry.reflect_critic_observation(torch.zeros(1, 61), RUNTIME_JOINT_NAMES)
    with pytest.raises(ValueError, match="Expected 12 joint values"):
        symmetry.reflect_joint_data(torch.zeros(1, 11), RUNTIME_JOINT_NAMES)


def test_runtime_observation_term_reordering_fails_explicitly():
    env = _FakeEnv()
    env.unwrapped.observation_manager.active_terms["policy"][0:2] = reversed(
        env.unwrapped.observation_manager.active_terms["policy"][0:2]
    )
    with pytest.raises(RuntimeError, match="observation term ordering changed"):
        symmetry.compute_symmetric_states(env=env, obs=None, actions=None)
