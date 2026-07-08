from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _guess_lgy_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Pingpong_TTRL").exists() and (parent / "unitree_rl_lab").exists():
            return parent
    return Path.cwd()


DEFAULT_POLICY = (
    _guess_lgy_root()
    / "Pingpong_TTRL/logs/a1_tt_v11/2026-07-07_10-48-31/exported/policy.onnx"
)


def generate_launch_description():
    policy_path = LaunchConfiguration("policy_path")
    start_arm_control = LaunchConfiguration("start_arm_control")
    publish_actions = LaunchConfiguration("publish_actions")
    publish_position_velocity = LaunchConfiguration("publish_position_velocity")
    enable_on_start = LaunchConfiguration("enable_on_start")
    enable_motors_on_start = LaunchConfiguration("enable_motors_on_start")
    right_arm_device = LaunchConfiguration("right_arm_device")
    ball_state_topic = LaunchConfiguration("ball_state_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument("policy_path", default_value=str(DEFAULT_POLICY)),
            DeclareLaunchArgument("start_arm_control", default_value="false"),
            DeclareLaunchArgument("publish_actions", default_value="true"),
            DeclareLaunchArgument("publish_position_velocity", default_value="false"),
            DeclareLaunchArgument("enable_on_start", default_value="false"),
            DeclareLaunchArgument("enable_motors_on_start", default_value="false"),
            DeclareLaunchArgument("right_arm_device", default_value="/dev/ttyCANR"),
            DeclareLaunchArgument("ball_state_topic", default_value="/ball/state"),
            Node(
                package="armcontrol",
                executable="inference_arm_control_node",
                name="inference_arm_control_node",
                output="screen",
                condition=IfCondition(start_arm_control),
                parameters=[
                    {
                        "controlled_arms": "right",
                        "control_rate_hz": 100.0,
                        "expected_action_rate_hz": 50.0,
                        "action_topic": "/model_action",
                        "enable_topic": "/model_control/enable",
                        "right_arm_device": right_arm_device,
                        "servo_enabled_on_start": False,
                        "enable_motors_on_start": ParameterValue(enable_motors_on_start, value_type=bool),
                        "publish_joint_states": True,
                        "action_format": "auto",
                        "interpolation_mode": "hermite",
                        "action_timeout_s": 0.15,
                    }
                ],
            ),
            Node(
                package="sim2real_bridge",
                executable="a1_policy_bridge",
                name="a1_policy_bridge",
                output="screen",
                parameters=[
                    {
                        "policy_path": policy_path,
                        "joint_state_topic": "/right_joint_states",
                        "ball_state_topic": ball_state_topic,
                        "action_topic": "/model_action",
                        "enable_topic": "/model_control/enable",
                        "control_hz": 50.0,
                        "publish_actions": ParameterValue(publish_actions, value_type=bool),
                        "publish_position_velocity": ParameterValue(publish_position_velocity, value_type=bool),
                        "enable_on_start": ParameterValue(enable_on_start, value_type=bool),
                        "hold_when_ball_stale": True,
                    }
                ],
            ),
        ]
    )
