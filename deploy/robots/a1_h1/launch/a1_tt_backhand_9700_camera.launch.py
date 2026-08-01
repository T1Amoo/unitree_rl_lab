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


BASE = _guess_lgy_root() / "Pingpong_TTRL/pretrained/a1_tt_backhand/base_9700_hitplane020"


def generate_launch_description():
    start_ball_bridge = LaunchConfiguration("start_ball_bridge")
    start_fsm = LaunchConfiguration("start_fsm")
    start_policy_bridge = LaunchConfiguration("start_policy_bridge")
    publish_actions = LaunchConfiguration("publish_actions")

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_ball_bridge", default_value="true"),
            DeclareLaunchArgument("start_fsm", default_value="true"),
            DeclareLaunchArgument("start_policy_bridge", default_value="true"),
            DeclareLaunchArgument("publish_actions", default_value="true"),
            Node(
                package="sim2real_bridge_cpp",
                executable="a1_vrpn_ball_state_bridge",
                name="a1_backhand_camera_ball_state_bridge",
                output="screen",
                condition=IfCondition(start_ball_bridge),
                parameters=[
                    {
                        "input_topic": "/pingpong_location",
                        "output_topic": "/ball/state",
                        # Camera x/y origin is the table center and camera z=0 is the tabletop.
                        # The 9700 policy uses robot_pos=(-1.8, 0.0, 0.0282), so no legacy y+0.76 shift.
                        "origin_in_training_world": [0.0, 0.0, 0.76],
                        "rotation_wxyz_to_training": [1.0, 0.0, 0.0, 0.0],
                        # Timestamp-aware alpha-beta tracker. Reject isolated
                        # stereo jumps, then extrapolate the accepted state from
                        # the ZED exposure time to local receipt time.
                        "temporal_filter_enabled": True,
                        "filter_alpha": 0.65,
                        "filter_beta": 0.10,
                        "max_innovation_m": 0.12,
                        "reacquire_innovation_m": 0.06,
                        "reacquire_max_speed_mps": 8.0,
                        # The Jetson empty-scene test is now zero-detection and
                        # this bridge still enforces trajectory consistency.
                        # Two frames recover one 60 Hz camera period of lead.
                        "acquire_frames": 2,
                        "reacquire_frames": 5,
                        "reset_gap_s": 0.25,
                        "max_source_age_s": 0.30,
                        "max_extrapolation_s": 0.16,
                        "gravity_mps2": -9.81,
                        "table_bounce_enabled": True,
                        "table_ball_center_z": 0.78,
                        "table_restitution": 0.95,
                        "max_speed_mps": 8.0,
                        "diag_every": 20,
                    }
                ],
            ),
            Node(
                package="sim2real_bridge_cpp",
                executable="a1_tt_fsm_supervisor",
                name="a1_tt_backhand_fsm_supervisor",
                output="screen",
                condition=IfCondition(start_fsm),
                parameters=[
                    {
                        "joint_state_topic": "/right_joint_states",
                        "action_topic": "/model_action",
                        "right_movej_topic": "/movej_right_angle",
                        "enable_topic": "/model_control/enable",
                        "policy_enable_topic": "/a1_tt/policy_enable",
                        "damping_topic": "/model_control/damping",
                        "joystick_topic": "/joystick_info",
                        "command_topic": "/a1_tt/fsm_command",
                        "state_topic": "/a1_tt/fsm_state",
                        "control_hz": 50.0,
                        "joint_timeout_s": 2.0,
                        "joystick_fixstand_code": 27,
                        "joystick_table_tennis_code": 28,
                        "joystick_passive_code": 21,
                        "require_ready_for_tt": True,
                        "hold_ready": True,
                        "default_q": [1.450, -0.762, -2.050, 1.445, 0.206, -0.827, 1.043],
                        "fixstand_timeout_s": 60.0,
                        "fixstand_interp_s": 2.0,
                        "fixstand_enable_settle_s": 0.5,
                        "enable_republish_ticks": 0,
                        "diag_every": 50,
                        "fixstand_use_movej": True,
                        "damping_exit_delay_s": 0.10,
                        "joystick_debounce_s": 0.50,
                    }
                ],
            ),
            Node(
                package="sim2real_bridge_cpp",
                executable="a1_policy_bridge_cpp",
                name="a1_tt_backhand_policy_bridge",
                output="screen",
                condition=IfCondition(start_policy_bridge),
                parameters=[
                    {
                        "policy_path": str(BASE / "policy/policy.onnx"),
                        "predictor_path": str(BASE / "policy/predictor.onnx"),
                        "use_predictor": True,
                        "joint_state_topic": "/right_joint_states",
                        "ball_state_topic": "/ball/state",
                        "action_topic": "/model_action",
                        "enable_topic": "/model_control/enable",
                        "policy_enable_topic": "/a1_tt/policy_enable",
                        "control_hz": 50.0,
                        "joint_timeout_s": 2.0,
                        "ball_timeout_s": 0.16,
                        # Between new camera messages, advance the filtered
                        # state at the 50 Hz policy clock instead of repeating
                        # an old point as if it were a fresh predictor sample.
                        "ball_coast_enabled": True,
                        "ball_coast_max_s": 0.12,
                        "ball_coast_gravity_mps2": -9.81,
                        "ball_coast_table_bounce_enabled": True,
                        "ball_coast_table_z": 0.78,
                        "ball_coast_table_restitution": 0.95,
                        "publish_actions": ParameterValue(publish_actions, value_type=bool),
                        # Training and the real SDK both use position targets
                        # with desired velocity/feedforward torque equal to zero.
                        "publish_position_velocity": False,
                        "policy_enabled_on_start": False,
                        "enable_on_start": False,
                        "hold_when_ball_stale": False,
                        "default_q": [1.450, -0.762, -2.050, 1.445, 0.206, -0.827, 1.043],
                        "robot_table_pos": [-1.8, 0.0, 0.0282],
                        "hit_plane_x": -1.243,
                        "home_y": 0.0,
                        "paddle_y_offset": -0.03,
                        "pred_sentinel": [-1.243, 0.041, 0.92],
                        "hit_target_y_range": [-0.025, 0.107],
                        "hit_target_z_range": [0.86, 0.98],
                        "zero_action_when_ball_invalid": True,
                        # The timestamp-aware trajectory filter is the acquire
                        # stage; do not pay another 20 ms confirmation delay.
                        # Keep the five-tick coast hysteresis on dropout.
                        "gate_confirm_frames": 1,
                        "gate_coast_frames": 5,
                        # The 9700 backhand target is y=[-0.025, 0.107].
                        # Reject the static false stereo cluster around y=-0.5
                        # and require a physically incoming ball.
                        "gate_y_abs": 0.35,
                        "gate_min_approach_vx": -0.50,
                        # Training action-target route at 50 Hz: first-order
                        # low-pass plus the matching per-joint velocity cap.
                        "servo_filter_enabled": True,
                        "servo_tau_s": [0.10, 0.10, 0.08, 0.10, 0.05, 0.05, 0.10],
                        "servo_velocity_limit": [1.0, 1.2, 1.8, 1.6, 4.0, 3.2, 8.0],
                        "qdes_slew_enabled": False,
                        "max_delta_per_tick": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "diag_every": 50,
                    }
                ],
            ),
        ]
    )
