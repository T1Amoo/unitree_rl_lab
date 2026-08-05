from pathlib import Path
import subprocess

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.logging import get_logger
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
    / "Pingpong_TTRL/logs/a1_tt_backhand_real_v2_r115_netclear_highslow_paddle075/2026-08-03_11-14-53_scratch_r115_netclear_highslow_paddle075_camera_tau_delay_5k10k5k/exported_model_15500/policy.onnx"
)


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clock_preflight(context):
    enabled = _is_true(LaunchConfiguration("clock_preflight_enabled").perform(context))
    camera_enabled = _is_true(LaunchConfiguration("start_vrpn_ball_bridge").perform(context))
    relative_timing = _is_true(LaunchConfiguration("relative_camera_timing_enabled").perform(context))
    if camera_enabled and relative_timing:
        get_logger("a1_clock_preflight").info(
            "Jetson-relative camera timing enabled; absolute clock offset is diagnostic-only"
        )
        return []
    if not enabled or not camera_enabled:
        get_logger("a1_clock_preflight").warning(
            "Jetson clock preflight skipped; this is allowed only for offline/no-camera diagnostics"
        )
        return []

    package_prefix = Path(get_package_prefix("sim2real_bridge_cpp"))
    checker = package_prefix / "lib/sim2real_bridge_cpp/check_jetson_clock_sync.py"
    command = [
        str(checker),
        "--host", LaunchConfiguration("clock_preflight_host").perform(context),
        "--user", LaunchConfiguration("clock_preflight_user").perform(context),
        "--samples", LaunchConfiguration("clock_preflight_samples").perform(context),
        "--max-offset-ms", LaunchConfiguration("clock_preflight_max_offset_ms").perform(context),
        "--max-rtt-ms", LaunchConfiguration("clock_preflight_max_rtt_ms").perform(context),
        "--max-spread-ms", LaunchConfiguration("clock_preflight_max_spread_ms").perform(context),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    detail = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise RuntimeError(
            "Jetson/local clock preflight failed; fail-closed before starting deployment nodes. "
            + (detail or f"checker rc={result.returncode}")
        )
    get_logger("a1_clock_preflight").info(detail)
    return []


def generate_launch_description():
    policy_path = LaunchConfiguration("policy_path")
    predictor_path = LaunchConfiguration("predictor_path")
    start_arm_control = LaunchConfiguration("start_arm_control")
    start_vrpn_ball_bridge = LaunchConfiguration("start_vrpn_ball_bridge")
    start_fsm = LaunchConfiguration("start_fsm")
    start_policy_bridge = LaunchConfiguration("start_policy_bridge")
    publish_actions = LaunchConfiguration("publish_actions")
    publish_position_velocity = LaunchConfiguration("publish_position_velocity")
    policy_enabled_on_start = LaunchConfiguration("policy_enabled_on_start")
    enable_on_start = LaunchConfiguration("enable_on_start")
    enable_motors_on_start = LaunchConfiguration("enable_motors_on_start")
    right_arm_device = LaunchConfiguration("right_arm_device")
    fixstand_use_movej = LaunchConfiguration("fixstand_use_movej")
    fixstand_enable_settle_s = LaunchConfiguration("fixstand_enable_settle_s")
    vrpn_ball_topic = LaunchConfiguration("vrpn_ball_topic")
    relative_camera_topic = LaunchConfiguration("relative_camera_topic")
    relative_camera_timing_enabled = LaunchConfiguration("relative_camera_timing_enabled")
    relative_camera_transport_delay_s = LaunchConfiguration("relative_camera_transport_delay_s")
    ball_state_topic = LaunchConfiguration("ball_state_topic")
    test_enabled = LaunchConfiguration("test_enabled")
    test_signal_type = LaunchConfiguration("test_signal_type")
    test_joint_index = LaunchConfiguration("test_joint_index")
    test_freq_hz = LaunchConfiguration("test_freq_hz")
    test_chirp_start_hz = LaunchConfiguration("test_chirp_start_hz")
    test_chirp_end_hz = LaunchConfiguration("test_chirp_end_hz")
    test_amplitude_rad = LaunchConfiguration("test_amplitude_rad")
    test_cycles = LaunchConfiguration("test_cycles")
    test_duration_s = LaunchConfiguration("test_duration_s")
    test_warmup_s = LaunchConfiguration("test_warmup_s")
    test_post_hold_s = LaunchConfiguration("test_post_hold_s")
    test_ramp_s = LaunchConfiguration("test_ramp_s")

    return LaunchDescription(
        [
            DeclareLaunchArgument("policy_path", default_value=str(DEFAULT_POLICY)),
            DeclareLaunchArgument("predictor_path", default_value=""),
            DeclareLaunchArgument("start_arm_control", default_value="false"),
            DeclareLaunchArgument("start_vrpn_ball_bridge", default_value="true"),
            DeclareLaunchArgument("start_fsm", default_value="true"),
            DeclareLaunchArgument("start_policy_bridge", default_value="true"),
            DeclareLaunchArgument("publish_actions", default_value="true"),
            DeclareLaunchArgument("publish_position_velocity", default_value="false"),
            DeclareLaunchArgument("policy_enabled_on_start", default_value="false"),
            DeclareLaunchArgument("enable_on_start", default_value="false"),
            DeclareLaunchArgument("enable_motors_on_start", default_value="false"),
            DeclareLaunchArgument("right_arm_device", default_value="/dev/ttyACM1"),
            DeclareLaunchArgument("fixstand_use_movej", default_value="true"),
            DeclareLaunchArgument("fixstand_enable_settle_s", default_value="0.5"),
            DeclareLaunchArgument("vrpn_ball_topic", default_value="/pingpong_location"),
            DeclareLaunchArgument("relative_camera_topic", default_value="/pingpong_location_relative"),
            DeclareLaunchArgument("relative_camera_timing_enabled", default_value="true"),
            DeclareLaunchArgument("relative_camera_transport_delay_s", default_value="0.015"),
            DeclareLaunchArgument("ball_state_topic", default_value="/ball/state"),
            DeclareLaunchArgument("test_enabled", default_value="false"),
            DeclareLaunchArgument("test_signal_type", default_value="sine"),
            DeclareLaunchArgument("test_joint_index", default_value="1"),
            DeclareLaunchArgument("test_freq_hz", default_value="0.5"),
            DeclareLaunchArgument("test_chirp_start_hz", default_value="0.1"),
            DeclareLaunchArgument("test_chirp_end_hz", default_value="3.0"),
            DeclareLaunchArgument("test_amplitude_rad", default_value="0.12"),
            DeclareLaunchArgument("test_cycles", default_value="8.0"),
            DeclareLaunchArgument("test_duration_s", default_value="0.0"),
            DeclareLaunchArgument("test_warmup_s", default_value="2.0"),
            DeclareLaunchArgument("test_post_hold_s", default_value="1.0"),
            DeclareLaunchArgument("test_ramp_s", default_value="0.5"),
            DeclareLaunchArgument("clock_preflight_enabled", default_value="true"),
            DeclareLaunchArgument("clock_preflight_host", default_value="192.168.1.231"),
            DeclareLaunchArgument("clock_preflight_user", default_value="jetson"),
            DeclareLaunchArgument("clock_preflight_samples", default_value="7"),
            DeclareLaunchArgument("clock_preflight_max_offset_ms", default_value="10.0"),
            DeclareLaunchArgument("clock_preflight_max_rtt_ms", default_value="40.0"),
            DeclareLaunchArgument("clock_preflight_max_spread_ms", default_value="6.0"),
            OpaqueFunction(function=_clock_preflight),
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
                        "interpolation_mode": "none",
                        "action_timeout_s": 0.15,
                        "kps": [300.0, 300.0, 300.0, 120.0, 120.0, 120.0, 60.0],
                        "kds": [3.5, 3.5, 3.5, 1.0, 1.0, 1.0, 0.5],
                        "torque_ff_scale": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "enable_mit_velocity": False,
                        # The trained/deployed q_des route is the per-joint tau_s
                        # filter in a1_policy_bridge_cpp.  armcontrol currently
                        # requires seven finite safety-limit entries, so use
                        # deliberately non-binding values instead of adding a
                        # second slew/velocity/acceleration trajectory filter.
                        "max_vel": [1000.0] * 7,
                        "max_acc": [100000.0] * 7,
                        "max_delta_per_cycle": [100.0] * 7,
                    }
                ],
            ),
            Node(
                package="sim2real_bridge_cpp",
                executable="a1_vrpn_ball_state_bridge",
                name="a1_backhand_camera_ball_state_bridge",
                output="screen",
                condition=IfCondition(start_vrpn_ball_bridge),
                parameters=[
                    {
                        "input_topic": vrpn_ball_topic,
                        "relative_input_topic": relative_camera_topic,
                        "relative_timing_enabled": ParameterValue(
                            relative_camera_timing_enabled, value_type=bool
                        ),
                        "relative_transport_delay_s": ParameterValue(
                            relative_camera_transport_delay_s, value_type=float
                        ),
                        "output_topic": ball_state_topic,
                        "origin_in_training_world": [0.0, 0.0, 0.76],
                        "rotation_wxyz_to_training": [1.0, 0.0, 0.0, 0.0],
                        "temporal_filter_enabled": True,
                        "filter_alpha": 0.65,
                        "filter_beta": 0.10,
                        "max_innovation_m": 0.12,
                        "reacquire_innovation_m": 0.06,
                        "reacquire_max_speed_mps": 8.0,
                        "acquire_frames": 2,
                        "reacquire_frames": 5,
                        "reset_gap_s": 0.25,
                        "min_source_age_s": 0.01,
                        "max_source_age_s": 0.09,
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
                        "joint_timeout_s": 10.0,
                        "joystick_fixstand_code": 27,
                        "joystick_table_tennis_code": 28,
                        "joystick_passive_code": 21,
                        "require_ready_for_tt": True,
                        "hold_ready": True,
                        "default_q": [1.450, -0.762, -2.050, 1.445, 0.206, -0.827, 1.043],
                        "fixstand_timeout_s": 60.0,
                        "fixstand_interp_s": 2.0,
                        "fixstand_enable_settle_s": ParameterValue(fixstand_enable_settle_s, value_type=float),
                        "enable_republish_ticks": 0,
                        "diag_every": 10,
                        "fixstand_use_movej": ParameterValue(fixstand_use_movej, value_type=bool),
                        "damping_exit_delay_s": 0.10,
                        "joystick_debounce_s": 0.50,
                        "test_enabled": ParameterValue(test_enabled, value_type=bool),
                        "require_ready_for_test": True,
                        "test_signal_type": test_signal_type,
                        "test_joint_index": ParameterValue(test_joint_index, value_type=int),
                        "test_freq_hz": ParameterValue(test_freq_hz, value_type=float),
                        "test_chirp_start_hz": ParameterValue(test_chirp_start_hz, value_type=float),
                        "test_chirp_end_hz": ParameterValue(test_chirp_end_hz, value_type=float),
                        "test_amplitude_rad": ParameterValue(test_amplitude_rad, value_type=float),
                        "test_cycles": ParameterValue(test_cycles, value_type=float),
                        "test_duration_s": ParameterValue(test_duration_s, value_type=float),
                        "test_warmup_s": ParameterValue(test_warmup_s, value_type=float),
                        "test_post_hold_s": ParameterValue(test_post_hold_s, value_type=float),
                        "test_ramp_s": ParameterValue(test_ramp_s, value_type=float),
                        "test_done_to_ready": True,
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
                        "policy_path": policy_path,
                        "predictor_path": predictor_path,
                        "joint_state_topic": "/right_joint_states",
                        "ball_state_topic": ball_state_topic,
                        "action_topic": "/model_action",
                        "enable_topic": "/model_control/enable",
                        "policy_enable_topic": "/a1_tt/policy_enable",
                        "control_hz": 50.0,
                        "publish_actions": ParameterValue(publish_actions, value_type=bool),
                        "publish_position_velocity": ParameterValue(publish_position_velocity, value_type=bool),
                        "policy_enabled_on_start": ParameterValue(policy_enabled_on_start, value_type=bool),
                        "enable_on_start": ParameterValue(enable_on_start, value_type=bool),
                        "hold_when_ball_stale": False,
                        "ball_timeout_s": 0.16,
                        "ball_coast_enabled": True,
                        "ball_coast_max_s": 0.12,
                        "ball_coast_gravity_mps2": -9.81,
                        "ball_coast_table_bounce_enabled": True,
                        "ball_coast_table_z": 0.78,
                        "ball_coast_table_restitution": 0.95,
                        "default_q": [1.450, -0.762, -2.050, 1.445, 0.206, -0.827, 1.043],
                        "robot_table_pos": [-1.8, 0.0, 0.0282],
                        "hit_plane_x": -1.243,
                        "home_y": 0.0,
                        "paddle_y_offset": -0.03,
                        "pred_sentinel": [-1.243, -0.03, 0.228],
                        "hit_target_y_range": [-0.06, 0.20],
                        "hit_target_z_range": [0.84, 1.38],
                        "zero_action_when_ball_invalid": True,
                        "gate_confirm_frames": 1,
                        "gate_coast_frames": 5,
                        "gate_y_abs": 0.35,
                        "gate_min_approach_vx": -0.50,
                        "servo_filter_enabled": True,
                        "servo_tau_s": [0.10, 0.10, 0.08, 0.10, 0.05, 0.05, 0.10],
                        "servo_velocity_limit": [1.0, 1.2, 1.8, 1.6, 4.0, 3.2, 8.0],
                        "qdes_slew_enabled": False,
                        "max_delta_per_tick": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    }
                ],
            ),
        ]
    )
