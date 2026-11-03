import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def _param(name: str):
    return ["-p", [name, ":=", LaunchConfiguration(name)]]


def _assign(name: str, value: str):
    return ["-p", [name, ":=", LaunchConfiguration(value)]]


def _package_share_root() -> Path:
    # Under --symlink-install, resolve() lands in the source tree.
    return Path(__file__).resolve().parents[1]


def _default_model_path(package_share_root: Path) -> str:
    checkpoint_dir = package_share_root / "checkpoints" / "onix_rough_standard"
    candidates = []
    for path in checkpoint_dir.glob("model_*.pt"):
        if not path.exists():
            continue
        try:
            step = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            step = -1
        candidates.append((step, path.name, path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    candidates.sort()
    return str(candidates[-1][2])


def generate_launch_description() -> LaunchDescription:
    package_prefix = get_package_prefix("onix_policy_bridge")
    package_share_root = _package_share_root()
    venv_python = os.path.join(package_prefix, "venv", "bin", "python3")
    default_python = venv_python if os.path.exists(venv_python) else sys.executable
    default_model_path = _default_model_path(package_share_root)
    default_env_yaml_path = str(package_share_root / "config" / "onix_rough_standard" / "env.yaml")

    arguments = [
        DeclareLaunchArgument("model_path", default_value=default_model_path),
        DeclareLaunchArgument("env_yaml_path", default_value=default_env_yaml_path),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("rate", default_value="20.0"),
        DeclareLaunchArgument("cmd_timeout", default_value="0.5"),
        DeclareLaunchArgument("state_timeout", default_value="0.5"),
        DeclareLaunchArgument("track_tread", default_value="0.56"),
        DeclareLaunchArgument("sprocket_radius", default_value="0.0703"),
        DeclareLaunchArgument("right_track_sign", default_value="1.0"),
        DeclareLaunchArgument("left_track_sign", default_value="1.0"),
        DeclareLaunchArgument("imu_yaw_offset", default_value="0.0"),
        DeclareLaunchArgument("cmd_deadband_linear", default_value="0.01"),
        DeclareLaunchArgument("cmd_deadband_angular", default_value="0.01"),
        DeclareLaunchArgument("use_stack_avoidance", default_value="true"),
        DeclareLaunchArgument("stuck_turn_rate_ratio", default_value="0.5"),
        DeclareLaunchArgument("stuck_detection_duration", default_value="2.0"),
        DeclareLaunchArgument("stuck_linear_random_amplitude", default_value="0.2"),
        DeclareLaunchArgument("stuck_linear_gain", default_value="1.0"),
        DeclareLaunchArgument("stuck_direction_switch_period", default_value="1.0"),
        DeclareLaunchArgument("debug_log_period", default_value="1.0"),
        DeclareLaunchArgument("use_last_action", default_value="auto"),
        DeclareLaunchArgument("use_joint_effort", default_value="auto"),
        DeclareLaunchArgument("use_directional_flipper_constraint", default_value="auto"),
        DeclareLaunchArgument("directional_flipper_switch_duration", default_value="2.0"),
        DeclareLaunchArgument("directional_flipper_turn_angular_threshold", default_value="0.2"),
        DeclareLaunchArgument("directional_flipper_turn_radius_threshold", default_value="0.25"),
        DeclareLaunchArgument("joint_effort_scale", default_value="1.0"),
        DeclareLaunchArgument("last_action_smooth_alpha", default_value="0.0"),
        DeclareLaunchArgument("use_height_scan", default_value="auto"),
        DeclareLaunchArgument("height_scan_topic", default_value="/height_scan"),
        DeclareLaunchArgument("height_scan_dim", default_value="4"),
        DeclareLaunchArgument("flipper_fold_mode_topic", default_value="/flipper_fold_mode"),
        DeclareLaunchArgument("teleop", default_value="false"),
        DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
        DeclareLaunchArgument("joy_deadzone", default_value="0.05"),
        DeclareLaunchArgument("joy_autorepeat_rate", default_value="20.0"),
        DeclareLaunchArgument("teleop_enable_button", default_value="4"),
        DeclareLaunchArgument("teleop_enable_turbo_button", default_value="5"),
        DeclareLaunchArgument("teleop_axis_linear", default_value="1"),
        DeclareLaunchArgument("teleop_scale_linear", default_value="0.6"),
        DeclareLaunchArgument("teleop_scale_linear_turbo", default_value="1.0"),
        DeclareLaunchArgument("teleop_axis_angular", default_value="0"),
        DeclareLaunchArgument("teleop_scale_angular", default_value="1.2"),
        DeclareLaunchArgument("teleop_scale_angular_turbo", default_value="2.0"),
        DeclareLaunchArgument("mode_topic", default_value="/chassis_mode_cmd"),
        DeclareLaunchArgument("mode_state_topic", default_value="/chassis_mode_state"),
        DeclareLaunchArgument("mode_state_period", default_value="1.0"),
        DeclareLaunchArgument("default_mode", default_value="manual"),
        DeclareLaunchArgument("drive_direction_topic", default_value="/chassis/direction"),
        DeclareLaunchArgument("cmd_vel_manual_topic", default_value="/cmd_vel_manual"),
        DeclareLaunchArgument("cmd_vel_manual_use_stamped", default_value="true"),
        DeclareLaunchArgument("cmd_vel_auto_topic", default_value="/cmd_vel_auto"),
        DeclareLaunchArgument("cmd_vel_auto_use_stamped", default_value="true"),
        DeclareLaunchArgument("cmd_vel_out_topic", default_value="/diff_drive_controller/cmd_vel"),
        DeclareLaunchArgument("cmd_vel_out_use_stamped", default_value="true"),
        DeclareLaunchArgument("cmd_vel_out_max_linear_velocity", default_value="0.4"),
        DeclareLaunchArgument("cmd_vel_out_max_angular_velocity", default_value="2.0"),
        DeclareLaunchArgument("cmd_vel_out_max_linear_acceleration", default_value="2.0"),
        DeclareLaunchArgument("cmd_vel_out_max_angular_acceleration", default_value="8.0"),
        DeclareLaunchArgument("cmd_vel_out_limit_dt_max", default_value="0.1"),
        DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
        DeclareLaunchArgument("imu_topic", default_value="/bno055/imu"),
        DeclareLaunchArgument("fallback_imu_topic", default_value="/livox/imu_172_16_94_141"),
        DeclareLaunchArgument("fallback_orientation_parent_frame", default_value="map"),
        DeclareLaunchArgument("fallback_orientation_child_frame", default_value="chassis/body"),
        DeclareLaunchArgument("fallback_tf_timeout", default_value="0.0"),
        DeclareLaunchArgument("flipper_cmd_topic", default_value="/flipper_joint_position_controller/commands"),
        DeclareLaunchArgument("flipper_manual_topic", default_value="/flipper_manual"),
        DeclareLaunchArgument("flipper_manual_timeout", default_value="0.5"),
        DeclareLaunchArgument("follower_status_topic", default_value="/simple_trajectory_follower/status"),
        DeclareLaunchArgument("follower_goal_flipper_angle_deg", default_value="90.0"),
        DeclareLaunchArgument("flipper_cmd_mode", default_value="position"),
        DeclareLaunchArgument("flipper_velocity_kp", default_value="3.0"),
        DeclareLaunchArgument("flipper_velocity_limit", default_value="1.5"),
        DeclareLaunchArgument("flipper_velocity_deadband", default_value="0.02"),
        DeclareLaunchArgument("flipper_manual_deadband", default_value="0.02"),
        DeclareLaunchArgument("fr_topic", default_value="/unused/flipper/fr"),
        DeclareLaunchArgument("fl_topic", default_value="/unused/flipper/fl"),
        DeclareLaunchArgument("rr_topic", default_value="/unused/flipper/rr"),
        DeclareLaunchArgument("rl_topic", default_value="/unused/flipper/rl"),
        DeclareLaunchArgument("flipper_joint_names", default_value="['chassis/rr_flipper_joint','chassis/rl_flipper_joint','chassis/fr_flipper_joint','chassis/fl_flipper_joint']"),
        DeclareLaunchArgument("flipper_cmd_joint_names", default_value="['chassis/fl_flipper_joint','chassis/fr_flipper_joint','chassis/rl_flipper_joint','chassis/rr_flipper_joint']"),
        DeclareLaunchArgument("right_track_vel_names", default_value="['chassis/fr_sprocket_axle','chassis/rr_sprocket_axle']"),
        DeclareLaunchArgument("left_track_vel_names", default_value="['chassis/fl_sprocket_axle','chassis/rl_sprocket_axle']"),
        DeclareLaunchArgument("ipc_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("ipc_port", default_value="29001"),
        DeclareLaunchArgument("python_executable", default_value=default_python),
    ]

    frontend_cmd = [
        LaunchConfiguration("python_executable"),
        "-m",
        "onix_policy_bridge.frontend",
        "--ros-args",
        *_param("use_sim_time"),
        *_param("rate"),
        *_param("cmd_timeout"),
        *_param("flipper_manual_timeout"),
        *_param("state_timeout"),
        *_param("sprocket_radius"),
        *_param("track_tread"),
        *_param("cmd_deadband_linear"),
        *_param("cmd_deadband_angular"),
        *_param("use_stack_avoidance"),
        *_param("stuck_turn_rate_ratio"),
        *_param("stuck_detection_duration"),
        *_param("stuck_linear_random_amplitude"),
        *_param("stuck_linear_gain"),
        *_param("stuck_direction_switch_period"),
        *_param("mode_topic"),
        *_param("mode_state_topic"),
        *_param("mode_state_period"),
        *_param("default_mode"),
        *_param("drive_direction_topic"),
        *_param("cmd_vel_manual_topic"),
        *_param("cmd_vel_manual_use_stamped"),
        *_param("cmd_vel_auto_topic"),
        *_param("cmd_vel_auto_use_stamped"),
        *_param("cmd_vel_out_topic"),
        *_param("cmd_vel_out_use_stamped"),
        *_param("cmd_vel_out_max_linear_velocity"),
        *_param("cmd_vel_out_max_angular_velocity"),
        *_param("cmd_vel_out_max_linear_acceleration"),
        *_param("cmd_vel_out_max_angular_acceleration"),
        *_param("cmd_vel_out_limit_dt_max"),
        *_param("joint_states_topic"),
        *_param("imu_topic"),
        *_param("fallback_imu_topic"),
        *_param("fallback_orientation_parent_frame"),
        *_param("fallback_orientation_child_frame"),
        *_param("fallback_tf_timeout"),
        *_param("height_scan_topic"),
        *_param("flipper_fold_mode_topic"),
        *_param("use_joint_effort"),
        *_param("use_height_scan"),
        *_param("height_scan_dim"),
        *_param("flipper_cmd_topic"),
        *_param("flipper_manual_topic"),
        *_param("follower_status_topic"),
        *_param("follower_goal_flipper_angle_deg"),
        *_param("flipper_cmd_mode"),
        *_param("flipper_velocity_kp"),
        *_param("flipper_velocity_limit"),
        *_param("flipper_velocity_deadband"),
        *_param("flipper_manual_deadband"),
        *_param("fr_topic"),
        *_param("fl_topic"),
        *_param("rr_topic"),
        *_param("rl_topic"),
        *_param("flipper_joint_names"),
        *_param("flipper_cmd_joint_names"),
        *_param("right_track_vel_names"),
        *_param("left_track_vel_names"),
        *_param("ipc_host"),
        *_param("ipc_port"),
    ]

    backend_cmd = [
        LaunchConfiguration("python_executable"),
        "-m",
        "onix_policy_bridge.backend",
        "--model_path",
        LaunchConfiguration("model_path"),
        "--env_yaml_path",
        LaunchConfiguration("env_yaml_path"),
        "--device",
        LaunchConfiguration("device"),
        "--track_tread",
        LaunchConfiguration("track_tread"),
        "--sprocket_radius",
        LaunchConfiguration("sprocket_radius"),
        "--right_track_sign",
        LaunchConfiguration("right_track_sign"),
        "--left_track_sign",
        LaunchConfiguration("left_track_sign"),
        "--imu_yaw_offset",
        LaunchConfiguration("imu_yaw_offset"),
        "--cmd_deadband_linear",
        LaunchConfiguration("cmd_deadband_linear"),
        "--cmd_deadband_angular",
        LaunchConfiguration("cmd_deadband_angular"),
        "--joint_effort_scale",
        LaunchConfiguration("joint_effort_scale"),
        "--last_action_smooth_alpha",
        LaunchConfiguration("last_action_smooth_alpha"),
        "--use_last_action",
        LaunchConfiguration("use_last_action"),
        "--use_joint_effort",
        LaunchConfiguration("use_joint_effort"),
        "--use_directional_flipper_constraint",
        LaunchConfiguration("use_directional_flipper_constraint"),
        "--directional_flipper_switch_duration",
        LaunchConfiguration("directional_flipper_switch_duration"),
        "--directional_flipper_turn_angular_threshold",
        LaunchConfiguration("directional_flipper_turn_angular_threshold"),
        "--directional_flipper_turn_radius_threshold",
        LaunchConfiguration("directional_flipper_turn_radius_threshold"),
        "--use_height_scan",
        LaunchConfiguration("use_height_scan"),
        "--height_scan_dim",
        LaunchConfiguration("height_scan_dim"),
        "--activation",
        "elu",
        "--ipc_host",
        LaunchConfiguration("ipc_host"),
        "--ipc_port",
        LaunchConfiguration("ipc_port"),
        "--flipper_joint_names",
        LaunchConfiguration("flipper_joint_names"),
        "--right_track_vel_names",
        LaunchConfiguration("right_track_vel_names"),
        "--left_track_vel_names",
        LaunchConfiguration("left_track_vel_names"),
    ]

    joy_cmd = [
        "ros2",
        "run",
        "joy",
        "joy_node",
        "--ros-args",
        *_param("use_sim_time"),
        *_assign("dev", "joy_dev"),
        *_assign("deadzone", "joy_deadzone"),
        *_assign("autorepeat_rate", "joy_autorepeat_rate"),
        "-r",
        ["joy:=", "/joy"],
    ]

    teleop_joy_cmd = [
        "ros2",
        "run",
        "teleop_twist_joy",
        "teleop_node",
        "--ros-args",
        *_param("use_sim_time"),
        *_assign("enable_button", "teleop_enable_button"),
        *_assign("enable_turbo_button", "teleop_enable_turbo_button"),
        "-p",
        ["axis_linear:=[0, ", LaunchConfiguration("teleop_axis_linear"), ", 0]"],
        "-p",
        ["scale_linear:=[0.0, ", LaunchConfiguration("teleop_scale_linear"), ", 0.0]"],
        "-p",
        ["scale_linear_turbo:=[0.0, ", LaunchConfiguration("teleop_scale_linear_turbo"), ", 0.0]"],
        "-p",
        ["axis_angular:=[0, 0, ", LaunchConfiguration("teleop_axis_angular"), "]"],
        "-p",
        ["scale_angular:=[0.0, 0.0, ", LaunchConfiguration("teleop_scale_angular"), "]"],
        "-p",
        ["scale_angular_turbo:=[0.0, 0.0, ", LaunchConfiguration("teleop_scale_angular_turbo"), "]"],
        "-r",
        ["joy:=", "/joy"],
        "-r",
        ["cmd_vel:=", LaunchConfiguration("cmd_vel_manual_topic")],
    ]

    keyboard_cmd = [
        "/usr/bin/xterm",
        "-e",
        [
            "ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=",
            LaunchConfiguration("cmd_vel_manual_topic"),
        ],
    ]

    return LaunchDescription(
        arguments
        + [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            ExecuteProcess(cmd=backend_cmd, output="screen"),
            ExecuteProcess(cmd=frontend_cmd, output="screen"),
            ExecuteProcess(cmd=joy_cmd, output="screen", condition=IfCondition(LaunchConfiguration("teleop"))),
            ExecuteProcess(cmd=teleop_joy_cmd, output="screen", condition=IfCondition(LaunchConfiguration("teleop"))),
            ExecuteProcess(cmd=keyboard_cmd, output="screen", condition=IfCondition(LaunchConfiguration("teleop"))),
        ]
    )
