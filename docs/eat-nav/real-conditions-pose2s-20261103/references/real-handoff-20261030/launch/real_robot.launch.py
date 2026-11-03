"""Preview by default. Existing policy bridge remains externally managed."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,EmitEvent,RegisterEventHandler,OpaqueFunction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def make_nodes(context):
    def value(name):return LaunchConfiguration(name).perform(context)
    params=value('params_file');enable=ParameterValue(LaunchConfiguration('enable_robot'),value_type=bool)
    planner=Node(package='eat_nav_real_robot',executable='planner',name='eat_nav_planner',output='screen',
        parameters=[params,dict(enable_robot=enable,repository=value('repository'),checkpoint=value('checkpoint'),
            navigation_mode=value('navigation_mode'),
            stall_risk_only=ParameterValue(LaunchConfiguration('stall_risk_only'),value_type=bool),
            replan_period_sec=ParameterValue(LaunchConfiguration('replan_period_sec'),value_type=float),
            checkpoint_sha256=value('checkpoint_sha256'),torch_site_packages=value('torch_site_packages'),
            odom_topic=value('odom_topic'),height_layer=value('height_layer'),
            use_sim_time=ParameterValue(LaunchConfiguration('use_sim_time'),value_type=bool),
            dem_max_age_sec=ParameterValue(LaunchConfiguration('dem_max_age_sec'),value_type=float),
            history_interval_sec=ParameterValue(LaunchConfiguration('history_interval_sec'),value_type=float),
            history_tolerance_sec=ParameterValue(LaunchConfiguration('history_tolerance_sec'),value_type=float))])
    gate=Node(package='eat_nav_real_robot',executable='command_gate',name='eat_nav_command_gate',output='screen',
        parameters=[params,dict(enable_robot=enable,
            use_sim_time=ParameterValue(LaunchConfiguration('use_sim_time'),value_type=bool))])
    nodes=[planner,gate]
    # No follower exists in preview mode, and preview creates no physical publishers.
    if value('enable_robot').lower()=='true' and value('start_follower').lower()=='true':
        share=Path(get_package_share_directory('simple_trajectory_follower'))
        follower=Node(package='simple_trajectory_follower',executable='simple_trajectory_follower_node',
            name='eat_nav_follower',output='screen',parameters=[str(share/'config/onix.yaml'),dict(
                path_topic='/eat_nav/path',cmd_vel_topic='/eat_nav/follower_cmd',cmd_vel_use_stamped=True,
                cmd_vel_frame_id='chassis/body',robot_frame='chassis/body',
                recovery_path_topic='/eat_nav/unused_recovery_path',publish_subcrawler_commands=False)])
        nodes.append(follower)
    actions=list(nodes)
    for node in nodes:
        actions.append(RegisterEventHandler(OnProcessExit(target_action=node,
            on_exit=[EmitEvent(event=Shutdown(reason='EAT-Nav process exited; stop pipeline'))])))
    return actions


def generate_launch_description():
    share=Path(get_package_share_directory('eat_nav_real_robot'))
    defaults=dict(params_file=str(share/'config/real_robot.yaml'),enable_robot='false',start_follower='true',
        repository=str(Path.home()/'visualnav-transformer-ros2'),
        checkpoint='',navigation_mode='r5_14',replan_period_sec='2.0',stall_risk_only='false',
        checkpoint_sha256='',torch_site_packages='',odom_topic='/diff_drive_controller/odom',height_layer='elevation',
        dem_max_age_sec='1.0',history_interval_sec='0.3333333333333333',history_tolerance_sec='1.0',use_sim_time='false')
    return LaunchDescription([DeclareLaunchArgument(k,default_value=v) for k,v in defaults.items()]+[OpaqueFunction(function=make_nodes)])
