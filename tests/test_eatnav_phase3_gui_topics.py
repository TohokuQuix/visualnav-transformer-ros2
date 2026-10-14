"""The GUI shares safe robot topic names without publishing actuator commands."""

import importlib.util
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _launcher():
    spec = importlib.util.spec_from_file_location(
        'eat_nav_phase3_gui_launch', ROOT/'scripts/eat_nav_phase3_gui_launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_uses_default_ros_domain():
    launcher = _launcher()
    class Result:
        stdout = b'ROS_DOMAIN_ID=227\0PATH=/usr/bin\0'
    with patch.object(launcher, 'ROS_OVERLAY', ROOT/'config/eat_nav.yaml'), \
         patch.object(launcher.subprocess, 'run', return_value=Result()):
        env = launcher._ros_environment()
    assert 'ROS_DOMAIN_ID' not in env
    assert env['RMW_IMPLEMENTATION'] == 'rmw_zenoh_cpp'


def test_gui_topic_contract_uses_robot_odom_but_never_robot_actuation():
    frontend = (ROOT/'scripts/eat_nav_phase3_gui_frontend.py').read_text()
    rviz = (ROOT/'config/eat_nav_phase3_gui.rviz').read_text()
    assert "create_publisher(Odometry, '/odom'" in frontend
    assert 'Value: /odom' in rviz
    assert "'/goal_pose'" in frontend
    assert "'/eat_nav/selected_path'" in frontend
    assert "'/eat_nav/mock/follower_cmd'" in frontend
    assert "'/eat_nav/mock/final_cmd'" in frontend
    assert "'/teaching_playback/path'" not in frontend
    assert "'/cmd_vel_auto'" not in frontend
    assert "'/diff_drive_controller/cmd_vel'" not in frontend
