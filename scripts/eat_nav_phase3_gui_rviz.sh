#!/usr/bin/env bash
set -eo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
set -u
unset ROS_DOMAIN_ID
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
exec rviz2 -d "${repo_root}/config/eat_nav_phase3_gui.rviz" \
  --ros-args -p use_sim_time:=true
