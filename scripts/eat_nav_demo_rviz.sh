#!/usr/bin/env bash
set -euo pipefail

demo_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
exec rviz2 -d "${demo_repo_root}/config/eat_nav_demo.rviz" --ros-args -p use_sim_time:=true
