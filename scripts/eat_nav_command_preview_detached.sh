#!/usr/bin/env bash
set -eo pipefail
LIMIT=${1:-all}
source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ROOT=/tmp/eatnav-candidate-coverage/command-preview-long-horizon
mkdir -p "$ROOT/follower-logs"
/usr/bin/python3 /tmp/eatnav-vnt-candidate-coverage/scripts/eat_nav_phase_c_ros_bridge.py \
  --host 0.0.0.0 --port 47819 --branches 8 \
  --follower /home/kojima/onix/build/simple_trajectory_follower/simple_trajectory_follower_node \
  --params /home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml \
  --diagnostic-dir "$ROOT/follower-logs" >> "$ROOT/bridge.log" 2>&1 &
BRIDGE_PID=$!
trap 'kill -KILL "$BRIDGE_PID" 2>/dev/null || true; wait "$BRIDGE_PID" 2>/dev/null || true' EXIT
for n in $(seq 1 100); do
  if /usr/bin/python3 -c 'import socket; s=socket.create_connection(("127.0.0.1",47819),timeout=.1);s.close()' 2>/dev/null; then break; fi
  sleep .1
done
export PYTHONPATH=/tmp/eatnav-vnt-candidate-coverage/src:/tmp/eatnav-vnt-candidate-coverage
if [[ "$LIMIT" == all ]]; then
  /usr/bin/python3 /tmp/eatnav-vnt-candidate-coverage/scripts/eat_nav_command_preview_run.py
else
  /usr/bin/python3 /tmp/eatnav-vnt-candidate-coverage/scripts/eat_nav_command_preview_run.py --limit "$LIMIT"
fi
