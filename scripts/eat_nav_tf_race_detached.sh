#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH=/tmp/eatnav-vnt-candidate-coverage/src:/tmp/eatnav-vnt-candidate-coverage/scripts:${PYTHONPATH:-}
ROOT=/tmp/eatnav-candidate-coverage/tf-race-diagnostic
mkdir -p "$ROOT/follower-logs"
/usr/bin/python3 /tmp/eatnav-vnt-candidate-coverage/scripts/eat_nav_tf_race_bridge.py \
 --host 0.0.0.0 --port 47819 --branches 8 \
 --follower "$ROOT/diagnostic-follower-build-r1/simple_trajectory_follower_node" \
 --params /home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml \
 --diagnostic-dir "$ROOT/follower-logs" >> "$ROOT/bridge.log" 2>&1 &
BRIDGE_PID=$!
trap 'kill -TERM "$BRIDGE_PID" 2>/dev/null || true; wait "$BRIDGE_PID" 2>/dev/null || true' EXIT
for n in $(seq 1 100); do
 if /usr/bin/python3 -c 'import socket; s=socket.create_connection(("127.0.0.1",47819),timeout=.1);s.close()' 2>/dev/null; then break; fi
 sleep .1
done
/usr/bin/python3 /tmp/eatnav-vnt-candidate-coverage/scripts/eat_nav_tf_race_run.py >> "$ROOT/runner.log" 2>&1
