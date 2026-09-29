#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH=/tmp/eatnav-vnt-candidate-coverage/src:/tmp/eatnav-vnt-candidate-coverage/scripts:${PYTHONPATH:-}
ROOT=/tmp/eatnav-candidate-coverage/clock-domain-diagnostic
mkdir -p "$ROOT/bridge-follower-logs"
setsid /usr/bin/python3 /tmp/eatnav-vnt-candidate-coverage/scripts/eat_nav_clock_bridge.py \
  --host 0.0.0.0 --port 47819 --branches 8 \
  --follower "$ROOT/diagnostic-follower-build/simple_trajectory_follower_node" \
  --params /home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml \
  --diagnostic-dir "$ROOT/bridge-follower-logs" >> "$ROOT/clock-bridge.log" 2>&1 &
BRIDGE_PID=$!
trap 'kill -KILL -- -"$BRIDGE_PID" 2>/dev/null || true; wait "$BRIDGE_PID" 2>/dev/null || true' EXIT
for n in $(seq 1 150); do
  if /usr/bin/python3 -c 'import socket; s=socket.create_connection(("127.0.0.1",47819),timeout=.1);s.close()' 2>/dev/null; then break; fi
  if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then echo 'bridge exited before readiness'; exit 1; fi
  sleep .1
done
/usr/bin/python3 /tmp/eatnav-vnt-candidate-coverage/scripts/eat_nav_clock_diagnostic_run.py
