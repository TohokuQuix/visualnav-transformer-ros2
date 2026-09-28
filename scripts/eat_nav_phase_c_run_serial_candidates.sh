#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID=77 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
root=/tmp/eatnav-quality-identifiability-gate/phase-c-remediation
for candidate_index in 1 2 3 4 5 6 7; do
  if docker exec isaac-lab-onix test -f "$root/sim-candidates/serial_candidate-${candidate_index}.marker.json"; then
    continue
  fi
  for bridge_pid in $(pgrep -f '^/usr/bin/python3 /tmp/eatnav-vnt/scripts/eat_nav_phase_c_ros_bridge.py'); do kill -KILL "$bridge_pid"; done
  for follower_pid in $(pgrep -f '^/home/kojima/onix/build/simple_trajectory_follower/simple_trajectory_follower_node.*eat_nav_follower_'); do kill -KILL "$follower_pid"; done
  sleep 2
  /usr/bin/python3 /tmp/eatnav-vnt/scripts/eat_nav_phase_c_ros_bridge.py \
    --follower /home/kojima/onix/build/simple_trajectory_follower/simple_trajectory_follower_node \
    --params /home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml \
    > "$root/serial-candidate-bridge-${candidate_index}.log" 2>&1 &
  sleep 2
  docker exec isaac-lab-onix bash -lc "export PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source:\$PYTHONPATH; /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_phase_c_remediation_sim.py --case serial_candidate --repeat-id ${candidate_index} --candidate-index ${candidate_index} --fixed-candidate-json $root/sim-candidates/serial_candidate-0.json --recorded $root/sim/duplicate.npz --headless --reference $root/reference/best.pt --reference-config /tmp/eatnav-phase-c/reference-eat.yaml --output $root/sim-candidates > $root/serial-candidate-${candidate_index}.log 2>&1"
  docker exec isaac-lab-onix test -f "$root/sim-candidates/serial_candidate-${candidate_index}.marker.json"
done
