#!/usr/bin/env bash
set -e
mkdir -p /tmp/krail-autonomous-gpu
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=188 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOG_DIR=/tmp/krail-autonomous-gpu/roslogs MPLCONFIGDIR=/tmp/krail-autonomous-gpu/mpl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/ubuntu/visualnav-transformer-ros2/src:${PYTHONPATH:-}
cd /home/ubuntu/visualnav-transformer-ros2
/tmp/nomad-gpu-venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda, "GPU", torch.cuda.get_device_name(0))'
pids=()
cleanup() { trap - EXIT INT TERM; kill -TERM "${pids[@]}" 2>/dev/null || true; wait "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
/tmp/nomad-gpu-venv/bin/python -m visualnav_transformer.deployment.src.navigate_robot --interface-config config/krail_baseline_robot.yaml --ros-args -p use_sim_time:=true -p device:=cuda -p path_topic:=/nomad/baseline_predicted_path -p goal_image:=/home/ubuntu/visualnav-transformer-ros2/docs/validation/2026-09-16-krail-autonomous/goal.jpg > /tmp/krail-autonomous-gpu/node.log 2>&1 &
pids+=("$!")
rviz2 -d /home/ubuntu/visualnav-transformer-ros2/config/rviz/krail_autonomous_baseline.rviz --ros-args -p use_sim_time:=true > /tmp/krail-autonomous-gpu/rviz.log 2>&1 &
pids+=("$!")
ros2 bag play '/media/ubuntu/Extreme SSD/20260803_experiment/2026_08_03-10_31_05' --start-offset 25 --clock 50 --rate 1.0 --delay 8 --loop --topics /front_camera/image_raw/compressed /tf /tf_static > /tmp/krail-autonomous-gpu/bag.log 2>&1 &
pids+=("$!")
printf 'GPU baseline node, RViz and looping bag started (ROS_DOMAIN_ID=188). Ctrl-C stops all.\n'
wait -n "${pids[@]}"
