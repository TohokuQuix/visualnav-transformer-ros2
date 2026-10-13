# EAT-Nav ROS2 + Isaac Sim GUI demo

## What runs

`eat_nav_demo_launch.py` starts two processes. Isaac Sim owns four Onix environments,
the RL Actor, the unchanged C++ `PureFollower` control law, the DEM history, and the
Pilot Baseline Evaluator. A host ROS2 Jazzy node transports goals and publishes state.
The processes exchange newline-delimited JSON over localhost. The external Zenoh
router is used by ROS2, but this launcher never starts or stops it. The ROS production
follower and bridge are not part of this control path.

The control sequence every 50 ms is root pose → follower core → CommandManager →
Actor observation (noise and latency preserved) → Actor inference → physics. The
current sample has four 250 ms spaced, robot-relative DEM frames. Replanning runs
every 10 control steps by default. Isaac sim time is published on `/clock`; map to
`base_link` TF, odometry, and sensor messages carry that time. The sensor's actual
last-update time is also recorded separately.

The bank contains the eight-point P1 Direct, Left +20°, Right −20° Paths for a
0.75 m goal. The templates are rotated and scaled toward a requested local goal,
at most 0.75 m away. Such transformed Paths are a demo input; the Pilot checkpoint
was trained on only one StateGroup and one seed, so its scores are uncalibrated
predictions of 2 s common-goal progress, in metres. Score differences here do not
establish navigation quality or generalization.

## Start and operate

Prerequisites: running `isaac-lab-onix` Docker container with GUI/X11 access,
NVIDIA driver and Isaac Sim, ROS2 Jazzy with `rmw_zenoh_cpp`, and an externally
managed Zenoh router. The Pilot Evaluator checkpoint is at
`~/eatnav-experiments/baseline-evaluator-pilot-20261013/train-common-goal-v1/checkpoint.pt`.
The RL checkpoint is `~/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt`.
The launcher validates their SHA values before control starts.

Checkout `exp/eatnav-ros2-gui-demo` from `visualnav-transformer-ros2`, then in one
terminal run:

```bash
cd ~/lib/visualnav-transformer-ros2
/usr/bin/python3 scripts/eat_nav_demo_launch.py --max-steps 3000 --replan-steps 10
```

In another terminal run `bash scripts/eat_nav_demo_rviz.sh`. It sets
`use_sim_time:=true`, which is required for RViz goal stamps to match Isaac
`/clock`. Keep the fixed frame as `map`. Use **2D Goal
Pose**; the `/goal_pose` PoseStamped stamp must match current `/clock` within
0.5 s. Select a point around 0.75 m in front of the robot for a pilot-like
geometry. An optional transport smoke uses
`/usr/bin/python3 scripts/eat_nav_demo_goal_once.py --distance 0.75`.
Publish `std_msgs/msg/Empty` on `/eat_nav/cancel` to stop following. Ctrl+C
terminates the launcher-owned simulator and frontend; it leaves the external
router alone.

The launcher prints the persistent output root under `~/lib/onix_isaac_lab/`
with prefix `.eatnav-ros2-gui-demo-v1-`. `launch-manifest.json`, logs, the inner
`input-ledger.json`, `trace.jsonl`, completion or failure marker, SHA ledger, and
host seal are saved there. Each attempt uses a fresh root. Prior attempts and
research evidence are left untouched.

## ROS interface

| Topic | Type | Use |
|---|---|---|
| `/goal_pose` | `geometry_msgs/PoseStamped` | RViz goal input; map or TF-transformable frame |
| `/eat_nav/cancel` | `std_msgs/Empty` | cancel and send zero command |
| `/clock` | `rosgraph_msgs/Clock` | simulation clock at 20 Hz |
| `/tf` | `tf2_msgs/TFMessage` | map → base_link |
| `/eat_nav/odom` | `nav_msgs/Odometry` | root pose and body velocity |
| `/eat_nav/dem/height` | `sensor_msgs/Image`, 32FC1 | robot-relative 20×20 DEM |
| `/eat_nav/dem/valid_mask` | `sensor_msgs/Image`, mono8 | valid cells |
| `/eat_nav/actor_pre_noise` | `std_msgs/Float32MultiArray` | 22-element Actor state |
| `/eat_nav/candidates` | `visualization_msgs/MarkerArray` | candidate lines, goal, predicted score labels |
| `/eat_nav/selected_path` | `nav_msgs/Path` | selected Path in map |
| `/eat_nav/scores` | `std_msgs/String` JSON | uncalibrated predicted common-goal progress [m] |
| `/eat_nav/status` | `std_msgs/String` JSON | idle, active, arrived, command |

RViz2 displays odometry, TF, the local DEM Image, all candidates, the selected
Path and score labels. A missing or stale goal, DEM, Actor input, checkpoint, or IPC connection
causes rejection or a fresh attempt failure; the control command is not continued
from an old trajectory. A goal closer than 5 cm is rejected by the candidate
contract. Once the global goal is within 15 cm and the follower reports Goal,
the demo holds zero command until a new goal or cancel.

## Limits

The demo uses boxes terrain at fixed difficulty, four environments for the
checkpoint-compatible Actor shape, and only env 0 is driven by the RViz goal.
The other three receive zero command and are not navigation trials. Goal-relative
template scaling can leave the Pilot Evaluator training distribution. There is
no held-out validation for this checkpoint, no demonstrated DEM-aware selection
benefit, no Pain head, and no high-curvature candidate bank. Closed-loop physical
motion is a pipeline demonstration; it is not a performance claim.
