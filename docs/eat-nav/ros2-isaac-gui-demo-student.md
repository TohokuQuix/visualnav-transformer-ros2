# EAT-Nav one-robot ROS2 + Isaac GUI demo

## Current handoff: ROS-separated 14-candidate demo

Use branch `exp/eatnav-ros2-gui-demo-terrain` after the ROS-separated integration. Install Git LFS, run `git lfs pull`, then verify the packaged checkpoints with `/usr/bin/python3 scripts/eat_nav_verify_checkpoints.py`. Start the external Zenoh router separately, then run from this repository:

```bash
unset ROS_DOMAIN_ID
RMW_IMPLEMENTATION=rmw_zenoh_cpp /usr/bin/python3 \
  scripts/eat_nav_phase3_gui_launch.py \
  --terrain-type boxes --terrain-level 0 --max-steps 300000
```

The default maximum is 300000 control steps; use a smaller `--max-steps` for a bounded check. For an RViz goal, add `--manual-goal` and open `config/eat_nav_phase3_gui.rviz` with the same ROS middleware and no domain override. The launcher reads the external router but never starts or stops it. It writes a new persistent run under `~/lib/onix_isaac_lab/.eatnav-ros2-phase3-evidence/`; inspect `result.json`, `sim-output/trace.jsonl`, `simulator.log`, and `host-seal.json` there.

This version uses the Stage C adopted 14-candidate Motion Predictor (`checkpoints/eat_nav/motion14/training/full-seed123/best.pt`), not the older MI-B Pilot six-slot checkpoint. It ranks predicted 2 s body-frame XY displacement by **global goal** distance reduction, then executes the selected Path with the ROS-free C++ `PureFollower` in Isaac Lab. The 14 candidates comprise forward and reverse Direct and ±30°/±60°/±90° arcs, each with eight points. This is a software integration demonstration; goal-directed quality and real-robot behavior remain unverified. See the [sealed GUI run and limits](motion-predictor-vnext/ros2-phase3-gui-adopted14-runtime.md).

The adopted Motion Predictor now receives a copy of the 22D Actor observation with velocity-command indices 6–8 fixed to zero **before Stage C normalization**. This is ON in the standard command above. The original Actor observation, RL Actor command, and follower command are unchanged. Stage C trained on zero velocity commands; feeding nonzero commands to the old demo caused large normalized input shifts and implausible predicted displacement. To reproduce the old predictor input explicitly, add `--no-predictor-zero-velocity-command` to the GUI launch command. `--predictor-zero-velocity-command` explicitly selects the new default. Startup logs and the run manifest record the setting.

The H4 check on boxes level 0, seed 2026101101 reduced predicted displacement scale and improved closed-loop goal approach, but did not establish goal arrival within 600 steps or generalization across terrains and seeds. Collection for repeated replanning, predictor retraining, removing velocity-command inputs from the network, broader goal-arrival evaluation, and real-robot verification remain unperformed. The handoff demo uses the XY predictor; a yaw predictor is not part of this branch's standard launch.

| Signal | Current demo topic | Real robot relationship |
| --- | --- | --- |
| Goal `PoseStamped` | `/goal_pose` | Same name and message type as the real planner input. |
| Odometry `Odometry` | `/odom` | Same name and type; the demo publishes in `map`, so verify frame semantics. |
| Selected Path | `/eat_nav/selected_path` | Visualization only. The real follower's `/teaching_playback/path` is deliberately not published on the shared domain. |
| Model DEM / Actor22 | `/eat_nav/dem`, `/eat_nav/actor22` | Demo-specific typed inputs; real sensor parity is not yet established. |
| DEM image | `/eat_nav/dem/height` | Visualization only; it is not the metric model input. |
| Follower and gated commands | `/eat_nav/mock/*` | Isaac-only transport; no `/cmd_vel_auto` or controller command is published. |

When using the default ROS domain, run only one EAT-Nav GUI instance and ensure that unrelated goal publishers are not sending `/goal_pose`. The selected Path display does not command a physical follower. A matching topic name alone does not establish real-robot input parity or an emergency stop guarantee.

## Legacy seven-candidate GUI demo

This demonstration connects an RViz2 goal to seven robot-forward Path candidates,
the MI-B Pilot Motion Predictor, the C++ `PureFollower`, the trained RL Actor, and Isaac
physics. It is a pipeline demonstration, not a validated navigation policy.
The predictor estimates 2 s body-frame XY displacement for each candidate.
The demo selects the candidate with the largest predicted distance reduction
to the full global goal. Direct points along the robot's current heading;
the other Paths turn left or right by 30°, 60°, or 90° over 0.75 m of arc
length. This selection rule has not been validated by
MI-B's held-out motion-prediction metrics.

## Setup

- Host: ROS2 Jazzy with `rmw_zenoh_cpp`, X11 display, NVIDIA driver, and
  the running `isaac-lab-onix` Docker container with the Onix Isaac Lab source.
- Start the external Zenoh router in a separate terminal before the demo:
  `ros2 run rmw_zenoh_cpp rmw_zenohd`. The launcher never manages it.
- MI-B Pilot Full seed-456 checkpoint (selected by aggregate validation loss):
  `~/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-b-pilot-20261014-01/training/full-seed456/best.pt`.
- RL Actor checkpoint:
  `~/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt`.
  The launcher/simulator verify their frozen SHA-256 values.

From this repository:

```bash
/usr/bin/python3 scripts/eat_nav_demo_launch.py \
  --num-envs 1 --terrain-type boxes --terrain-level 0 \
  --max-steps 3000 --replan-steps 10
```

In another terminal, run `bash scripts/eat_nav_demo_rviz.sh` and select the
`2D Goal Pose` tool in RViz. The goal is transformed to `map` and accepted
only with a recent simulation stamp. For a repeatable 0.75 m forward goal:

```bash
RMW_IMPLEMENTATION=rmw_zenoh_cpp /usr/bin/python3 \
  scripts/eat_nav_demo_goal_once.py --distance 0.75
```

To cancel the active goal:

```bash
RMW_IMPLEMENTATION=rmw_zenoh_cpp ros2 topic pub --once \
  /eat_nav/cancel std_msgs/msg/Empty '{}'
```

The selected Path and scores are visible on `/eat_nav/selected_path`,
`/eat_nav/candidates`, and `/eat_nav/scores`; robot pose, DEM, and status
are published on `/eat_nav/odom`, `/eat_nav/dem/height`, and
`/eat_nav/status`. Ctrl+C stops the launcher. All run roots are created
under `~/lib/onix_isaac_lab/.eatnav-ros2-gui-demo-terrain-v9-body-arcs-*`.
The earlier `terrain-v2` roots use the old 40 s timeout behavior and are
preserved as separate evidence.
`sim-output/trace.jsonl`, terrain provenance, input/SHA ledgers, and
host seal allow a run to be audited. The old v1 roots are independent.

## Terrain selection

`--terrain-type` accepts `k_rail_step_field`, `pyramid_stairs`,
`pyramid_stairs_inv`, `random_rough`, `boxes`, `hf_pyramid_slope`, and
`k_rail_field`. `--terrain-level` accepts 0–4. The original Isaac Lab
terrain generator creates five curriculum rows for the selected type;
the robot is placed on the chosen row. With the standard Onix rough
`difficulty_range=(0,1)`, level 0 covers `[0,0.2)`, level 1 `[0.2,0.4)`,
level 2 `[0.4,0.6)`, level 3 `[0.6,0.8)`, and level 4 `[0.8,1)`.
Isaac Lab draws one actual difficulty within the row band from the
recorded seed. The run records that draw, tile geometry SHA, tile origin,
and robot spawn pose. The generator's terrain parameters are unchanged.

`--num-envs 4` is retained as an optional compatibility mode. Each robot
uses a different column of the selected terrain row; only env 0 follows
the RViz goal, as in the original demo. Candidate count remains seven
and is independent of env count.
The MI-B model was trained with six candidates and ±30° Arcs. This demo
uses two six-slot inference batches to score seven forward Paths and masks
unused slots. The ±60° and ±90° inputs extend beyond the measured training
geometry. The resulting selector is an
exploratory demo with a different input distribution, not an MI-B Gate result.

## Checks and limitations

- If ROS topics are absent, verify the external router, `RMW_IMPLEMENTATION`,
  the frontend log, and the simulator's `startup-ready.json`.
- If a goal is rejected, inspect the frontend log for stale/future stamp
  or TF failure. RViz should use the simulator `/clock` and `map` frame.
- If DEM is stale or nonfinite, the simulator fails closed and writes
  `failure.json`; review its sensor update age.
- If the robot does not move, check `trace.jsonl` for a `selection` event,
  nonzero `command`, Actor action, and changing `pose_post` separately.
- Cancel and local Goal set the command to zero. The trained RL policy and
  physics can still produce residual motion after zero command; zero
  command alone is not proof of a physical emergency stop.
- The current demo resets only when the existing tilt threshold is reached.
  Timeout is disabled; an invalid numeric state stops the run rather than
  resetting. On a fall reset, the Goal and Path are cleared, and the operator
  must send a new Goal. Flipper fold mode is fixed to `0` at every reset.
- The demo applies a ROS-free turn-stall command preprocessor after the
  follower and before the RL Actor. It uses the production bridge launch
  thresholds (2 s at at most 50% yaw-rate response, then up to 0.2 m/s
  alternating forward/backward offset every 1 s). The measured input is
  Isaac body yaw rate, a proxy for the production bridge's IMU yaw rate.
  Trace rows distinguish `follower_command`, processed `command`, and the
  `turn_escape` diagnostic.
- The single-env demo disables the stock startup USD sprocket filtered-pairs
  edit. That edit is already skipped in multi-env mode and otherwise
  invalidates the single-env PhysX tensor view during reset. Reset events,
  policy, and robot dynamics are unchanged.
