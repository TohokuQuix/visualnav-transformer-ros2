# EAT-Nav one-robot ROS2 + Isaac GUI demo

This demonstration connects an RViz2 goal to three frozen Path candidates,
the pilot Evaluator, the C++ `PureFollower`, the trained RL Actor, and Isaac
physics. It is a pipeline demonstration, not a validated navigation policy.

## Setup

- Host: ROS2 Jazzy with `rmw_zenoh_cpp`, X11 display, NVIDIA driver, and
  the running `isaac-lab-onix` Docker container with the Onix Isaac Lab source.
- Start the external Zenoh router in a separate terminal before the demo:
  `ros2 run rmw_zenoh_cpp rmw_zenohd`. The launcher never manages it.
- Pilot Evaluator checkpoint:
  `~/eatnav-experiments/baseline-evaluator-pilot-20261013/train-common-goal-v1/checkpoint.pt`.
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
under `~/lib/onix_isaac_lab/.eatnav-ros2-gui-demo-terrain-v2-*`.
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
the RViz goal, as in the original demo. Candidate count remains three
and is independent of env count.

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
- The single-env demo disables the stock startup USD sprocket filtered-pairs
  edit. That edit is already skipped in multi-env mode and otherwise
  invalidates the single-env PhysX tensor view during reset. Reset events,
  policy, and robot dynamics are unchanged.
