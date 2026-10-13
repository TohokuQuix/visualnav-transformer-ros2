# EAT-Nav student environment and GUI startup

This is the verified **research demo**, not a real-robot controller or a
physical emergency stop. The implementation lives on branch
`exp/eatnav-ros2-gui-demo-terrain` of `visualnav-transformer-ros2`; its
terrain-v2 base is commit `b42c2ea4f91abd3d0df7796904095436dee51d2f`.
Use `git rev-parse HEAD` and the final handoff report to record the exact
subsequent diagnostic/documentation commit. The host Onix checkout used for
the tests is `/home/kojima/lib/onix_isaac_lab` at
`87fe97d266ae2f64d802a80471727d3429b05ff4`; it is not modified by this
handoff. Keep the worktree and experiment roots in persistent storage before
a reboot; `/tmp` worktrees may be removed by the host.

The current demo protocol is `eatnav_ros2_isaac_gui_demo_terrain_v3_fall_only`.
Only the existing 85-degree tilt termination resets the robot. The 40 s
episode length remains in the RL configuration, but its timeout termination
is disabled for the demo. An invalid numeric robot state stops the run with
an error instead of triggering an automatic reset. Each initial spawn and
fall reset sets the flipper fold mode to `0` (no directional fold constraint);
the flipper angle, speed, and front/rear interference limits still apply.
After a fall, the Goal, selected Path, and DEM history are cleared; a new
Goal must be sent. The ROS `/clock` continues, while the reset raycaster's
episode-local update time is mapped onto that continuous clock.

## Installed environment checked on 2026-10-13

| Component | Verified value / responsibility |
| --- | --- |
| Host ROS | Jazzy; `/usr/bin/python3` 3.12.3 imports `rclpy` |
| RMW | `rmw_zenoh_cpp`; external router must be running |
| Isaac container | `isaac-lab-onix`, image ID `sha256:e211f8c944c196fca39fb0b27f83f5715514b28fb119610e2dd13b956100d71b` |
| Simulator Python | `/isaac-sim/python.sh`, Python 3.11.13, PyTorch 2.7.0+cu128 |
| GUI | Host X11/`DISPLAY` and NVIDIA access passed through the existing container |
| Demo frontend | Host `/usr/bin/python3 scripts/eat_nav_demo_ros_frontend.py`, started by launcher |
| Offline test interpreter | `/home/kojima/anaconda3/envs/tinyvla/bin/python`; PyTorch 2.0.1 is **below** `pyproject.toml`'s `torch>=2.3`, so it is only the recorded test interpreter, not a recommended fresh installation |

The launcher expects the existing container, the Onix checkout, and the two
checkpoint paths below. It stages the demo source/config into a new container
`/tmp` directory for each run; raw output is written through the bind mount
to persistent `/home/kojima/lib/onix_isaac_lab/.eatnav-ros2-gui-demo-*-*`.
The launcher does not create the Docker container or manage the router.

```bash
docker ps --filter name=isaac-lab-onix
nvidia-smi
echo "$DISPLAY"
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 topic list
```

Start the external router in its own terminal if it is not already running:

```bash
source /opt/ros/jazzy/setup.bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

Keep that terminal alive. Do not run a second router on the same endpoint.
The observed external logs were under
`/home/kojima/eatnav-router-logs/eatnav-p1-paired-noise-20261010-v7/`.
The demo launcher never starts or stops it.

## Checkpoints and integrity

```bash
sha256sum \
  /home/kojima/eatnav-experiments/baseline-evaluator-pilot-20261013/train-common-goal-v1/checkpoint.pt \
  /home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt
```

Expected SHA-256 values, in the same order:

```text
329341fbf56ccb6bbc2f05b3db6b143cc1a1e9553ba2796e3dec3c785f30e470
b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d
```

The simulator checks these values before control. The Evaluator is a
train-only pilot from one StateGroup×seed, not a terrain-generalized model.

## One-robot GUI run

From the checked-out repository root, with the external router already up:

```bash
/usr/bin/python3 scripts/eat_nav_demo_launch.py \
  --num-envs 1 --terrain-type boxes --terrain-level 0 \
  --max-steps 3000 --replan-steps 10
```

In another terminal from the same root:

```bash
bash scripts/eat_nav_demo_rviz.sh
```

RViz uses simulation time and the `map` frame. Select **2D Goal Pose** to
publish a goal. The automatically verified equivalent transport path is:

```bash
RMW_IMPLEMENTATION=rmw_zenoh_cpp /usr/bin/python3 \
  scripts/eat_nav_demo_goal_once.py --distance 0.75
```

Cancel with:

```bash
RMW_IMPLEMENTATION=rmw_zenoh_cpp ros2 topic pub --once \
  /eat_nav/cancel std_msgs/msg/Empty '{}'
```

The 20 Hz loop publishes `/clock`, `/eat_nav/odom`, TF `map→base_link`,
`/eat_nav/dem/height`, `/eat_nav/dem/valid_mask`, `/eat_nav/candidates`,
`/eat_nav/selected_path`, `/eat_nav/scores`, and `/eat_nav/status`.
`Ctrl+C` ends the launcher; check its printed run-root path and then
`sim-output/complete.json`, `sim-output/sha-ledger.json`, and
`sim-output/host-seal.json`. A `failure.json` means the run did not pass.
The trace uses one row per control step with `stamp_ns=step×50 ms`.
New roots use `.eatnav-ros2-gui-demo-terrain-v3-fall-only-*`; earlier
`terrain-v2` roots remain separate evidence.

`--terrain-type` accepts `k_rail_step_field`, `pyramid_stairs`,
`pyramid_stairs_inv`, `random_rough`, `boxes`, `hf_pyramid_slope`, and
`k_rail_field`. `--terrain-level` is 0–4; the Isaac generator draws an
actual difficulty inside the selected row's 0.2-wide band. Inspect
`sim-output/terrain-provenance.json` for the actual draw, seed, tile SHA,
origin, and spawn pose. `--num-envs 4` is an optional compatibility check;
the student GUI demonstration defaults to one robot.

## Troubleshooting

| Symptom | First check |
| --- | --- |
| No ROS topics or Goal | External router process/log, `RMW_IMPLEMENTATION`, `ros-frontend.log`; never restart it through the demo launcher |
| Goal rejected | `/clock` advancing, RViz fixed frame `map`, `ros-frontend.log` stale/future stamp or TF message |
| GUI absent | `DISPLAY`, X11 permissions, container GPU/X11 mount, `simulator.log`; a headless pass does not prove GUI |
| Wrong terrain | CLI type/level, selected row and geometry SHA in `terrain-provenance.json` |
| No Path/score | `selection` rows in `trace.jsonl`, four fresh DEM frames, `failure.json`, checkpoint SHA |
| No movement | Compare command, Actor input/action, and `pose_post` separately; nonzero command alone is not movement |
| Cancel | Command should become zero, but physical movement may continue; **Cancel is not a safety stop** |

The [experiment pipeline](experiment_pipeline.md) and
[research roadmap](research_roadmap.md) describe what is implemented and
what still requires new data or validation.
