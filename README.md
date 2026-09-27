# General Navigation Models: GNM, ViNT and NoMaD

This repository is a port of [visualnav-transformer](https://github.com/robodhruv/visualnav-transformer) to ROS2. Its purpose is to make running the models more straightforward by providing a Dockerfile with all dependencies set up. For more details on the models, please refer to the original repository.

### Running the code

1. Clone the repository:

```bash
git clone https://github.com/RobotecAI/visualnav-transformer-ros2
cd visualnav-transformer-ros2
```

2. Download the default NoMaD weights and build the Docker image:

From the repository root, run the setup script. It uses Docker to download the default weights to `model_weights/nomad.pth`, then builds the image. An existing nonempty weight file is left untouched.

```bash
./scripts/setup_docker.sh
```

The defaults are ROS 2 Jazzy, NVIDIA PyTorch, and the image tag `visualnav_transformer:latest`. Set `ROS_DISTRO`, `TORCH_VARIANT` (`nvidia`, `amd`, or `cpu`), or `IMAGE_TAG` to change them:

```bash
ROS_DISTRO=humble TORCH_VARIANT=amd ./scripts/setup_docker.sh
# For CPU only:
TORCH_VARIANT=cpu ./scripts/setup_docker.sh
```

To download the default weights again, remove `model_weights/nomad.pth` and rerun the script.

3. Run the Docker container:

The run script resolves the repository from its own location. It mounts that repository at `/visualnav-transformer`, regardless of your current directory. Run it from the repository root as shown below, or invoke it by its absolute path from elsewhere.

```bash
# NVIDIA (default; requires the NVIDIA Container Toolkit)
./scripts/run_docker.sh

# AMD (requires ROCm drivers and an image built with TORCH_VARIANT=amd)
TORCH_VARIANT=amd ./scripts/run_docker.sh

# CPU (requires an image built with TORCH_VARIANT=cpu)
TORCH_VARIANT=cpu ./scripts/run_docker.sh
```

Set `IMAGE_TAG` to the same value used during setup if you chose a custom image tag. `ROS_DOMAIN_ID` is passed through to the container. Additional arguments to the script are passed to `docker run`; for example, if the GPU is not detected, try `./scripts/run_docker.sh --group-add video --group-add render --security-opt seccomp=unconfined`.

Edits to local code and weight files are visible in the container immediately. Restart the model process to load changed weights. The Python environment stays in the image at `/opt/visualnav-venv`; rerun the setup script after changing Python dependencies.

4. Run the model:

Inside the container, run the following commands:
```bash
uv run python src/visualnav_transformer/deployment/src/explore.py
```

> [!TIP]
> Instead of prefixing every command with `uv run`, you can activate the environment once inside Docker with `source /opt/visualnav-venv/bin/activate` and then just use `python`.

This will run the model and publish the predicted waypoints to a ROS2 topic, but your robot will not move yet. Next to running the model you have to run a script that will publish the movement commands to the robot.

```bash
uv run python scripts/publish_cmd.py
```

Now the robot should start moving based on the model's predictions.

To visualize the waypoints that the model is outputting you can run the following command:
```bash
uv run python scripts/visualize.py
```
A window should appear with the camera image and the model outputs updated in real time.

### Optional local Python setup (uv)

The Docker workflow above installs Python dependencies inside the image and downloads the default model weights to this repository. You do not need a local Python environment to run the model in Docker. Set one up only if you want to run tools such as `uv run python scripts/visualize.py` on the host.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -p 3.12   # ROS2 Jazzy; use -p 3.10 for Humble
uv sync
```

The Python version must match your ROS distro: 3.12 for Jazzy, 3.10 for Humble.

Pick the torch build for your hardware:
```bash
# amd
uv pip install -U torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
export UV_NO_SYNC=1

# cuda
uv pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu126
export UV_NO_SYNC=1

# cpu
# automatically installed by uv sync
```

> [!NOTE]
> `UV_NO_SYNC=1` stops `uv run` from syncing the environment back to the lockfile, which would restore pytorch to CPU-only. Requires a ROS2 installation sourced in your shell.

### Creating a topomap of the environment

In order to navigate to a desired goal location, the robot needs to have a map of the environment. To create a topomap of the environment, you can run the following command:
```bash
uv run python src/visualnav_transformer/deployment/src/create_topomap.py
```
The script will save an image from the camera every second (this interval can be changed with the `-t` parameter). Now you can drive the robot around the environment manually (using your navigation stack or teleop) and the map will be saved automatically. After you have driven around the environment, you can stop the script and proceed to the next step.

### Navigation
Having created a topomap of the environment, you can now run the navigation script:
```bash
uv run python src/visualnav_transformer/deployment/src/navigate.py
```
By default the robot will try to follow the topomap to reach the last image captured. You can specify a different goal image by providing an index of an image in the topomap using the `--goal-node` parameter.

## Connecting NoMaD to the ONIX path follower

`navigate_robot.py` receives `/front_camera/image_raw/compressed` and publishes
`nav_msgs/Path` to `/teaching_playback/path`. It uses the existing NoMaD networks,
DDPM loop and goal-image selection. It does not add a classifier or publish velocities.

1. Source your ROS 2 environment (Jazzy / Python 3.12, or Humble / Python 3.10).
   ROS Python packages must be visible to the Python interpreter running inference.
2. Run `uv sync` with your GPU/CPU setup as described above.
3. Edit `config/robot_interface.yaml` and provide:
   - A model YAML and an **architecture-compatible** NoMaD checkpoint.
   - `topomap`: a directory of contiguous `0.png`, `1.png`, ... teaching images
     (JPEG is also accepted; duplicate image numbers are rejected).
   - `context_interval_sec`: the image interval used in training.
   - `action_scale_m`: the training normalization scale in meters. For an
     unnormalized model this must be `1.0`. Do not guess it from robot speed.
   Paths in this YAML are resolved relative to the YAML's location.
4. Start the interface:

   ```bash
   uv run python -m visualnav_transformer.deployment.src.navigate_robot \
     --interface-config config/robot_interface.yaml
   ```

   Bag playback:

   ```bash
   uv run python -m visualnav_transformer.deployment.src.navigate_robot \
     --interface-config config/robot_interface.yaml --ros-args -p use_sim_time:=true
   ```

5. Run `simple_trajectory_follower` with its `nomad_follower.launch.py` from the
   separate ONIX repository (`feat/nomad-path-following`). Use `use_sim_time:=true`
   there too when replaying a bag. Do not also run `scripts/publish_cmd.py`.

### Interface contract

- Images are decoded from compressed BGR to RGB. The model still receives its
  configured number of chronological frames and its existing image normalization.
- All model output points from the first sampled trajectory are retained. The
  observation-time robot position is prepended to the Path.
- The model's local XY must be based on `chassis/body` position and yaw. Metric
  scaling is applied once, followed by observation-time map translation and yaw.
  Roll/pitch are not applied to the model's planar training coordinates.
- TF `map <- chassis/body` must exist **at the image stamp**, not just at the latest
  time. The bag's camera frame name is not used as the trajectory reference frame.
- Path and Pose headers use `map` and the observation stamp. Pose orientation is
  the identity quaternion, not a predicted vehicle heading. The follower uses XY.
- Path QoS is Reliable / Transient Local / KeepLast(1), matching the follower.
- The network computation runs in one worker. New images replace pending input;
  old-generation predictions cannot be published after a reset or stop.
- A steady-clock watchdog clears the Path during image loss or stalled inference;
  it also checks the age of the published Path independently of incoming images.
  TF failure, invalid output and clock rewind also publish an empty Path.
- Final goal detection retains the legacy nearest-goal-image rule and enters a
  terminal stop state. Restart the node to begin a new navigation session.
- Ctrl-C sends an empty Path before shutdown. A nonreturning inference call can
  delay process exit; the stop message is sent first and the follower has its own
  timeout for crashes or lost communication.

The default observation/receipt deadlines are 0.8 seconds versus the follower's
initial 1.0-second timeout. Adjust them to measured inference latency and training
image interval. Missing history or a sampling gap above 1.5 times the interval
requires refilling the history. Duplicate frames do not refresh the receipt timer.

The prior `krail_directional_camera_switchback_conditioned_2026_09_06_17_30_01`
checkpoint contains extra conditioning layers. This interface deliberately rejects
it with the standard NoMaD architecture, instead of silently ignoring those weights.
The selected baseline checkpoint, training scale/coordinate frame/image interval,
and actual goal images must be confirmed before real-robot operation.

During bag playback, remap or exclude the recorded `/teaching_playback/path` to
avoid conflicting publishers, and enable `/clock`. Verify map alignment in RViz
before connecting the command consumer.

### Tests

Source ROS 2, install pytest in the Python environment, then run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  ROS_DOMAIN_ID=188 uv run python -m pytest -q tests
```

ROS tests use `/test_camera` and `/test_path` with synthetic TF and predictor
fixtures. The separate inference test runs real NoMaD modules on CPU with a
seeded synthetic checkpoint and compares against the original navigation code.
These tests do not establish navigation accuracy or replace bag/real-robot tests.
