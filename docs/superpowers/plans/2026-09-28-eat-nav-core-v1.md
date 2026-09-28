# EAT-Nav Core v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Isaac collection → quality dataset → scratch NoMaD-style DEM policy → Onix ROS 2 trajectory pipeline.

**Architecture:** Keep EAT configuration, DEM conversion, dataset, model, and training in `visualnav-transformer-ros2`. Add a collection-only Isaac environment and collector in `onix_isaac_lab`; add EAT node, launch, and command guard in the Onix ROS workspace. One versioned config and checkpoint contract carries `context_size`, `len_traj_pred`, trajectory interval, and derived horizon across every stage.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, existing NoMaD Transformer/`ConditionalUnet1D`/DDPM code, IsaacLab and RSL-RL, ROS 2 Jazzy, GridMap, TF2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-eat-nav-core-v1-design.md`

## Repository map

- `VNT` = `/home/kojima/lib/visualnav-transformer-ros2` (this repository): shared EAT package, dataset, model, training, pure tests.
- `ISAAC` = `/home/kojima/lib/onix_isaac_lab`: collection-specific environment, recorder and Isaac integration tests.
- `ONIX` = `/home/kojima/onix/dep/onix_automation`: ROS node packaging, config, launch and transport tests.
- The shared `visualnav_transformer.eat_nav` package must be importable in both Isaac and ROS Python environments. Install this repository in each environment without pulling training-only dependencies into ROS; declare or isolate runtime dependencies accordingly.
- `ISAAC` and `ONIX` are outside the current writable roots. At execution, arrange approved writable workspaces for those repositories before editing. This plan itself changes only VNT docs.

## Global Constraints

- The collection checkpoint defaults to `/home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt`; CLI/config may override it. Load with `weights_only=True` when using PyTorch.
- RL control and raw Critic logging are 20 Hz. DEM collection defaults to 4 Hz. Record timestamps and all Critic values; compute P10 only in the dataset.
- Each of 100 environments receives an independent terrain seed, command seed and initial pose/yaw. Commands remain piecewise constant for initially 0.5–3.0 s and inside the trained RL ranges.
- Simulated source DEM matches the real 20×20, 0.1 m GridMap, map-fixed axes and unknown-cell behavior. The shared adapter outputs two robot-centric channels: scalar-offset height and validity mask. Do not fit a local plane or level roll/pitch.
- Default `context_size=3` means four DEM frames; `context_size=5` means six. Default `len_traj_pred=8`, `trajectory_interval_sec=0.25`, and derived `trajectory_horizon_sec=2.0`. Point count and interval are independent config inputs; horizon is derived and any explicit horizon must match.
- NoMaD's 4 Hz bag preprocessor and stride 1 imply a nominal 0.25 s point interval, but its mixed preprocessed datasets lack timestamps; inheriting eight points does not establish an exact NoMaD physical horizon.
- NoMaD pretrained weights are never loaded. Reuse its Transformer, diffusion U-Net, scheduler and XY-delta trajectory representation; replace RGB encoding with a shared CNN DEM encoder. Goal and Quality use separate MLP tokens.
- Train-only action statistics and train-only Quality ECDF apply to all splits. Keep failed episodes and their termination reasons. Quality disabled is a config-selected baseline. Run Quality overlap and fixed-input q-sensitivity pilot gates before full collection or full training; do not add a quality loss or change model structure for these diagnostics.
- Network trajectories are in the base frame. ROS converts goals to base and generated paths to the configurable reference frame (default `map`).
- EAT and planner adapter are launched exclusively. Do not modify follower source or its config file. The EAT launch may override the follower command topic and interpose a command guard.

## Review Focus

1. GridMap circular-buffer index, axis inversion and unknown cells: Task 2 tests against an actual message or bag and asymmetric fixtures.
2. Episode reset at the end of an Isaac step: Task 4 proves the terminal state/reason is logged before reset and never joined to the next episode.
3. Irregular DEM timestamps: Task 5 rejects windows outside tolerance and never silently substitutes an adjacent frame.
4. Different `len_traj_pred` checkpoints: Tasks 5–7 test that load fails explicitly; Task 8 tests that Path/model/evaluation point counts follow config.
5. Quality tied to terrain/goal or ignored by the model: Tasks 5 and 7 test overlap and fixed-input q sensitivity before full collection/training.
6. Stale follower output after EAT input loss: Task 9 transport tests prove guarded velocity becomes zero within TTL and stays zero until a new valid path.

---

### Task 1: Shared EAT configuration contract

**Files:** Create `VNT/config/eat_nav.yaml`, `VNT/src/visualnav_transformer/eat_nav/config.py`, `VNT/tests/eat_nav/test_config.py`.

**Interfaces:** `load_eat_config(path: Path) -> EATConfig`; immutable `EATConfig` with `context_size`, `dem_history_interval_sec`, `len_traj_pred`, `trajectory_interval_sec`, derived `trajectory_horizon_sec`, DEM geometry, quality and training settings. `assert_checkpoint_compatible(config: EATConfig, metadata: Mapping) -> None`.

- [ ] Write tests for defaults (3 history + current, 8 future points, 0.25 s, 2.0 s), 12 points at 0.2 s → 2.4 s, invalid count/interval/explicit horizon, and metadata mismatch in count, interval, history or DEM geometry.
- [ ] Run `python -m pytest tests/eat_nav/test_config.py -q`; expect failure because the contract is absent.
- [ ] Implement loader, validation, derived horizon and versioned metadata serializer; make unsupported config and checkpoint differences explicit errors.
- [ ] Rerun the focused test; expect pass. Commit the config and test.

### Task 2: Shared DEM adapter and real GridMap contract

**Files:** Create `VNT/src/visualnav_transformer/eat_nav/dem.py`, `VNT/tests/eat_nav/test_dem.py`; add an adapter for `grid_map_msgs/msg/GridMap` in `ONIX/onix_automation/scripts/eat_nav_grid_map.py` only after checking a real message/bag.

**Interfaces:** `SourceGrid(height: ndarray, valid: ndarray, origin_xy: tuple, resolution_m: float, start_index_xy: tuple, frame_id: str, stamp_ns: int)`; `robot_dem(grid: SourceGrid, pose_xy_yaw: tuple, config: EATConfig) -> ndarray[2,H,W]`. ROS adapter yields `SourceGrid`; Isaac adapter yields the same type.

- [ ] Test 20×20 geometry, world-fixed asymmetric axes at yaw 0/45/90°, circular-buffer rollover, translation, NaN/out-of-bounds masks, scalar-height shift invariance and preserved terrain slope. Include a test that fails if a plane is subtracted.
- [ ] Run `python -m pytest tests/eat_nav/test_dem.py -q`; expect failure before implementation.
- [ ] Implement source-grid conversion and one shared bilinear/mask rule. Check actual GridMap layer layout, start index, frame and measured update rate before wiring the ROS adapter; do not infer these from YAML alone.
- [ ] Rerun tests; expect pass. Save measured GridMap findings in `VNT/docs/eat_nav-validation.md`. Commit in each touched repository.

### Task 3: Isaac collection environment and checkpoint probe

**Files:** Create `ISAAC/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/config/onix/eat_nav_env_cfg.py`, `ISAAC/scripts/rsl_rl/eat_nav_probe.py`, `ISAAC/tests/test_eat_nav_env.py`. Keep the existing RL scanner and policy observation group unchanged.

**Interfaces:** `make_collection_cfg(base_cfg, config: EATConfig, seed: int)`; `probe_checkpoint(path: Path, env_cfg) -> CheckpointInfo`; `sample_command(env_id: int, time_s: float) -> (vx, 0, omega)`. Add a separate `ray_alignment="world"` collection RayCaster with 1.9 m span and 0.1 m spacing for 20×20 ray centers.

- [ ] Test world-fixed ray centers under robot yaw, independent terrain/command/pose seeds for 100 envs, command hold periods and learned bounds, and unchanged RL actor/critic observations.
- [ ] Run `python -m pytest tests/test_eat_nav_env.py -q` from ISAAC; expect failure.
- [ ] Implement collection subclass and a short Isaac-runtime probe. Restore observation toggles from the run's `params/env.yaml`; safely load the specified checkpoint, compare actual Actor/Critic tensor shapes, verify velocity command enters the Critic, and record value range. If any mismatch appears, stop collection and document the four-part discrepancy analysis required by the source request.
- [ ] Rerun unit and Isaac probe tests; expect pass with the selected checkpoint. Commit ISAAC changes.

### Task 4: Raw collector, terminal records and resumable chunks

**Files:** Create `ISAAC/scripts/rsl_rl/collect_eat_nav.py`, `ISAAC/source/onix_isaac_lab/onix_isaac_lab/eat_nav/recorder.py`, `ISAAC/tests/test_eat_nav_recorder.py`.

**Interfaces:** CLI accepts checkpoint/config override, env count, duration, collection rate, seed and output path. Chunk schema records per-tick `timestamp, env_id, episode_id, terrain_id/seed, pose, twist, command, action, critic_value, reward, termination flags/reason`; 4 Hz and terminal DEM records hold source and converted grids. Manifest records schema version, hashes, configs, seeds and checksums.

- [ ] Test raw 20 Hz Critic preservation, pre-reset terminal record with `termination_reason`, failed episode retention, per-env reset, chunk atomicity and restart without duplicate or overwritten chunks.
- [ ] Run `python -m pytest tests/test_eat_nav_recorder.py -q`; expect failure.
- [ ] Implement pre-reset hook and chunk writer; never compute P10 here. Run 100-env short collection, inspect command/Critic/reward/termination variation and terrain diversity, and measure throughput plus GB per 100 robot-hours before the 3600 s/env run.
- [ ] Rerun tests and short probe; expect pass and a manifest with unique seeds. Commit ISAAC changes.

### Task 5: Episode dataset, split, quality and variable-length targets

**Files:** Create `VNT/src/visualnav_transformer/eat_nav/dataset.py`, `quality.py`, `quality_diagnostics.py`, `VNT/tests/eat_nav/test_dataset.py`, `test_quality.py`, `test_quality_diagnostics.py`.

**Interfaces:** `EATDataset(manifest: Path, split: str, config: EATConfig)` returns DEM `[context_size+1,2,20,20]`, goal `[2]`, action `[len_traj_pred,2]`, valid action mask `[len_traj_pred]`, raw Critic slice and normalized quality. `fit_train_statistics(train_windows)` produces action stats and ECDF; validation/test only load them. `diagnose_quality_overlap(windows, config) -> OverlapReport` uses only DEM-history/mask/goal similarity to form independent-episode neighborhoods, then counts train-ECDF q bins.

- [ ] Test fixed terrain/episode split, 3→4 and 5→6 history frames, 8 and 12 point extraction from timestamps, irregular-frame rejection, base-frame XY transform, bounded goal, terminal padding/mask, no cross-episode frames, and corresponding raw Critic samples.
- [ ] Test P10 from raw sequence, train-only ECDF, unchanged validation/test mapping, q in [0,1], low-quality retention, and action statistics recomputed on valid train deltas when `len_traj_pred` changes.
- [ ] Test overlap diagnostics on matched DEM/goal with multiple q bins, terrain/goal-to-q one-to-one fixtures, duplicate adjacent windows, and insufficient pilot samples. Assert the report includes neighborhood/bin counts, multi-bin anchor fraction and per-terrain/goal-stratum values; the one-to-one fixture must fail the gate.
- [ ] Run `python -m pytest tests/eat_nav/test_dataset.py tests/eat_nav/test_quality.py tests/eat_nav/test_quality_diagnostics.py -q`; expect failure.
- [ ] Implement indexed windows, frozen split/statistics artifacts and predeclared mask IoU/height RMSE/goal-distance matching (defaults 0.8/0.05 m/0.1 m). Use four q bins, independent episodes, and report the initial gate of at least 50 distinct multi-bin anchors across independent episodes and 5% of all valid pilot windows. Rerun tests, expect pass, then commit.

### Task 6: DEM/Goal/Quality tokens and NoMaD diffusion model

**Files:** Create `VNT/src/visualnav_transformer/eat_nav/model.py`, `VNT/tests/eat_nav/test_model.py`; reuse `vint_train/models/nomad/nomad_vint.py`, `nomad.py` and existing diffusion utilities without copying the decoder.

**Interfaces:** `EATNav(config: EATConfig)`; `encode(dem, goal, quality) -> context[B,256]`; `noise_loss(action_delta, valid_mask, context) -> scalar`; `sample(dem, goal, quality) -> XY[B,len_traj_pred,2]`.

- [ ] Test shared CNN yields one 256-D token per DEM frame; goal and quality MLPs yield 256-D tokens; token count is 6/8 with quality at context 3/5; baseline removes quality token. Test model outputs and masked loss for lengths 8 and 12, with zero contribution from padded steps.
- [ ] Run `python -m pytest tests/eat_nav/test_model.py -q`; expect failure.
- [ ] Implement scratch initialization, existing Transformer and `ConditionalUnet1D`/DDPM path with `len_traj_pred` tensor dimensions; preserve normalized delta and cumulative XY semantics. Do not load pretrained weights.
- [ ] Rerun focused tests; expect pass. Commit.

### Task 7: Training, checkpoint compatibility and evaluation

**Files:** Create `VNT/src/visualnav_transformer/train/train_eat_nav.py`, `VNT/src/visualnav_transformer/eat_nav/checkpoint.py`, `evaluation.py`, `visualize.py`, `VNT/tests/eat_nav/test_training.py`, `test_conditioning_sensitivity.py`.

**Interfaces:** `train(config_path, manifest, output_dir)`; `save_checkpoint(model, optimizer, stats, ecdf, config, split_hash, path)`; `load_checkpoint(path, config) -> (model, stats, ecdf)`; `evaluate(model, loader, config)` reports loss and pointwise trajectory errors for exactly `len_traj_pred` points. `diagnose_conditioning_sensitivity(model, heldout_windows, q_values, seeds) -> SensitivityReport` keeps DEM/goal and diffusion noise paired across q.

- [ ] Test one-batch train/validation for lengths 8 and 12, masked loss, train-only statistics, quality-enabled and disabled runs, best EMA checkpoint on improving validation metric, patience stop before max_epochs, resume, and rejection of count/interval/history/DEM mismatch. Test evaluation and visualization dimensions for both lengths.
- [ ] Test sensitivity diagnostics with a model that ignores q and one that changes trajectories with q. Assert q=[0.1,0.5,0.9,0.95], at least 50 held-out fixed inputs (otherwise inconclusive), identical noise seeds across q, separate same-q stochastic baseline, and report of per-point XY, endpoint, path length and curvature changes.
- [ ] Run `python -m pytest tests/eat_nav/test_training.py tests/eat_nav/test_conditioning_sensitivity.py -q`; expect failure.
- [ ] Implement max_epochs=100 upper bound, configurable validation metric, patience, minimum improvement and best selection. Store schema/config/hash/stats/ECDF/model/EMA/optimizer; verify strict compatibility before constructing an inference model.
- [ ] Rerun focused tests, then perform a small Quality-enabled pilot training run on short-collection data; run paired-q sensitivity against its checkpoint and Quality-disabled baseline. Record curves, quality distribution and the sensitivity-to-sampling comparison. If q effects are at numerical noise level or below same-q sampling variation, report that conditioning use is unconfirmed. Commit.

### Task 8: ROS inference and dynamic Path size

**Files:** Create `ONIX/onix_automation/scripts/eat_nav_node.py`, `ONIX/onix_automation/config/eat_nav.onix2.yaml`, `ONIX/onix_automation/launch/eat_nav_onix2.launch.py`, `ONIX/onix_automation/test/test_eat_nav_ros.py`; modify `ONIX/onix_automation/CMakeLists.txt`, `package.xml` to install/runtime-declare the scripts. Reuse VNT shared package; leave follower files unchanged.

**Interfaces:** ROS node subscribes `/elevation_map` GridMap and `/goal_pose` PoseStamped, publishes `/teaching_playback/path` Path and ready heartbeat. Reference frame defaults to `map`; quality target defaults to 0.95. Each pose stamp uses observation time plus `k * trajectory_interval_sec` for `k=1..len_traj_pred`.

- [ ] Test actual GridMap and RViz goal transport, historical DEM timing, TF goal→base and predicted base→configured reference, dry-run debug output, exact `len_traj_pred` Path poses and pose stamps for 8 and 12, checkpoint mismatch rejection, and invalidation on DEM/TF/goal/timestamp failure.
- [ ] Run `colcon test --packages-select onix_automation --event-handlers console_direct+`; expect new tests to fail.
- [ ] Implement node and dry-run launch. Keep EAT and planner adapter exclusive. Rerun tests; expect pass. Commit ONIX changes.

### Task 9: Independent follower command guard

**Files:** Create `ONIX/onix_automation/scripts/eat_nav_command_guard.py`, `ONIX/onix_automation/test/test_eat_nav_guard.py`; modify `ONIX/onix_automation/launch/eat_nav_onix2.launch.py` and `CMakeLists.txt`. Do not modify follower source or config files.

**Interfaces:** Guard consumes follower `/eat_nav/follower_cmd_vel`, EAT ready heartbeat and Path freshness; guard alone publishes `/cmd_vel_auto`. Live EAT launch overrides follower `cmd_vel_topic` and includes the guard.

- [ ] Test heartbeat/command TTL 0.3 s, Path TTL 0.8 s, zero within TTL after each input loss, rearm only after a newly generated Path, and guard-only publisher of `/cmd_vel_auto`.
- [ ] Run `colcon test --packages-select onix_automation --event-handlers console_direct+`; expect new guard test to fail.
- [ ] Implement guard and launch override, then rerun tests; expect pass. Check the downstream controller's current 3.0 s timeout behavior before enabling live launch. Commit ONIX changes.

### Task 10: End-to-end gates and delivery

**Files:** Update `VNT/docs/eat_nav-validation.md`, `VNT/README.md`, and user-facing run configs only as needed.

**Interfaces:** Deliver commands for collection probe/full run, dataset build, training/baseline, checkpoint evaluation, ROS dry-run and live launch.

- [ ] Run VNT unit tests and 8/12-point model/training smoke tests; expect all pass.
- [ ] Run 100-env short Isaac integration and report seed uniqueness, Critic/command/termination distributions, throughput and storage estimate. Build its pilot dataset, run Task 5 Quality overlap diagnosis and Task 7 small training plus paired-q sensitivity diagnosis. Record the two reports and gate status before starting the 3600 s/env collection or full training; if either gate fails or is inconclusive, stop those full runs and report the evidence without changing loss or architecture. Record actual completeness if full collection proceeds.
- [ ] Build a dataset from collected chunks, train a scratch checkpoint, reload it, and produce a Path from recorded or real GridMap plus goal in ROS dry-run. Confirm frame and point count from checkpoint config.
- [ ] Test guard loss/recovery with follower in dry-run; verify downstream timeout and publisher exclusivity before any live motion.
- [ ] Record executed tests, parameter count, training/validation curves, ROS topics/TF/launch commands, measured limits and any skipped GPU/real-message tests. Commit docs, then request whole-branch review before integration.
