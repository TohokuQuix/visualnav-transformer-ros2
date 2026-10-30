# R6 sensor-bag extraction and input audit

This is an offline input audit, not a model result or a demonstration of physical avoidance. No ROS node, player, publisher, gate, follower, or robot command was started. No threshold was fitted to this bag.

## Sources and integrity

Source: `/home/ubuntu/eat_nav_planner_20261029_210926/bag`.
The handoff `/home/ubuntu/eat_nav_retraining_handoff_20261030/manifest.json` contains 55 entries; every source file was read and its size and SHA256 matched. Required `metadata.yaml`, `bag_0.mcap`, parent `planner_params.yaml`, and `record_qos.yaml` are present. Source digests and artifact digests are retained in `bag-intermediate/index.json`.

MCAP SHA256: `f2ba42fdc16c8a86853fd79e508eb98b0ca64638ef527a75cca941baa7127ba3`.
Metadata SHA256: `8e8efeaad54d2c2f1ca2ade0e66611495040bfd771590db36ba87a858571e042`.
The prior report references `sensor_only_210926_planner_status.json`, which is not included in the handoff. Its reported R5 behavior cannot be substituted for new-model evaluation.

## Actual coverage

5,655 messages; log interval `1793275767193966714..1793275793226995695` ns, 26.033 seconds. UTC start: 2026-10-29 12:09:27.193967.

| Topic | ROS type | Count |
|---|---|---:|
| /filtered_elevation_map | grid_map_msgs/msg/GridMap | 87 |
| /bno055/imu | sensor_msgs/msg/Imu | 2605 |
| /joint_states | sensor_msgs/msg/JointState | 521 |
| /diff_drive_controller/odom | nav_msgs/msg/Odometry | 261 |
| /tf | tf2_msgs/msg/TFMessage | 616 |
| /tf_static | tf2_msgs/msg/TFMessage | 3 |
| /eat_nav/plan_lease | std_msgs/msg/String | 521 |
| /eat_nav/status | std_msgs/msg/String | 520 |
| /eat_nav/gate_ready | std_msgs/msg/Bool | 521 |

Declared zero-count topics: /goal_pose, /eat_nav/candidates, /eat_nav/path, /eat_nav/selected_path, /eat_nav/cancel, /flipper_fold_mode. No executed candidate labels or counterfactual candidate outcomes can be recovered from these messages.

## Clocks and causality

The MCAP log timestamp is the extraction's availability clock; this is a conservative recorded-availability boundary, not proof of physical acquisition time. All selected input records have log time <= decision time. Sensor header clocks remain unchanged. Each DEM is rasterized using a body TF received no later than that DEM. Decision-time gravity comes from the latest already-received map-to-body TF. No nearest future receipt or interpolation with future receipts is used.

| Stream | Median log minus header seconds |
|---|---:|
| DEM | -0.319915 |
| IMU | -0.517270 |
| JointState | -0.516680 |
| Odometry | -0.516801 |
| map to chassis/body TF | -0.429760 |

Header timestamps lead log timestamps systematically. This does not establish physical anticipation: clock synchronization is unresolved. The extraction does not shift timestamps to hide the discrepancy. Downstream physical timing/lead-time claims must retain that limitation.

Four distinct real maps are selected at nominal 1/3-second spacing ending at the latest received map, with actual log/header times retained. This differs from the frozen training's exact 250ms history. Header intervals min/median/max: 0.200160/0.300000/0.399845s. Receive ages for oldest/second/third/latest frames across the 13 inputs span approximately [0.904,1.136]/[0.591,0.830]/[0.300,0.524]/[0.00693,0.220]s. These values are not replaced by synthetic 250ms ages. The latest DEM, sensor state, and body TF each require receipt freshness <=1s; older historical DEMs naturally exceed that current-sensor limit.

Fourteen decision attempts were made at 2s spacing from the first recorded receipt: the initial attempt lacked history, and 13 produced finite observations. Invalid attempts and reasons are in `index.json`; no invalid attempt is converted into a safe label.

## Physical contracts and unresolved calibration

Raw DEM is 20x20 at 0.1m resolution, 2mx2m, frame map. `elevation_inpainted` is present, all cells finite, raw height range -0.320413..0.518117m. The handoff's existing `inputs.decode_grid_map` and `sample_dem` implement layout unwrapping, bilinear interpolation, yaw alignment and subtraction of the median reference annulus (radius 0.3..0.8m). Unknown/out-of-extent cells retain explicit validity masks; zero invalid heights do not mean free ground. The physical scale is unchanged.

Recorded planner parameters use `elevation_inpainted`; the handoff generic configuration defaults to `elevation`. Extraction explicitly uses the recorded layer. Full raw serialized maps, schema definitions, raster heights, centers, resolutions, and source timestamps remain available.

Actor22 layout remains the handoff contract: angular velocity x0.25, projected gravity, three excluded command placeholders, fold mode, flipper radians in RR/RL/FR/FL order, right/left sprocket speed x0.05, grouped effort x0.01. Body9 is projected gravity, body linear m/s, body angular rad/s. The model's state28 conversion removes actor command columns6..8 from the concatenated actor22+body9. The extractor does not normalize by training statistics.

IMU orientation covariance[0] is -1 throughout: numeric quaternion values are preserved in raw records but not used for gravity. Gravity uses the received map-to-body TF orientation. Angular velocity uses IMU measurement transformed by the received body-to-bno055 static mount (identity here). This is an explicit alternate orientation source, with localization and mount calibration limitations; it does not prove parity with a calibrated IMU pipeline. Per-sample `imu_orientation_valid` is false.

Measured flipper positions: RR about0.71389, RL0.76284, FR0.82770, FL0.96615rad. Simulation reset references define rest0 and folded pi. The adapter directly uses measured radians; no verified encoder-origin/sign calibration is available. Raw values are preserved; no training-mean substitution is applied. Fold mode has no messages and uses the handoff's declared zero initial default, recorded as such in the index.

## Reproduction and training interface

ROS reading requires the installed Python3.12 system environment. Default Python3.10 cannot load the Python3.12 ROS extensions. `grid_map_msgs` is absent, so a bounded CDR1 parser reads its embedded MCAP schema; all standard messages use ROS deserialization. No additional ROS package installation is required.

```bash
/usr/bin/python3 scripts/eat_nav_r6_extract_bag.py \
  --bag /home/ubuntu/eat_nav_planner_20261029_210926/bag \
  --output /tmp/new-r6-bag-extraction
/usr/bin/python3 -m unittest discover -s tests/eat_nav -p test_r6_bag_extract.py -v
```

The output directory must not already exist. Actual intermediates are in `/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/bag-intermediate`:

- `serialized_records.bin`: exact serialized contents of every original message, concatenated; 5,438,478 bytes.
- `records.jsonl`: all5,655 original topics, log times and raw offsets/lengths; standard message fields include original TF and header timestamps. GridMap headers are explicit; full contents remain in binary.
- `message_definitions.json`: original embedded ROS schema definitions.
- `raw_dem.npz`: all87 unwrapped raw maps, validity, centers, yaw, resolution and both timestamps, independent of observation sampling.
- `observations.npz`: 13 causal inputs and timing/validity diagnostics.
- `index.json`: source and artifact hashes, common goal, units, orientation provenance, missing calibration, attempted decisions, clock offsets and limitations.

Import `scripts/eat_nav_r6_extract_bag.py` from any NumPy environment and call `arrays, index = load_observations(directory)`; this path imports no ROS modules. The loader verifies observation SHA256. Keys:

| Key | Shape | Meaning |
|---|---|---|
| dem | 13,4,2,20,20 | height[m], binary validity |
| actor_state | 13,22 | old unnormalized actor contract |
| body_state | 13,9 | gravity, velocity, angular velocity |
| decision_receive_ns | 13 | decision availability clock |
| dem_receive_ns / dem_header_ns | 13,4 | actual history clocks |
| state_receive_ns / state_header_ns | 13,3 | IMU, joints, odom clocks |
| tf_receive_ns | 13 | orientation/pose TF availability |
| pose_map | 13,4 | map xyz[m], yaw[rad]; geometry-only, not model input |
| imu_orientation_valid | 13 | false; supplied orientation covariance marks invalid |
| valid | 13 | all stored observations valid; failed attempts stay in index |

The evaluator adds the exact frozen 14-candidate path bank/directions; paths are not inferred from this goal-free bag. Receipt age arrays can be calculated without any timestamp guess: `decision_receive_ns[:,None] - dem_receive_ns`, and similarly for state. Do not use `pose_map` or later raw streams as extra model inputs.

Common provisional map goal for all frozen profiles: **(-0.0825648920573,-0.706100836189)m**, exactly2m ahead of the first received map-to-body TF. Initial position (-0.0670918514374,1.2938393091684,-0.1855965604091)m; yaw -1.5785329242835rad. Raw DEM geometry and `pose_map` support later shared goal/obstacle overlays. These are geometry inputs, not a new-predictor result.

## Verification performed

Five unit tests pass: CDR alignment/bounds, embedded-schema synthetic GridMap decoding and truncation/trailing-byte rejection, inclusive receipt causality, distinct causal history, and yaw-relative provisional goal. Actual extraction counts match bag metadata, every artifact SHA256 was rechecked, all stored observations are finite with documented shapes, all selected receipt times precede decisions, four DEM frames are distinct, and IMU orientation invalidity is preserved. No threshold or model evaluation has been performed by this extractor.
