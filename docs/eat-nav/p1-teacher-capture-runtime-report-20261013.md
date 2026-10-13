# P1 G3 Evaluator teacher capture: implementation and small runtime check

## Decision

The ROS-free, Direct-only teacher capture diagnostic is protocol-valid. The four fixed 2 s labels can be reconstructed from one sealed rollout. This verifies logging on one StateGroup and one seed; it does not authorize the second-graded crossover, Evaluator training, or collection.

## Provenance and preservation

- Protocol: `eatnav_p1_teacher_capture_v1`, control protocol `eatnav_p1_g3_rosfree_crossover_v1`.
- Successful root: `/home/kojima/lib/onix_isaac_lab/.eatnav-p1-teacher-capture-v1-attempt-20261013-2/batch-first-differential-0-period-0`.
- Manifest SHA: `4c9cc841ad6cec74d747c8556adf17d22bfd8ea6d9134abb9c1806c38807f46e`; assignment SHA: `9bb65ecee7ddd747a127b021c792b7f6278c255e0ec63eed6453efa3b032ae9f`; simulator SHA: `1dbb06305c0294ed7237fb28969e0e2af480d31cee4967b1b22beb86e58ea744`; staged host SHA: `434995bc41e901e3f21d48194874eed0628cc0f107f1f09f599266f98c12821a`.
- Host seal SHA: `9380c428497a8a421c9e069d190f057ea6237d2347ec53e2cdecd269a43ee2eb`. Read-only seal verification: 324 files, valid. Chunk verification: 3 chunks, 344/344 rows, NPZ shapes/dtypes and gzip row counts valid. Host/container SHA checked by the host runner.
- The original sealed 12-trial crossover roots were not changed. The failed staging root `...attempt-20261013-0` and prelaunch import failure root `...attempt-20261013-1` remain separate. No Isaac execution occurred in those roots.

## Capture contract

The new simulator logs resolved reward term names and weights, weighted contributions, raw values where the resolved weight is nonzero, termination contribution, and returned total reward. RewardManager is read after its normal compute; no second reward computation occurs. The frozen sum tolerance is `atol=rtol=1e-5`. Across all 344 rows, the largest weighted-term sum error was `5.96e-8`; teacher total and existing trace reward differed by `0`.

Joint index/name/actuator mapping has 52 joints; body index/name mapping has 60 bodies. Each row includes pre-step joint position, velocity, applied torque, available actuator targets, and post-step applied torque. The logged torque is simulated applied joint effort in N·m, not a measured hardware torque. Body net contact force is world-frame N at pre/post control step. The physics tap reads `contact_physx_view.get_net_contact_forces(dt=0.005)` after each of ten actual physics steps per 50 ms control step and stores per-body peak vector/index, vector time integral in N·s, and nonzero substep count. It does not reuse the 50 ms ContactSensor history as a substep sample. This remains a body-net measurement, not per-contact-point force or local stress.

The compact writer uses compressed NumPy NPZ for fixed tensors and gzip JSONL for ragged rows, with a bounded 128-row CPU buffer, per-chunk SHA, row count, complete marker, and outer host seal. The current ragged contact table is empty by design: per-point contact data is not collected. Each row includes env/step/stamp, root pre/post, follower/injected command, Actor/Critic input, action/value, joint/contact/load, reward terms, and goal status; Path and DEM provenance are linked by sealed env-specific manifests. Allowables use `eatnav_pain_allowables_v1`; missing engineering values produce `missing_allowables`, so normalized Pain is not reported. The offline interface supports separate joint/body components, mean, peak, p95, and exceedance duration when defensible allowables are supplied. A synthetic numerical test exercises that formula.

## Verification

- Focused new tests: 20 passed. Related ROS-free control, crossover, and follower-core tests: 26 passed.
- One runtime process, four Direct trials, 20-step settling, 20 Hz control: all four arrived and completed 40-step hold. Control steps: 86; per-env trace rows: 86; no termination or reset. Physical/shared DEM/Actor/material, command/Actor/physics order, resource, cleanup, and seal gates passed.
- Fixed 40-step labels: env 0/1/2/3 common-goal progress `0.494/0.537/0.574/0.483 m`; accumulated, undiscounted logged RL reward `15.709/15.798/15.487/15.727`. Forward absolute motion error `0.068/0.053/0.100/0.075 m/s`; yaw absolute motion error `0.056/0.051/0.077/0.040 rad/s`. Env 2 reached its local Path goal at step 39; the others reached after the fixed label window. All four Pain labels are explicitly unavailable because allowables are missing. The separate machine-readable analysis is `docs/eat-nav/p1-teacher-capture-runtime-analysis-20261013.json`.
- Logging-off comparison: the existing sealed crossover period-0 root has identical freeze SHA, pre-Path gate SHA, shared DEM SHA, and terrain material fingerprint. Env 0/1 were Direct in both runs; their 86 overlapping rows have zero differences in every control/Actor/physics/reward field except `timing_protocol`, whose metadata reflects different protocol/termination lengths. Arrival steps match exactly (44/41). This supports noninterference for those two paired branches. Env 2/3 have different Path assignments in the prior run, so they cannot serve as logging-off controls.

## Measured resources and projection

The successful 4-env process took 37.63 s, with device-total GPU peak 5201.94 MiB, GPU attribution gap peak 103.98 MiB, container anonymous+shared memory peak 3.236 GiB, and final root size 10.19 MB. The compact teacher chunks occupy 0.536 MB for 344 env-steps (about 1.56 kB/env-step). The existing verbose JSON traces in the same run occupy 5.809 MB. No owned residual process remained; native exit was 0.

For 100 robot-hours at 20 Hz, linear storage extrapolation from this short trace gives about 121.6 GB for verbose JSON, 9.7 GB for compact fields without substep summaries, and 11.1 GB with the measured substep peak/integral fields. Retaining all ten raw `[60,3]` float32 body-force samples would add about 51.8 GB before compression or metadata, yielding at least roughly 61.6 GB with the compact base; this full-substep mode was not implemented or measured. These are storage planning figures for the observed schema and compression, not a 100-env runtime validation. GPU/RAM and wall time must be measured in a separate 100-env calibration; the four-env peaks cannot be scaled reliably.

## Remaining work before the 24-trial second-graded pilot

Freeze a new pilot manifest with two independent seed bundles, 3 balanced periods × 4 env per seed, its Path/source/input SHA ledger and this logger schema. Verify resolved reward manifest, 52-joint/60-body mapping, contact tap availability and 5 ms count, material/pre-Path gates, resource caps, chunk/host seals, and stop on any protocol/resource failure. The same fixed 2 s window and four separate labels should be used; missing Pain allowables must remain missing while raw load is retained. Per-contact-point data, local stress, and real mechanical allowables remain unvalidated. The prior ROS-free crossover outcomes are not silently pooled with this new logger series.
