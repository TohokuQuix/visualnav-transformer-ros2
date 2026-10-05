# Portable Evaluator input integration: I3 design-review stop

## Decision

`portable_input_contract_requires_joint_subset_redesign`. Work stopped at I3-4, before a canonical adapter, source fixture, or logging parity run. The 52-joint vector in the frozen portable contract cannot be reconstructed from the sampled physical `/joint_states` without inventing values. The instruction requires a design review before choosing a reduced portable contract or a simulator-only 52-joint ablation.

## Completed audit

- Worktree: `/tmp/eatnav-vnt-portable-input`, branch `exp/eatnav-portable-input`, base `be5635674e996ac4881a61aac27645c4c19077d2`. The worktree was clean before this audit.
- Both prior artifact ledgers were rechecked: 22 M0–M10 files and 19 M11–M13 files, **41/41 SHA matches**. The frozen E1 simulator source is SHA `9e2a40f0b283d4581219302c3f2f0782b00ca825e07653bbe3924dd5c47cec07`. The known compressed trace and its fixed prefix, suffix, map Path, and RL checkpoint match the SHA values saved by that run. See [source-provenance.json](source-provenance.json).
- The existing 56-field future logging schema was classified provisionally as A 12, B 26, C 18. Seven fields presently logged after command update would need a separately timestamped pre-Path anchor capture before they could serve as online inputs. Six required online concepts, including Path geometry/profile and availability, are not fields of that 56-field collection schema. See [field-classification.json](field-classification.json). This classification is an audit, not an approved online message schema.
- The known live articulation order has 52 joints. Three historical real-robot bags, sampled read-only, each publish the same 16 `/joint_states` names: eight arm joints and eight chassis joints. Four simulator flipper names have direct name-normalization candidates (`chassis_fl_*` → `chassis/fl_*`, etc.). The 16 simulator left/right sprockets do not map one-to-one to the four front/rear physical sprocket names. The 32 simulator subrollers were not published in these bags. Position/velocity signs and offsets remain unverified. The [52-row joint audit](joint-order-contract.json) marks each as published, ambiguous, or absent from the sampled stream; it does not assert mechanical nonexistence.
- Across 48 historical bag metadata files, two were empty and none of the 46 readable files had a nonempty odometry topic. A short bag includes `map → chassis/body` TF and `/bno055/imu`, but its first IMU sample has covariance sentinel `-1` for orientation and angular velocity. Pose-difference velocity might be possible after timing/filter/reset validation; commanded velocity cannot substitute for measured base velocity. The current ROS bridge sends selected joint values and IMU data, not a full joint vector or a measured base linear velocity. See [rosbag-availability-audit.json](rosbag-availability-audit.json).
- The simulator's pre-step capture reads `ActionManager._prev_action`. The physical bridge backend caches a processed six-dimensional action, but the present frontend payload neither exposes a timestamped applied-action cache nor proves hardware adoption. This is another unresolved physical source, not a value to fill from `cmd_vel`.

## Contract choice needed

| Option | Required definition before implementation | Consequence |
| --- | --- | --- |
| Reduced portable body-state contract | Define a physically meaningful chassis subset and derive matching simulator channels. Confirm how four physical sprockets relate to the sixteen simulator sprockets; verify sign, offset, units, and timestamps. | One model input can exist in both domains after physical source and parity checks. The 52-vector is removed from the primary portable contract. |
| Simulator-only 52-joint branch | Keep the full simulator vector in a separately versioned ablation and define the portable primary input without it. | Simulator study can use all 52 joints, but its result cannot be claimed physically portable. |

The four flipper channels alone are the only direct name candidates. No sprocket aggregation or projection is selected here. Zero-filling missing joints would misrepresent the physical state. Whichever contract is chosen must receive a new schema version, field-order hash, normalization statistics, checkpoint compatibility check, and simulator/ROS2 validator tests.

## Scope and cleanup

No production source, follower, Actor/Critic, control law, exact-stamp protocol, old raw trace, bag, checkpoint, or old report was changed. No simulator, bridge, follower, router, or monitor was started by this audit. No fresh run or Outcome was generated. The host process check found no matching live router, simulator, bridge, follower, or monitor process; an escalated `tmux ls` reported no server. A historical router provenance file was SHA-frozen, but **current PID/start tick, ZID, endpoint and connectivity were not established**. Thus I9 would remain blocked even after input design resolution until the operator-managed router is independently verified.

## Pending after design review

Resolve the portable joint representation and measured base velocity source; specify pre-Path previous applied action and IMU/body frame timing. Then implement and test the canonical schema, simulator/ROS adapters, validator, tensor converter, separated loggers, source fixture, and the single logging-off/on 120-step parity pair under a verified external router. Only after those pass can the small-training plan and positive readiness decision be written. No training or collection was begun.
