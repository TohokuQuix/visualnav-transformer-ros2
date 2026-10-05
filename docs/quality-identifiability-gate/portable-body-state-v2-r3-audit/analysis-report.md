# Portable body-state v2: source comparison and R3 stop

## Decision

`sim_robot_aggregation_mismatch`. The user's clarification establishes the local `onix_policy_bridge` observation/control implementation as the physical reference, superseding the earlier R0 identity stop for this audit. The source comparison then finds that its two track channels do **not** use the same joint population as Isaac Lab. R4's reduced primary contract is therefore not frozen, and R8/R10 fixtures, adapters, and simulator logging parity were not started.

## Executed observation paths

The local installed package resolves through `/home/kojima/onix/build/onix_policy_bridge` to `/home/kojima/onix/dep/onix_automation/onix_policy_bridge`. Its frontend looks up `/joint_states` by name and sends four flipper positions plus front/rear sprocket velocities for each side to the backend. The backend builds an Actor tensor from the current config. Its `model_8000.pt` accepts 22 inputs; the frozen simulator `model_10000.pt` also accepts 22. Current configured terms total 22, so the bridge's dimension-based toggle reconciliation does not change this layout.

| Actor slice | Meaning | Isaac Lab | Local bridge |
| --- | --- | --- | --- |
| 0–2 | body angular velocity | simulated body angular velocity ×0.25 | IMU angular velocity after frame transform ×0.25 |
| 3–5 | projected gravity | world gravity in body frame | IMU orientation → body gravity |
| 6–8 | velocity command | CommandManager command | selected frontend command |
| 9 | flipper fold mode | simulator mode | `/flipper_fold_mode` |
| 10–13 | flipper positions RR, RL, FR, FL | `joint_pos_rel`, default offset 0 | name-reordered `/joint_states.position`, no code offset |
| 14–15 | right, left track angular speed | **mean of 24 joints per side** | **mean of 2 sprockets per side** |
| 16–21 | flipper and track effort | 4 flippers, sum of 24 per side | 4 flippers, sum of 2 per side |

The exact per-slice sources and transforms are in [bridge observation schema](onix-policy-observation-schema.json) and [Isaac Actor schema](isaac-actor-observation-schema.json). Both policies exclude base linear velocity and previous action from the **Actor** 22-vector. The simulator Critic includes a last-action term; it must not be mistaken for Actor input. Raw effort remains outside the proposed portable primary contract.

## First mismatch: track channels

Isaac Lab's `track_mean_velocity` uses eight main sprockets, eight front subrollers, and eight rear subrollers on **each** side, then scales the mean by `0.05`. The bridge averages physical front and rear sprocket velocities on each side, also scales by `0.05`; it has no subroller values. These are distinct aggregate measurements even though both produce right/left values in the same tensor positions. No one-to-one raw joint mapping was assumed.

The saved simulator anchor verifies the simulator formula numerically: right 24-joint mean ×0.05 = `0.067405242721`, Actor pre-noise slice = `0.067405246198`; left = `0.252876854936` versus `0.252876847982`. Errors are below `1e-8`. See [track aggregation contract](track-aggregation-contract.json). This validates the source interpretation; it does **not** establish that a physical two-sprocket mean equals a simulator sprocket-plus-subroller mean under load or slip. The existing robot policy therefore has an unquantified sim-to-real observation gap on these two channels.

## Other findings relevant to a future contract

- Flipper names and RR/RL/FR/FL order align after name normalization. The simulator's `joint_pos_rel` subtracts a configured zero default position; the bridge takes raw JointState position without a sign/offset conversion. The physical encoder sign and zero still need independent parity evidence. [Mapping](flipper-mapping-contract.json).
- The bridge's primary IMU path uses `/bno055/imu` if timely and otherwise can use Livox angular velocity plus `map → chassis/body` TF. It does not inspect covariance sentinel `-1` when selecting the primary sample. Freshness uses callback receipt time, not propagated message header stamps. IMU extrinsic/time parity remains open. [IMU audit](imu-contract.json).
- The bridge's processed six-dimensional `last_action` is assigned **before** frontend publication. Its `last_action_obs` is raw policy action, and the current Actor does not consume it. Neither proves hardware adoption; `previous_commanded_action` and `previous_applied_action` must remain distinct. [Action audit](previous-action-contract.json).
- Base linear velocity is absent from both 22-dimensional Actor configs. The backend has an optional wheel-kinematic estimate, which is not measured body twist. It belongs only in an optional ablation. [Velocity decision](base-linear-velocity-decision.json).
- Eight physical arm joints remain excluded. A short existing bag's 92 JointState samples show arm position ranges at most 0.0031 rad, but that 4.7-second segment cannot establish a fixed pose generally. The future portable contract requires `arm_configuration = fixed_stowed_pose`. [Arm scope](arm-configuration-scope.json).

## Next design decision

An agreed mapping or new measurement must establish how the two-sprocket physical summary relates to the 24-joint simulator summary. The choices to review include changing the **offline Evaluator input derivation** to a simulator sprocket-only proxy and quantifying the proxy error against the frozen Actor slice, or retaining the current two summaries as domain-specific features with explicit domain and parity limits. This report selects neither and changes no RL policy, follower, simulator dynamics, checkpoint, loss, or collection configuration. A source-equivalence decision is required before `portable_evaluator_input_v2` can be frozen.

The previous artifacts were not overwritten. No simulator, router, bridge, follower, monitor, training, collection, or real-robot command process was started by this audit; no owned process needs cleanup.
