# P1 G3 ROS-free follower Phase A

## Scope and provenance

This is an offline implementation and replay only. No Isaac process, Path rollout, router, production ROS node, existing G3 evidence, evaluator training, or collection was changed. The ROS-free experiment is a **new protocol series**; v2/v3/v4 outcomes remain separate. The core is based on commit `938f797`. Its production semantic reference is `simple_trajectory_follower.cpp` SHA `75249a834a90feb25fc2967e2e253eeaeffa498491c68692434bd08099a8d2be`.

The parity thresholds and time boundary are in `config/eat_nav/p1_g3_rosfree_phase_a_parity.json`. The output ledger, including input trace and Path-publication SHA values, is `docs/eat-nav/p1-g3-rosfree-phase-a-parity-20261012.json`. The replay reads source traces only; it never writes into experiment roots.

## Implemented behavior

`p1_follower_core.cpp` retains the existing nominal projection, tangent, carrot, heading scaling, speed caps, goal line and goal hold. It now implements the single-Path G3 subset of the production control loop: bounded motion history, progress best-so-far and timeout, command-vs-motion linear and angular stall checks, frozen carrot-history backtrack, linear recovery completion, two-second angular boost, and sustained zero command after Goal. `p1_follower_core.py` exposes one-step command plus diagnostics and a full deep-copy `clone()` for all internal history. The old four-field `state()/restore()` remains for the original short-run API; it is **not** a full long-horizon snapshot. Phase B must instantiate one fresh core per env and use `clone()` only if a complete follower-state copy is needed.

The core accepts arbitrary nonnegative absolute logical stamps. The first stamp is the episode-local origin. G3 supplies consecutive 50 ms stamps, and offline replay rejects a gap or duplicate. Motion samples are pruned only when age is **greater than** 20 s; a motion window is ready when newest minus oldest is **at least** 20 s. Progress stalls when elapsed since last qualifying improvement is **at least** 20 s. Angular boost is active while `stamp < boost_until`. No 9-second absolute-stamp limit remains.

The C ABI returns command, carrot, projection, remaining distance, heading, direction, Goal/recovery flags and transition events, motion accumulators and pre-command diagnostic state, progress age, backtrack size/length, and elapsed time. The pre-command fields correspond to the exact-stamp diagnostic's `command_pre_publish` record, before `updateMotionTracking()` in the production control loop.

## Offline comparison and coverage

The replay feeds the *recorded exact-stamp TF pose* (`tf_x/y/yaw`) and original `map_xy` Path to a fresh core instance per trace. Command, carrot and heading use 1e-6 absolute tolerances; Goal/recovery flags and transition steps require exact equality. Carrot and heading are compared only while tracking: the ROS diagnostic retains its last values after Goal although the actual command is zero. Motion sample count, ready flag, oldest/newest stamp require exact equality, and motion accumulators/ratio use 1e-6. The stored trace validates its own 50 ms and TF/command stamp before replay.

Results: **16 output groups, 11,268 control rows, zero mismatches**. This comprises 2,400 G2 600-step rows, 6,632 rows from all 48 protocol-valid direct-repeat G3 trials, and 2,236 rows from the three first-differential crossover periods. The last crossover period is previously offline-audited conditional evidence, not a host protocol-valid seal; its replay is command-parity evidence only. Maximum observed command, tracking carrot/heading and motion-state error is 0 in the stored decimal representation. The original P1 3,360-command comparison also passes unchanged after relocating its fixture path to the persistent backup.

There are **no measured recovery steps** in these ROS traces. Synthetic tests cover progress stall at the 20 s boundary, linear command-vs-motion stall, frozen backtrack and exit after 0.3 m, angular boost start, Goal hold, and 640-step nonarrival. These establish implementation boundaries, **not measured ROS recovery parity**. Runtime and closed-loop parity remain unverified.

## Production and exact-stamp differences

The production control loop's order is Goal check → projection/direction → progress tracking → carrot/heading → progress/linear/angular recovery triggers → command publication → motion-history update → linear-recovery exit (`simple_trajectory_follower.cpp:317-515`). This order is retained for the single-Path core. Production `updateRobotPoseInPathFrame()` uses latest TF; the G3 diagnostic patch requests the exact logical TF stamp and emits only one command per stamp (`scripts/eat_nav_exact_stamp_build.py:21-68`). The pure core receives the already converted tile-local/map pose directly. It has no ROS timer, TF lookup, Path subscription, status publisher, recovery-Path subscription, or async callback. The G3 protocol publishes one Path only; production Path replacement and external recovery-Path behavior are outside this Phase A core and must remain excluded from the ROS-free G3 contract.

## Phase B direct G3 connection

1. Fork a new G3 simulator/host protocol and manifest; leave v2/v3/v4 untouched. Remove `FollowerBridgeClient` (`scripts/eat_nav_p1_g3_crossover_v2_sim.py:679-690`) and the ROS/Zenoh follower and bridge launcher (`scripts/eat_nav_p1_g3_crossover_v4_inner_host.py:704-760`). No router lifecycle requirement belongs in the new host.
2. Keep existing `local_pose_from_root`, `path_in_map`, four assigned Paths and tile-local origins (`v2_sim.py:671-679`). At Path activation, create one `PureFollower` per env and call it once at `logical_t0`, using the four actual poses. Preserve the published-path geometry and one-publication semantic as immutable input records.
3. Preserve `CommandOverride`, `capture_actor_critic`, paired noise, shared DEM, `env.step`, exposure windows, terminal capture, trace and Outcome logic (`v2_sim.py:752-862`). Before each physics step, write the direct core command to the same command-manager term. After physics, read fresh poses and call each core once for the next exact 50 ms stamp. Never call a core twice for one stamp.
4. Replace ROS-specific readiness/bridge acceptance fields with explicit direct-core construction, Path SHA, pose frame, per-env core identity, stamp monotonicity and command-injection evidence. Preserve pre-Path physical/shared DEM/Actor gates, episode/resource/SHA/seal gates, and duplicate Outcome diagnostics. Do not relabel these new records as ROS command acceptance.
5. Bridge evidence before any candidate comparison: same recorded Path and poses must match exact-stamp ROS commands at every step offline (Phase A); then run an authorized, small paired closed-loop ROS-vs-pure control and compare command, Goal/arrival, tracking, recovery and trajectory under the same initial state/noise. Assess state/physics variation separately. Only after a new protocol passes that gate should direct/curved candidate comparisons be considered. The two runtime variants are never silently pooled as one protocol.

## Decision

Phase A offline nominal and long-horizon Goal parity passes. The C++ core is ready for **Phase B implementation and a separately authorized runtime parity gate**, with the explicit limitation that measured recovery parity and closed-loop Outcome parity are still unavailable. This report does not authorize Isaac runtime or the remaining v4 processes.
