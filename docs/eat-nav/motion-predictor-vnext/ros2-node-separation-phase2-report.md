# EAT-Nav ROS2 node separation — Phase 2 CPU report

## Scope and state

This implementation is on `exp/eatnav-ros2-adapter-design` in the isolated worktree `/tmp/eatnav-ros2-adapter-design`, based on `0e74f6746006aef55ed9b6c54943d5940b82dea2`. It adds a CPU-only ROS2 mock pipeline. It does not modify the running Stage B/C worktree, the existing GUI demo, a production follower, the external router, Isaac Lab, or any checkpoint/evidence file. `Phase 3` remains unstarted.

The three design amendments are frozen in [the Phase 1 design](ros2-node-separation-phase1-design.md), sections 8.1–8.3. The new system is isolated from robot actuation by construction: all Path, follower-command and applied-command test topics are under `/eat_nav/mock/`.

## Implemented components

| Component | Responsibility | CPU evidence |
|---|---|---|
| `eat_nav_interfaces` | Versioned DEM, 22D Actor, bank, evaluation, selection, goal and status messages | Isolated Jazzy `colcon` build; ROS serialization and topic round-trip |
| Input adapter | Typed DEM/Actor codecs, explicit GridMap circular layout, 250 ms history, 22D term ordering | Float32 byte parity; negative heights and invalid cells; four circular starts, nonzero robot yaw |
| Inference node | Four-frame assembly, exact-stamp map→body TF, fixed 14-bank, goal score | GUI-less sealed sample → selected Path over ROS topics |
| Model worker | Python 3.10 CPU PyTorch isolated from ROS Python 3.12; three explicit modes | Pinned seven-forward legacy parity; synthetic adopted-14 provenance/output test; mock14 transport |
| Supervisor/gateway | Goal generation, bounded DEM and Path lease, Cancel/timeout/feedback stop, no real topic | Normal jitter/degraded/outage cases; stale generation rejection; mock final command zero |
| Independent mock robot | Holds only a mock Path; separate 100 ms status and 50 ms command watchdog | Nonzero mock command while valid; Path invalidated and zero applied command after Cancel/communication loss |
| Display node | RViz MarkerArray/status subscriber only | 14 Path markers; no navigation publisher |
| Sealed source replay | Read-only sealed Stage B `prepath.npz` for one CPU fixture | Outer-ledger SHA and DEM/Actor float32 round-trip; source pose from tile-local root |

The mock transport mode is the Phase 2 default. `legacy6_demo` uses the pinned MI-B Full seed456 checkpoint and exactly seven forward arcs in two six-slot batches; the worker's output matches the existing loader. `learned14_adopted` remains disabled until Stage C writes an adoption-Pass result and all result/dataset ledgers, selected checkpoint SHA, source SHA, candidate order and geometry pass validation. It cannot silently fall back to legacy or mock. The model's goal input remains absent; selection uses predicted two-second global-goal distance reduction and frozen tie ordering.

## Stop contract

The 250 ms DEM cadence is unchanged. A missing fresh quartet at sensor age ≤300 ms defers replanning while an earlier Path lease is still live. At 300–500 ms the state is degraded and still cannot issue a new Path. Sensor age >500 ms or lease expiry invalidates the Path generation and closes the mock command gate. Duplicate sensor stamps do not refresh age. Cancel, timeout and follower feedback loss also revoke the generation. The independent mock robot clears its held Path on inactive status and emits zero when status or command heartbeat expires. This proves only the mock interface; the production ROS follower's internal Path invalidation and physical stop are a separate, unapproved integration gate.

## CPU verification and limitations

- The isolated interface package builds with `/usr/bin/python3` 3.12. The system-Python suite passed **29/29** tests covering wire, input, safety, replay, topic, display, and mock pipeline. The `tinyvla` Python 3.10 PyTorch worker suite passed **2/2** tests, CPU only. Input/source/test SHA values are in [the Phase 2 ledger](ros2-node-separation-phase2-ledger.json).
- A sealed Stage B pre-Path sample was read without modification. Its `outer-seal.json` ledger hash, `prepath.npz` file hash and valid decision were checked before encoding. The four DEM arrays and Actor pre-noise 22D state round-trip bitwise through the generated messages. Replay stamps are shifted to a local ROS clock for live topic tests; these are **not** claimed as original sensor update timestamps.
- The mock end-to-end test sends an actual nonzero mock follower command, observes a nonzero mock applied command under an active Path, then cancels and observes Path clearing and zero. A separate-process ROS smoke selected one Path across source/inference/supervisor processes. Fast DDS emitted UDP socket permission warnings in this sandbox; the isolated same-host test still delivered topics, but external network transport was not certified.
- The real `/elevation_map` storage layout, pose convention, update cadence, and the real 22D Actor term provenance remain unmeasured. The GridMap converter requires an explicit layout and rejects a rotated map pose. There is no real-source ROS node enabled by default and no real follower connection. Real-source input parity is **Unknown**, not inferred from the synthetic GridMap test.
- Stage C's actual selected 14-model checkpoint does not yet exist. Synthetic 14-model loading tests prove the loader's shape and fail-closed checks, not predictive performance or adoption. Exact model-output parity with the eventual checkpoint is pending.
- A single sealed sample and mock predictor demonstrate communication and assembly, not navigation behavior. The production follower versus collection PureFollower difference and real stop acknowledgement remain unresolved. No Isaac integration, robot command, data collection or model training was performed.

## Isolated GUI-less invocation

After building `ros2/eat_nav_interfaces` into a private prefix, source `/opt/ros/jazzy/setup.bash` and set `PYTHONPATH` to this worktree's `src` plus the private interface `lib/python3.12/site-packages`, and `LD_LIBRARY_PATH` to the private interface `lib`. Use a private `ROS_LOG_DIR` and `ROS_DOMAIN_ID`; the tests use Fast DDS to avoid touching the externally managed Zenoh router.

Run `scripts/eat_nav_ros2_phase2.py` as separate `supervisor`, `inference`, `mock_robot`, and optional `display` processes, then run `sealed_source --sealed-root <verified completed Stage B process root>` for a one-shot no-GUI fixture. The supervisor and inference default to `mock14_transport`. `legacy6_demo` requires the pinned old checkpoint and Python 3.10 worker interpreter on both process commands; `learned14_adopted` additionally requires sealed Stage C `results.json` and `normalization.json`. The source replay is diagnostic only; it never publishes to production velocity or follower topics.

## Next gate

After Stage B/C has fully finished and the user separately authorizes it: inspect a saved real GridMap message and actual 22D source terms, verify actual adopted-checkpoint tensor/output parity if Stage C passes, then plan Isaac integration. If Stage C does not pass, keep the 14-candidate learned mode disabled. The independent mock stop contract does not authorize attachment to the real follower or robot.
