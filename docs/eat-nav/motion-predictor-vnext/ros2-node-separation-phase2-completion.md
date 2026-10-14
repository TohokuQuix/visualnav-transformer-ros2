# EAT-Nav ROS2 node separation — Phase 2 completion record

Status: **Phase 2 complete (CPU-only scope), accepted 2026-10-14**. Phase 3 has not started. This record fixes the reproducible implementation state; it does not certify Isaac Lab or robot operation.

## Frozen implementation

| Item | Record |
|---|---|
| Branch | `exp/eatnav-ros2-adapter-design` |
| Isolated worktree at completion | `/tmp/eatnav-ros2-adapter-design` |
| Stage B/C base | `0e74f6746006aef55ed9b6c54943d5940b82dea2` |
| Phase 2 implementation HEAD | `4a41f59dcf2a3e24cc9d6974e92d62e548b505b5` |
| Design and stop contract | [Phase 1 design](ros2-node-separation-phase1-design.md), especially §§8.1–8.3 |
| Implementation and verification | [Phase 2 CPU report](ros2-node-separation-phase2-report.md) |
| Source, test and fixture hashes | [Phase 2 SHA ledger](ros2-node-separation-phase2-ledger.json) (`eatnav_ros2_phase2_cpu_source_ledger_v1`) |
| Implementation plan | [Phase 2 plan](../../superpowers/plans/2026-10-14-eatnav-ros2-adapter-phase2.md) |

The implementation history is `46e71e2` (design), `c76fc0e` (wire contract), `961516c` (input adapters), and `4a41f59` (CPU pipeline). The implementation worktree was clean at this record's creation. The SHA ledger covers 27 source files, 12 test files and four frozen dependencies; it also records the pinned legacy checkpoint and the read-only sealed Stage B fixture hashes. The ledger is the file-level identity record. A later documentation-only commit does not change that implementation identity.

To verify a checkout, check out the implementation commit in a private worktree, hash every `source_sha256`, `test_sha256` and `frozen_dependency_sha256` path in the ledger, and compare the two read-only fixture hashes against their original files. Do not rewrite the fixture or the active Stage B/C tree. Build `ros2/eat_nav_interfaces` with ROS Jazzy and `/usr/bin/python3` in a **private colcon prefix**; run the ROS/input/mock suites under system Python 3.12 and the two PyTorch worker tests under `tinyvla` Python 3.10 on CPU. Keep `ROS_DOMAIN_ID`, `ROS_LOG_DIR`, build and install paths private. The existing test modules are `tests/test_eatnav_ros2_phase2_*.py`; the separate worker interpreter boundary is described in the Phase 2 report. `/tmp/eatnav-ros2-adapter-build/` held the original private build and logs and is not the durable source of record.

## Verified result and limits

- The isolated ROS interface build succeeded. The final system-Python CPU suite passed **29/29** tests, and the CPU PyTorch worker suite passed **2/2**. The sealed Stage B pre-Path DEM and Actor fixture survived float32 ROS message round-trip bitwise, including its four-frame order; this is one read-only fixture, not real sensor parity.
- The 14-candidate `mock14_transport` pipeline selected and published a mock Path. The independent mock robot observed a nonzero applied command while the lease was valid, then invalidated the held Path and emitted zero after Cancel. Normal DEM jitter, degraded input, outage and stale-generation handling are specified in the design and covered by CPU tests. All actuation test topics use `/eat_nav/mock/`.
- `legacy6_demo` is separately pinned to the old forward-only checkpoint. `learned14_adopted` remains disabled until a sealed Stage C adoption **Pass**, selected checkpoint and normalization pass strict identity checks. Synthetic loader tests are not evidence that a Stage C model has been adopted.
- The actual GridMap layout, update stamps and frame behavior; real 22D Actor term source, order and units; and the production follower's Path invalidation and physical stop are **Unknown**. External ROS network transport, Isaac integration and navigation performance have not been certified. The mock stopping result is not a real robot stopping guarantee.

The running Stage B/C environment, data, checkpoints, manifests and evidence were not changed for this completion record. No Isaac Lab launch, GUI DEMO edit, real follower connection, model training or collection was performed. Phase 3 requires the separate approval and gates in [the Phase 3 verification plan](ros2-node-separation-phase3-verification-plan.md).
