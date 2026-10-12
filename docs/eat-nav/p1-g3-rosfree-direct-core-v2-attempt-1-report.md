# P1 G3 ROS-free Direct core v2, attempt 1

## Decision

**Protocol fail; stop after the one authorized process.** The output-path contract passed and the run reached Path activation. The simulator then raised a PyTorch inference-tensor in-place update error during observation reconciliation after one control step. Four 1-row traces were preserved. There is no valid closed-loop Outcome and no ROS comparison was performed.

## Frozen provenance

- Protocol: `eatnav_p1_g3_rosfree_direct_core_v2`
- Attempt root: `/home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-direct-core-v2-attempt-1/`
- Persistent alias: `/home/kojima/eatnav-experiments/p1-g3-rosfree-direct-core-v2-attempt-1/`
- Attempt manifest SHA-256: `9805581b98541504dc968a521609d05605fa1a5b1c0b616f3a7ecb299e41a37f`
- Simulator source SHA-256: `7bb4cb2872109bf2ee3dffd41e52be3c2f1e433cf6f689060df0c5743e680d0d`
- Host source SHA-256: `2261ed37eab4621e24c7d7098da4b8d5c7a107dbdde6ba901b8131dd1881e34c`
- Path-contract module SHA-256: `97fbcbc572056090fe4876a411a120e233718c8a0cbf8e825d06b16d706a6b51`
- Assignment SHA-256: `e41c4a86652472bc3cd4298a789e8ccc0e1846026fa3794d59c9c500fe4acbaa`

## Offline gate

The new output-path contract checks the authorized attempt root, Docker bind mapping, runtime authorization and identity, traversal, symlink escape, and sealed-root exclusion. A path-contract test suite passed 10 cases. The relevant static/synthetic suite passed 77 tests; the Torch-enabled control/adapter suite passed 7. The Phase A replay remained 11,268 rows with zero command mismatches. Host and container checks confirmed the live bind mapping and staged source SHA before launch.

## Runtime evidence

- The output-path contract wrote `output/path-contract.json` and passed.
- Pre-Path physical and shared DEM gates passed. Native DEM equality remained a separate diagnostic. The direct Actor input gate reported maximum difference zero against its frozen `1e-4` limit.
- Path activation occurred once; four independent C++ core instances each had one call at activation. No ROS follower, bridge, TF service, or Zenoh router was launched by this run.
- `output/diagnostic-error.json` identifies `reconcile_command_observation()` in `p1_g3_rosfree_control.py`, line 162: assignment into an inference tensor outside `InferenceMode`. The exception occurred after the first recorded control step, during preparation of the next Actor observation. This is an observation-adapter implementation failure; the 600-step control loop and Outcome gates were not reached.
- Simulator native exit was 0, but the host classified `diagnostic-error.json` as failure and exited 1. Owned residual processes: zero. Peak GPU device-total memory: 5180.875 MiB; peak container anon+shmem: 3,601,522,688 bytes; peak persistent batch size in the resource series: 3,550,147 bytes; monitored elapsed time: 15.16 s. No resource cap was exceeded.
- The failed batch host seal verifies: 302 files, stage `G3_rosfree_failed_or_inconclusive`. Attempt 0 and prior ROS evidence were not modified.

The failure occurred after Path activation, so the authorized no-retry rule applies. A future attempt requires fixing the observation tensor ownership/inference-mode handling, adding a live-relevant synthetic regression, freezing new source and manifest hashes, and separate runtime authorization. ROS closed-loop parity and recovery parity remain unverified.
