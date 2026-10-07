# R3 eight-pose candidate-free settling pilot plan

1. Freeze the offline proposal, the full virtual scan, source assets, and old inventory contract by SHA. Derive exact rank, terrain, row/column/env ID, requested XY/yaw, and virtual DEM SHA for eight primary poses. Preserve the remaining R3 choices in a separate reserve manifest that the runner never reads.
2. Partition ranks 1/3/5/7 into shard A and 2/4/6/8 into shard B. Each shard runs the same 112-environment scene, normal RL reset, selected XY/yaw only, zero 6D action, 200 physics steps, four DEM frames, Actor captures, and termination-aware contact trace as the first inventory.
3. Reuse the frozen physical/eligibility thresholds and contact capacity. Adapt only the selected-pose mapping and requested XY/yaw. Check the generated terrain mesh SHA against the completed offline scan before any physics step.
4. Add fail-closed host lifecycle: independent subroots, source/input SHA, PID/start tick, resource and heartbeat monitor, native exit, per-file host/container SHA, seal, and residual-process check. Gate shard B on shard A's protocol validity, including valid physical terminations.
5. After both shards, compare only eligible settled frame-03 DEMs against the frozen virtual class and profiles. Produce class supply, plots, decision JSON, and SHA ledger. Stop without reserve poses, second seed, Path, follower, Outcome, or learning.

Infrastructure-only repair is bounded to two cycles with a new root, RED→GREEN test, and source SHA. No repair after a physical candidate run has started if the issue concerns state, dynamics, classification, or eligibility.
