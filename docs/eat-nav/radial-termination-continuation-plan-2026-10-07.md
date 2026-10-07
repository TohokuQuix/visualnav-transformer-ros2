# Radial termination-aware continuation plan

1. Test and implement a pure termination record validator: selected versus non-selected resets, multiple selected resets, term flags, counters, pre/post-reset separation, timeout and unknown reasons.
2. Capture terminal evidence in the simulator at the Onix pre-reset boundary. Close only terminated selected streams; continue the other selected poses to physics step 200. Preserve all frozen scene and physics settings.
3. Extend shard verification and host lifecycle to accept either 200 complete steps or a fully evidenced physical termination per selected pose. Preserve atomic markers, SHA, resource and ownership checks.
4. Build a new continuation root using byte-identical pose manifests and the unchanged 112-pose inventory. Add a read-only reuse ledger for shards 00–14. Update only measurement source hashes and new root paths.
5. Run tests, then one new `shard-15`. Continue shards 16–22 in order only if the shard-15 gate passes. Stop on any integrity, resource, lifecycle, or unknown termination condition.
6. Aggregate all 112 poses offline into A/B/C/D, report terrain/direction and interaction supply, seal the SHA ledger, commit only source/tests/small reports, and stop before candidate Path, second seed, or learning.
