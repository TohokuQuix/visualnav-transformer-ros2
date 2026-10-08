# P1 resource-attribution smoke: stop report

**Decision: `p1_resource_attribution_fail`.** The frozen 28-process P1 manifest was not started. X1, X2, follower, bridge, candidate Path publication and Outcome collection were not started.

## What passed

- The earlier 4-anchor radial reconstruction parity was reused from its SHA-verified ledger; the three-shard parity fixture was not rerun.
- The representative shard was selected before launch from the highest prior GPU peak: original shard-B / batch seed 20261007 (5,906 MiB in the prior fixture).
- The collector's offline tests passed: 12 initial GPU classification tests, then 16 after the startup-race test. The final related suite passed 50 tests.
- Both candidate-free simulators eventually completed 200 physics steps with zero physical terminations. Their 89 files each were copied to a separate read-only audit root and SHA-matched against the container. No owned simulator process remains.

## Why the resource gate failed

The first failing GPU sample in each attempt saw the new simulator as a compute process but reported no owned simulator identity. The collector classified that process as `unexpected_compute` and `unresolved_gpu_process`, as the frozen contract requires. The samples were at about 2.9 s and 3.2 s; device-total use and bracket gap were below the caps. The matching GPU rows carried exact host PID/start tick, Docker cgroup and the frozen simulator output path. A single-row offline lookup succeeds, while the live owner resolver returned empty. Multiple matching launcher processes in the live table are a plausible cause, but that table was not saved; the exact ambiguity cannot be proven from these artifacts.

The first change moved launcher resolution into the same NVML bracket and passed its RED→GREEN test. The second smoke still failed. No second code repair or third smoke was attempted.

## Protocol and authorization deviations

The monitor copied each run while the simulator continued in Docker. The local stopped-run copy already contained physics step 50 data, although the three GPU samples had no heartbeat. The remote simulator later wrote its 200-step completion marker. Thus the first attempt was not safely pre-control when the second attempt was launched; that second attempt exceeded the permitted remediation boundary. The source-based 28-row output overlay for this fresh attempt was also not frozen before the smoke. These deviations are recorded and neither run is promoted to a valid resource measurement.

The operator's router files were snapshotted separately after stopping. The original launch PID and child were no longer live. Codex did not start, stop or modify the router.

## Preserved evidence

- Attempt roots: `/tmp/eatnav-p1-resource-rollout-20261008-attempt1` and `/tmp/eatnav-p1-resource-rollout-20261008-attempt2`.
- Full remote-run read-only copies and SHA ledgers: `/tmp/eatnav-p1-resource-rollout-poststop-audit-20261008`.
- Durable backup of both attempts and the audit copy: `/home/kojima/eatnav-backups/2026-10-08-p1-resource-attribution-stop` (275 files, all SHA-matched).
- The frozen P1 model, anchors, Paths, order, seed, resource thresholds and control rules were not changed.

The next run needs a separately reviewed collector that records the complete bracket process table, resolves launcher and simulator identities without ambiguity, and guarantees exact owned-process cleanup before any retry. This report makes no P1 signal or teacher-quality claim.
