# P1 G3 crossover v4: offline runtime preparation

## Scope and decision

The v4 source and contract are prepared offline. No Isaac process, candidate Path, or router operation was started. The existing period 0/1 seals, conditionally audited period 2, failed v3 batch, offline audit, and 12-trial dataset remain unchanged. The v4 manifest sets `runtime_authorized: false` and lists no authorized process IDs.

The first remaining frozen row is `batch-first-differential-1-period-1`. Its frozen cross-period gate requires a protocol-valid `batch-first-differential-1-period-0` seal. The only existing period 0 seal for that seed is **protocol-invalid**, so direct continuation of the 14 remaining rows is blocked. No gate or assignment was relaxed. A future decision must authorize a fresh period 0 measurement under v4 in a new root, or explicitly redesign the cross-period reference rule before any period 1 candidate is published. The present preparation authorizes neither.

## Protocol and source freeze

| Item | SHA-256 |
| --- | --- |
| [v4 manifest](p1-g3-crossover-v4-manifest.json) | `b41f38e37c671671426d05ad0fe6a13cfd0acdb116dab69fa4690bdf16db024d` |
| [monitoring contract](p1-g3-crossover-v4-monitoring-contract.json) | `4d63bc4c94a3705f44755fb40d510ab7bb89877428c56faaea5b5201f2ff79b4` |
| [v4 inner host](../../scripts/eat_nav_p1_g3_crossover_v4_inner_host.py) | `41f8b53a32ca5c1dccd48617c2194f394a560851a5551c64012bc3132e47476d` |
| [v4 outer host](../../scripts/eat_nav_p1_g3_crossover_v4_outer.py) | `2a6602e6f42c3df0474190d49a2eb4b1cccecfd8d48b32e69fa4267583bac1d7` |
| [router observer](../../scripts/eat_nav_p1_g3_router_observer_v4.py) | `45027c8cd7a8e08faec85c44afcb6a39e2fedaf57ea26b75ce732621354a1452` |
| [evidence validator](../../scripts/eat_nav_p1_g3_runtime_evidence_v4.py) | `5b3e037bb9658c138ff2bdb3ac9ba77754ba3c740d41c81890161e07375513e1` |
| v3 parent manifest | `3f9c588bf002d9967632425f66cf32a57a6efdd0c7a795b0e993eba8e260b426` |

The manifest retains the exact 18-process assignment, Path geometry, simulator SHA, checkpoint, follower, physics, material, numeric gates, and resource caps. The v4 inner host is a copy of v3 with only router deleted-log FD fallback, GPU bracket wall timestamps, and terminal-boundary classification added. The v4 outer host adds a separate observer and an aggregate seal. It refuses the checked-in preparation manifest until a separately approved attempt manifest names its authorized process IDs.

## External router operating procedure

The operator owns `rmw_zenohd` outside candidate runners and supplies an absolute persistent log directory containing `router-launch.pid`, `router.pid`, `router.stdout.log`, and `router.stderr.log`. The router must be alive before the observer starts. Its PID/start tick, parent PID, session ID, cgroup, command line, ZID, each declared endpoint, and the 7447 listening socket are sampled every **0.5 s**. The observer starts before the inner host and finishes with a final sample after host exit and cleanup. Each JSONL sample and its heartbeat sequence is flushed and fsynced. A gap over **1.5 s**, missing heartbeat sequence, missing launcher/router, PID reuse, changed ZID, missing listener, unavailable log, or unreachable declared endpoint invalidates the lifecycle evidence. No missed period is interpolated.

The observer never starts, stops, signals, or configures the router. It reads stdout/stderr from persistent pathnames or the launcher's open regular-file descriptors if the pathnames are removed, and keeps fsynced experiment-owned byte copies plus source SHA, size, and FD target. It will not read a pipe FD, which could block. The experiment-owned copies are checked against the last observer sample SHA. This preserves evidence even when an original pathname disappears; the router's own logs remain externally managed.

## Independent communication and GPU gates

Lifecycle sampling cannot guarantee detection of every sub-0.5-second network interruption. The separate per-step gate checks all four traces for contiguous 20 Hz control steps, exact logical stamps and `/clock → TF → follower command → bridge acceptance → physics` order, successful exact-stamp TF, accepted bridge command, no reset/termination/truncation, and no missing or duplicate step. The inner host also checks four-follower final status, protocol errors, arrival/censor, 40-step hold, live material, and trace/host/container SHA. Both lifecycle and per-step communication evidence are required; one does not substitute for the other.

GPU samples now record wall-clock bracket start and end. Any process identity, attribution, device cap, or 256 MiB gap failure **before** a verified simulator completion marker remains fail-closed. After a marker with validated trace SHA and owner identity plus native exit 0, only known owned-process exit flags can be classified as normal terminal disappearance. A sample spanning the marker with an exit flag is inconclusive. Unknown native exit, invalid marker, unrelated GPU process, or numerical resource violation cannot use the terminal exemption. The raw bracket and any waived flags remain recorded. Existing v3 resource series and seals are untouched.

## Static and synthetic verification

The new tests cover router normal operation, disappearance, PID reuse, launcher loss, ZID change, endpoint or listener loss, heartbeat gaps, deleted log pathname with open regular FD, nonblocking rejection of a pipe FD, and durable log copies. They also cover post-completion PID disappearance, active GPU failure, ambiguous boundary overlap, invalid completion, per-step command/TF/step failures, explicit runtime authorization, source freeze, and post-seal mutation detection. Related v3 tests remain in the verification command.

Runtime behavior remains unverified: listener ownership visibility under the actual host permissions, observer endpoint polling load, 0.5 s sampling jitter during Isaac startup, complete outer cleanup, and v4 terminal classification under live shutdown must be checked in a separately authorized candidate-free preflight before any candidate Path process.

Verification on this worktree: **72 passed** across new v4 and related v3 lifecycle/attribution suites. The v4 per-step communication validator also accepted the 4 × 233 rows of the prior failed batch in a read-only compatibility check; this does not change that batch's invalid status. A repository-wide `pytest -q --maxfail=1` stopped during collection of unrelated corridor tests because the host `/usr/bin/python3` lacks `torch`; this is not a v4 gate result. No dependency was installed or runtime condition changed.

## Resume plan and resources

1. Keep all existing evidence immutable. Obtain approval for a fresh v4 attempt manifest/root and resolve the invalid period 0 predecessor. Do not mark the failed v3 period 0 valid.
2. Stage the unchanged inputs and v4 host sources in a new persistent root, verify every manifest/source/input SHA, and audit the external router identity. Run a candidate-free observer/GPU preflight before Path publication.
3. If a fresh `first-differential / batch 1 / period 0` measurement is separately authorized and becomes protocol-valid, run the 14 remaining rows in their frozen order, one process at a time. Check inner and outer seals, lifecycle and communication series, GPU/resource caps, and cleanup before the next row. Stop on any protocol failure; do not retry a Path-published row automatically.
4. Analyze complete seed×env Direct/Left/Right pairs only after all requisite periods are valid. Keep the existing 12-trial conditional analysis and any v4 results provenance-separated until an approved integration review.

The one recent failed-v3 process took **101.85 s**, used **5,212.06 MiB** peak device memory, **3.50 GB** container anonymous+shared memory, and **29.3 MB** of batch storage. This is a calibration point, not a promise for all terrains. At the same observed pace, a separately authorized recovery period 0 plus 14 rows would take about **25.5 minutes** of inner-host time, before staging/audit/cleanup. The frozen hard cap remains **600 s per process**, so 15 process caps alone sum to **2.5 hours**, excluding overhead. A fresh preflight and capacity estimate are required before any runtime approval.
