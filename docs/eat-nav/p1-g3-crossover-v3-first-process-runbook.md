# P1 G3 crossover v3: first remaining process

## Frozen scope

- New host protocol: `eatnav_p1_g3_path_crossover_v3_assignment_validator`.
- The simulator continues to use the unchanged crossover v2 protocol and source.
- The only authorized process is `batch-first-differential-1-period-0`, seed `2026101102`, four env slots with `Left, Right, Direct, Direct`.
- Prelaunch and post-Path call the same frozen-assignment validator; the host and shared-module source SHA are recorded in `p1-g3-crossover-v3-manifest.json`.
- Root: `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-crossover-v3-first-remaining-attempt-0/`. The batch is staged without an output directory. No other process is staged in this root.

## Runtime gate

1. Verify staged input SHA ledger, container and GPU readiness, and the externally owned Zenoh router identity. The runner does not start, stop, or reconfigure the router.
2. Execute only the staged batch using the existing G3 outer launcher, with console and exit records outside the sealed batch.
3. Verify live material, physical/shared DEM/Actor pre-Path gates; four follower readiness; exact-stamp and command acceptance; same episode; arrival/censor/hold; resource series; native exit; zero residual; host/container SHA; host seal and post-launcher read-only seal verification.
4. On any protocol/resource failure, stop without another process or outcome-driven retry. A valid batch is never repeated.

The existing caps remain: 600 s per process, GPU device total ≤7,680 MiB, attribution gap ≤256 MiB, container anonymous plus shared memory ≤6 GiB, persistent root ≤1 GB, and heartbeat ≤90 s. The three measured pilot process durations were 77.14, 77.87, and 133.26 s; their GPU peaks were 5,169–5,224 MiB and batch storage peaks were 21.4–30.3 MB. These measurements inform a planning range, not a changed cap.

## Remaining 14 and full-analysis design

The remaining 14 processes require separate authorization. Run in the frozen 18-process assignment order and check each seal before the next process. At the observed 77–133 s per process, 14 processes imply about 18–31 minutes of host execution before review overhead; the existing 600 s hard cap gives a conservative maximum of 140 minutes. The first v3 process supplies an updated resource estimate.

For full crossover analysis, pair candidate minus Direct within each StateGroup × seed × env ID. Aggregate four env differences within seed, then describe sign, magnitude, and spread across seeds separately by StateGroup. Keep arrival/censor, local and common-goal progress, tracking, recovery, contact force, and torque separate. Report Path length, curvature, and endpoint offset alongside physical response. The older two host-valid periods and one offline-audited period retain distinct provenance in any combined analysis. Path-command response does not by itself establish DEM/Actor/Critic-based quality identifiability.
