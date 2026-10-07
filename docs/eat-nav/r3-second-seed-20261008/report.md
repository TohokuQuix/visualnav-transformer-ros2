# R3 second seed: zero-step scan, robustness, candidate-free settling

## Decision

`r3_second_seed_physical_supply_pass_ready_for_goal_candidate_bank_design`. The independent second batch seed supplied at least one protocol-valid, physically eligible anchor in each of the four virtual DEM classes. This is a gate to **design** goal bearings and a candidate bank; it is not evidence of terrain-wide generalization. No additional seed, reserve, Path, follower, bridge, router, Outcome, model training, or robot control was started.

## Frozen provenance and order

The seed was frozen before reading second-seed terrain geometry: SHA-256 first-four-byte big-endian uint32 of `eatnav-r3-second-seed-v1` gave batch seed **3773891777**, without collision with batch seed 20261007 or its seven terrain seeds. The [seed freeze](seed-freeze.json) SHA-256 is `aef592cc262a4774a6d124a42dffa7c647bc36ae549f15b3cc14dd538883bf72`. The fixed [robustness definition](robustness-freeze.json) SHA-256 is `19d7b0f3884dab9c390a5bb0070d11e457b39dc2d64aadc979d5cd97ddc06899`: robot-local translations `(0,0), (±0.02,0), (0,±0.02)` m combined with yaw offsets `0, ±2°`, 15 conditions including nominal. The perturbations resample the same zero-step source grid; unknown remains unknown. Robustness had no role in R3 rank, pilot selection, or hard eligibility.

The 448-pose scan used seven terrain types, 16 directions at 22.5° spacing, radii 0.30/0.40/0.50/0.60 m, difficulty 0.5, outward yaw, the same 20×20 0.1 m robot-relative DEM, collision proxy, classifier and R3 rule. The scanner recorded **448/448 complete finite DEMs**, **6,720/6,720 valid perturbation DEMs**, and physics step counter **0→0**. A separate [geometry contract audit](geometry-contract-audit.json) checked source/target shape, frame, axis, resolution, origins, yaw and finite values for all 448 poses. The source/classifier/R3 hashes matched the first seed. The exact scan CLI, source SHA and Docker identity are in [scan-preflight.json](scan-preflight.json); the raw radius scans remain in `/tmp/eatnav-r3-second-seed-20261008/scan-v1/`.

The scan executed from commit `ce2b196`, and the frozen pilot launcher from commit `e4f030c`, on worktree `/tmp/eatnav-vnt-r3-second-seed` and branch `exp/eatnav-r3-second-seed`.

| Virtual class | First seed, 448 poses | Second seed, 448 poses | Second-seed R3, 112 poses |
| --- | ---: | ---: | ---: |
| transition | 4 | 5 | 5 |
| differential | 140 | 150 | 36 |
| graded | 89 | 80 | 36 |
| flat-only | 215 | 213 | 35 |

R3 still selected all five available second-seed transition poses; the other classes have broader geometric support. The 8-pose [pilot manifest](pilot-manifest.json), SHA-256 `d493060b2cf30622710e5986a29f40c0252319adfb320224fbb14a92db574510`, was frozen before physics. Shard A used the first distinct source family per class and shard B the second. Both shards had four distinct environment indices. The 104 unexecuted poses are in [reserve-rank-freeze.json](reserve-rank-freeze.json).

## Candidate-free physical result

Both 112-environment shards passed their seal, native exit, contact/DEM/Actor completeness, termination-aware record, resource and cleanup gates. Each selected pose had a measurement-valid 200-step result, four DEM frames and Actor prephysics/step-200 observations. Neither shard had a physical termination. The original reset, root-z rule `env_origin.z + 0.35 m`, XY/yaw-only placement, zero root velocity, zero 6D action, 0.005 s physics, final40 window and capacity 4096 were retained. No remediation cycle or retry was used.

| Pilot slot | Class / terrain | Eligibility | Primary settled class | 15-condition retention | Physical supply |
| ---: | --- | --- | --- | ---: | --- |
| 1 | transition / boxes | eligible | transition | 1.000 | yes |
| 2 | differential / k_rail_step_field | eligible | differential | 1.000 | yes |
| 3 | graded / k_rail_step_field | eligible | graded | 0.133 | yes |
| 4 | flat-only / k_rail_field | numeric ineligible: speed | NA | 0.800 | no |
| 5 | transition / k_rail_step_field | numeric ineligible: angular speed | NA | 0.667 | no |
| 6 | differential / boxes | eligible | differential | 1.000 | yes |
| 7 | graded / k_rail_field | eligible | graded | 0.933 | no: settled onset unknown |
| 8 | flat-only / pyramid_stairs_inv | eligible | flat-only | 0.800 | yes |

Class supply counts were **transition 1/2, differential 2/2, graded 1/2, flat control 1/2**. For the flat control, the [pre-physics supply contract](supply-contract.json) requires known DEM coverage and a known absence of interaction; a positive onset would contradict `flat-only`. For the three interaction classes it requires a known onset within 0–0.6 m. Slot 7 was numerically eligible and class-matched, but its primary settled DEM had no known onset, so it was not counted. Slot 2's onset changed from virtual 0.15 m to settled 0.05 m; exact onset agreement was diagnostic, not a hard supply gate. The secondary DEM class of an ineligible pose was not promoted to primary eligibility.

## Robustness interpretation and limits

Among the **six eligible** poses, all six primary settled classes matched virtual class; there was no eligible class-mismatch group to compare. The two numeric-ineligible poses are recorded as class-match `NA`, not mismatches. Mean 15-condition retention was 0.811 for eligible matches and 0.733 for the ineligible group. It was 0.787 for five supplied poses and 0.800 for three non-supplied poses. In particular, slot 3 supplied graded despite retention 0.133, while slot 7 retained graded at 0.933 but lacked settled onset. This small sample does not support a robustness threshold or a predictive claim. The [pilot manifest](pilot-manifest.json) retains all 15 condition records per pose; [per-pose results](per-pose-results.json) and [robustness comparison](robustness-vs-settled.json) retain onset ranges and threshold margins.

The independent unit is terrain seed/source family, not the 8 poses or their future candidate Paths. The 8 pilot poses cover only four second-seed source families; the five supplied poses cover three. Passing the second-seed gate does not establish generalization to seven terrain types or to new goals.

## Resource and integrity

The scan peaked at **5,537 MiB GPU** and approximately **25.1 MB monitored storage**. Settling peaked at **5,861/5,867 MiB GPU** for shards A/B; peak monitored storage was **57.0/71.6 MB**. Combined shard wall time was **89.6 s**; each process was far below the 900 s timeout. GPU, container anon+shmem, host available memory and storage stayed inside the frozen caps. Both simulator processes exited 0 and left zero owned residuals. [Integrity checks](integrity-check.json) verify scan files, both shard ledgers, first-seed and transition-reserve artifacts, markers and resource caps. The older anchor-placement ledger has one documented dynamic-log mismatch: `analysis-v9.result.log` was empty at ledger creation and later received stdout; no immutable scan or analysis artifact differs. Exact per-shard PID/start ticks and resource series are under the raw root.

## Next design review

The [goal/candidate-bank proposal](goal-candidate-bank-proposal.json) uses bearings `−20°, −10°, 0°, +10°, +20°` at 0.6 m and the seven proposed families: direct, three left amplitudes and three right amplitudes. Every candidate from one anchor/bearing is judged against the same global goal. All bearings and candidates from one anchor stay in one split; terrain seed/source family remains the independence unit. One balanced design option is one supplied anchor per class per seed (8 anchors, 280 *definitions*), but that set and its runtime budget require a separate review and freeze. No Path was generated or executed in this task.

The ten requested figures are under [figures](figures). Source code and small JSON/figures are committed on `exp/eatnav-r3-second-seed`; raw DEM, trace and run roots remain outside Git at `/tmp/eatnav-r3-second-seed-20261008/`.
