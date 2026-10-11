# P1 long-horizon G3 attempt-0: pre-Path gate stop

Date: 2026-10-11 (JST). Code commit: `c920d1c`. This report describes the one authorized G3 attempt. No retry was made.

## Decision

G3 did **not** reach a protocol-valid candidate trial. Batch 0 (`first-differential`, noise block 0, seed `2026101101`) stopped before Path publication because the direct-duplicate four-frame DEM numeric gate raised `ValueError: direct DEM numeric gate`. Batches 1–3 were not started. The execution-quality identifiability gate is **not assessable**; this is a pre-Path measurement gate failure, not evidence for or against candidate Path quality. Do not advance to Evaluator training or full collection.

## Frozen scope and what ran

- Manifest: 2 StateGroups × 2 independent noise blocks × 4 slots = 16 planned trials. Per batch slots were direct duplicate 0/1, −20°, +20°. Frozen source SHA `ea5b137f49cac8c13c510bca9d9fd3bfb9a299b575c3cb1458c4eda9f0cc2ab8`; assignment SHA `f5ce459bf26ffe0a2316e37f046f2105d0637060ac17758bacf6da46fe49608f`.
- Four-env `first-differential` batch 0 completed the 20 candidate-free settling steps. Initial-state and terrain identity checks pass when applied offline with the exact copied G2 runtime module: all root/joint spreads are zero. Canonical terrain mesh and composed collision fingerprints match G2. The six frozen G2 input files copied into G3 are byte-identical.
- The new G3 DEM gate sampled four settling frames per env at steps 4, 9, 14, 19. It requires identical frame identity, timestamp, shape, robot-relative frame, and resolution, and numeric direct-duplicate limits of height RMS ≤`1e-5` m, max absolute height ≤`1e-4` m, mask IoU ≥`1.0`. One or more numeric limits failed. The code raised before saving the frame arrays or individual metric values; therefore **the failing frame, metric, and magnitude cannot be recovered from this attempt**. The sealed evidence must not be modified or represented as a quantitative DEM comparison.
- `output/status.json` and `output/diagnostic-error.json` record `Path_publications: 0`. No G3 follower command, candidate physics trace, local arrival, tracking, recovery, body-load Outcome, or geometry-only comparison was produced. Completed candidate trials: **0/16**.

## Runtime and preservation

The simulator wrote the DEM gate exception; the host recorded native exit 0 despite that diagnostic failure. The host GPU monitor sampled once during process disappearance and reported `owned_start_tick_mismatch,simulator_disappeared_in_bracket`. This is secondary to the simulator's earlier DEM exception; the resource series contains 21 passing attribution samples and one disappearance sample. No resource cap exceedance was observed. The secondary monitor failure prevents a protocol-valid host result, as intended by fail-closed monitoring.

| Evidence | Value |
| --- | ---: |
| GPU peak device total | 5,178.06 MiB / 7,680 MiB cap |
| Largest measured attribution gap | 108.85 MiB / 256 MiB cap |
| Host available memory minimum | 83.11 GiB |
| Container anon + shmem peak | 3.20 GiB / 6 GiB cap |
| Persistent G3 root size after stop | 4.27 MB / 1,000 MB cap |
| Batch host elapsed | 24.06 s / 600 s cap |
| Owned simulator and bridge residual | 0 / 0 |
| Router PID/start tick | 6026 / 306994, unchanged before/after |
| Batch-0 host seal | `7bdf0a995fe70d99a8511b4dd6b45896e09256c31defef4070cf445a69cd597f` |
| Batch-0 seal verification | valid, 229 files |

The G1 and G2 host-seal SHA values were rechecked after this attempt and remain `c5712eb6bc816f49222f01218bb593f116a38a00f8ce07eb79bd5b637e420609` and `73ec7b6cac44dfdb717a8f8a8dd2a4cce4f19f9dc8d7b565df6055a0f251f1b0`. No earlier evidence, production follower, router, driver, checkpoint, threshold, Path geometry, or initial condition was changed.

## Follow-up needed before another pilot authorization

The first issue is to determine whether the direct-clone DEM difference comes from raycaster history, terrain sampling at distinct world tiles, numerical physics variation, or the new DEM transform/capture implementation. The current attempt cannot distinguish these because its frame arrays were not persisted before raising. A future diagnostic should save the four frames and numeric comparison **before** applying the unchanged gate, in a new root. It should also report simulator diagnostic failure ahead of the expected process-exit GPU disappearance while preserving fail-closed attribution. These are proposals only; no rerun or threshold change is authorized by this report.

## Evidence paths

- Stage manifest/status: `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-attempt-0/g3-assignment.json`, `stage-status.json`.
- Sealed batch-0 evidence: `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-attempt-0/batch-0/` (`output/diagnostic-error.json`, `output/heartbeat.json`, `output/initial-state.json`, `output/terrain.json`, `resource-series.jsonl`, `g3-host-status.json`, `cleanup-evidence.json`, `host-seal.json`).
