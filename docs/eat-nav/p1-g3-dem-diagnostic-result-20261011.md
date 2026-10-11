# P1 G3 candidate-free DEM diagnostic: one-run result

## Decision

The authorized **first-differential / noise block 0, four-env, 20-step** diagnostic was executed once. The unchanged G3 `direct_dem_gate` **failed** at the first captured frame. No Path was published and no candidate trial ran. This is a DEM-gate diagnosis, not an execution-quality result. The existing G3 attempt-0 and G1/G2 artifacts and seals were verified before and after; they were not changed.

The persistent run is `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-dem-diagnostic-attempt-0/`. Its diagnostic manifest SHA-256 is `d11debaad30f5eb8e708b84accd86246499614ccbe9cf39409ba5812142474cf`. Simulator source SHA-256 is `90f7545e344f4921a22b6250bc2bfe167ba3890b15889109586dbbdacee9f201`. The dedicated manifest restricts execution to one first-differential/block-0 run, steps 4/9/14/19, four envs, and zero Path publications. Frozen input file SHAs are in the manifest.

## First observed divergence and stage-by-stage measurements

The first captured frame is **settling step 4 (zero-based)**. Earlier steps were not captured, so the first physical divergence time remains unknown. Direct duplicates are env 0 and env 1; their tile origins differ by 8 m. World-coordinate ray differences therefore include the tile offset; the table uses tile-local comparisons where appropriate.

| Frame / actual sensor update | Pre-clone local root XY difference | Yaw difference (env 1 − env 0) | Raw ray-hit height max | Raw validity difference | Transformed DEM RMS / max | Cells above frozen 0.1 mm max threshold |
|---|---:|---:|---:|---:|---:|---:|
| step 4 / 0.25 s | (3.0005, 1.3423) mm | 0.0009486 rad | 0.089 µm | 0 | 0.0891 / 0.6261 mm | 32 |
| step 9 / 0.50 s | (2.9968, 1.3409) mm | 0.0009419 rad | 0.060 µm | 0 | 0.0884 / 0.6217 mm | 32 |
| step 14 / 0.75 s | (3.0014, 1.3454) mm | 0.0009586 rad | 0.119 µm | 0 | 0.0900 / 0.6326 mm | 32 |
| step 19 / 1.00 s | (3.0077, 1.3475) mm | 0.0009709 rad | 0.089 µm | 0 | 0.0912 / 0.6408 mm | 33 |

At step 4, the first raw height difference by exact float comparison is ray-grid cell `(row 0, column 4)` and is only about 0.030 µm. The first transformed DEM cell exceeding the unchanged `1e-4 m` maximum threshold is `(row 0, column 10)`, with env-0 minus env-1 height `+0.3943 mm`; the largest difference is at `(15, 18)`. The height RMS also exceeds the unchanged `1e-5 m` threshold. Both source and transformed validity masks agree exactly (IoU 1.0). All 400 ray hits are finite in both direct envs for each frame.

World ray origins/hits differ by nearly the expected 8 m tile translation. After subtracting each env origin, ray-origin and ray-hit XYZ differences are at most about 3.00 mm. These track the pre-clone body-position difference; raw terrain-hit **height** differences remain below 0.12 µm. Joint positions and velocities also differ during capture (at step 4, maximum 0.0168 rad and 0.543 rad/s, respectively). The subsequent `clone_internal_branch_state` reports zero maximum difference in root-local state, joint position/velocity, and command, but it runs **after** all four DEM frames were captured and does not clone raycaster history.

The sensor's actual last-update stamps are equal between direct envs for every pair: about 0.25, 0.50, 0.75, and 1.00 s. Its outdated flag was false at each capture. The source's synthetic `timestamp_ns` values are 0.20, 0.45, 0.70, and 0.95 s because it uses the zero-based settling index; this 50 ms labeling offset is common to both envs and does not explain their numeric difference. It remains a metadata issue for any later protocol revision.

An offline counterfactual recomputed env 1's `robot_dem` from its **unchanged raw heights, validity, ray starts, and root XY**, replacing only its yaw with env 0's yaw. This reduced the env-0/env-1 DEM max difference to `0.000052–0.000283 mm` across four frames, with zero cells above `0.1 mm`. Recorded DEMs were also reproduced from their stored raw inputs within `1.2e-8 m`. This isolates the observed yaw mismatch as sufficient to explain the numeric-gate failure for these saved frames. It does not establish why the two environments' physics states began to diverge during settling, or whether all future scenes behave similarly.

## Evidence integrity and runtime

- Sixteen raw frame NPZs, four difference maps, metadata, comparison JSON, and a SHA ledger were flushed and fsynced **before** the frozen gate rejected the run. All 22 ledger entries match host and Docker copies. Four-frame data include raw world ray hits and starts, source height and validity, transformed DEM, env origin, root pose, joint positions/velocities, and actual sensor update stamps. The full numeric measurements are in `p1-g3-dem-diagnostic-analysis-20261011.json`.
- The simulator wrote `diagnostic-error.json` with `ValueError: direct DEM numeric gate` and `Path_publications: 0`. Native exit was 0, but the corrected host reported failure explicitly and exited 1. The simulator's native exit-0 behavior remains unexplained; it no longer masks protocol failure at the host boundary.
- GPU attribution passed 21/21 samples, including 17 owned-GPU samples. Peak device memory was 5,192.7 MiB, peak attribution gap 104.0 MiB, peak container anon+shmem 3.43 GB, and total persistent storage about 1.93 MB. Elapsed host time was 13.73 s. These were within the frozen resource caps.
- Cleanup found zero owned residual processes and no bridge process was launched. The external router was not used or changed, since this diagnostic ends before follower/Path interaction. All writers closed before sealing. The host seal SHA-256 is `1e8687f6e2ea746b6cc3c973c76fceeb645e8447d9e1a32e5884c917d4d390ec`; read-only verification passed for 248 files. The lifecycle phase is `RUNNING` in host status because the DEM gate rejected before protocol completion; this is not a successful G3 batch.

## Interpretation and limit

The capture-before-clone hypothesis is **supported for this run**: the directly measured pre-clone yaw difference is large enough to produce the transformed DEM discrepancy, while raw ray heights and masks nearly coincide and sensor update times coincide. The offline yaw substitution removes the threshold violation without changing terrain rays. The experiment does not determine the first step or physics mechanism that created the yaw difference. No G3 candidate pilot or protocol change was made; the G3 experiment design should be reviewed against this evidence before any further candidate trial.
