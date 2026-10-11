# P1 G3 Path crossover v2: pilot stop report (2026-10-12)

## Decision

The authorized first-differential / batch 0 crossover pilot is **incomplete**: period 0 and 1 are protocol-valid (8/12 trials), while period 2 stopped before simulator startup and Path publication. No remaining crossover processes, Evaluator training, or full collection were started. The 3-period paired analysis and execution-quality identifiability decision are pending.

The frozen v1 crossover attempt and all 48 Direct-repeat trials were left unchanged. Raw attempts 0/1/2 are under `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-crossover-v2-pilot-attempt-{0,1,2}`. Attempt 0 failed before Path due Docker-root-owned `output` directory permissions. Attempt 1 produced four traces, then sealed as failed due host variable shadowing (`rows` overwritten by NVML process rows). A read-only copy of attempt 1 passed the corrected post-Path validator; its failed seal was **not** promoted. Attempt 2 used new source SHA and preserved both earlier attempts.

## Frozen evidence and gates

| Item | Value |
|---|---|
| Attempt-2 assignment SHA | `859e5d4f31125ca975a1c0f5cb79bc627b0dd2e6c234c77c8a62b01106af54f8` |
| Attempt-2 manifest SHA | `1099036714055ccf424bf71ab00ce8f65dea597a8d2407f062dcaef8fe01c623` |
| Attempt-2 simulator SHA | `9c4a8838f0fa6fc1a86ca5c5244bb0b4ebd9a76378fc47253fb6148ca57b21b3` |
| Attempt-2 host SHA | `ef988448e66dda2e4cc5b6611a185caa4fce45003045957d72aad1212bcc0712` |
| Period-0 host seal SHA | `43481f2ee59c58a352ef2e41a777d6009b11b47a5ce7e6ead10468a81d612e90` |
| Period-1 host seal SHA | `50e02055999d7acd7c1d973befa9a297f17b9bbb9937d993c8e8077349dc3a80` |
| Live material JSON SHA, both periods | `84157dffcf951941a38c805cd37e2782ebe2f2af3d18a7e214efd868f223adbc` |

Material gate v2 passed in both valid periods. One live terrain collision material binding had composed values matching the declared fingerprint. All 1,084 robot collision prims were recorded as USD-unbound; the effective PhysX tensor was `[4,271,3]`, all triples `(1.0,1.0,0.0)`. The 60 per-env link slices covered all collision shapes, including uniform multi-shape links. The resolved `OnixRoughEnvCfg` startup event was friction `(1.0,1.0)`, restitution `0.0`, 128 buckets, and `make_consistent=true`; the old base material `(0.8,0.6,0.0)` was not used as an expectation.

| Period | Status | Trace steps × env | Cross-period strict fields | Native exit | GPU peak | Container anon+shmem peak | Batch storage peak | Host time | Seal / cleanup |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| 0 | protocol-valid | 165 × 4 | pass | 0 | 5,169.19 MiB | 3.245 GiB | 21.39 MB | 77.14 s | valid; residual 0 |
| 1 | protocol-valid | 167 × 4 | pass, 0 strict and 0 diagnostic differences vs period 0 | 0 | 5,169.06 MiB | 3.244 GiB | 21.66 MB | 77.87 s | valid; residual 0 |
| 2 | prelaunch invalid | 0 | not reached | — | — | — | — | 0.10 s | no simulator, Path, or seal |

Both valid periods passed four-follower readiness, exact-stamp, same-episode, resource, SHA, and outer-launcher post-seal verification. Each had 97–98 owned GPU attribution samples. The direct duplicate pair in period 0 failed the existing reproducibility gate on arrival time and common-goal progress; it remains a valid variability observation, not a protocol failure.

## Eight valid trial outcomes

All arrivals were observed (no right-censoring) and recovery count was zero. Progress values use the fixed first 2 s. Force and torque below are means over that fixed window; tracking is the recorded Path error mean.

| Period/env | Path | Arrival s | Local progress m | Common-goal progress m | Tracking m | Contact N | Torque abs |
|---|---|---:|---:|---:|---:|---:|---:|
| 0/0 | Direct | 2.40 | 0.446 | 0.446 | 0.028 | 6.98 | 186.07 |
| 0/1 | Direct | 2.00 | 0.562 | 0.562 | 0.023 | 9.32 | 248.43 |
| 0/2 | Left +20° | 6.20 | 0.302 | 0.341 | 0.085 | 8.62 | 210.72 |
| 0/3 | Right −20° | 2.65 | 0.392 | 0.404 | 0.033 | 7.94 | 179.20 |
| 1/0 | Left +20° | 6.30 | 0.323 | 0.366 | 0.087 | 8.30 | 227.32 |
| 1/1 | Right −20° | 2.60 | 0.439 | 0.470 | 0.044 | 8.90 | 215.59 |
| 1/2 | Direct | 2.00 | 0.561 | 0.561 | 0.027 | 7.88 | 157.86 |
| 1/3 | Direct | 2.45 | 0.438 | 0.438 | 0.028 | 6.81 | 160.36 |

For the four available same-seed/env comparisons, candidate minus Direct common-goal progress was −0.080 m (env 0, Left), −0.093 m (env 1, Right), −0.220 m (env 2, Left), and −0.034 m (env 3, Right). Local progress differences were −0.123, −0.124, −0.259, and −0.046 m, respectively. These are **partial, descriptive** comparisons: each env is still missing its third Path. The curved Paths have length 0.670 m and endpoints 0.260 m from the common goal, versus Direct length 0.656 m and zero endpoint offset; local arrival and tracking alone do not determine navigation quality.

## Period-2 stop and recovery boundary

The frozen balanced assignment intentionally gives period 2 `Right, Left, Right, Left`. The attempt-2 host source incorrectly required `{Direct, Left, Right}` **within every period**. That assertion raised `ValueError: crossover four-branch assignment invalid` before router audit, Docker launch, or Path publication. The external launcher recorded exit 1 and missing seal; there is no simulator output. This is a host validation defect, not a material or candidate measurement failure.

A regression test now covers the period-2 assignment and a corrected, **unexecuted** host validator compares all four rows against the frozen manifest. The corrected host source SHA is `a5b5b580e31633c23d7fdd599b5f82de02252dc67d139d614ab97739461ddf63`; the new proposed manifest SHA is `dfd7b0063b94ea1ca25217fdc8fdd459ea77faead041cdbaa2eedfcbf973929e`. All 55 relevant static/synthetic tests pass. No attempt was run with this source.

Completing period 2 while retaining the protocol-valid periods 0/1 requires an explicit cross-attempt reference to their sealed pre-Path snapshots and an explicit decision about the different host-validation SHA. Re-running periods 0/1 would violate the instruction not to repeat protocol-valid processes. Therefore this pilot stops at 2/3 valid processes; the remaining 15 crossover processes are **not ready** to start.
