# Moving-state 2s endpoint evaluation

Role: audit_eval_geometry_hash_nonoverlap_subset. Checkpoint SHA-256: `75fa9659d6924b4f6bae421498a45f485fb737d8e6b86e1a9a91a245eb9457b1`.

641 executed-candidate windows from 6 independent parents. Pose supervision: complete 2s only. Missing pose: 0; invalid predicted rotations: 0.

| Model | Pose count | Mean 3D error (m) | P95 3D (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|---:|
| model | 641 | 0.06615 | 0.175673 | 0.105723 |
| no_motion | 641 | 0.237346 | 0.468491 | 0.161024 |
| constant_body_twist | 641 | 0.262565 | 0.55667 | 0.805954 |

| Event | Positive | Negative | Unknown | Positive parents | AP | AUROC | Brier | BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tip_over | 0 | 641 | 0 | 0 | N/A | N/A | 2.5145e-06 | 0.000401414 |
| near_fall | 0 | 641 | 0 | 0 | N/A | N/A | 3.31984e-06 | 0.000545241 |
| linear_stall | 11 | 630 | 0 | 2 | 0.138828 | 0.92482 | 0.0160643 | 0.0666898 |
| turn_stall | 11 | 630 | 0 | 4 | 0.0731032 | 0.820635 | 0.0164191 | 0.0746092 |

AP/AUROC require both classes. Event probabilities are uncalibrated estimates from unweighted BCE; 0.5 is an evaluation threshold. Per-head confusion counts and denominators, 10-bin reliability, axis diagnostics, wrapped Euler diagnostics, gimbal-lock counts, state strata, and parent bootstrap 95% intervals are in metrics.json. Windows are not independent runs.

Candidate geometry is compared only in initial-yaw horizontal XY and world yaw; it makes no terrain z/roll/pitch prediction. Qualified ascent/descent/state results require explicit qualified metadata. Pose errors exclude unavailable 2s endpoints, leaving survival/censoring selection bias.

CPU batch1 K14 NN median/P95: 0.209658/0.221338 ms. Input transform + NN: 0.346017/0.404011 ms. Host measurement, not robot-PC performance.

Moving-only same-holdout model comparison

Reference checkpoint: `docs/eat-nav/moving-state-pose2s-20261103/models/M_stop/checkpoint.pt` (SHA-256 `a34e1e4face53181af761d585a5427a1258636303d3fb1741fe92113addc54b7`). Moving means a subsequent window (`is_initial=false`), including low-speed/stalled states.

484 windows from 5 independent parents. Negative paired error differences favor the current model.

| Model | Pose count | Mean 3D error (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|
| current | 484 | 0.0667217 | 0.10968 |
| reference | 484 | 0.0835969 | 0.128378 |

| Paired current minus reference | Difference | Parent bootstrap 95% CI |
|---|---:|---|
| position_3d_m | -0.0168752 | [-0.0218677, -0.0108945] |
| so3_rad | -0.0186979 | [-0.0225544, -0.0139599] |

Per-event moving-only Brier differences and both-model state strata are recorded in metrics.json; reference pose strata are also included in state-stratified.csv.
