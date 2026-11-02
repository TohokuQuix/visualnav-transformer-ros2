# Moving-state 2s endpoint evaluation

Role: audit_eval. Checkpoint SHA-256: `3f64a6617702a5d50ac704ccc099cb18fc9e171aabf427e51b2865bd800d272b`.

168 executed-candidate windows from 13 independent parents. Pose supervision: complete 2s only. Missing pose: 0; invalid predicted rotations: 0.

| Model | Pose count | Mean 3D error (m) | P95 3D (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|---:|
| model | 168 | 0.0975842 | 0.278805 | 0.132318 |
| no_motion | 168 | 0.275626 | 0.555986 | 0.183755 |
| constant_body_twist | 168 | 0.482554 | 1.11835 | 1.31765 |

| Event | Positive | Negative | Unknown | Positive parents | AP | AUROC | Brier | BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tip_over | 0 | 168 | 0 | 0 | N/A | N/A | 1.55913e-07 | 0.000267858 |
| near_fall | 0 | 168 | 0 | 0 | N/A | N/A | 5.8905e-07 | 0.000563022 |
| linear_stall | 3 | 165 | 0 | 1 | 0.0169775 | 0.329293 | 0.0182195 | 0.0998112 |
| turn_stall | 1 | 167 | 0 | 1 | 0.0166667 | 0.646707 | 0.00621146 | 0.0398129 |

AP/AUROC require both classes. Event probabilities are uncalibrated estimates from unweighted BCE; 0.5 is an evaluation threshold. Per-head confusion counts and denominators, 10-bin reliability, axis diagnostics, wrapped Euler diagnostics, gimbal-lock counts, state strata, and parent bootstrap 95% intervals are in metrics.json. Windows are not independent runs.

Candidate geometry is compared only in initial-yaw horizontal XY and world yaw; it makes no terrain z/roll/pitch prediction. Qualified ascent/descent/state results require explicit qualified metadata. Pose errors exclude unavailable 2s endpoints, leaving survival/censoring selection bias.

CPU batch1 K14 NN median/P95: 0.205729/0.25078 ms. Input transform + NN: 0.340699/0.354904 ms. Host measurement, not robot-PC performance.

Moving-only same-holdout model comparison

Reference checkpoint: `/home/ubuntu/eatnav-moving-state-pose2s-data-20261103/models/M_move/checkpoint.pt` (SHA-256 `75fa9659d6924b4f6bae421498a45f485fb737d8e6b86e1a9a91a245eb9457b1`). Moving means a subsequent window (`is_initial=false`), including low-speed/stalled states.

168 windows from 13 independent parents. Negative paired error differences favor the current model.

| Model | Pose count | Mean 3D error (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|
| current | 168 | 0.0975842 | 0.132318 |
| reference | 168 | 0.0928727 | 0.136461 |

| Paired current minus reference | Difference | Parent bootstrap 95% CI |
|---|---:|---|
| position_3d_m | 0.00471152 | [-0.0108512, 0.0218149] |
| so3_rad | -0.00414334 | [-0.011152, 0.00511146] |

Per-event moving-only Brier differences and both-model state strata are recorded in metrics.json; reference pose strata are also included in state-stratified.csv.

Moving-only same-holdout model comparison

Reference checkpoint: `/home/ubuntu/eatnav-moving-state-pose2s-data-20261103/models/M_stop/checkpoint.pt` (SHA-256 `a34e1e4face53181af761d585a5427a1258636303d3fb1741fe92113addc54b7`). Moving means a subsequent window (`is_initial=false`), including low-speed/stalled states.

168 windows from 13 independent parents. Negative paired error differences favor the current model.

| Model | Pose count | Mean 3D error (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|
| current | 168 | 0.0975842 | 0.132318 |
| reference | 168 | 0.116429 | 0.187211 |

| Paired current minus reference | Difference | Parent bootstrap 95% CI |
|---|---:|---|
| position_3d_m | -0.0188444 | [-0.0339494, -0.00556114] |
| so3_rad | -0.0548929 | [-0.106564, -0.0225006] |

Per-event moving-only Brier differences and both-model state strata are recorded in metrics.json; reference pose strata are also included in state-stratified.csv.
