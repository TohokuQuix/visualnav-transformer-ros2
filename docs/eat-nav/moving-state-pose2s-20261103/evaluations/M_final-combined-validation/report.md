# Moving-state 2s endpoint evaluation

Role: validation. Checkpoint SHA-256: `3f64a6617702a5d50ac704ccc099cb18fc9e171aabf427e51b2865bd800d272b`.

959 executed-candidate windows from 21 independent parents. Pose supervision: complete 2s only. Missing pose: 3; invalid predicted rotations: 0.

| Model | Pose count | Mean 3D error (m) | P95 3D (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|---:|
| model | 956 | 0.0907678 | 0.246837 | 0.13947 |
| no_motion | 956 | 0.238513 | 0.506542 | 0.17277 |
| constant_body_twist | 956 | 0.350259 | 0.894529 | 1.02632 |

| Event | Positive | Negative | Unknown | Positive parents | AP | AUROC | Brier | BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tip_over | 3 | 956 | 0 | 2 | 0.0109881 | 0.801953 | 0.00313014 | 0.0211644 |
| near_fall | 5 | 954 | 0 | 3 | 0.0130528 | 0.71174 | 0.00542853 | 0.0352913 |
| linear_stall | 30 | 926 | 3 | 3 | 0.0359185 | 0.575126 | 0.0338198 | 0.151581 |
| turn_stall | 4 | 952 | 3 | 3 | 0.0266202 | 0.845063 | 0.00449317 | 0.031334 |

AP/AUROC require both classes. Event probabilities are uncalibrated estimates from unweighted BCE; 0.5 is an evaluation threshold. Per-head confusion counts and denominators, 10-bin reliability, axis diagnostics, wrapped Euler diagnostics, gimbal-lock counts, state strata, and parent bootstrap 95% intervals are in metrics.json. Windows are not independent runs.

Candidate geometry is compared only in initial-yaw horizontal XY and world yaw; it makes no terrain z/roll/pitch prediction. Qualified ascent/descent/state results require explicit qualified metadata. Pose errors exclude unavailable 2s endpoints, leaving survival/censoring selection bias.

CPU batch1 K14 NN median/P95: 0.214641/0.222042 ms. Input transform + NN: 0.332303/0.342991 ms. Host measurement, not robot-PC performance.

Moving-only same-holdout model comparison

Reference checkpoint: `/home/ubuntu/eatnav-moving-state-pose2s-data-20261103/models/M_move/checkpoint.pt` (SHA-256 `75fa9659d6924b4f6bae421498a45f485fb737d8e6b86e1a9a91a245eb9457b1`). Moving means a subsequent window (`is_initial=false`), including low-speed/stalled states.

746 windows from 19 independent parents. Negative paired error differences favor the current model.

| Model | Pose count | Mean 3D error (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|
| current | 743 | 0.0871298 | 0.134531 |
| reference | 743 | 0.0822213 | 0.127021 |

| Paired current minus reference | Difference | Parent bootstrap 95% CI |
|---|---:|---|
| position_3d_m | 0.00490847 | [-0.00284079, 0.0106385] |
| so3_rad | 0.00751038 | [-0.000154957, 0.0152777] |

Per-event moving-only Brier differences and both-model state strata are recorded in metrics.json; reference pose strata are also included in state-stratified.csv.
