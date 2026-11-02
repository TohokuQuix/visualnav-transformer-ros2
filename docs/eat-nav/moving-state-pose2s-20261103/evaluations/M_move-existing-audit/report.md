# Moving-state 2s endpoint evaluation

Role: audit_eval. Checkpoint SHA-256: `75fa9659d6924b4f6bae421498a45f485fb737d8e6b86e1a9a91a245eb9457b1`.

807 executed-candidate windows from 8 independent parents. Pose supervision: complete 2s only. Missing pose: 0; invalid predicted rotations: 0.

| Model | Pose count | Mean 3D error (m) | P95 3D (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|---:|
| model | 807 | 0.0761422 | 0.204501 | 0.122921 |
| no_motion | 807 | 0.233285 | 0.467363 | 0.175563 |
| constant_body_twist | 807 | 0.26629 | 0.557717 | 0.900761 |

| Event | Positive | Negative | Unknown | Positive parents | AP | AUROC | Brier | BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tip_over | 0 | 807 | 0 | 0 | N/A | N/A | 2.21144e-06 | 0.000402795 |
| near_fall | 0 | 807 | 0 | 0 | N/A | N/A | 3.16403e-06 | 0.00060096 |
| linear_stall | 19 | 788 | 0 | 3 | 0.192828 | 0.888124 | 0.0218783 | 0.0886736 |
| turn_stall | 11 | 796 | 0 | 4 | 0.0621886 | 0.833029 | 0.013123 | 0.0616386 |

AP/AUROC require both classes. Event probabilities are uncalibrated estimates from unweighted BCE; 0.5 is an evaluation threshold. Per-head confusion counts and denominators, 10-bin reliability, axis diagnostics, wrapped Euler diagnostics, gimbal-lock counts, state strata, and parent bootstrap 95% intervals are in metrics.json. Windows are not independent runs.

Candidate geometry is compared only in initial-yaw horizontal XY and world yaw; it makes no terrain z/roll/pitch prediction. Qualified ascent/descent/state results require explicit qualified metadata. Pose errors exclude unavailable 2s endpoints, leaving survival/censoring selection bias.

CPU batch1 K14 NN median/P95: 0.209658/0.221338 ms. Input transform + NN: 0.346017/0.404011 ms. Host measurement, not robot-PC performance.

Moving-only same-holdout model comparison

Reference checkpoint: `/home/ubuntu/eatnav-moving-state-pose2s-data-20261103/models/M_stop/checkpoint.pt` (SHA-256 `a34e1e4face53181af761d585a5427a1258636303d3fb1741fe92113addc54b7`). Moving means a subsequent window (`is_initial=false`), including low-speed/stalled states.

590 windows from 6 independent parents. Negative paired error differences favor the current model.

| Model | Pose count | Mean 3D error (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|
| current | 590 | 0.075452 | 0.124775 |
| reference | 590 | 0.0901733 | 0.142057 |

| Paired current minus reference | Difference | Parent bootstrap 95% CI |
|---|---:|---|
| position_3d_m | -0.0147213 | [-0.0209673, -0.00836523] |
| so3_rad | -0.0172813 | [-0.0219445, -0.013129] |

Per-event moving-only Brier differences and both-model state strata are recorded in metrics.json; reference pose strata are also included in state-stratified.csv.
