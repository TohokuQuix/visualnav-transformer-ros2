# Moving-state 2s endpoint evaluation

Role: audit_eval. Checkpoint SHA-256: `3f64a6617702a5d50ac704ccc099cb18fc9e171aabf427e51b2865bd800d272b`.

807 executed-candidate windows from 8 independent parents. Pose supervision: complete 2s only. Missing pose: 0; invalid predicted rotations: 0.

| Model | Pose count | Mean 3D error (m) | P95 3D (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|---:|
| model | 807 | 0.0891033 | 0.234073 | 0.136661 |
| no_motion | 807 | 0.233285 | 0.467363 | 0.175563 |
| constant_body_twist | 807 | 0.26629 | 0.557717 | 0.900761 |

| Event | Positive | Negative | Unknown | Positive parents | AP | AUROC | Brier | BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tip_over | 0 | 807 | 0 | 0 | N/A | N/A | 9.67395e-06 | 0.00144898 |
| near_fall | 0 | 807 | 0 | 0 | N/A | N/A | 2.95161e-05 | 0.00248772 |
| linear_stall | 19 | 788 | 0 | 3 | 0.0403037 | 0.48417 | 0.0249061 | 0.125444 |
| turn_stall | 11 | 796 | 0 | 4 | 0.0444676 | 0.77958 | 0.0132961 | 0.0656428 |

AP/AUROC require both classes. Event probabilities are uncalibrated estimates from unweighted BCE; 0.5 is an evaluation threshold. Per-head confusion counts and denominators, 10-bin reliability, axis diagnostics, wrapped Euler diagnostics, gimbal-lock counts, state strata, and parent bootstrap 95% intervals are in metrics.json. Windows are not independent runs.

Candidate geometry is compared only in initial-yaw horizontal XY and world yaw; it makes no terrain z/roll/pitch prediction. Qualified ascent/descent/state results require explicit qualified metadata. Pose errors exclude unavailable 2s endpoints, leaving survival/censoring selection bias.

CPU batch1 K14 NN median/P95: 0.207661/0.214885 ms. Input transform + NN: 0.338789/0.352377 ms. Host measurement, not robot-PC performance.

Moving-only same-holdout model comparison

Reference checkpoint: `/home/ubuntu/eatnav-moving-state-pose2s-data-20261103/models/M_move/checkpoint.pt` (SHA-256 `75fa9659d6924b4f6bae421498a45f485fb737d8e6b86e1a9a91a245eb9457b1`). Moving means a subsequent window (`is_initial=false`), including low-speed/stalled states.

590 windows from 6 independent parents. Negative paired error differences favor the current model.

| Model | Pose count | Mean 3D error (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|
| current | 590 | 0.0889037 | 0.136364 |
| reference | 590 | 0.075452 | 0.124775 |

| Paired current minus reference | Difference | Parent bootstrap 95% CI |
|---|---:|---|
| position_3d_m | 0.0134518 | [0.00841719, 0.0218897] |
| so3_rad | 0.0115889 | [0.00592525, 0.0183461] |

Per-event moving-only Brier differences and both-model state strata are recorded in metrics.json; reference pose strata are also included in state-stratified.csv.
