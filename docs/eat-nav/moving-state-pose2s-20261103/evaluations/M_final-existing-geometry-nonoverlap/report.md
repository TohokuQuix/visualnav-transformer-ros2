# Moving-state 2s endpoint evaluation

Role: audit_eval_geometry_hash_not_in_combined_train_subset. Checkpoint SHA-256: `3f64a6617702a5d50ac704ccc099cb18fc9e171aabf427e51b2865bd800d272b`.

641 executed-candidate windows from 6 independent parents. Pose supervision: complete 2s only. Missing pose: 0; invalid predicted rotations: 0.

| Model | Pose count | Mean 3D error (m) | P95 3D (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|---:|
| model | 641 | 0.0810583 | 0.217029 | 0.119122 |
| no_motion | 641 | 0.237346 | 0.468491 | 0.161024 |
| constant_body_twist | 641 | 0.262565 | 0.55667 | 0.805954 |

| Event | Positive | Negative | Unknown | Positive parents | AP | AUROC | Brier | BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tip_over | 0 | 641 | 0 | 0 | N/A | N/A | 1.02058e-05 | 0.00139786 |
| near_fall | 0 | 641 | 0 | 0 | N/A | N/A | 3.37665e-05 | 0.00251782 |
| linear_stall | 11 | 630 | 0 | 2 | 0.0129406 | 0.32886 | 0.0198489 | 0.110108 |
| turn_stall | 11 | 630 | 0 | 4 | 0.0541837 | 0.777778 | 0.0165855 | 0.077966 |

AP/AUROC require both classes. Event probabilities are uncalibrated estimates from unweighted BCE; 0.5 is an evaluation threshold. Per-head confusion counts and denominators, 10-bin reliability, axis diagnostics, wrapped Euler diagnostics, gimbal-lock counts, state strata, and parent bootstrap 95% intervals are in metrics.json. Windows are not independent runs.

Candidate geometry is compared only in initial-yaw horizontal XY and world yaw; it makes no terrain z/roll/pitch prediction. Qualified ascent/descent/state results require explicit qualified metadata. Pose errors exclude unavailable 2s endpoints, leaving survival/censoring selection bias.

CPU batch1 K14 NN median/P95: 0.207661/0.214885 ms. Input transform + NN: 0.338789/0.352377 ms. Host measurement, not robot-PC performance.

Derived from saved predictions with no NN rerun or audit tuning. Selection uses combined-train geometry hashes. Same known terrain classes remain; this is not unseen-class generalization. Unknown evaluation geometry hashes are excluded. See subset-selection.json for counts/bindings. Baseline event prevalence retains the source evaluation dataset train split. Benchmark values are copied from the source evaluation, not newly timed on this subset.
