# Moving-state 2s endpoint evaluation

Role: audit_eval_geometry_hash_not_in_combined_train_subset. Checkpoint SHA-256: `3f64a6617702a5d50ac704ccc099cb18fc9e171aabf427e51b2865bd800d272b`.

140 executed-candidate windows from 11 independent parents. Pose supervision: complete 2s only. Missing pose: 0; invalid predicted rotations: 0.

| Model | Pose count | Mean 3D error (m) | P95 3D (m) | Mean SO(3) (rad) |
|---|---:|---:|---:|---:|
| model | 140 | 0.100878 | 0.297535 | 0.121826 |
| no_motion | 140 | 0.268834 | 0.560438 | 0.16576 |
| constant_body_twist | 140 | 0.515568 | 1.16307 | 1.36949 |

| Event | Positive | Negative | Unknown | Positive parents | AP | AUROC | Brier | BCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tip_over | 0 | 140 | 0 | 0 | N/A | N/A | 7.11767e-08 | 0.000190981 |
| near_fall | 0 | 140 | 0 | 0 | N/A | N/A | 2.88902e-07 | 0.000418974 |
| linear_stall | 3 | 137 | 0 | 1 | 0.0225838 | 0.396594 | 0.0214906 | 0.111317 |
| turn_stall | 1 | 139 | 0 | 1 | 0.0217391 | 0.676259 | 0.00734766 | 0.0445632 |

AP/AUROC require both classes. Event probabilities are uncalibrated estimates from unweighted BCE; 0.5 is an evaluation threshold. Per-head confusion counts and denominators, 10-bin reliability, axis diagnostics, wrapped Euler diagnostics, gimbal-lock counts, state strata, and parent bootstrap 95% intervals are in metrics.json. Windows are not independent runs.

Candidate geometry is compared only in initial-yaw horizontal XY and world yaw; it makes no terrain z/roll/pitch prediction. Qualified ascent/descent/state results require explicit qualified metadata. Pose errors exclude unavailable 2s endpoints, leaving survival/censoring selection bias.

CPU batch1 K14 NN median/P95: 0.205729/0.25078 ms. Input transform + NN: 0.340699/0.354904 ms. Host measurement, not robot-PC performance.

Derived from saved predictions with no NN rerun or audit tuning. Selection uses combined-train geometry hashes. Same known terrain classes remain; this is not unseen-class generalization. Unknown evaluation geometry hashes are excluded. See subset-selection.json for counts/bindings. Baseline event prevalence retains the source evaluation dataset train split. Benchmark values are copied from the source evaluation, not newly timed on this subset.
