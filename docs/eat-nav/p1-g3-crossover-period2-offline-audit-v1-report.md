# P1 G3 period-2 sealed-evidence offline audit v1

## Decision and preservation

**`audit_pass`**: the failed period-2 host seal remains failed and unchanged. Periods 0/1 remain 2 host-protocol-valid processes and 8 trials. Period 2's 4 trials are **offline-audited conditional analysis candidates**, not host-protocol-valid trials. The new 12-trial dataset is a separate analysis version. No simulation or retry was run.

The prespecified criteria were frozen as `criteria.json` before opening the Outcome traces (SHA-256 `d3f90e0b6148632d3cab3bd938505c2f741dd3eff5d216512d40abc48c024d30`). The audit read the three original sealed batches, copied period 2 to a disposable `/tmp` shadow for the remaining host-validator replay, and wrote results only to the new persistent audit root. The original period-2 failure remains `ValueError:crossover Path publication contract failed` and its seal stage remains `G3_crossover_batch_failed_or_inconclusive`.

## Validator and gate evidence

The v3 host source changes only the assignment check relative to the executed v2 host: prelaunch and post-Path both call `validate_frozen_assignment()`. It compares each frozen slot's role and Path SHA; it has no per-period distinct-Path-count requirement. Static/synthetic tests accept periods 0/1/2, including period 2's `Right, Left, Right, Left`, and reject wrong roles, wrong SHA, or duplicate/swapped env IDs. The sealed host source was not modified.

All 12 frozen audit categories passed: both prior seals; period-2 failed-seal integrity; manifest/source/input SHA; four publications and assignment; cross-period strict pre-Path equality; live terrain/robot material; four-follower readiness; exact-stamp and clock→TF→command→bridge→physics order; same episode without reset/termination; arrival/censor/hold and complete traces; resource/native-exit/cleanup; and host/container SHA. The post-Path validator reached its end on the disposable copy after removing only the faulty three-kind requirement. Four period-2 traces have 227 contiguous steps each, at 50 ms logical intervals. There were no direct duplicate pairs in the period-2 assignment, so that diagnostic is not applicable to this period.

Period-2 resource evidence: 142 resource samples; GPU device-total peak **5,223.94 MiB**; container anonymous plus shared memory peak **3,533,545,472 bytes**; persistent batch peak **30,290,747 bytes**; host elapsed **133.26 s**; native exit **0**; owned simulator and bridge residual **0**. The GPU attribution and frozen caps passed. Original seals were reverified after the audit.

## Descriptive 12-trial crossover result

Each of four env IDs has Direct, Left +20°, and Right −20° once. All 12 trials reached the local Path endpoint; none was censored and recovery count was zero. The table gives the mean **candidate minus Direct paired difference** over the four env IDs within one StateGroup and one seed. The complete per-trial rows and paired differences are in `integrated-dataset-v1.json`.

| Candidate | Arrival time (s) | Local progress at 2 s (m) | Common-goal progress at 2 s (m) | Tracking error (m) | Mean contact force (N) | Mean absolute joint torque | Recovery count |
|---|---:|---:|---:|---:|---:|---:|---:|
| Left +20° | +5.100 | −0.1961 | −0.1516 | +0.0737 | +1.0721 | −5.9135 | 0 |
| Right −20° | +0.425 | −0.0854 | −0.0506 | +0.0165 | +0.3577 | +6.7984 | 0 |

Curved Path length is about **0.670 m** versus **0.656 m** for Direct; its endpoint is about **0.260 m** from the Direct common goal. Curvature differs as well. The differences above demonstrate Path-command-dependent execution response in this one seed, with measured body load, but geometry and endpoint changes are part of that response. They do **not** establish a terrain-aware quality advantage, a preferred Path, or statistical generalization.

## Runtime restart proposal and limits

The proposed next host protocol is `eatnav_p1_g3_path_crossover_v3_assignment_validator_proposal`. A future fresh manifest must stage both the v3 host and shared validator module, freeze both source SHA values, keep the existing simulator/Path/seed/physics/follower contracts, and use the same four-slot assignment check before and after Path publication. The old host-valid 8 trials and offline-audited 4 trials must retain distinct provenance. This is a proposal only: no remaining crossover process is authorized or started.

Related static/synthetic tests: **51 passed**. The repository-wide test collection was attempted in the offline `tinyvla` interpreter and stopped with 12 unrelated import errors for missing `diffusion_policy` or ROS `geometry_msgs`; no failing related test was observed.
