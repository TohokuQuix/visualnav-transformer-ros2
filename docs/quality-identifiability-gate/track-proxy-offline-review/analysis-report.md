# Portable track-state proxy: offline review

## Decision

**`portable_2sprocket_contract_information_loss`**, relative to the current simulator 24-joint diagnostic. The source-defined four-sprocket representation is reproducible and separates the frozen candidates from exact direct repeats, but candidate-distance ranking and its descriptive relationship to reward/goal change materially for several StateGroups. This is a gate result, **not** proof that 24 joints predict outcomes better: seven StateGroups and six alternatives per group cannot establish predictive superiority. The [v2 schema](portable-evaluator-input-contract-v2.proposal.json) is therefore a deferred proposal, not a frozen input contract. No training, new rollout, adapter change or robot control occurred.

## Sources and semantic mapping

The new isolated worktree started clean at `0efb8620c6657120d23489c71865e347068450ab`. The P0/P1 source and analysis config were frozen in commit `dfba097` **before** P2 metrics. [Provenance](source-provenance.json) records all 63 compressed trace SHA values, run/candidate identities, RL checkpoint/config, bridge frontend/backend/config/launch, simulator observations, URDF/xacro and previous M0–M13/portable-state reviews. All 63 trace and summary SHA values and 41 prior-ledger entries passed; no previous output was modified. The generated physical URDF was used only as a read-only expanded source fixture.

| Physical `/joint_states` joint | Simulator joint (52-order index) | Side/end | Placement x (m) | Axis |
|---|---|---|---:|---|
| `chassis/fr_sprocket_axle` | `chassis_r08_sprocket_axle` (17) | right/front | +0.255 | +Y |
| `chassis/rr_sprocket_axle` | `chassis_r01_sprocket_axle` (10) | right/rear | −0.255 | +Y |
| `chassis/fl_sprocket_axle` | `chassis_l08_sprocket_axle` (9) | left/front | +0.255 | +Y |
| `chassis/rl_sprocket_axle` | `chassis_l01_sprocket_axle` (2) | left/rear | −0.255 | +Y |

The [semantic mapping](sprocket-semantic-mapping.json) checks joint names, body parent, sprocket child, x/y placement, shared +Y joint axis and 0.0703 m radius in the simulator and expanded physical URDF. Simulator sprockets are 0.075 m farther laterally from the center than the physical joint origins. Physical right/left transmissions use −1/+1 reductions; `/joint_states` is interpreted in URDF joint coordinates. Encoder sign under load and dynamic/contact equivalence have **not** been measured. The candidate canonical order is right, left; each side is the arithmetic mean of front/rear raw rad/s, clipped to ±100 rad/s, multiplied by the prospectively fixed 0.05 scale. No fitted scale or sign correction was used. The 24-joint Actor mean uses its existing eight main and sixteen subroller joints per side and remains untouched.

## P2: reconstruction and 2-joint comparison

Across 63 × 120 = **7,560** same-step records, the 24-joint expression reconstructs the Actor `pre_noise[14:16]` with maximum absolute error **6.06×10⁻⁸**, below the pre-fixed `1×10⁻⁵` tolerance. No missing joint, wrong shape, nonfinite value or identity/SHA mismatch occurred. Only after this check were proxy metrics calculated. [Full comparison](track-proxy-comparison.json) includes left/right absolute and relative errors, bias, RMSE, Pearson/Spearman, sign agreement, motion, terrain, StateGroup and candidate strata. Relative error uses denominator `max(|24j|,1e-6)`, so means rise near zero; median relative error is more interpretable.

| Side | MAE | RMSE | Median relative error | Pearson | Spearman | Active-sign agreement |
|---|---:|---:|---:|---:|---:|---:|
| Right | 0.0147 | 0.0609 | 2.54% | 0.883 | 0.965 | 97.6% |
| Left | 0.0205 | 0.0649 | 3.31% | 0.805 | 0.921 | 96.1% |

Errors above are in **rad/s × 0.05**. Mean bias (2j−24j) is −0.00287 right and +0.00058 left. The largest terrain mean absolute errors are boxes/right 0.0297 and pyramid_stairs/left 0.0281. By frozen command strata, right/left MAE is 0.0169/0.0218 while stopped, 0.0087/0.0169 while straight, and 0.0023/0.0245 while turning. Turning classification uses `|yaw command|≥0.15 rad/s`; the left turn error is notably larger. See [scatter](track-proxy-scatter.svg), [120-step example](track-proxy-timeseries.svg), and [StateGroup/terrain variation](track-proxy-group-variation.svg). The compressed derived [timeseries](/tmp/eatnav-track-proxy-review-20261006/track-proxy-timeseries.jsonl.gz) is external to Git.

## P3: candidate information

The seven StateGroups, rather than 105 within-group candidate pairs, are the independent comparison units. Each group has three exact `direct_reference` repeats and six distinct candidate Paths. The direct repeats have **zero** track/command/action/XY trajectory difference in the saved trace, so candidate/duplicate ratios are **undefined** and are not reported as large finite numbers. All seven groups have nonzero candidate pairwise track RMS in both representations; in scaled units the group ranges are 0.038–0.064 for 24j and 0.067–0.182 for 2j. Follower command, Actor action and realized XY candidate pairwise RMS are also nonzero in all groups. This shows sensitivity to candidate execution, not outcome sufficiency.

| StateGroup / terrain | Candidate-distance pair order concordance (2j vs 24j) | Spearman of distance to direct vs direct-relative reward, 24j → 2j | Goal reduction, 24j → 2j |
|---|---:|---:|---:|
| expanded-train-000 / boxes | 8/15 | −0.14 → +0.49 | +0.03 → +0.77 |
| expanded-train-001 / hf_pyramid_slope | 11/15 | +0.14 → +0.09 | +0.26 → −0.37 |
| expanded-train-002 / k_rail_field | 7/15 | +0.94 → −0.26 | +0.60 → −0.54 |
| expanded-train-003 / k_rail_step_field | 12/15 | −0.03 → −0.14 | +0.09 → +0.20 |
| expanded-train-004 / pyramid_stairs | 11/15 | −0.77 → −0.37 | −0.37 → −0.49 |
| g0e-train-000 / pyramid_stairs_inv | 12/15 | −0.03 → +0.03 | +0.83 → +1.00 |
| g0e-train-001 / random_rough | 8/15 | −0.31 → +0.26 | +0.09 → −0.49 |

These are **descriptive six-candidate within-group correlations**, with no p-value or independent-pair claim. `discounted_reward_2s` equals the saved `reward` field; `direct_relative_reward_2s` uses `direct_relative_reward`; `goal_distance_reduction_2s` uses `goal`. All are listed in [information retention](candidate-information-retention.json), alongside command, action, XY and auxiliary Critic drawdown. The mean order concordance is 0.657 across seven group values. The k_rail_field relationship flips strongly for both reward and goal; random_rough and boxes also reorder. The underlying sprocket velocities may capture different physical behavior from subrollers; current outcomes cannot establish which is best. No baseline weights or thresholds were fit to these 63 traces.

## P4/P5 and next gate

The [v2 proposal](portable-evaluator-input-contract-v2.proposal.json) specifies four DEM frames with valid masks, base-frame relative goal and 8×2 candidate Path, deterministic geometry/profile/coverage, base angular velocity, projected gravity, RR/RL/FR/FL flipper position, right/left sprocket velocity, and explicit availability/domain masks. It gives frame, unit, order, anchor timestamp, prospective normalization, clip and missing handling. Simulator 52-joint vectors and 24-joint mean stay in simulator-only diagnostics/ablation; effort, previous commanded/applied action and arm axes stay out of primary; base linear velocity is an optional ablation. `arm_configuration=fixed_stowed_pose` is required, with future fixed tolerance and abstention if unknown/outside it.

The separate [physical parity gates](remaining-robot-parity-gates.yaml) cover four-sprocket encoder sign/timestamps, flipper sign/zero/order, IMU extrinsic and body-frame gravity/rate, header versus callback times and multi-sensor freshness, missing/stale handling, arm-stowed evidence, train-only normalization and field-order/checkpoint hash, and logging parity. The immediate next evidence is a source fixture followed by synchronized real-bag parity; logging parity follows before training. Existing short arm data are insufficient to certify always-stowed pose. Small Evaluator training is **not authorized**: portable information-retention, real-robot parity, prospective schema compatibility and logging parity remain open. NoMaD retraining stays downstream of the Evaluator readiness decision.
