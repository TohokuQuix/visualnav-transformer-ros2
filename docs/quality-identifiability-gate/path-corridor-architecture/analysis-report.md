# Path-corridor architecture offline comparison

All 28 StateGroups and 252 saved candidate traces were retained. No rollout or label change occurred.

## Outcome-blind terrain support

Frozen config SHA: `aca4a0cebd7a8d9fe5b7613dc02678e6e782e23e329761bb8c0f3ceac3351e4d`. Classes: `{'differential_corridor': 1, 'flat_corridor': 15, 'transition_corridor': 12}`.
Layers with fewer than three StateGroups are insufficient support for performance claims.

Parameter counts: `{'A1': 267990, 'A2': 221254, 'A3': 233014, 'A4': 274918}`. Completed fits: 480; early stops: 0; best at last epoch: 474.
If training loss is still improving at epoch 200, this is a training-limit caveat and does not justify changing this frozen comparison.

| Terrain | flat | common rough | differential | transition | insufficient/unknown |
|---|---:|---:|---:|---:|---:|
| boxes | 4 | 0 | 0 | 0 | 0 |
| hf_pyramid_slope | 2 | 0 | 0 | 2 | 0 |
| k_rail_field | 1 | 0 | 0 | 3 | 0 |
| k_rail_step_field | 0 | 0 | 1 | 3 | 0 |
| pyramid_stairs | 1 | 0 | 0 | 3 | 0 |
| pyramid_stairs_inv | 3 | 0 | 0 | 1 | 0 |
| random_rough | 4 | 0 | 0 | 0 | 0 |

## Held-out results

| Split | Model | reward MAE | goal MAE | reward pair accuracy | top-1 | reward regret | Pareto precision |
|---|---|---:|---:|---:|---:|---:|---:|
| E-anchor | A1 | 0.7456 | 0.0440 | 0.5351 | 0.1746 | 0.0944 | 0.3749 |
| E-anchor | A2 | 0.7549 | 0.0505 | 0.5420 | 0.1429 | 0.0863 | 0.3095 |
| E-anchor | A3 | 0.7452 | 0.0519 | 0.4701 | 0.0794 | 0.1201 | 0.2958 |
| E-anchor | A4 | 0.7091 | 0.0612 | 0.4792 | 0.1111 | 0.1112 | 0.3635 |
| E-goal | A1 | 0.1596 | 0.0162 | 0.5635 | 0.1905 | 0.0799 | 0.2929 |
| E-goal | A2 | 0.3827 | 0.0340 | 0.5136 | 0.1429 | 0.0915 | 0.3242 |
| E-goal | A3 | 0.4411 | 0.0326 | 0.4898 | 0.0952 | 0.1261 | 0.2726 |
| E-goal | A4 | 0.2384 | 0.0311 | 0.5045 | 0.1190 | 0.1216 | 0.2159 |
| E-random-group | A1 | 0.6353 | 0.0456 | 0.4932 | 0.1071 | 0.1028 | 0.3988 |
| E-random-group | A2 | 0.6582 | 0.0608 | 0.5300 | 0.1786 | 0.0879 | 0.3589 |
| E-random-group | A3 | 0.6346 | 0.0567 | 0.4512 | 0.0952 | 0.1170 | 0.3351 |
| E-random-group | A4 | 0.5749 | 0.0593 | 0.4722 | 0.1429 | 0.0971 | 0.3349 |
| E-terrain | A1 | 0.6535 | 0.0565 | 0.5351 | 0.0833 | 0.0960 | 0.4754 |
| E-terrain | A2 | 0.5297 | 0.0710 | 0.5380 | 0.1548 | 0.0886 | 0.4188 |
| E-terrain | A3 | 0.6181 | 0.0662 | 0.5000 | 0.0714 | 0.1119 | 0.4315 |
| E-terrain | A4 | 0.6385 | 0.0669 | 0.4949 | 0.1310 | 0.1011 | 0.3946 |

## Paired StateGroup confidence intervals

Effects are model minus comparator; negative MAE/regret and positive accuracy are improvements. Intervals are 95% group bootstrap intervals with 1,000 resamples. They are not multiple-comparison adjusted.

| E-anchor metric | Comparison | Effect | 95% CI | Interpretation |
|---|---|---:|---|---|
| reward_mae | A2-A1 | 0.0093 | [-0.2456, 0.3349] | inconclusive |
| reward_mae | A3-A1 | -0.0004 | [-0.2590, 0.2681] | inconclusive |
| reward_mae | A3-A2 | -0.0097 | [-0.1367, 0.1103] | inconclusive |
| reward_mae | A4-A3 | -0.0361 | [-0.1555, 0.0726] | inconclusive |
| goal_mae | A2-A1 | 0.0065 | [-0.0116, 0.0297] | inconclusive |
| goal_mae | A3-A1 | 0.0079 | [-0.0068, 0.0284] | inconclusive |
| goal_mae | A3-A2 | 0.0014 | [-0.0058, 0.0085] | inconclusive |
| goal_mae | A4-A3 | 0.0093 | [0.0017, 0.0171] | positive |
| oracle_reward_regret | A2-A1 | -0.0081 | [-0.0304, 0.0133] | inconclusive |
| oracle_reward_regret | A3-A1 | 0.0257 | [0.0069, 0.0449] | positive |
| oracle_reward_regret | A3-A2 | 0.0338 | [0.0132, 0.0550] | positive |
| oracle_reward_regret | A4-A3 | -0.0089 | [-0.0224, 0.0035] | inconclusive |
| reward_pair_accuracy | A2-A1 | 0.0068 | [-0.0401, 0.0514] | inconclusive |
| reward_pair_accuracy | A3-A1 | -0.0650 | [-0.1262, -0.0083] | negative |
| reward_pair_accuracy | A3-A2 | -0.0718 | [-0.1217, -0.0196] | negative |
| reward_pair_accuracy | A4-A3 | 0.0091 | [-0.0181, 0.0378] | inconclusive |

## Interpretation

A3−A1 reward MAE is -0.0004 with CI [-0.25899171319217995, 0.26807514822421646]; this does not establish a regression gain.
A3−A1 reward pair accuracy is -0.0650 with CI [-0.12622826908541193, -0.008295540438397597], and oracle reward regret is +0.0257 with CI [0.006896126603745972, 0.04493164292284086]. These are adverse candidate-selection effects in E-anchor under unadjusted CIs; they do not justify an architecture change on this sparse terrain-interaction support.
Only 1 independent source anchor family has multiple candidate corridor clusters. The current data cannot establish terrain-conditioned selection generalization.


## Decision

- **corridor_support_sufficient_for_architecture_comparison:** False
- **a2_profile_adds_heldout_signal:** inconclusive
- **a3_corridor_cnn_adds_heldout_signal:** inconclusive
- **a3_outperforms_a2:** inconclusive
- **a4_full_dem_adds_signal_beyond_corridor:** inconclusive
- **terrain_conditioned_candidate_selection_supported:** insufficient_terrain_interaction_support
- **distinct_interaction_group_count:** 2
- **architecture_ready_for_larger_training:** False
- **additional_state_group_collection_needed:** True
- **scope_note:** E-goal shared-anchor and E-terrain diagnostic-only; unadjusted multiple comparisons.
- **distinct_interaction_anchor_family_count:** 1
- **distinct_interaction_anchor_families:** ['indep-train-006']

The A1 flatten branch uses the same anchor-aligned DEM as A4; this differs from historical B4 preprocessing.
The common branch retains the historical 64-point path DEM profile in all four models, so this isolates the added representation rather than all Path-conditioned terrain information.
E-goal shares anchors; E-terrain is diagnostic-only. Candidate pairs are descriptive and never bootstrap units.
A constant predictor can have Pareto recall 1 by marking all candidates Pareto; precision, set size and all-tied rate must accompany recall.
