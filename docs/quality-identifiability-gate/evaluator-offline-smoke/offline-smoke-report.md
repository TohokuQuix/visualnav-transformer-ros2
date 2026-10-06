# Evaluator offline smoke: Actor-compatible teacher and portable student

## Scope and provenance

The seven-group offline smoke used the 63 previously sealed E1 simulator traces (7 StateGroups × [3 direct repeats + 6 distinct curves]). No simulator rollout, collection, robot command, NoMaD training, or production bridge change was made. The isolated branch is `exp/eatnav-evaluator-offline-smoke` at base `350b67b8a2d84a6db52fc833309f87f4a0885c73`. The source, checkpoint, bridge, labels, candidate identities, paths, configs and trace SHA values are in [source-provenance.json](source-provenance.json). The valid `r1` training root is `/tmp/eatnav-evaluator-offline-smoke-20261006-r1`; its 216 fixed jobs finished in 613 s. A preceding implementation-invalid run was stopped before evaluation because ranking weights were applied at half their configured values. Its files are isolated under `/tmp/eatnav-evaluator-offline-smoke-20261006/` and were not used in this analysis.

## A1: physical Critic dependency

The local physical `onix_policy_bridge` loads only `actor.*` weights from the RL checkpoint. `backend.handle_request` builds the Actor observation, evaluates `ActorOnly`, processes the Action and returns commands. The frontend/backend have no Critic inference or Critic observation consumer in the control, watchdog, arbitration, recovery or follower path. The serialized `env.yaml` contains a `critic` observation section for the training environment, but the physical backend does not construct or evaluate it. Verdict: `critic_not_used_for_robot_control` ([audit](critic-runtime-dependency-audit.json)).

The bridge's `_base_lin_vel_obs` computes a pseudo linear speed from measured left/right sprocket angular velocities and sprocket radius. `observations.policy.base_lin_vel` is null in the inspected 22D policy config, so this pseudo speed is absent from the active Actor input. The function is conditional on a non-null policy observation term or model-dimension reconciliation; the current 22D configuration already matches the checkpoint. Missing named joint values fall back to zero in the bridge, an existing runtime behavior and a physical logging/parity concern. `cmd_out.linear_x` is instead derived from the processed *policy action*, not this measured pseudo speed. The portable Evaluator contract excludes pseudo ground speed, physical Critic observation/value, and Critic input-dimension compatibility. Production code was not altered.

## A2–A4: input meaning and temporal audit

The raw four sprocket fields retain `rad/s` units and physical front/rear names. Model 2S uses `0.05 × clip(mean(front,rear), -100, 100)` independently for right and left; `0.05` is Actor-observation scaling, not a wheel-radius or m/s conversion. The 24-joint aggregate remains simulator-only. The [sprocket contract](sprocket-angular-state-contract.json) and [field hashes](input-field-order-hashes.json) distinguish these inputs.

The only body and DEM source for all variants is the saved `summary.anchor`, captured before Path publication by the frozen E1 source. The candidate-specific inputs are the 8-point base-frame Path and deterministic geometry, DEM profile and coverage derived from that Path. Per-step `trace` and `post_follower_anchor` are never read as inputs. All 63 trace/label/Path identities and SHA values match; all four DEM identities remain in the same episode at or before the anchor step. Within each StateGroup, the DEM, goal, Actor22, body, flipper, 2S, 24J, arm mask and timestamp are identical for all nine runs ([alignment audit](label-alignment-audit.json)). The 7 groups have distinct source windows, episode keys, environment IDs and terrain seeds ([split contract](data-split-contract.json)).

Simulator body angular velocity and projected gravity come from the pre-noise Actor terms, and flipper positions, fold mode and 2S come from the same synchronous anchor. The simulator has no arm joints, so the arm applicability mask is explicitly `0` for these fixtures. This is **not** proof that a physical arm is stowed: its reference/tolerance and logging parity remain unfrozen. DEM unknowns and Path coverage carry masks; no missing physical measurement was inferred as a zero-valued observation. All seven groups have one identical relative goal, and each terrain has only one group. This limits both goal and terrain conclusions.

## A5–A7: frozen training and observed result

The [training config](training-smoke-config.yaml) was fixed before learning: B0–B7, seeds 123/456/789, StateGroup-level LOSO, train-only StateGroup-balanced scalers, a shared 2-layer MLP, Huber regression on observed 2 s reward/direct-relative reward/goal reduction plus endpoint/path diagnostics, and lower-weight strict-pair Bradley–Terry losses. Three direct repeats stay clustered in their StateGroup and are averaged at the candidate level for ranking. Terminal Critic and residual labels are Phase B-only auxiliaries; they were not used as primary teachers or trained in this smoke. Safety remained disabled. B7 used B3 output as a training-only auxiliary loss while retaining B5 portable inputs and observed primary labels ([teacher/student contract](teacher-student-contract.yaml)). No LOSO test group selected a checkpoint.

The single-batch test produced mean primary MAE 0.0055–0.013 for B1 and B3–B7. B0 and B2 retain an irreducible collision because their input is identical for the two different candidates; their MAE was 0.0494. In seven-group memorization, B2 mean primary MAE was 0.0483 versus 0.0389 for B4 and 0.0387 for B7. The best train epoch was usually the 300-epoch cap; this is a pipeline sanity check, not established convergence or generalization.

| Variant | Reward MAE | Direct-relative MAE | Goal MAE | Reward pair accuracy | Top-1 recall | Oracle reward regret |
|---|---:|---:|---:|---:|---:|---:|
| B0 constant | 0.273 | 0.075 | 0.031 | 0.558 | 0.286 | 0.063 |
| B1 geometry | 0.465 | 0.135 | 0.039 | 0.449 | 0.000 | 0.081 |
| B2 Actor22 | 0.222 | 0.078 | 0.027 | 0.558 | 0.286 | 0.063 |
| B3 Actor22 + context | 0.435 | 0.142 | 0.025 | 0.497 | 0.095 | 0.085 |
| B4 Portable Core | 0.395 | 0.138 | 0.024 | 0.476 | 0.095 | 0.085 |
| B5 Core + 2S | 0.335 | 0.131 | 0.033 | 0.512 | 0.095 | 0.083 |
| B6 Core + 24J | 0.337 | 0.127 | 0.033 | 0.528 | 0.143 | 0.078 |
| B7 Core + 2S + distillation | 0.313 | 0.119 | 0.031 | 0.528 | 0.048 | 0.088 |

These are means of seven held-out StateGroups, with the same three seeds for every variant; full fold/head values are in [LOSO results](loso-diagnostic-results.json). The candidate decision metrics are weak and mixed. B4 beats B1 on goal MAE in five folds but on reward MAE in only four and direct-relative MAE in two. B5 beats B4 on reward/direct-relative/goal MAE in 4/3/1 folds respectively. B6 is not a clear gain over B5. B7 improves aggregate regression but worsens top-1 recall (0.048 versus 0.095) and oracle regret (0.088 versus 0.083), so distillation fails the observed-Outcome adoption rule. The exact per-fold comparisons, Spearman, RMSE, collapse, pair preference, Pareto, conflict, direct-relative sign, direct delta and abstention diagnostics are in [ablation comparison](ablation-comparison.json) and [multiobjective evaluation](multiobjective-evaluation.json).

B0 and B2 predict the same score for every candidate within a StateGroup. Their Pareto-front recall can appear artificially perfect because every tied candidate is included in the predicted front; predicted-front size, precision and an all-tied flag are provided to prevent that misreading. Neither can identify a Path-specific winner. No single variant is approved as a selector by these results.

Figures: [prediction versus target](fig-head-prediction-target.png), [input ablation](fig-input-ablation.png), [fold metrics](fig-fold-metrics.png), and [teacher/student](fig-teacher-student.png).

## A8: decision and next evidence

The [machine-readable decision](final-decision.json) records `actor22_not_sufficient_for_path_selection`, `portable_2s_adds_signal=false`, `portable_core_learns_signal=false`, `distillation_supported=false`, and `insufficient_state_group_support=true`. Actor22 learned some group-level outcome signal, but candidate-specific Actor22 distillation is unsupported because all candidates share the same input. The 24J and privileged Actor22 variants do not establish a portable advantage. The observed reward/goal conflicts remain in the labels and diagnostics, but the learned selection results do not reliably resolve them.

The next evidence is more independent StateGroups with goal variation, followed by physical DEM/IMU/flipper/arm logging parity and a prospective arm-stowed reference. Model redesign can be considered after those data and parity gates, not selected from this seven-group smoke. Formal Evaluator training, online selection, NoMaD retraining, full collection and physical control remain unapproved.

## Verification

The SHA/label/anchor audit completed on all 63 records. The targeted offline smoke and existing portable-input/multi-head suites passed **40 tests** with the `tinyvla` Python environment. Python compilation passed for the audit, training and analysis scripts. The saved runner status is `complete: 216/216`, and the [artifact ledger](artifact-sha-ledger.json) covers the review outputs.
