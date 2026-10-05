# EAT-Nav execution-aware Evaluator design review

## Decision

**Gate: `needs_label_revision`.** The seven sealed StateGroups establish reproducible, candidate-dependent Critic and embodiment responses, but they do not support ΔV as a standalone notion of “better.” The next approval should settle the primary teacher and the production input contract, then revisit outcome-blind hard-state support. No new rollout, candidate generation, model learning, formal validation, scaled collection, or hardware experiment was run.

This review treats the StateGroup as its statistical unit. The 21 pairs within a seven-candidate group are descriptive correlated comparisons. GUI env10 viewing is excluded.

## Confirmed facts

### D0 source and identity

- Base worktree was clean at `e6c0bff11c017644e2bee5e375c024ded5a234bb`; this review uses an isolated branch/worktree. The source RL checkpoint SHA is `b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d`, and the reference config SHA is `caa88735e765fc0358930dfa41579ecaac9eb96dc4052dc01a81390377279992`.
- Seven distinct source trajectories/environment instances/terrain seeds, each source episode 0, contribute one anchor per terrain. Two groups are sealed earlier E1 outcomes corrected by the map-frame addendum; five are Expanded E1. Each has three direct repeats and six non-direct candidate outcomes: **63 live traces**. Five additional anchor-only processes are never labels.
- The read-only audit verified 120 nonterminal steps, all listed output SHA values, run-complete and protocol-valid markers, manifest identity, selected anchor episode/step, duplicate gates and corrected geometry. It recomputed raw Critic/reward/body outcomes and the three coordinate-corrected fields (`cross_track_p95_m`, `endpoint_error_m`, `progress_m`); the saved values matched within `1e-10`. The previously transferred first direct repeat has no local stage marker in the recovered root, while its run-complete, protocol-valid and output SHA checks pass. No prior artifact was overwritten.
- The source DEM's `axis=row=+world_y,col=+world_x` describes the **world grid**. The saved `transformed` tensor is a **base-yaw raster** produced by `robot_dem`, with only a scalar reference-height offset removed. The Path/DEM geometry audit uses that transformed tensor. The realized XY comparison uses the coordinate-corrected environment-local map frame.
- Frozen candidate geometry is cubic Bezier: one direct, left/right three amplitudes, common start and goal, eight future waypoints, 2 s Path horizon, 2 m DEM scope. The production follower, fixed RL Actor and exact-stamp protocol are unchanged. P→C holds in 7/7 groups, C→R in 5/7, R→Y in 7/7. Actor action changes are observable in 7/7. `random_rough` and `boxes` have seven command modes but one clustered realized XY mode. This does not imply identical Critic/body response.
- The 63 traces have **0 failure-positive**, **0 verified stacks**, and **0 terminations**. In the five new groups 44/45 live trajectories reach near their Path endpoint within 0–2 s. The 6 s continuation is diagnostic after a 2 s Path, so 0–6 s progress is not a pure Path-following score.

### D1 labels and repeat variation

Every direct three-repeat set is numerically identical for the outcomes used here. Direct-repeat range, standard deviation, pairwise RMS and max absolute difference are **0 in all seven groups**. Consequently, candidate-to-duplicate ratios have a zero denominator and are recorded as `NA`; no epsilon was introduced. Candidate ΔV ranges exceed the observed repeat variation in 7/7 groups, but this only shows deterministic differences under the same seed/protocol. It does not measure variation across seeds, episode conditions or hardware.

ΔV candidate range is 0.174–0.434 Critic units by group. Its StateGroup-macro rank concordance with H-step advantage is Spearman **0.209**, Kendall **0.197**, raw pair agreement **0.599**, top-1 agreement **1/7**. With cumulative reward these are **0.015**, **0.007**, **0.503**, **0/7**. With corrected 0–6 s progress they are **−0.158**, **−0.129**, **0.435**, **0/7**. ΔV versus 0–2 s cross-track has macro Spearman **0.168** and top-1 **3/7**; versus 0–2 s endpoint error macro Spearman is **−0.194** and top-1 **0/7**. These are descriptive seven-group summaries, not significance claims. Particularly, `k_rail_step_field` has ΔV versus advantage Spearman **−0.79** and versus reward **−0.86**. The Critic cannot be treated as sole ground truth for physical progress or safety.

An exploratory [horizon diagnostic](horizon-diagnostic.json) recomputed ΔV and bootstrapped advantage over the Path's first 40 control steps (2 s), using the saved pre-step Critic at trace index 40. Their macro Spearman rises to **0.684**, but top-1 agreement is still only **2/7** and `random_rough` remains negative (**−0.179**). This supports investigating horizon mismatch and the post-Path continuation as causes of disagreement; it does **not** select a new teacher or overwrite the frozen 6 s Outcomes.

Raw joint effort maxima are saturated at 3000 in these runs; they cannot rank candidates. Max contact can mean traction or impact and has no universal monotonic quality direction. Drawdown (higher, closer to zero is preferable) and recovery are separate: larger recovery can follow a worse dip. The label table retains initial/final Critic, ΔV, H-step advantage, reward, progress, endpoint encounter/error, cross-track, drawdown/recovery, tilt/contact/effort, stack, termination and censoring. Absent support is `NA`, never a synthetic zero or negative.

### D2 input and leakage

The pre-choice portable input is the four-frame base-yaw DEM history with unknown masks, relative goal, base-frame orientation/velocity/angular velocity, ordered joint position/velocity, previous action, and candidate Path. All 4 DEM frames are spaced by five 20 Hz control steps (0.25 s), covering 0.75 s from first to current. Candidate Path has eight future points and an implicit start. Path and terrain geometry features are deterministic from those inputs; unknown cells remain masked.

The anchor-only simulator snapshot includes a 256-dimensional Actor latent, but use would couple the Evaluator to the exact RL checkpoint and latent API. Compare an optional latent model against the portable model. **Do not use live first-step latent**: it is downstream of Path publication. Raw joint effort is present in live trace at a pre-step capture phase, but **not in the saved anchor-only snapshot**; real-robot equivalent unit, order and timing are unverified. It is excluded from the primary input. Physical GridMap, odometry/IMU, joint-state and previous-action parity are likewise not established. Normalization is fitted on train only, with unknown/missing masks retained.

Future follower command, Actor action/latent response, realized trajectory, contact, effort, reward, Critic, termination and outcome are labels/diagnostics, never primary inputs. Candidate ID, family/amplitude, run order, absolute environment origin and future-derived features are excluded. See the field-by-field [input contract](evaluator-input-contract.json), [availability audit](online-input-availability.json), [feature contract](path-dem-feature-contract.json), and [exclusion list](leakage-exclusion-list.json).

## Design recommendations

### Teacher and learner

1. Define the operational preference before training. ΔV alone is **not approved** as the primary teacher. Compare ΔV, H-step advantage, 0–2 s progress/tracking, drawdown and cause-specific contact/tilt on new independent StateGroups. Do not collapse contradictory labels with weights chosen from these seven groups.
2. After teacher revision, use within-StateGroup ranking as the selection baseline; use absolute Huber regression for calibration/diagnostic heads. The multi-task ranking architecture is the leading design because it exposes rather than hides disagreement. Its weights, tie rules and calibration plan require prospective freeze. The seven groups provide no Evaluator performance estimate.
3. Train/evaluate by source environment/terrain seed/episode clusters; keep every candidate, direct repeat and anchor derivative together. Report StateGroup-macro pairwise accuracy, oracle recall@1/@3, regret, comparison to direct, worse-than-direct rate, calibration, uncertainty versus misranking, and supported terrain/geometry strata. A held-out terrain claim requires genuinely independent terrain support. Current 1 group per terrain is insufficient.
4. Keep the Candidate Generator frozen. The present issue is teacher semantics and hard-state exposure, not evidence that the seven Bezier curves should be changed.

### Outcome-blind hard-state collection support

The prospective geometry screen was fixed in [hard-state-support-config.yaml](hard-state-support-config.yaml) before counting: complete-only train, source episode 0, prefix ≤80, exclude the seven existing environments and three previously probed anchors, finite four-frame DEM, frozen seven Path centerlines within 2 m, direct relief/roughness/jump/slope exposure, and a detour with a differing known DEM height profile. It reads no reward, Critic, termination or prior candidate Outcome. Its `0.04 m` relief/jump, `0.015 m` roughness, `0.2` slope and `0.02 m` detour-profile difference are prospective definitions, not optimized on the E1 labels. The first two ranked candidates are an audit output, not a frozen rollout selection.

| Terrain | Supported windows | Independent episode/seed support | Two-group target |
| --- | ---: | ---: | --- |
| boxes | 0 | 0 | no |
| hf_pyramid_slope | 0 | 0 | no |
| k_rail_field | 71 | 10 | yes |
| k_rail_step_field | 97 | 9 | yes |
| pyramid_stairs | 7 | 2 | yes, marginal |
| pyramid_stairs_inv | 7 | 2 | yes, marginal |
| random_rough | 16 | 8 | yes |

Thus the proposed new 14 groups cannot be selected under the stated short-prefix rules. Do not silently extend prefix or weaken geometry thresholds. Any revised exposure or runtime envelope needs a new prospective config and review before collecting. The input-copy manifest and two needed raw chunks were SHA-checked. This is an offline support count, not validation/held-out selection.

At the measured 50-process mean, 140 new serial processes would take **6.63 h point estimate, 7.96 h with 20% margin**. Estimated final saved size is **1.19 GB point, 1.42 GB with 20% margin**, beyond the previous 4 h/1 GB caps. Serial topology suggests GPU device total near the prior **5,960 MiB** peak and simulator PSS near **5.38 GiB** (anon+shared memory **3.72 GiB**), but these are measured references, not guarantees. External Zenoh readiness, fresh-process lifecycle, per-run seals and resource monitoring remain required. No 140-process schedule is authorized.

### Safety head

Do not train a Safety head. There are no failure-positive live trajectories or verified stacks, and nonterminal completions cannot stand in for cause-specific safety negatives. Separate verified stack, excessive tilt, cause-specific termination, recovery, high effort, contact anomaly, progress failure, censored timeout and infrastructure failure. The prospective support proposal requires, **per event type**, at least 40 positive train StateGroups, 10 positive validation, 10 positive test, 40 negative train, positive coverage on at least three terrain types and ten terrain seeds. This is an engineering review threshold, not inferred from the current zero count.

## Answers to the review questions

1. **ΔV alone?** No; ordering conflicts strongly with advantage, reward and progress.
2. **Rank agreement?** Macro Spearman 0.209 / 0.015 / −0.158 against advantage / reward / 0–6 s progress. Top-1 1/7, 0/7, 0/7 respectively.
3. **Above direct noise?** 7/7 show nonzero ΔV differences, but the duplicate denominator is zero and stochastic robustness is unmeasured.
4. **One XY mode?** `boxes` and `random_rough` retain candidate-dependent Actor/Critic/body signals. Whether those differences are useful preferences is unresolved.
5. **Regression or ranking?** Group-relative ranking is the operational baseline after label revision; absolute regression remains a calibration head/baseline.
6. **Multi-task?** It preserves disagreements and physical diagnostics; it needs more independent groups and prospectively fixed weights.
7. **Actor latent?** Portable primary excludes it. Test an anchor-only latent branch later under the frozen checkpoint; post-Path latent is leakage.
8. **Unverified physical inputs?** DEM/GridMap parity, body velocity/orientation timing, 52-joint order/units, previous action availability, Actor latent API, and especially raw effort equivalence.
9. **Split unit?** Source environment/terrain seed/episode cluster, including all anchors and candidate derivatives.
10. **Why no Safety head?** Zero failure-positive and zero verified stack support; timeout/infrastructure do not supply robot-failure labels.
11. **Hard-state support?** Independent episodes: boxes 0, hf slope 0, rails 10/9, pyramid types 2/2, random rough 8, under the frozen screen.
12. **140-process resources?** 6.63 h point / 7.96 h with margin; 1.19 / 1.42 GB; GPU ~5,960 MiB reference, simulator PSS ~5.38 GiB reference.
13. **Next approval?** Teacher definition and physical input parity, then prospective hard-state selection revision. Neither Evaluator training nor 14-group rollout yet.
14. **Change Generator?** No evidence-based need from this review.

## Figures and scope

The eight SVGs in `figures/` show outcome range/noise, rank comparisons, macro concordance, candidate effect with zero duplicate denominator, stage retention, input/output contract, hard-state support, and the uncertainty-versus-misranking evaluation plan. The last diagram is a plan with `NA` performance because no Evaluator exists. No terrain generalization or Safety-head readiness is claimed.

**Facts** are the sealed trace audit, numeric label comparisons and train geometry support above. **Recommendations** are ranking/multi-task architecture, portable inputs and clustered splits. **Hypotheses for the next phase** are whether hard-state exposure creates aligned execution quality labels, whether an Evaluator selects better candidates than direct, and whether its uncertainty predicts misranking. None was tested here.
