# R6 threshold candidates, policy geometry and selection

This document describes the implemented threshold derivation and selection protocol. The validation dataset, 12 derived profiles and profile-specific error margins now exist; the full closed-loop comparison, top-three selection, unused test and bag inference are still pending. Actual numeric configurations are in `verified-training/production-profiles.json` and the artifact-root `production-profiles/index.json`. Test outcomes and the real bag are never inputs to derivation or freezing.

## Interfaces

`visualnav_transformer.eat_nav.r6_policy` is NumPy-only. It never imports a simulator, reads a bag, publishes ROS messages or loads an old progress head.

```python
shape = Footprint(front_m=measured_front_with_flippers,
                  rear_m=measured_rear_with_flippers,
                  half_width_m=measured_half_width,
                  stopping_margin_m=chosen_stopping_margin)
result = select_candidate(trajectory, event_probs, dem, goal_body, profile,
                          footprint=shape, mode='pose',
                          event_support=[True, True, True, True])
```

The caller supplies measured collision/flipper dimensions. No guessed robot footprint is silently substituted. Stopping-margin default0.06m preserves the explicit handoff comparator; it is a geometry assumption to evaluate, not a data-derived safety margin.

`trajectory` is (14,20,9), at0.1..2.0seconds: relative xyz[m], sin/cos(relative yaw), sin/cos(gravity roll), sin/cos(gravity pitch). Positions are in the initial yaw-aligned world-up frame. Sin/cos pairs must have unit norm within0.02. Event probabilities are (14,4) for tip_over, near_fall, linear_stall, turn_stall **within2seconds**. DEM is the latest physical2mx2m raster (2,20,20), height[m] plus binary validity, in the current local yaw frame. Goal is local xy[m]. Inputs require fixed shapes, finite known values, valid probabilities and schema `eatnav_r6_two_second_policy_v1`. Old10-second horizon is rejected. Event/hybrid modes require actual positive-and-negative training support for all four heads.

Modes: `event` applies supported2second event cutoffs plus map coverage; `pose` applies predicted maximum absolute roll/pitch and conditional DEM step cutoffs plus map coverage; `hybrid` applies both. Current implementation does not claim that event-only geometry is obstacle-safe. Profile probability0.5 is explicitly an uncalibrated comparator, not an empirical calibration result. The validation comparison must measure its behavior.

All candidates excluded produces `selected_index=None`, `stop=True`. Among admissible candidates, selection maximizes **2second endpoint reduction in goal distance**. Legacy10second progress is never read. Return fields include per-candidate exclusion reasons, coverage, clearance diagnostic, DEM geometry, maximum predicted roll/pitch, predicted goal progress and processing seconds.

## Swept raster geometry and uncertainty

All20 predicted poses and the current origin are joined with circularly unwrapped headings. Pose sampling uses <=0.04m translation and <=0.05rad yaw increments. Each pose samples the complete measured body/flipper rectangle on a <=0.04m grid. Both front and rear extend by the stopping margin, covering reversals conservatively. Native raster cell areas are [-1,1)m in each dimension; no extrapolation is used. Any sampled footprint cell unknown or outside the map excludes the candidate. Coverage counts unique swept cells, preventing a long dwell on known cells from diluting an unknown future corridor.

This is a finite-resolution raster screen. It is not continuous collision checking, a dynamics/contact model, a guarantee of clearance between sample points, or a recovery of unseen wall geometry. A range/footprint/stop-margin conflict is `map_unknown`, distinct from an obstacle exclusion. The2m DEM may legitimately exclude long trajectories: unknown ground is not free ground.

For each swept longitudinal footprint ray, duplicate native cells are removed. Stable treads separate individual risers. At least two transitions separated by observed stable treads classify `stairs`; one abrupt transition is conservatively `wall`; no abrupt transition is `flat_or_ramp`. A narrow unsupported peak remains a transition even without a stable top tread. Repeated risers are not summed into one wall height. The fixed geometry definition uses minimum tread depth0.20m and flatness0.04m, preserved as explicit geometric assumptions requiring evaluation. These are not represented as statistically learned values. Lateral height difference alone never excludes stairs; predicted roll may still exclude a candidate with excessive lateral tilt.

Clearance is the signed nearest distance from sampled swept footprint points to cells adjoining abrupt height transitions, minus half a cell diagonal. It is a diagnostic with no learned safety meaning; climbable stair risers can give negative clearance. Root/body clearance and obstacle identity require additional calibrated geometry and validation; this implementation does not invent a bottom-clearance teacher.

## Deriving at least10 actual candidate combinations

`derive_profiles(records)` accepts **validation-only** observed candidate records:

```json
{"split":"validation","group_id":"source-initial-state-group",
 "complete":true,"failed":false,
 "max_abs_pitch_deg":12.0,"max_abs_roll_deg":7.0,"step_cm":9.0,
 "geometry_kind":"stairs","field":"pyramid_stairs","level":2}
```

`failed` must be an actual boolean outcome under a recorded event definition. Caller must state whether the outcome is any of the four2second events or a narrower endpoint. `complete=false` records are excluded and counted, not converted to successes. For early observed failures, partial maximum-tilt observations are lower bounds; preserve censor metadata and explicitly choose a conservative treatment rather than silently treating them as complete2second successful runs. Source groups and candidates must remain in the same split. `step_cm` is an observed DEM riser metric; do not label level numbers as centimetres. Keep actual simulator geometry alongside the record when available.

Default support requirements are30 admissible branch records,10 observed event-free outcomes and5 observed failures. Empirical mixture quantiles for pitch are0.50/0.65/0.80/0.95 and roll0.50/0.75/0.95:12 crossed candidates. Each quantile includes both successful and failed observed magnitudes. Step quantile is the arithmetic mean of pitch and roll quantiles. A numerical positive floor0.001 is used in the respective units. Failures at small tilt do not imply a monotonic posture-failure boundary; the previously proposed lower-failure-quantile cap is not used. These are candidates requiring actual closed-loop validation, not proof that lowering any one angle prevents failure.

Each profile writes explicit `pitch_deg`, `roll_deg`, `step_cm`, quantiles, source hash, sample counts, failure rates, confidence intervals and conditional shape limits. At least10 **distinct numeric triples** must exist; changing only configuration names does not qualify. If support or distinctness is insufficient, status is `insufficient_support`, profiles are empty and requirements/reason are saved. No55degree/30cm baseline is called learned.

Separate wall/stair/flat-ramp step limits require at least5 successes and3 failures in that shape group. Unsupported conditional values are `null`; that geometry yields `geometry_threshold_unsupported` and STOP rather than inventing an observed bound. Unknown map is always rejected regardless of conditional support. Base derive_profiles output starts with margin0 and explicitly requires a separate error fit. Production uses eat_nav_r6_calibrate_profiles.py: measure max(0, observed peak minus predicted peak) across roll/pitch, including aligned0.1s errors and observed20Hz peaks. Each profile applies the error quantile equal to the mean of its pitch/roll quantiles. This grid was fixed before production closed-loop results. The shared q95 remains diagnostic only; applying it to every profile was rejected before production because it made11/12 angular envelopes universally STOP. No margin is clamped or retuned using rollout/test/bag outcomes. The applied production margins range from0 to6.29degrees. Horizontal position-error q95 is a bag-clearance diagnostic, not hidden control-footprint inflation.

Wilson95% intervals accompany observed branch counts. Since candidate branches can share source groups, those branch intervals are marked descriptive with a correlation warning. A second interval uses source groups with any observed failure; it is a different estimand. Sparse field/level/shape cells must retain their counts and be reported as uncertain. Confidence intervals do not remove dependencies between terrain seeds or guarantee generalization.

## CLI and immutable selection

```bash
/usr/bin/python3 scripts/eat_nav_r6_thresholds.py derive \
  --records /absolute/path/validation-records.jsonl \
  --output /absolute/path/new-threshold-directory
/usr/bin/python3 scripts/eat_nav_r6_thresholds.py freeze \
  --profiles /absolute/path/new-threshold-directory/index.json \
  --evaluations /absolute/path/closed-loop-validation-summary.json \
  --evaluation-binding /absolute/path/validation-evaluation/frozen-binding.json \
  --output /absolute/path/frozen-top-three.json
```

Derivation writes each exact profile JSON and an index with hashes and refuses an existing output. Production calibrated12 profiles from6,440 actual validation candidate records (236 observed failures,6,204 event-free2s outcomes). These branches share56 source episode groups and are not6,440 independent experiments. Validation tip/near-fall positives are absent; the failures here are stall outcomes. Event-free2s is never renamed successful2m climbing. For the actual calibrated production path use eat_nav_r6_calibrate_profiles.py with --dataset, --checkpoint, --records and a fresh --output; execution-records retains the executed command. The uncalibrated derive command above documents the pure derivation step only.

`rank_validation(rows)` consumes summary rows with unique id, positive equal `trials`, integer `fall_count`, `mean_stop_sec` **total commanded STOP plus unavailable residual time per trial for ranking**, `mean_progress_m`, `linear_windows`, `turn_windows`, `linear_stall_sec`, `turn_stall_sec`, and `split='validation'`. Caller must aggregate the same field/level/seed conditions and fixed evaluation duration across policies, preserve raw field/level metrics and macro means, and record failure/censor/remaining-time accounting. Unequal trial counts or test rows are rejected.

The ranking is strictly lexicographic:

1. fewer observed falls plus trials with an unknown fall outcome after censoring, with observed count, unknown count, denominator and conservative Wilson95% interval separately reported;
2. lower mean total commanded STOP plus unavailable residual seconds per trial;
3. equal weights0.2 for five normalized metrics: higher route progress, fewer linear/turn stall windows, fewer mean linear/turn stall seconds.

Third-priority scales are validation-only min/range, fixed once and saved (minimum positive range1e-9). Raw metrics and normalized components are both retained. No lower-priority gain compensates a fall or STOP disadvantage. Supplying frozen scales permits later independent reporting without refitting to test.

`freeze_top_three(profiles,evaluations)` requires at least10 pose profiles actually evaluated on validation before selecting3. It retains exact configurations, all validation rankings, scale origins/ranges, and a SHA256 of canonical JSON. CLI adds source file hashes before calculating its final freeze hash. Output is exclusive-create. Before ranking, the CLI requires all planned validation jobs completed, all35 field/level cells represented, at least30planned trials/cell, and positive common initial-valid support in every cell. Actual denominators and intervals remain explicit when initialization exclusions reduce counts. It binds the complete runtime-source inventory, model SHA, evaluation capacity, cohort and comparison profiles into the freeze digest. Only then may unused test and requested bag model inference run. Downstream queue and bag planner enforce this digest/model/source binding and never refit scales, thresholds or margins from test/bag outcomes.

## Verification

Unit tests exercise actual synthetic swept stair and wall rasters, all-unknown and out-of-map STOP, all-candidate predicted-roll rejection, normalized orientation/wrap behavior, bad schemas/shapes, event training support, hybrid event veto,10second rejection, lexicographic fall priority, required10-profile validation selection, real quantile dependence, sparse/constant-data refusal and Wilson bounds. These are implementation regressions, not physical evaluation trials.

## Production execution and interpretation

The ten pose profiles were chosen before observing production outcomes by derivation-index `(0,2,3,4,5,6,8,9,10,11)`. All12 remain documented. Old reference, direct2s-event and hybrid comparators are also evaluated. The event/hybrid comparison profile is pose_q3_2 in validation and remains the same in test; it is not silently replaced by the selected best pose profile.

Validation uses fixed initial-forward2m goal,60seconds,30-environment capacity, all7fields×levels0–4. K-Rail level4 has90planned parents per condition under the initialization addendum frozen before production. Completed paired jobs alone count toward results. Whole-batch policy-equivalent reuse requires every active planner choice to match throughout all actual source traces and preserves correlated-sample/timing disclosures. Any divergent choice requires an actual new rollout.

For each decision, the actual eval_sim source sets decision_all_excluded only at the2second replan boundary. Its trial sum is a decision count, not20Hz STOP exposure. Diagnostic reports divide by actual decision_count; protective_stop_sec is separately computed from every20Hz protective_stop flag. Goal hold and interrupted residual time remain separate.

Mesh vertex-height gaps include base/border geometry and must not be interpreted as traversable riser heights. The saved causal-DEM centreline/swept-shape metrics and known fixture dimensions are the shape evidence; a field/level label or overall mesh bound alone is insufficient to claim a climbable-centimetre limit. Successful reaching may also involve a detour, so goal-reaching rate is not automatically a measured per-riser climbing probability.
