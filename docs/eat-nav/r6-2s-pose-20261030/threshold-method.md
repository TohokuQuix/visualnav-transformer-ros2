# R6 threshold candidates, policy geometry and selection

This document specifies implementation and statistical prerequisites. It contains no measured rollout result, selected profile or safety claim. Actual threshold values must be derived after the new validation dataset is available. Test outcomes and the real bag are not inputs to derivation or freezing.

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

Default support requirements are30 complete branch records,10 observed successes,5 observed failures. Success-envelope quantiles for pitch are0.50/0.65/0.80/0.95 and roll0.50/0.75/0.95:12 crossed candidates. Step quantile is their arithmetic mean. Each threshold is the smaller of the successful-outcome quantile and the failed-outcome tenth percentile, with a numerical positive floor0.001 in the respective units. This uses failures as well as successes; it remains an empirical proposal, not causal proof that lowering any one angle prevents failure.

Each profile writes explicit `pitch_deg`, `roll_deg`, `step_cm`, quantiles, source hash, sample counts, failure rates, confidence intervals and conditional shape limits. At least10 **distinct numeric triples** must exist; changing only configuration names does not qualify. If support or distinctness is insufficient, status is `insufficient_support`, profiles are empty and requirements/reason are saved. No55degree/30cm baseline is called learned.

Separate wall/stair/flat-ramp step limits require at least5 successes and3 failures in that shape group. Unsupported conditional values are `null`; that geometry yields `geometry_threshold_unsupported` and STOP rather than inventing an observed bound. Unknown map is always rejected regardless of conditional support. Nonzero pose-prediction uncertainty margins require an additional validation error fit; current generated margin is0 with status explicitly recording that no error margin has been fitted.

Wilson95% intervals accompany observed branch counts. Since candidate branches can share source groups, those branch intervals are marked descriptive with a correlation warning. A second interval uses source groups with any observed failure; it is a different estimand. Sparse field/level/shape cells must retain their counts and be reported as uncertain. Confidence intervals do not remove dependencies between terrain seeds or guarantee generalization.

## CLI and immutable selection

```bash
/usr/bin/python3 scripts/eat_nav_r6_thresholds.py derive \
  --records /absolute/path/validation-records.jsonl \
  --output /absolute/path/new-threshold-directory
/usr/bin/python3 scripts/eat_nav_r6_thresholds.py freeze \
  --profiles /absolute/path/new-threshold-directory/index.json \
  --evaluations /absolute/path/closed-loop-validation-summary.json \
  --output /absolute/path/frozen-top-three.json
```

Derivation writes each exact profile JSON and an index with hashes. It refuses an existing output. No actual validation records or real threshold artifacts have been produced by this implementation task.

`rank_validation(rows)` consumes summary rows with unique id, positive equal `trials`, integer `fall_count`, `mean_stop_sec` **total commanded STOP per trial**, `mean_progress_m`, `linear_windows`, `turn_windows`, `linear_stall_sec`, `turn_stall_sec`, and `split='validation'`. Caller must aggregate the same field/level/seed conditions and fixed evaluation duration across policies, preserve raw field/level metrics and macro means, and record failure/censor/remaining-time accounting. Unequal trial counts or test rows are rejected.

The ranking is strictly lexicographic:

1. fewer falls, with total denominator, rate and Wilson95% interval;
2. lower mean total commanded STOP seconds per trial;
3. equal weights0.2 for five normalized metrics: higher route progress, fewer linear/turn stall windows, fewer mean linear/turn stall seconds.

Third-priority scales are validation-only min/range, fixed once and saved (minimum positive range1e-9). Raw metrics and normalized components are both retained. No lower-priority gain compensates a fall or STOP disadvantage. Supplying frozen scales permits later independent reporting without refitting to test.

`freeze_top_three(profiles,evaluations)` requires at least10 pose profiles actually evaluated on validation before selecting3. It retains exact configurations, all validation rankings, scale origins/ranges, and a SHA256 of canonical JSON. CLI adds source file hashes before calculating its final freeze hash. Output is exclusive-create. Only then may the unused simulation test and requested bag be evaluated. The filesystem/hash does not enforce scientific discipline by itself: downstream evaluation must verify the frozen digest and must not refit scales, thresholds or margins using test/bag outcomes.

## Verification

Unit tests exercise actual synthetic swept stair and wall rasters, all-unknown and out-of-map STOP, all-candidate predicted-roll rejection, normalized orientation/wrap behavior, bad schemas/shapes, event training support, hybrid event veto,10second rejection, lexicographic fall priority, required10-profile validation selection, real quantile dependence, sparse/constant-data refusal and Wilson bounds. These are implementation regressions, not physical evaluation trials.

## Pre-execution correction: nonmonotonic failure types

Before any real threshold fit, a regression with low-angle stall failures showed that capping posture thresholds at the lower quantile of all failures wrongly assumes every failure is monotonic in tilt. The implemented v2 candidate derivation uses empirical quantiles of the joint observed success/failure magnitude distribution, reports both distributions and each envelope's conditional failure rate/CI, and leaves selection to actual validation rollouts. Low-angle stalls do not force all pitch/roll limits to near-zero. Constant or insufficient distributions still cannot produce ten distinct profiles.
