# EAT-Nav v2 coordinate audit and GS1–GS4 reanalysis

Status: **coordinate_bug_confirmed_analysis_recomputed**. The original `local_subgoal_execution_not_supported` result is retained as `superseded_pending_coordinate_audit`; original files were not overwritten. No simulator rollout, model training, threshold, candidate, Path, or raw trace changed.

## Frame finding

The old analyzer compared Isaac `root_state_w` (world) to Path and goal (environment-local `map`). `local_pose()` subtracts `scene.env_origins` before Path generation and bridge TF publication, while the trace saves `root_state_w` unchanged. Isaac `env_origins` is translation only, so canonical root XY is world XY minus the state origin. The four inferred origins were cross-checked across all 11 live runs per state and step-0 anchor invariants.

| State | Terrain | env origin world XY (m) | norm (m) | residual norm range after subtraction (m) |
|---|---|---:|---:|---:|
| v2-train-000 | k_rail_step_field | [11.850000381469727, -35.849998474121094] | 37.76 | 0.101–0.240 |
| v2-train-001 | pyramid_stairs | [28.0, -36.0] | 45.61 | 0.086–0.376 |
| v2-train-002 | pyramid_stairs_inv | [20.0, -20.0] | 28.28 | 0.079–0.195 |
| v2-train-003 | boxes | [-12.0, 12.0] | 16.97 | 0.033–0.236 |


The legacy 17–45 m errors are explained primarily by those origins. The residual is real candidate/path deviation and varies by candidate; no yaw rotation or axis swap was needed. See `coordinate-frame-spec.json` for the variable-by-variable frame table and source lines.

## Recomputed execution

- 0–2 s reach ≤0.2 m: logged 50.0%, ID 100.0%, near-OOD 100.0%, far-OOD 0.0%.
- 0–6 s reach ≤0.2 m: logged 75.0%, ID 100.0%, near-OOD 100.0%, far-OOD 0.0%.
- Logged/ID/near-OOD combined primary reach: 83.3%; state support 2/4. Far-OOD remains diagnostic only.
- Median candidate 0–2 s cross-track P95: 0.031 m. The Path ends at 2 s; the 2–6 s metrics describe goal arrival and subsequent motion, not Path tracking to a new 6 s plan.
- Post-reach departure beyond 0.2 m: 10/32 candidates. Catastrophic termination: 0.

## Other gates and scope

- GS2 P/C/R: R mode counts [2, 3, 3, 3]; old R pair distances/modes are unchanged by common origin translation. P and C modes and nominal variation are separately recorded.
- GS3: ΔV, advantage, drawdown, contact, tilt, stack, and effort agree exactly with old raw-series results. State pass count 4/4. Canonical goal and cross-track correlations are new read-only diagnostics.
- GS4 simulator schema: True. Real robot equivalent raw effort remains `NA / not verified`; simulator availability does not establish field availability or semantics on hardware.
- Counterfactual Evaluator collection decision: **local_subgoal_execution_not_supported**, subject to the existing staged approval process. This offline audit alone does not authorize starting collection.
- NoMaD retraining: not required by this coordinate bug alone. Far-OOD reach is 0/8 and could motivate a separate proposal, without changing this pilot's result.

The original GS1 implementation counted ID and near-OOD (16 samples), whereas this requested audit counts logged, ID, and near-OOD (24 samples). The reach threshold (0.2 m), 75% aggregate requirement, 3-of-4 state requirement, and 2 s primary horizon are unchanged. The corrected aggregate clears 75%, but only 2 states clear the state-level requirement. The old and new decisions have the same label for different reasons.

## Requested answers

1. **17–45 m legacy error:** yes; the state origin norms are 16.97, 28.28, 37.76, and 45.61 m. Remaining vector error is candidate-specific motion/Path deviation, not another shared offset.
2. **Frames:** goal starts in robot local, is rotated and translated into environment-local `map`; planned Path is cumulative robot-local waypoints transformed into that same `map`; Isaac root and post-root are world. Bridge TF is `map → body` using the origin-subtracted pose.
3. **0–2 s reach:** logged 4/8, ID 8/8, near-OOD 8/8, far-OOD 0/8; core logged/ID/near-OOD 20/24 = 83.3%.
4. **0–6 s reach:** logged 6/8, ID 8/8, near-OOD 8/8, far-OOD 0/8; core 22/24 = 91.7%.
5. **Goal classes:** ID and near-OOD reached consistently; logged reached later in two additional runs; far-OOD did not reach. Four states and two seeds per goal class are recorded in the JSON aggregates.
6. **Cross-track:** median of candidate-level 0–2 s P95 is 0.031 m; class-specific medians are logged 0.012, ID 0.016, near 0.043, far 0.047 m.
7. **After reach:** 10 candidates later moved beyond 0.2 m from goal. The per-run maximum deviation is saved in the corrected execution JSON.
8. **GS2:** corrected R pair distances and mode counts equal the old values exactly; all four states have at least two R modes. P/C mode counts are newly reported because the old output had distances but no P/C clustering verdict.
9. **GS3:** the raw ΔV, advantage, drawdown, recovery, tilt, contact, stack, termination, and effort verdict is unchanged: four of four states pass. Only canonical goal/cross-track correlations are added.
10. **Trajectory and outcome diversity:** yes, candidates create multiple realized R modes and distinct Critic-derived outcomes, while the precommitted GS1 state support gate fails.
11. **Evaluator collection:** do not advance under this gate result; a new design decision would require separate approval. No collection was started by this audit.
12. **NoMaD retraining:** not required to correct the analysis bug. Any retraining proposal should address a separately defined limitation, such as the far-OOD 0/8 result.
13. **Real robot inputs:** equivalent raw joint effort remains unverified; the simulator field availability and source-level frame mapping do not establish hardware timing, calibration, or equivalence.

## Preservation and figures

Old decision SHA: `94fda75cf17a9582e69981068886eab2135086a37339edd117c45fcf59100376`. New decision and full 48-run SHA ledger are separate. Figures: old_vs_corrected_goal_metrics.png, origin_vs_legacy_error_vector.png, goal_path_realized_overlay.png, world_vs_env_local_by_state.png, pcr_outcomes_gates.png. Frames, metres, origins, robot start, goal, and Path endpoint are labelled in the coordinate and overlay plots.
