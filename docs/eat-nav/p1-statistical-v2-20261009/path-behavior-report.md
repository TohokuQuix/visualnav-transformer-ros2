# P1 v2: read-only Path behavior and Outcome analysis

**Decision:** Investigate the Path-to-command-to-motion response before collecting more P1 data. The 2 s progress deficit is mostly a shorter traveled distance, not the lateral-distance penalty of an intentional detour. The existing records cannot assign that shorter travel to follower speed scheduling, Actor response, traction, or contact. This is a descriptive decision from **two independent batch seeds per StateGroup**, not a Path ranking.

## Data availability

| Quantity | Status | Exact source or limit |
| --- | --- | --- |
| Root XY, quaternion/heading, linear and angular velocity | Direct | `root_state` and `post_root_state`, every control step; velocity uses Isaac root-state components 7–12. |
| Actual XY, traveled distance, forward/lateral displacement, goal progress, roll/pitch | Derivable | Root state and frozen initial-frame goal `[0.75, 0]`. Roll/pitch are orientation measurements, not a safety score. |
| Frozen candidate/direct Path and geometry | Direct/derivable | `path-publication.json` contains the exact published local-map polyline. `initial-state.json` supplies each tile origin; the root provides the initial pose. |
| Follower command, total reward, Critic value, Actor observation/action, logical timestamp, identifiers | Direct | Per-step trace and frozen assignment. Command is `[forward, lateral, angular]`; production follower's internal target/error is not logged. |
| Per-step flipper joint position/velocity/effort, reward components, contact forces/collisions | Unavailable | Initial joint/actuator snapshot exists, but no per-step physical or reward decomposition. Actor action is not substituted for joint motion. |

All 56 sealed trials have 120 contiguous steps and no physical termination. Existing 2 s discounted reward and global-goal progress were recomputed exactly from the trace and checked against the frozen Outcome file. The 6 s measures below are **post-hoc diagnostics** and do not change the 2 s Outcome or its practical margins.

## Geometry and actual motion

The direct Path is 0.65625 m from its first to last waypoint. The ±10° family is 0.65954 m (+0.5%) and reaches 0.130 m nominal lateral deviation; the ±20° family is 0.66956 m (+2.0%) and reaches 0.257 m deviation. These are genuinely different published geometries. Discrete curvature and heading-change values are in `per-trial.json`.

The fixed, outcome-blind example is `first-differential / batch 0`. The ±20° Paths turn visibly, and follower angular commands increase; robot trajectories remain much closer to the direct corridor over the first 2 s. Across all alternative trials, mean absolute lateral position at 2 s is 0.015 m while the Path lateral target interpolated at the *actual attained forward position* averages 0.040 m in absolute value. The nearest-Path distance is 0.041 m for alternatives versus 0.029 m for direct, a descriptive difference of 0.012 m. Because Paths begin roughly 0.1 m ahead and some trials barely advance, this nearest-polyline statistic alone is not a follower-failure verdict.

Across the 16 StateGroup × alternative contrasts, candidate-minus-batch-direct mean 2 s global-goal progress is −0.068 m (all 16 point estimates negative). Mean forward displacement difference is −0.068 m; mean traveled-distance difference is −0.063 m (all 16 negative). The progress-minus-forward contrast averages +0.0001 m, with at most 0.0021 m absolute residual per alternative. Thus the observed progress loss comes from less forward travel, while the lateral penalty in the exact goal-distance metric is small at 2 s. The nominal Path-length increase alone is also much smaller than the observed movement loss.

The follower command responds to Path geometry: all 16 alternatives increase mean absolute commanded angular speed (mean contrast +0.114 rad/s) and lower mean commanded forward speed (mean contrast −0.025 m/s). Actual heading and lateral motion change in the intended sign on average, but much less than the full Path offset in the first 2 s; response differs by StateGroup and side. This links Path geometry to command and measured motion without identifying a unique physical cause.

| StateGroup | 2 s progress Δ (m) | 2 s traveled distance Δ (m) | 2 s discounted reward Δ | 6 s progress Δ (m), post-hoc |
| --- | ---: | ---: | ---: | ---: |
| first-differential | −0.095 | −0.094 | −0.075 | +0.008 |
| second-differential | −0.075 | −0.068 | −0.176 | +0.002 |
| first-graded | −0.066 | −0.055 | −0.094 | +0.040 |
| second-graded | −0.037 | −0.036 | −0.003 | −0.130 |

Each table cell averages four alternative Path contrasts; each contrast first subtracts the three direct slots **within each batch**, then averages the two batches. These 16 contrasts are not 16 independent replications. Eleven of 16 alternatives have the same 2 s progress-difference sign across both batches; the other five change sign. Observation noise is active and unpaired across Paths.

## Locomotion and horizon interpretation

Total 2 s discounted reward decreases in 12/16 contrasts and increases in 4/16. Root roll/pitch RMS and angular-speed RMS have mixed directions, and neither establishes a consistent body-disturbance pattern. Reward components, per-step flipper motion, effort, and contact are unavailable. A locomotion-quality difference is possible, but the present traces cannot distinguish it from lower commanded forward speed or terrain response. No stability or safety score was invented.

At 6 s, the candidate-minus-direct goal-progress contrast is positive for 8 alternatives and negative for 8. Eight alternatives reverse their negative 2 s contrast by 6 s. The first three StateGroups largely catch up; `second-graded` remains behind by 0.130 m on average. Longer-horizon behavior therefore changes the practical reading of the initial deficit, but it does not replace the approved primary endpoint.

## Research decision and next smallest test

The evidence is **mixed A/D** under the requested decision framework: Path geometry reaches the follower command, but intended lateral displacement is small and tracking/locomotion attribution is incomplete. A pure geometric-detour explanation (B) is not supported by the 2 s kinematics; a distinct locomotion-quality effect (C) is not established. The smallest next experiment is a bounded, independent direct-versus-±20° Path response check on fixed anchors with per-step production-follower target/cross-track/speed-limit state and actual joint/contact signals, while preserving the P1 primary Outcome. It should test where forward speed and lateral response diverge before scaling collection. The recommendation follows observed command/motion differences, not merely wide intervals.

## Artifacts and provenance

- Source view: `/tmp/eatnav-p1-statistical-v2-recovered-view-20261009/` (read-only links to seven original sealed batches and the recovered sealed batch).
- Frozen Outcome: `/tmp/eatnav-p1-statistical-v2-recovery-20261009/analysis/outcomes.json`.
- New analysis: `/tmp/eatnav-p1-path-behavior-readonly-20261009-final/` (`per-trial.json`, `per-path.json`, `ledger.json`, four PNG figures).
- Analysis code: `scripts/eat_nav_p1_path_behavior_readonly.py`. The assignment SHA remains `709fa5ccc98c152e75dc24cbd659dc8ea249c740efeb14bdabe31b5cc5affc69`.
- No simulator, ROS bridge, follower process, new rollout, or learning was run for this analysis.
