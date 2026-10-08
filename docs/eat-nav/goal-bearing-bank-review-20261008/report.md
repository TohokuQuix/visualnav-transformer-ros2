# Goal-bearing / candidate-bank offline design review

**Decision:** `candidate_bank_follower_compatibility_insufficient`. This is an offline Path-design decision only; no candidate was run.

## Frozen inputs

- 8 anchors: 2 seeds × 4 DEM classes, selected by prior R3 rank. Independent source families: 6.
- 87 source files SHA-checked, including all selected four-frame DEM histories, Actor body states, traces and seals.
- Global goals are 0.60 m away at −20°, −10°, 0°, +10° and +20° in robot-local base frame.
- Within a StateGroup, anchor, four DEM frames, body state and global goal are fixed; only Path changes.
- The 560 anchor-specific definitions contain 65 unique Path hashes because a bearing/candidate geometry is deliberately reused across anchors; definitions are not independent trajectories.

## Global parameter selection

- Bank A peak lateral amplitudes: [0.04, 0.16, 0.2] m.
- Bank B endpoint offsets: [0.12, 0.18, 0.24] m; terminal control: `strict_goal_facing`.
- The supplied pools and lexicographic outcome-blind priorities were used across all 40 StateGroups.
- Bank B terminal alternatives: strict goal-facing max curvature 33.93 m⁻¹ / initial turn estimates 64; blended goal-connected max curvature 10.94 m⁻¹ / initial turn estimates 48.
- Strict goal-facing wins the frozen geometry-priority ordering despite its higher curvature. This is a design concern for review, not evidence of tracking.

## Bank comparison

| Metric | A common endpoint | B local subgoal |
| --- | ---: | ---: |
| Hard-valid candidates | 280/280 | 280/280 |
| Seven hard-valid StateGroups | 40/40 | 40/40 |
| Seven distinct geometry modes | 40/40 | 40/40 |
| Follower-compatible screen | 120/280 | 40/280 |
| StateGroups with ≥4 follower-screen candidates | 0/40 | 0/40 |
| Common-known terrain-separated pairs | 229 | 368 |
| StateGroups with ≥1 known-terrain-separated pair | 29/40 | 35/40 |
| Path-aligned class proxy matching source class | 83/280 known | 92/280 known |
| Max curvature (m⁻¹) | 4.44 | 33.93 |
| Initial turn-in-place estimates | 0 | 64 |
| Reverse estimates | 0 | 0 |
| Mean local-endpoint→global-goal residual (m) | 0.000 | 0.154 |

The 1.236 m⁻¹ criterion is a production-follower command screen, not a physical vehicle limit. The first-command estimate does not establish realized tracking.
Known-terrain separation uses only common known DEM samples; unknown-only differences are excluded.
Path-aligned three-band terrain class is a diagnostic proxy using the prior thresholds; it is not the frozen axis-aligned anchor classifier.
Known coverage for all four collision-bound proxy families and per-StateGroup separation are in bank-comparison.json.

## Pilot and forecast

- Proposed 4-anchor / 3-bearing pilot: 12 StateGroups; 3 direct repeats plus 6 non-direct Paths = 108 fresh processes for one bank.
- Across both banks the direct Path is identical within each StateGroup: 180 processes with three shared direct repeats, or 216 with three independent repeats per bank, before anchor-only replay.
- One-bank 108-process reference: 4.54 h point, 5.61 h P95+20%, 1.10 GB saved with 20% margin.
- Minimum bearing comparison with all four classes is 4 anchors × two bearings (−20°, +20°), 72 processes; the staged 0° then ±20° design is 36 + 72.
- External Zenoh, per-process SHA seal, cleanup and resource checks require separate rollout approval.

## Interpretation and limits

- Bank B retains a common global navigation goal. Its shifted endpoint leaves a structural global-goal residual; a short-horizon teacher may penalize this before replan. No Outcome exists here to quantify that bias.
- Geometry, swept area, known terrain profile and initial follower command are four distinct offline claims. Realized trajectory and Outcome remain unmeasured.
- Bearing-derived groups and candidate pairs are correlated. Split by terrain seed/source family; no 40-independent-StateGroup claim.
- Representative footprint metrics use frozen USD collision-bound corner rectangles, not a contact-response simulation.

## Decision reasons

- Bank A: 40/40 hard-valid groups; 40/40 geometry-mode groups; 0/40 groups have >=4 follower-screen candidates; 229 common-known terrain pairs
- Bank B: 40/40 hard-valid groups; 40/40 geometry-mode groups; 0/40 groups have >=4 follower-screen candidates; 368 common-known terrain pairs

Raw candidate definitions and 12 figures are in the output root; selected small artifacts are copied into Git for review. No follower, bridge, router, simulator, Outcome or training run was started.
