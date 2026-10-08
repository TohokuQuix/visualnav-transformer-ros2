# Local motion primitive P1: role-aware terrain gate

## Decision

`role_aware_terrain_support_insufficient`. The new taxonomy was frozen before reading the eight anchors (SHA-256 `b787e879bc0417132ab79b429d90ad44031dbdd075297b05d34aa6d798a1c60a`). The prior decision, `local_motion_primitive_terrain_support_fail`, remains valid under the prior class-uniform gate. This is a separate v1 role-aware gate, not a reinterpretation of that result.

The eight SHA-verified physical anchors and 40 frozen Path/DEM combinations have known three-band DEM support over arc 0–0.60 m. The five primitive Paths passed the prior hard-geometry and production-follower static screens at all eight anchors (40/40). No candidate Outcome was read for this gate.

| Primary role | First seed | Second seed | Total | Needed |
|---|---:|---:|---:|---:|
| R-differential | 1 | 1 | 2 | One from each seed |
| R-transition-common | 2 | 2 | 4 | One separate anchor |
| R-graded-common | 0 | 0 | 0 | One separate anchor |
| R-flat-control | 0 | 0 | 0 | One separate anchor |

The two differential anchors have 7 and 9 primary-interval terrain-separated candidate pairs, respectively. The four transition-primary anchors each have at least one transition-class Path with known onset at or before 0.60 m. No anchor has five graded-class Paths, and no anchor has five flat-class Paths. The two originally named `flat-only` anchors have four Paths classified differential by the frozen three-band diagnostic and one classified transition; their original inventory labels were not substituted for the new Path-level measurement. Likewise, the originally named `graded` anchors have a transition-class direct Path and differential-class curved Paths. The one Path classified graded on `first-transition` does not make all five Paths share a graded corridor.

The taxonomy intentionally interprets **common graded support** as all five Paths classified graded on known bands. This is a conservative interpretation of “shared graded/rough corridor,” fixed before applying the eight-anchor data. A different interpretation would require a separately versioned design decision and new gate; it is not applied post hoc here. The flat-control shortage alone is sufficient for failure under the user's explicit all-five-flat requirement.

## Stop boundary

No five-anchor selection or 35-process manifest was frozen. X1/X2, the external Zenoh router audit, simulator, bridge, follower and candidate rollout were not started. The empty selection and manifest artifacts explicitly record the gate stop. There is no P1 Outcome, duplicate, P/C/A/R/Y, relabeling or runtime result to interpret. The missing roles cannot be filled by reusing a differential anchor's secondary flag because each selected role requires a separate physical anchor.

The role grid and decision count are in [Path-level terrain grid](figures/01-path-terrain-roles.png) and [gate decision](figures/12-gate-decision.png). Machine-readable per-Path classes, known fractions, onset, slope, step, roughness, source identity and pair metrics are in `role-aware-terrain-gate.json`. `source-provenance.json` records the verified radial and settling ledgers, Path/geometry audit, follower and RL source checks. `artifact-sha-ledger.json` seals the gate outputs.

## Verification

The role classifier/selection tests passed (4/4). The previously relevant offline suite plus the initial three role tests passed (80/80); the additional identity-gate test passed afterward. The prior artifact ledger (26 files), eight physical anchor histories and immutable Path SHAs were verified before classification. No previous artifact was altered.

Further action would require a new design review for the absent graded and flat roles or a newly authorized measured anchor set. This report does not propose a threshold or Path change based on candidate Outcome.
