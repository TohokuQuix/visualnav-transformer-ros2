# EAT-Nav vNext Phase A: motion and frame contract

Status: **Phase A source audit**. This contract does not promote the 12-trial pilot to an independent validation set.

## Provenance

- Base branch `exp/eatnav-ros2-gui-demo-terrain`, commit `11b5ef90ed18b8ce8dfc7bd050357cb72159afd8`.
- Baseline Goal-progress checkpoint SHA-256 `329341fbf56ccb6bbc2f05b3db6b143cc1a1e9553ba2796e3dec3c785f30e470`.
- RL Actor checkpoint SHA-256 `b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d`.
- Existing 12-trial ROS-free crossover has one independent StateGroup × seed. It can check label extraction and parity, but cannot supply separate train/validation/test groups.

## Existing implementation audit

| Contract | Source | Finding |
| --- | --- | --- |
| Pilot model | `src/visualnav_transformer/eat_nav/evaluator/model.py` | CNN maps eight flattened DEM channels to 64 features; Actor-state MLP to 32; Path MLP to 32; fusion currently emits one common-goal score. It accepts `[B,8,20,20]`, `[B,22]`, `[B,K,8,2]`. |
| DEM preprocessing | `src/visualnav_transformer/eat_nav/evaluator/preprocessing.py` | Four height/mask frames are flattened from `[B,4,2,20,20]` to `[B,8,20,20]`; valid-mask and resolution checks are required. |
| Current label | `src/visualnav_transformer/eat_nav/evaluator/labels.py` | A complete 40 × 50 ms window is already required. The current common-goal label uses tile-local root XY and a goal; terminal or short windows are censored. |
| Pilot dataset | `src/visualnav_transformer/eat_nav/evaluator/dataset.py` | Seal/SHA, Path frame, DEM frame, Actor pre-noise and group split checks exist, but the hard-coded crossover loader supports only the single existing StateGroup. It must not be used as evidence of held-out performance. |
| Current GUI Path | `src/visualnav_transformer/eat_nav/demo_contract.py:candidates_for_goal` | The old three-Path bank is rotated and scaled toward the Goal. vNext uses a separately versioned fixed body-frame bank; the old mode stays intact. |
| Runtime control | `scripts/eat_nav_demo_isaac.py` | At a 50 ms boundary: capture body pose and DEM, choose Path, call PureFollower, inject command, reconcile noise/latency, infer Actor, then physics. The Goal currently affects Path geometry and the old score. |
| Follower | `src/visualnav_transformer/eat_nav/p1_follower_core.cpp` | The actual core limits yaw rate to 1.6 rad/s, stops forward command at heading error ≥0.6 rad, and uses 0.5 m lookahead. Initial command tests can screen Arc geometry but do not establish closed-loop feasibility. |

## New model and label

The model consumes DEM `[B,4,2,20,20]`, pre-Path/pre-noise Actor `[B,22]`, fixed body-frame candidate Paths `[B,K,8,2]` and a boolean mask `[B,K]`. The Goal is not an input. Output is a finite `[B,K,2]` predicted 2 s root displacement in the **start** robot-base-yaw frame. The old checkpoint schema and model remain separate.

For a complete same-episode 40-step trace, let `p0` be the root XY immediately before Path activation, `p40` the post-physics root XY after control step 39, and `yaw0` the root yaw at activation. `delta_body = R(-yaw0) @ (p40 - p0)`. The terminal yaw must never enter this rotation. A terminal or missing step before 40 marks the label unavailable and preserves the termination reason; it is never filled with zero. Root positions may be world coordinates because the origin cancels in the difference. The Path must be in the same start-yaw frame; no Goal-driven rotation is allowed.

Input and output normalization are fitted to **train StateGroups only**. StateGroup, terrain seed, scene and randomization dependencies define split units. Candidates from one group cannot cross split boundaries. A held-out test group is reserved for the final evaluation; validation selects the checkpoint. Existing sealed 12 trials are retained as source evidence and are not relabeled in place.

For current map Goal `g_map`, root `p_map`, yaw `yaw`, compute `g_body = R(-yaw) @ (g_map - p_map)`. For each predicted displacement `d_i`, `score_i = ||g_body|| - ||g_body - d_i||`. Invalid candidates are masked. Goal changes recompute scores only; motion predictions are invariant when DEM, Actor state and Paths are fixed. Save both predicted XY and scores, with names distinct from the old Evaluator score.

## Timestamp and gate boundaries

- Source control period is 50 ms; 40 completed intervals are exactly 2 s. The last record is step 39, not step 40.
- Four DEM frames occur every 250 ms; the latest frame must align to pre-Path Actor state. Actual sensor update time is recorded separately from the logical control stamp.
- Phase A contract is ready for offline geometry implementation. The Arc Bank and paired rollout still require their own gates. The one-group pilot does **not** pass the independent-split gate needed for training claims.
