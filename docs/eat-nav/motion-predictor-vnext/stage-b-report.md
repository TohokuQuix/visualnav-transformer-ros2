# EAT-Nav vNext Phase A/B gate report

## Decision

**Stop before Phase C.** The 13-Arc geometry and the ROS-free runtime protocol passed, but the intended full directional bank has not passed the two-second follower feasibility gate. No independent-StateGroup pilot, training, full collection or GUI integration was started.

## Preserved source audit

The sealed ROS-free crossover source yielded 12/12 complete 40-step body-frame motion labels. It contains only one independent StateGroup × seed and cannot support a validation/test claim. The old Goal-progress evaluator and checkpoints were not changed. [Machine-readable audit](source-12trial-audit.json).

## Arc definition and offline audit

The new bank samples constant-curvature arcs of **0.75 m arclength** at eight equal-arclength points including the origin. The angle names denote **change of tangent**, not endpoint bearing. For example, the +90° arc ends at approximately `(0.477, 0.477) m`, bearing +45°. All 13 Paths passed origin, finiteness, waypoint spacing, analytic curvature, symmetry and self-intersection checks. The ±180° paths retain distinct identities and opposite sides. The actual C++ PureFollower was called at the initial pose for every Path: 0/±30/±60/±90 began with positive forward command; ±120/±150/±180 began with zero forward command and a turn-in-place command. This offline check did not establish closed-loop feasibility. [Full geometry and initial-command audit](arc-bank-audit.json).

## One-seed follower feasibility sweep

The frozen Stage B schedule ran all 13 candidates on `random_rough` level 2, seed `2026101101`, four env tiles per candidate. All 13 processes and 52 trials completed 40 steps with finite 2 s labels and valid inner/outer seals. Each candidate's pre-Path root, joint, Actor pre-noise and four-frame DEM arrays matched the Direct process exactly; terrain/material identity matched. The actual RL Actor, noise/latency observation adapter, C++ PureFollower and physics were used without changing their control rules.

| Arc tangent change | Mean 2 s body XY magnitude (m) | Mean turn-in-place steps (of 40) | Mean integrated forward command (m) |
| ---: | ---: | ---: | ---: |
| 0° | 0.443 | 0.0 | 0.581 |
| −30° / +30° | 0.408 / 0.375 | 0.0 / 1.3 | 0.565 / 0.516 |
| −60° / +60° | 0.372 / 0.175 | 0.0 / 20.5 | 0.522 / 0.249 |
| −90° / +90° | 0.193 / 0.052 | 16.0 / 31.8 | 0.295 / 0.102 |
| −120° / +120° | 0.069 / 0.019 | 35.8 / 40.0 | 0.052 / 0.000 |
| −150° / +150° | 0.045 / 0.022 | 37.8 / 40.0 | 0.027 / 0.000 |
| −180° / +180° | 0.031 / 0.023 | 39.0 / 40.0 | 0.012 / 0.000 |

The right/left asymmetry is also present in angular response: for +120°, the mean integrated yaw command was +0.699 rad but mean observed yaw changed only +0.057 rad. For −120°, the corresponding values were −0.669 and −0.233 rad. This does not establish whether the disparity comes from terrain, contact, RL Actor response or a broader directional bias; it does show that the unchanged stack did not realize the intended 2 s motion for the positive Stage 2 arcs in these four tiles. No candidate reached its Path endpoint within 40 steps, including Direct; therefore endpoint arrival is not used as the gate.

The largest observed device-total GPU use was 5,761 MiB against the frozen 6,656 MiB cap. Minimum host available RAM was 80.69 GiB. Maximum wall time per process was 20.82 s; 13 process wall time summed to 266.4 s. All 13 raw attempt roots together occupied 12.65 MB. [Per-env outcomes and resource-linked seal references](stage-b-analysis-v2.json) are preserved outside Git under `/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/`.

## Recommended design decision

Keep the follower and Actor unchanged. Do not silently treat the 13 arcs as a feasible directional bank. A focused next gate should repeat the Stage 2 and +90° response on an independent terrain seed and a mirrored initial yaw/terrain condition, retaining the same 2 s horizon and the same path assignment. If the near-zero forward response persists, freeze a new candidate specification: either a reduced two-second Arc Bank, or an explicit turn-then-translate primitive/horizon with new labels and a new protocol. The current Phase C pilot config is marked blocked; it must not be run with an implicitly altered bank.

The one-seed sweep is evidence of **Path-dependent command and motion differences**, not model learning, independent generalization or successful navigation. The Motion Predictor model skeleton, split validation and collector are implemented and synthetically tested, but no checkpoint was trained.
