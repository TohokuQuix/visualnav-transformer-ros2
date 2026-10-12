# ROS-free G3 crossover v1: first-differential / seed 2026101101

## Decision

All three periods and all 12 trials are protocol-valid. This is one StateGroup and one seed, with four env IDs nested in each process. It demonstrates a repeatable Path-command-dependent response under matched pre-Path inputs; it does not establish statistical significance, general Path quality, or a learned evaluator's ability to rank Paths from DEM/Actor/Critic input.

## Frozen assignment and provenance

Assignment SHA: `8c910f1204502cdf749c2d265887c3b29026bcdef4647674b8548d49a6277d29`.

| Period | Env 0 | Env 1 | Env 2 | Env 3 | Authorized attempt manifest SHA |
|---:|---|---|---|---|---|
| 0 | Direct | Direct | Left | Right | `9b4b3f21ff9266b9883e5e4b23fa560215ecddf7619c6bc269addaa0d2e2df92` |
| 1 | Left | Right | Direct | Direct | `65b6b17584b8b629f26d58de9b9b9f8f6ea15ca786a982d5b2569c59cebd79c6` |
| 2 | Right | Left | Right | Left | `68a687d3e3300f6691050913b8a29888b69ef3e6a8ce580b1e2247534ff05286` |

Every env experienced each role once. Periods 1 and 2 used separate new persistent roots. The simulator source SHA remained `0132a4e4bcfc6fa651f27fba13420ec5286a43aae34e7f4be553fc08347c80f5`; the C++ follower core SHA remained `f92004fe2c3d958b166ce718260c8c46c90f0e236af4495c72c69bfee4c78ff7`. Only host staging, authorization, and the output-root mapping module were extended for later periods. No ROS follower, bridge, TF service, or Zenoh router was used in the control path.

All three pre-Path snapshot files have identical SHA `a046ad14dbe549bec2d1db04c5f698032d865cd850ae403cde7915f94b14222c`. Thus the frozen strict fields, including root/joint state, Actor observations/action, shared DEM source frames, noise, command, material fingerprints, episode length and reset count, matched exactly. The period 1/2 cross-period gates recorded zero strict failures and zero diagnostic differences. Native DEM history remains a separate diagnostic and is not claimed equal.

The same read-only analyzer version verified each period's original seal, manifest, assignment, full control trace, cross-period gate, native exit, cleanup, and simulator/follower SHA before producing the paired dataset. The [machine-readable 12-trial analysis](/home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-analysis-20261013/analysis.json) has SHA `be3304b612eb769ab2737cdeee2ee48706d104b775ba1b9cb5dd19977fe614aa`; its analyzer source SHA is `9d5ecf48275d0eccfec10366c7f718e1f557d5c43457de9483b4e3d09c7bd7dc`. The [SHA ledger](/home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-analysis-20261013/analysis-ledger.json) is stored outside the three sealed roots.

## Runtime and resource gates

Each batch passed pre-Path physical/shared-DEM/Actor/live-material gates, four independent core instances, exactly-once 50 ms control ordering, command injection and Actor observation consistency, same-episode trace validation, native exit 0, cleanup with zero owned residual, host/container SHA, and the 305-file host seal. All 12 Paths reached local Goal and completed the 40-step hold; none were censored or terminated. The common-goal metric is kept separate from local arrival.

| Period | Rows per env | Wall (s) | GPU peak (MiB) | Container anon+shmem peak (GiB) | Root size (MB) | Seal SHA |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 145 | 53.48 | 5,188.0 | 3.24 | 13.69 | `c398e8825a242d1ee40a4c83a9225ec1d36cfadd45e4de4a790fcda51edfe6a4` |
| 1 | 149 | 53.29 | 5,187.9 | 3.28 | 13.95 | `38b91f7d3c7f2fd1dde0dfb05717557af0167947809a70dc1b7bdf15f3b94113` |
| 2 | 169 | 59.21 | 5,183.9 | 3.28 | 15.36 | `0947485ac69679e534f0575818f1de6c9c917a490bedaca17886a546932d1eba` |

The frozen limits were 600 s/process, 7,680 MiB GPU device total, 6 GiB container anon+shmem, and 1 GB persistent storage. Seals were verified again after analysis.

Related static/synthetic tests: 61 passed. A repository-wide `pytest -q` attempt stopped during collection with 12 unrelated missing-dependency errors (`diffusion_policy`, `rclpy`, `geometry_msgs`) in the tinyvla environment; it did not report a failure in the targeted ROS-free crossover tests.

## Trial outcomes

Contact and torque are means over the fixed first 2 s. Tracking is the mean Path distance before arrival. Local and common-goal progress are different for curves because their endpoints differ.

| Period | Env | Path | Arrival (s) | Local progress 2 s (m) | Common-goal progress 2 s (m) | Tracking (m) | Recovery | Contact (N) | Abs torque |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | Direct | 2.20 | 0.494 | 0.494 | 0.022 | 0 | 8.21 | 173.4 |
| 0 | 1 | Direct | 2.05 | 0.537 | 0.537 | 0.021 | 0 | 9.66 | 256.3 |
| 0 | 2 | Left | 5.20 | 0.320 | 0.361 | 0.078 | 0 | 7.86 | 238.3 |
| 0 | 3 | Right | 3.00 | 0.348 | 0.368 | 0.035 | 0 | 7.23 | 185.7 |
| 1 | 0 | Left | 5.40 | 0.324 | 0.369 | 0.101 | 0 | 7.15 | 181.4 |
| 1 | 1 | Right | 2.80 | 0.385 | 0.413 | 0.041 | 0 | 9.11 | 236.5 |
| 1 | 2 | Direct | 1.95 | 0.574 | 0.574 | 0.025 | 0 | 12.17 | 225.2 |
| 1 | 3 | Direct | 2.25 | 0.483 | 0.483 | 0.024 | 0 | 6.54 | 202.6 |
| 2 | 0 | Right | 2.80 | 0.382 | 0.408 | 0.044 | 0 | 7.48 | 186.3 |
| 2 | 1 | Left | 6.40 | 0.308 | 0.348 | 0.088 | 0 | 11.26 | 274.5 |
| 2 | 2 | Right | 2.55 | 0.456 | 0.515 | 0.056 | 0 | 8.12 | 225.5 |
| 2 | 3 | Left | 5.85 | 0.355 | 0.399 | 0.093 | 0 | 6.65 | 234.5 |

## Signed candidate-minus-Direct differences

These comparisons pair the same StateGroup, seed, env ID, and identical pre-Path snapshot across periods. Positive arrival time or tracking error means the curve took longer or tracked farther from its own Path. Contact/torque signs are descriptive body-load changes, not a combined quality score.

| Env | Path | Arrival Δ (s) | Local 2 s Δ (m) | Common-goal 2 s Δ (m) | Tracking Δ (m) | Contact Δ (N) | Abs torque Δ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Left | +3.20 | −0.170 | −0.124 | +0.078 | −1.06 | +8.0 |
| 0 | Right | +0.60 | −0.112 | −0.086 | +0.022 | −0.73 | +12.9 |
| 1 | Left | +4.35 | −0.229 | −0.189 | +0.067 | +1.60 | +18.2 |
| 1 | Right | +0.75 | −0.152 | −0.124 | +0.020 | −0.56 | −19.8 |
| 2 | Left | +3.25 | −0.254 | −0.212 | +0.053 | −4.30 | +13.1 |
| 2 | Right | +0.60 | −0.117 | −0.058 | +0.032 | −4.05 | +0.2 |
| 3 | Left | +3.60 | −0.128 | −0.084 | +0.069 | +0.11 | +31.9 |
| 3 | Right | +0.75 | −0.135 | −0.115 | +0.011 | +0.70 | −16.9 |

Across the four paired env IDs, mean Left-minus-Direct arrival was +3.60 s, common-goal progress −0.153 m, and tracking +0.067 m. Mean Right-minus-Direct arrival was +0.675 s, common-goal progress −0.096 m, and tracking +0.021 m. These means are descriptive within a single seed; the four env rows are not four independent seed replicates.

## Heading and motion mechanism

The analyzer preserves each step's heading error, the C++ core's nominal heading-dependent forward cap, follower/injected forward command, measured forward velocity projected into the robot heading, and follower/root yaw rates. The table averages each trial's first 40 steps, then averages the four trials per Path. The nominal cap follows the unchanged core rule: zero at `|heading| ≥ 0.6 rad`; otherwise `0.3 × (1 − |heading|/π)` m/s. Recovery can override that nominal rule, but no recovery occurred here.

| Path | Mean absolute heading (rad) | Nominal cap (m/s) | Follower forward (m/s) | Actual forward (m/s) | Absolute commanded yaw rate (rad/s) |
|---|---:|---:|---:|---:|---:|
| Direct | 0.035 | 0.295 | 0.295 | 0.256 | 0.018 |
| Left | 0.454 | 0.214 | 0.214 | 0.186 | 0.227 |
| Right | 0.344 | 0.267 | 0.267 | 0.212 | 0.172 |

The Left Path incurred 6–10 first-window steps with zero nominal forward cap in all four envs, and Right incurred none. The curved Paths also differ in length: Direct 0.65625 m, Left/Right 0.66956 m. The observed progress gap is therefore consistent with a substantial geometry-to-heading-to-command effect. Differences between commanded and actual body motion remain visible in the time series; this pilot does not isolate terrain-specific embodiment quality from controller geometry.

## Next decision

Further StateGroup/seed pilots are useful to test whether candidate effects persist after accounting for geometry and the follower's heading cap, and whether contact/torque responses change with terrain. This 12-trial pilot alone does not justify Evaluator training or full collection. No additional seed, ROS experiment, training, or collection was started.
