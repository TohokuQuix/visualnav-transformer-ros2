# EAT-Nav relative trajectory quality diagnostic

## Main result

Matched pairs rank **Critic delta** first for trajectory RMS separation (0.111 m cross-minus-control); this is a diagnostic association, not a causal ranking.

## Scope and timing

Frozen complete windows: train 49,598, validation 6,171, held-out 6,208. Primary aligned windows: {'test': 6208, 'train': 49576, 'val': 6164}. The 29 terminal-endpoint windows have auxiliary terminal returns but no complete bootstrap Critic and are excluded equally from all Probe comparisons.
The collector records a pre-action Critic, then a post-transition reward/pose. We anchor at the recorded post-step pose and use the next record Critic. Each 2 s window spans 40 control rewards with gamma=0.99 per control step. True terminal bootstraps zero; timeout uses the PPO time-limit correction. Reward decomposition was not saved. All future-derived labels are post-hoc and unavailable to online inference.

## Held-out matched neighborhoods

| Metric | Cross | Control | RMS gap (m) | episode bootstrap 95% CI |
|---|---:|---:|---:|---|
| absolute q | 91 | 111 | 0.027 | [-0.017, 0.070] |
| Critic delta | 114 | 115 | 0.111 | [0.052, 0.169] |
| H-step advantage | 117 | 116 | 0.077 | [0.030, 0.123] |
| drawdown | 115 | 116 | 0.092 | [0.035, 0.154] |
| recovery | 114 | 117 | 0.069 | [0.014, 0.117] |
| H-step return | 101 | 113 | 0.022 | [-0.033, 0.070] |
| Q-multi | 115 | 112 | 0.109 | [0.048, 0.163] |

Thresholds came only from train. Pairs share terrain and similar DEM/goal; no episode or window is reused within each comparison. Different metrics select different pairs, so gap sizes are not a direct head-to-head causal comparison.

## Confounding

| Metric | goal Spearman | DEM roughness Spearman | progress Spearman | goal+DEM held-out R² |
|---|---:|---:|---:|---:|
| absolute q | -0.348 | -0.261 | -0.150 | 0.124 |
| Critic delta | 0.061 | 0.056 | -0.006 | 0.002 |
| H-step advantage | -0.106 | 0.014 | -0.078 | 0.007 |
| drawdown | -0.201 | 0.053 | -0.126 | 0.013 |
| recovery | 0.314 | 0.056 | 0.141 | 0.053 |
| H-step return | -0.279 | -0.256 | -0.122 | 0.064 |

## Validation-selected deterministic Probes

Mean-trajectory validation MSE: 1.0498. Mean-trajectory held-out MSE: 0.9202.

| Probe | held-out MSE | physical RMS (m) | R² | ΔR² vs fixed q | RMS gain vs fixed q | status |
|---|---:|---:|---:|---:|---:|---|
| probe_0 | 0.3692 | 0.092 | 0.599 | 0.000 | 0.0% | reference |
| probe_abs | 0.3697 | 0.091 | 0.598 | -0.001 | 0.9% | reference |
| probe_adv | 0.3405 | 0.082 | 0.630 | 0.031 | 10.6% | pass |
| probe_drawdown | 0.3461 | 0.085 | 0.624 | 0.025 | 8.0% | pass |
| probe_multi | 0.3240 | 0.077 | 0.648 | 0.049 | 16.0% | pass |

The scalar Probe uses the same Quality MLP/token path for fixed q=0.5 and real q. Multi adds only 192 parameters. The train-only action scale and identical seeds/order/optimizer were used. Epoch selection used aggregate validation only; held-out was evaluated once.

## Gate details

- **probe_adv: pass**. Episode CI for held-out MSE gain [0.023740845720567404, 0.034203610182656094]; permutations 100, all beaten=True; checks={'positive_episode_ci': True, 'physical_rms_gain_5pct': True, 'delta_r2_0.02': True, 'beats_every_permutation': True, 'better_than_absolute_q': True, 'common_terrain_direction': True, 'matched_trajectory_difference': True, 'not_single_episode_dependent': True, 'not_goal_dem_proxy_only': True}; underfit=False.
- **probe_drawdown: pass**. Episode CI for held-out MSE gain [0.019035373191051426, 0.027087073204751823]; permutations 100, all beaten=True; checks={'positive_episode_ci': True, 'physical_rms_gain_5pct': True, 'delta_r2_0.02': True, 'beats_every_permutation': True, 'better_than_absolute_q': True, 'common_terrain_direction': True, 'matched_trajectory_difference': True, 'not_single_episode_dependent': True, 'not_goal_dem_proxy_only': True}; underfit=False.
- **probe_multi: pass**. Episode CI for held-out MSE gain [0.03865677592694191, 0.05189860986697514]; permutations 100, all beaten=True; checks={'positive_episode_ci': True, 'physical_rms_gain_5pct': True, 'delta_r2_0.02': True, 'beats_every_permutation': True, 'better_than_absolute_q': True, 'common_terrain_direction': True, 'matched_trajectory_difference': True, 'not_single_episode_dependent': True, 'not_goal_dem_proxy_only': True}; underfit=False.

Terrain-specific results are NA when a held-out terrain has fewer than the configured q-bin support. The train/validation/held-out common support is ['k_rail_step_field', 'pyramid_stairs', 'pyramid_stairs_inv', 'random_rough']; held-out supports ['boxes', 'k_rail_step_field', 'pyramid_stairs', 'pyramid_stairs_inv', 'random_rough']. No claim is made for all seven terrains. Timeout is not labeled as locomotion failure. A single tilt episode cannot support independent-window significance.

## Interpretation and next step

The rank and Probe gates above measure associations in the existing on-policy log. A positive relative label result would motivate an explicitly approved oracle best-of-K candidate rollout from shared simulator states. It does not authorize a new rollout, full collection, or model/loss change. If no candidate passes, the log is insufficient to conclude that a scalar relative score is useful; counterfactual candidate rollouts may be needed. Reward decomposition was not saved, so progress reward and body stability cannot be separated.

## Figures

- [relative-quality-distributions](plots/relative-quality-distributions.png)
- [terrain-confounding](plots/terrain-confounding.png)
- [goal-confounding](plots/goal-confounding.png)
- [quality-vs-trajectory-difference](plots/quality-vs-trajectory-difference.png)
- [probe-comparison](plots/probe-comparison.png)
- [advantage-drawdown-recovery](plots/advantage-drawdown-recovery.png)
