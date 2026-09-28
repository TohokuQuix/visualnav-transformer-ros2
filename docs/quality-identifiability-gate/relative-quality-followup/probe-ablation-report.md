# Relative Quality Probe ablation

The 61,948 frozen aligned windows, action normalization, source labels, three seeds and earlier Probe checkpoints were verified before training. All new runs started from scratch and selected checkpoints using aggregate validation loss only. The 29 terminal-endpoint windows are excluded equally.

| Probe | params | selected epochs | held-out MSE | physical RMS m | R² | RMS gain vs fixed q | decision |
|---|---:|---|---:|---:|---:|---:|---|
| probe_0 | 58,360 | 39,26,20 | 0.3692 | 0.0919 | 0.599 | 0.0% | reference |
| probe_abs | 58,360 | 31,30,20 | 0.3697 | 0.0911 | 0.598 | 0.9% | reference |
| probe_delta | 58,360 | 25,31,20 | 0.3337 | 0.0792 | 0.637 | 13.8% | pass |
| probe_recovery | 58,360 | 35,18,20 | 0.3547 | 0.0859 | 0.615 | 6.5% | fail |
| probe_return | 58,360 | 25,26,30 | 0.3656 | 0.0906 | 0.603 | 1.4% | fail |
| probe_adv | 58,360 | 21,14,20 | 0.3405 | 0.0821 | 0.630 | 10.6% | reference |
| probe_drawdown | 58,360 | 20,14,20 | 0.3461 | 0.0845 | 0.624 | 8.0% | reference |
| probe_delta_drawdown | 58,424 | 20,21,20 | 0.3310 | 0.0792 | 0.640 | 13.8% | pass |
| probe_relative_triplet | 58,488 | 20,21,20 | 0.3315 | 0.0790 | 0.640 | 14.0% | pass |
| probe_multi_reward_free | 58,552 | 30,21,20 | 0.3284 | 0.0779 | 0.643 | 15.2% | pass |
| probe_multi_no_v0 | 58,488 | 13,12,20 | 0.3290 | 0.0785 | 0.642 | 14.5% | pass |
| probe_multi | 58,552 | 25,21,20 | 0.3240 | 0.0772 | 0.648 | 16.0% | reference |

## Paired comparisons and controls

- **probe_delta**: fixed-q episode CI [0.030272522551496053, 0.04072912336735454]; absolute-q episode CI [0.03043852273042053, 0.041637074449815806]; matched trajectory RMS gap 0.111 m, CI [0.05243209826349953, 0.16947622057008777]; terrain shuffles 100 (all beaten=True); minimum leave-one-episode-out gain 0.0345; decision **pass**.
- **probe_recovery**: fixed-q episode CI [0.010020746424224217, 0.01855062401415409]; absolute-q episode CI [0.010354113907205923, 0.019101142693917387]; matched trajectory RMS gap 0.069 m, CI [0.013654834263861214, 0.11743881290719009]; terrain shuffles 100 (all beaten=True); minimum leave-one-episode-out gain 0.0140; decision **fail**.
- **probe_return**: fixed-q episode CI [0.0014477749057471765, 0.005702119035620031]; absolute-q episode CI [0.0020067690393537254, 0.006258309810348028]; matched trajectory RMS gap 0.022 m, CI [-0.03277683548138051, 0.07012738578242962]; terrain shuffles 10 (all beaten=True); minimum leave-one-episode-out gain 0.0032; decision **fail**.
- **probe_delta_drawdown**: fixed-q episode CI [0.032795590460950895, 0.043709789139491556]; absolute-q episode CI [0.03313012343074195, 0.04428149570732043]; matched trajectory RMS gap 0.119 m, CI [0.060363450524102726, 0.17765999337446903]; terrain shuffles 100 (all beaten=True); minimum leave-one-episode-out gain 0.0372; decision **pass**.
- **probe_relative_triplet**: fixed-q episode CI [0.032390388402631594, 0.043098021572626675]; absolute-q episode CI [0.03269193773185968, 0.04380032059096688]; matched trajectory RMS gap 0.135 m, CI [0.08226459744811052, 0.19066901053397964]; terrain shuffles 100 (all beaten=True); minimum leave-one-episode-out gain 0.0367; decision **pass**.
- **probe_multi_reward_free**: fixed-q episode CI [0.03428086046150782, 0.047268952861141925]; absolute-q episode CI [0.035037390282632704, 0.04774702249059479]; matched trajectory RMS gap 0.117 m, CI [0.052421378826971704, 0.1740663174655369]; terrain shuffles 100 (all beaten=True); minimum leave-one-episode-out gain 0.0396; decision **pass**.
- **probe_multi_no_v0**: fixed-q episode CI [0.03467681058106265, 0.04589400190311049]; absolute-q episode CI [0.03496476858117595, 0.04648359678935298]; matched trajectory RMS gap 0.092 m, CI [0.036390258558421505, 0.14075295264700685]; terrain shuffles 100 (all beaten=True); minimum leave-one-episode-out gain 0.0391; decision **pass**.

On identical held-out windows, delta beat advantage (episode CI [0.0032582800394222434, 0.009509721689283818]) and drawdown (CI [0.008737861731654665, 0.015834575802196695]). Reward-free multi was worse than the existing advantage-containing multi (old-minus-new MSE CI [-0.006188193180452129, -0.002821988526002715]); adding V0 to the Critic-only triplet improved prediction (CI [0.0009982453829374476, 0.0054064429583212115]). The advantage-containing model with V0 also beat its no-V0 variant (with-V0-minus-no-V0 MSE CI [-0.008614211707313751, -0.0023919634438078013]). Reward-free multi beat scalar delta on matched held-out windows (delta-minus-multi MSE CI [0.002522823711240207, 0.008270798173181982]).

## Questions requested for Phase A

- delta_beats_absolute_q: True
- delta_beats_advantage: True
- delta_beats_drawdown: True
- reward_free_multi_at_least_existing_multi: False
- reward_free_multi_point_rms_vs_existing_multi_pct: -0.8954291174413074
- v0_improves_reward_free_vs_triplet: True
- v0_improves_adv_multi_vs_no_v0: True
- recovery_single_pass: False
- return_v0_spearman: 0.755010403381817
- advantage_v0_spearman: -0.21936279866538477
- return_initial_state_confounding_supported: True
- scalar_vs_multi_parameter_difference: 192
- reward_free_multi_vs_delta_mse_gain_ci95: [0.002522823711240207, 0.008270798173181982]
- reward_free_multi_beats_delta_ci: True

## Terrain support and paired MSE gain

| Terrain | count | probe_delta | probe_recovery | probe_return | probe_delta_drawdown | probe_relative_triplet | probe_multi_reward_free | probe_multi_no_v0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| boxes | 1242 | 0.0397 | 0.0152 | 0.0082 | 0.0381 | 0.0373 | 0.0417 | 0.0412 |
| k_rail_step_field | 1242 | 0.0296 | 0.0115 | 0.0026 | 0.0346 | 0.0347 | 0.0382 | 0.0347 |
| pyramid_stairs | 1861 | 0.0454 | 0.0138 | 0.0038 | 0.0495 | 0.0486 | 0.0527 | 0.0507 |
| pyramid_stairs_inv | 1242 | 0.0255 | 0.0137 | -0.0018 | 0.0289 | 0.0282 | 0.0291 | 0.0305 |
| random_rough | 621 | 0.0288 | 0.0228 | 0.0064 | 0.0302 | 0.0310 | 0.0312 | 0.0365 |

All 36 train/validation curves are saved in [probe-ablation-curves.json](probe-ablation-curves.json).

The scalar and vector Probes share the exact DEM encoder, Quality-token path, trajectory head and optimizer. Vector width adds 64, 128 or 192 first-layer parameters; this small difference alone does not prove that dimensionality, rather than capacity, caused the gain. A positive post-hoc Probe result does not establish online candidate ranking. Train-only thresholds were used for matched vector controls. Different metrics select different matched pairs, so pair RMS gaps are not a head-to-head ranking; the Probe comparison uses identical held-out windows. All 100-shuffle expansions followed the frozen boundary rule.

Common train/validation/held-out terrain support: ['k_rail_step_field', 'pyramid_stairs', 'pyramid_stairs_inv', 'random_rough']. Terrain-specific NA cells retain 'insufficient support'; no all-seven-terrain claim is made.

## Figures

- [Probe ablation](plots/probe-ablation.png)
- [Scalar versus multi](plots/scalar-vs-multi.png)
- [Reward-free versus advantage](plots/reward-free-vs-advantage.png)
