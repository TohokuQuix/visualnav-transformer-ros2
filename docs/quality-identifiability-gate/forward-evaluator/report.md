# Phase B: Forward Relative-Outcome Evaluator

## Decision

**Deployable trajectory signal: present.** On the frozen 6,208 held-out windows, GT (DEM history, relative goal, logged candidate trajectory) beat G and the simple baselines for all three scratch seeds. The specified predictive, trajectory benefit, common-support, seed, and matched-neighborhood gates passed. The next **oracle best-of-K simulator rollout is recommended for a separate decision**, and was not started.

**Body-state and actor-latent benefit: unavailable.** The 360,000 raw records contain no exact 22-D pre-action `obs["policy"]` or actor latent. Neither can be reconstructed from the saved fields and policy observation noise. GB, GBT, GL, GLT and corresponding V0 variants were not trained. This result does not establish embodiment-aware benefit.

## Inputs and method

The six frozen Phase A SHA checks passed (manifest, relative labels, Probe code, and train/validation/test NPZ). The aligned counts remained 49,576 / 6,164 / 6,208; the 29 terminal endpoint windows remained excluded. Source audit: [source-sha-audit.json](source-sha-audit.json). Split seed, target definition, reward alignment, gamma and train-only target statistics were unchanged.

The logged candidate is eight robot-local XY increments at 0.25 s intervals; cumulative sum gives the eight waypoints. The DEM history has shape 4×2×20×20. No future reward, Critic, pose or termination field entered a model. The fixed CNN/MLP model has 43,564 parameters in every variant; disabled branches receive zero inputs. Four variants (G, GT, GV0, GTV0) used the same training schedule and seeds 11, 29, 47. Checkpoints were selected only by aggregate validation standardized MSE; all runs stopped by patience before epoch 60 and none met the underfit condition. The 12 runs took 40.8 s in the Docker environment. The smoke took 1.73 s and peaked at 34.7 MB PyTorch allocated VRAM.

V0 is privileged: the 267-D Critic observation includes simulator-only height scan, contact and control-latency information. GV0/GTV0 are upper bounds, not deployable evidence. The saved actor observation config has no history, empirical normalization is disabled, and policy observation corruption is enabled. The saved policy fields are angular velocity, projected gravity, velocity command, flipper fold mode, flipper position, mean track velocity and joint effort; last action and height scan are disabled. Exact field ordering and the noisy values cannot be recovered from this log.

## Held-out results

Aggregate MSE uses train-standardized delta, advantage, drawdown and recovery, averaged across heads. Lower is better. The R² columns are the three primary heads for GT.

| Seed | G MSE | GT MSE | GTV0 MSE | GT delta R² | GT advantage R² | GT drawdown R² | GT delta Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.608 | 0.440 | 0.407 | 0.324 | 0.300 | 0.240 | 0.649 |
| 29 | 0.616 | 0.447 | 0.388 | 0.338 | 0.286 | 0.183 | 0.683 |
| 47 | 0.612 | 0.469 | 0.424 | 0.285 | 0.258 | 0.184 | 0.620 |

Simple baseline MSE: train mean 0.626, goal+DEM roughness linear 0.613, trajectory statistics linear 0.586. The GT advantage over G in paired standardized loss was 0.167 [0.142, 0.194], 0.169 [0.143, 0.197], and 0.143 [0.121, 0.167] for seeds 11, 29, 47 (episode-cluster 95% CI). The GT advantage over trajectory statistics linear was 0.146 [0.120, 0.171], 0.140 [0.111, 0.168], and 0.117 [0.096, 0.140]. V0 further improved GT by 0.034–0.059 MSE, but cannot support the deployable gate. Terrain train mean was retained as a secondary baseline. A body-observation MLP baseline was unavailable for the same input-provenance reason as GB/GBT.

Common support across train, validation and held-out is `k_rail_step_field`, `pyramid_stairs`, `pyramid_stairs_inv`, and `random_rough`. GT beat G on all four for every seed. Held-out also contains `boxes`, which has no validation support. Held-out lacks `hf_pyramid_slope` and `k_rail_field`; this does **not** demonstrate generalization across all seven terrains. Full per-head raw MAE/RMSE, standardized MSE and episode CI, R², Spearman, calibration, terrain, seed and paired comparisons are in [heldout-results.json](heldout-results.json). See [summary figure](phase-b-summary.png).

The return target was kept diagnostic only, outside the four-head training loss. Its held-out Pearson correlation with V0 is 0.743, supporting the prior concern about initial-state dependence; see [return-diagnostic.json](return-diagnostic.json).

## Matched-neighborhood ranking

For the pre-frozen Phase A high-difference, cross-episode near-neighborhood pairs, GT delta ordering accuracy was 0.939 [0.848, 1.000], 0.956 [0.867, 1.000], and 0.930 [0.823, 0.992] across seeds (114 non-tied pairs; bootstrap resamples both episodes in each pair). These pairs were selected by the frozen geometry and relative-label criterion, independent of the forward predictions. This is a **near-neighborhood, on-policy ranking proxy**, not a comparison of two counterfactual trajectories from the same state. Separate metrics can use different pair sets, so their accuracies should not be directly ranked.

## Interpretation and limits

The dataset's relative goal is derived from the logged future trajectory endpoint, clipped to 0.8 m. In held-out, it exactly equals the candidate endpoint for 80.2% of windows. This goal construction and the on-policy candidate source limit transfer of the observed GT advantage to newly generated candidates. The G comparison still used the same goal, and the trajectory statistics linear control was much weaker than GT, but neither resolves the counterfactual question. An oracle rollout would test it directly.

The first held-out serialization attempt failed because terrain-constant baseline predictions yielded undefined Spearman. A fixed-checkpoint evaluation-only rerun wrote undefined rank as `null`. The comparison and pair CI implementations were then corrected to the specified paired episode bootstrap, with both pair episodes resampled. No checkpoint, input, threshold or architecture was changed after held-out inspection, and held-out never selected a model. The runner log records these evaluation-only corrections.

No new simulator rollout, full collection, Diffusion retraining, model/loss change, or EAT policy integration occurred.
