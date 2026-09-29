# Direct episode-start replay diagnostic

Date: 2026-09-29. The original seven-state selection, candidate Paths, and prior artifacts were not changed. No candidate or formal rollout was run.

## Frozen method

The same 100-env scene and RL checkpoint were instantiated in a fresh process for each run. The full baseline replayed the entire recorded action prefix. The direct run performed the normal initial reset, advanced only the selected env's reset stream to the recorded episode number without physics steps, and replayed the recorded episode-local action prefix. Neither run used same-process snapshot restore or copied the recorded episode-start state. Both used the actual ROS2 follower bridge for a 40-step nominal Path. Numerical tolerances were frozen in [diagnostic-config.yaml](diagnostic-config.yaml) before the comparisons.

## Comparison

| State | Episode | Full/direct result | Earliest recorded divergence | 40-step XY RMS |
|---|---:|---|---|---:|
| Short state #42 | 0 | Pass | None | 0 m |
| boxes #7 | 3 | Fail | Episode-local prefix step 0, command vs collection: max absolute difference 0.306441, tolerance 0.0001 | 0.0465184 m, tolerance 0.0105 m |

For boxes #7, the full replay matched the logged anchor position and DEM history exactly; its prefix command maximum difference was zero. The direct run differed at the first recorded episode-local command. At the anchor its position difference from the logged anchor was 0.0670 m, command maximum difference 0.4106, and DEM-history SHA differed. Root/joint state, delay state, Actor observations and latent/action, Critic observation/value, and nominal contact/DEM/reward also differed. Both nominal runs reached 40 steps. The first *recorded* divergence is command at local step 0; the data do not establish that this was the first underlying state divergence.

The per-env initial-pose RNG and command RNG are persistent across episodes. Command RNG consumes draws during each earlier episode, while joint/noise and other state also depend on prior runtime. Advancing only reset calls does not reconstruct that history. This is a cause candidate consistent with the observed command mismatch, not a proof that no alternative direct reconstruction is possible. The tested direct method is rejected under the pre-frozen gate.

The initial boxes baseline attempt was interrupted by a diagnostic-runner bug that treated the expected step-399 timeout reset as unexpected. A later combined anchor/live attempt reached the anchor but failed at nominal start because its bridge client was not initialized. Those runner failures were recorded separately and are not counted as direct-replay comparisons. The successful full baseline subsequently took 1084.02 s; direct replay took 86.58 s.

## Outcome-blind cost-aware selection v2

After gate failure, a new selection was frozen from all 49,598 complete-only train windows, with goal distance at least 0.30 m, episode-local prefix at most 80 steps, at most one selected state per episode, and SHA-256 ranking seed 20260929. No candidate outcome, reward, Critic value, termination, or maneuver effect was used. All seven terrains had support; none required a relaxed condition. The 5,796 eligible windows had local-prefix min/median/p90/max 15/45/75/80; the seven selected states had 25/45/58/70.

| Terrain | New state ID | Episode key | Local prefix | Goal distance (m) |
|---|---|---|---:|---:|
| boxes | v2-000 | [0,52,3] | 25 | 0.700 |
| hf_pyramid_slope | v2-001 | [0,58,0] | 40 | 0.800 |
| k_rail_field | v2-002 | [0,55,2] | 35 | 0.601 |
| k_rail_step_field | v2-003 | [0,51,3] | 70 | 0.320 |
| pyramid_stairs | v2-004 | [0,26,2] | 45 | 0.389 |
| pyramid_stairs_inv | v2-005 | [0,93,3] | 45 | 0.516 |
| random_rough | v2-006 | [0,24,3] | 50 | 0.800 |

The exact state IDs, selection rule, per-terrain support, input SHA-256 values, and prefix distributions are in [selection-v2.json](cost-selection-v2/selection-v2.json). Its file SHA-256 is `e54024f51294513e4d48d632ee25f33fa39a8c8e5e546bfecb85d75c151e9dbe`. Its status is `awaiting_approval`. The old state selection and its outputs remain untouched. V2 rollout requires new approval.

## Evidence

- [Short-state comparison](runs/state-042/comparison.json)
- [boxes comparison](runs/state-007/comparison.json)
- Full/direct boxes summaries and 40-step traces under `runs/state-007/full/` and `runs/state-007/direct/`, each with input/output SHA marker
- [Diagnostic-runner failure record](boxes-anchor-live-runner-failure.json)
- [V2 stage status](cost-selection-v2/status.json) and `cost-selection-v2/runner.log`
- Tests: `PYTHONPATH=. /usr/bin/pytest -q -p no:launch_testing tests/eat_nav/test_direct_episode_inputs.py tests/eat_nav/test_direct_episode_compare.py tests/eat_nav/test_direct_episode_v2_select.py` — 9 passed

No Phase B Evaluator, NoMaD, follower control law, maneuver bank, loss, scheduler, Phase D/E, full collection, or robot trial was changed or started.
