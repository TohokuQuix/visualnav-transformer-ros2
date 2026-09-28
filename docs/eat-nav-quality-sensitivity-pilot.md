# EAT-Nav post-training Quality sensitivity gate (2026-09-28)

**Gate: fail. Full collection was not started.** The trained Quality-enabled model does respond consistently to `q`, but the trajectory effect is much smaller than diffusion sampling variation under the approved paired-noise comparison. No quality loss, token, decoder, or model structure was changed after this result.

## Training and held-out conditions

The [extended 100-env pilot](/tmp/eatnav-isaac/docs/pilot-100env/extended-pilot-review.md) supplied raw DEM, pose, and Critic. Dataset windows and action statistics were built from the train terrain-seed split only; the ECDF was fitted on train raw Critic P10 only. Each policy started from scratch and used `context_size=3` (four DEM frames), eight 0.25-second XY-delta points, a shared DEM CNN, separate goal/Quality MLP tokens (Quality policy), the four-layer/four-head Transformer, and the lockfile-pinned `ConditionalUnet1D` plus 10-step DDPM scheduler. Neither model loaded pretrained NoMaD weights.

Both policies used 4,096 stratified train windows, 512 validation windows, batch size 64, eight-epoch maximum, early-stopping patience three, and validation-selected EMA weights. The Quality policy's validation diffusion loss improved from 0.832 to **0.322** at epoch 8; the Quality-disabled baseline improved from 0.899 to **0.315** at epoch 8. Both were still improving at the cap, so this is a small-training diagnostic, not converged full training. Curves and config are in [Quality history](pilot-quality-sensitivity/quality-history.json), [baseline history](pilot-quality-sensitivity/baseline-history.json), [Quality config](pilot-quality-sensitivity/quality.yaml), and [baseline config](pilot-quality-sensitivity/baseline.yaml).

Held-out test terrain seeds supplied 25 matched low/high Quality pairs, **50 fixed DEM-history + relative-goal inputs** across 42 episodes and five terrain types. Pair selection used the predeclared goal distance ≤0.1 m, four-frame mask IoU ≥0.8, height RMSE ≤0.05 m, cross-episode condition, and the same terrain type. The 50 inputs have 25 examples in q bin 0 and 25 in bin 3; 6,208 complete held-out windows were eligible. For each input, the DEM and goal were fixed while `q = 0.1, 0.3, 0.5, 0.7, 0.9, 0.95` was swept. Three diffusion seeds (11, 29, 47) were reused identically across q. All inference ran in eval mode. The comparison baseline is the same input at q≈0.5 under different seeds. The gate was predeclared as a paired q effect greater than both 1 mm and this same-q sampling variation.

## Measured response

| Model | Median 0.1→0.95 trajectory RMS difference | Same-q sampling RMS difference | Effect/sampling |
| --- | ---: | ---: | ---: |
| Trained Quality-enabled | 0.0094 m | 0.0957 m | **0.098** |
| Trained Quality-disabled | 0.0000 m | 0.0924 m | 0.000 |
| Untrained Quality-enabled control | 0.0050 m | 0.1325 m | 0.038 |

For the trained Quality model, the median 0.1→0.95 endpoint difference was 0.0140 m. Per-point differences grew from 0.0028 m at point 1 to 0.0140 m at point 8. Adjacent q steps changed the full trajectory by 0.0013 m (0.1→0.3), 0.0018 m (0.3→0.5), 0.0024 m (0.5→0.7), 0.0034 m (0.7→0.9), and 0.0010 m (0.9→0.95). The effect is numerically distinguishable from zero and exceeds the untrained model's 0.0050 m response, but remains about **one tenth** of sampling variation. The disabled baseline's paired-q response was exactly zero, supporting the paired-noise implementation.

The q response was directionally consistent: the median rank correlation between q and generated forward progress was −1.0, and 94% of input/seed comparisons had the prevailing sign. In the matched held-out data, high-Quality trajectories had median 0.083 m less forward progress, 0.098 m shorter path length, and 0.394 rad less turning than their low-Quality partners. The generated 0.1→0.95 changes had the same signs (−0.012 m forward progress, −0.012 m path length, −0.030 rad turning), but substantially smaller magnitudes. Directional agreement alone does not overcome the effect-size gate failure. Full per-input results and q-pair metrics are in the [machine-readable report](pilot-quality-sensitivity/report.json).

## Interpretation and next decision

The result does **not** support treating Quality conditioning as an effective control input yet. Plausible causes, not established fixes, are: eight epochs and 4,096 training windows were insufficient (validation had not plateaued); the current train set contains relatively few true failure episodes compared with timeout episodes; measured DEM/goal similarity leaves unobserved contact and robot state that can determine Critic quality; and the goal from the realized endpoint may already explain much of the trajectory, reducing additional information carried by q. Diffusion sampling variation is large relative to the learned q shift. The trained q response being only modestly larger than the untrained control reinforces this caution.

The approved gate requires reporting and reassessing the data/learning condition before full collection. No automatic loss or model redesign was made. The pilot checkpoints and full logs are archived outside Git at `/tmp/eatnav-sensitivity-pilot` (Quality checkpoint SHA-256 `fcaf3b9397f3344a0017a52e324ad11ce879c29d9737e1b924d3800293e00987`; baseline `c8187d5b57baf6ddfb6ab36029de7b644b8dcfa699553bb6fbe260942c7465fb`). The run used `diffusion-policy` commit `2356a7fbbfaf0ccccf9883b972b5a434d0f5ede3` from the existing lockfile and `diffusers 0.35.1` in the disposable GPU container.
