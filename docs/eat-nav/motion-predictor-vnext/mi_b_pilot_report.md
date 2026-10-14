# MI-B Pilot: sealed MI-A motion learning

**Pilot decision: Pass under the predeclared descriptive criterion. Formal MI-B Gate: not defined, so no formal Pass is claimed.** The Full model overfit two training groups below the fixed 15 mm error limits. On the four held-out terrain seeds, both Full training initializations improved Arc-minus-Direct XY RMSE over the train-only Direction mean and No Path. The gain is modest and uneven across terrain. DEM and Actor-state contributions, and Front/Rear interaction prediction, remain inconclusive.

## Source audit and split

The importer read the MI-A v3.1 formal root without writing to it. It verified all 192 frozen assignment rows against their inner/outer seals and Gate P, checked every candidate Path SHA, recomputed every 40-step body-frame XY label from raw trace, checked 1e−6 pre-Path agreement across candidates and produced 96 grouped examples (terrain seed × initial posture × env tile), each containing the six paired Path labels. All 576 XY and diagnostic yaw labels, DEMs, Actor states and Paths are finite. Input shapes are DEM `[96,4,2,20,20]`, pre-noise Actor `[96,22]`, Path `[96,6,8,2]`, direction `[96,6,1]`, mask `[96,6]`, and XY target `[96,6,2]`. The observed valid DEM heights span −0.2262 to +0.1635 m. The dataset and source seal ledger are stored separately in `/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-b-pilot-20261014-01`.

Each of the four terrain types contributes one seed to each split. Train seeds are `2026101401, 2026101502, 2026101603, 2026101701`; validation seeds are `2026101402, 2026101503, 2026101601, 2026101702`; test seeds are `2026101403, 2026101501, 2026101602, 2026101703`. Each split has four independent seeds, 32 paired groups and 192 candidate trials. The same seed's posture, tiles and candidates cannot cross splits. **Within each terrain, level and split are fully confounded** because there is only one seed at each level; held-out error can include a level shift. DEM height, Actor and target normalizers use train groups only; Path remains in metres and direction remains ±1. Test labels were audited during dataset construction, but were not used for normalization, training or checkpoint selection. Test metrics were computed after all eight checkpoints had been selected by aggregate validation loss.

## Model and training

The pilot uses a small three-convolution DEM encoder, Actor-state MLP, Path-plus-direction MLP and two-coordinate fusion head. Goal is absent. Full, No DEM, No Body and No Path share the same architecture and parameter count; the named modality is zeroed before encoding. No Path still receives direction, so it must predict the same displacement for Direct and both Arcs within a direction. Training loss is normalized absolute XY MSE plus four normalized Arc-minus-same-direction-Direct XY MSE terms. The frozen settings are batch 16 groups, AdamW LR 1e−3, weight decay 1e−4, maximum 240 epochs, patience 35, no scheduler and training seeds 123/456. The best aggregate validation loss selects each checkpoint. Host `tinyvla` PyTorch 2.0.1 had no usable CUDA; all training ran on CPU.

The two-group Full overfit probe ran 1,500 steps and reached maximum absolute XY error **5.08 mm** and maximum candidate-relative XY error **1.84 mm**, both below the predeclared 15 mm limits. It proves the pipeline can fit a tiny set; the two groups come from the same Front-folded training StateGroup and do not test posture generalization.

## Held-out metrics

XY RMSE is the square root of mean squared **vector norm**, in metres. Contrast RMSE is computed over the four Arc-minus-Direct vectors per paired group. Values for seed 123 / 456 are separate model initializations, not additional terrain samples.

| Model | Absolute XY RMSE (m) | Arc−Direct XY RMSE (m) |
| --- | ---: | ---: |
| Train-only Direction mean | 0.1342 | 0.0834 |
| Full | 0.1130 / 0.1136 | **0.0713 / 0.0724** |
| No DEM | 0.1416 / 0.1308 | 0.0746 / **0.0699** |
| No Body | 0.1126 / 0.1236 | 0.0736 / 0.0742 |
| No Path | 0.1219 / 0.1391 | 0.0834 / 0.0834 |

Full improves the primary contrast RMSE by 13–15% relative to the Direction mean. No Path equals the zero-contrast baseline by construction, supporting a Path-input contribution. Removing Actor state makes contrast RMSE 2–3 mm worse in both initializations, but the four-seed evidence is weak. Removing DEM is worse for seed 123 and **better** for seed 456; a reliable DEM contribution is not established. These ablation results do not prove that the network extracts terrain interaction rather than mainly learning the fixed bank's average Path effect.

For Full seed 123 / 456, the per-seed contrast RMSEs (m) are: `pyramid_stairs` 0.0633/0.0648, `pyramid_stairs_inv` 0.0668/0.0652, `random_rough` 0.0862/0.0866 and `k_rail_step_field` 0.0665/0.0708. Baseline values are 0.0865, 0.0893, 0.0891 and 0.0666, respectively. Thus most of the improvement comes from the two pyramid terrain seeds; `random_rough` improves only about 3 mm, and one Full initialization is worse than baseline on K-rail. These are four seed-level descriptions, not a population confidence interval. Front-folded contrast RMSE is 0.0793/0.0819 m, versus Rear-folded 0.0623/0.0614 m. Full absolute RMSE is about 0.113 m overall; Forward and Reverse subgroup values are retained in `pilot-subgroups-v2.json`.

The held-out mean Arc−Direct vectors (measured → Full seed-123 predicted, metres) are Forward +30° `(−0.038,+0.020) → (−0.041,+0.020)`, Forward −30° `(−0.060,−0.017) → (−0.038,−0.018)`, Reverse +30° `(+0.008,−0.019) → (−0.036,−0.016)`, and Reverse −30° `(+0.047,+0.019) → (+0.055,+0.018)`. Reverse +30° has the wrong mean X sign and only 62–69% per-group vector-direction agreement across Full initializations. The worst Full absolute-error group is Rear-folded, K-rail level 3, env tile 1, with group RMSE 0.379/0.346 m. This is an important failure case despite the aggregate pilot Pass.

The paired Rear-minus-Front Arc-effect interaction RMSE is 0.0964/0.0980 m for Full, against 0.0995 m for No Path. Thus the network has **not demonstrated useful prediction of posture-dependent Path effects**. Posture metadata was not fed to the model; the measured 22D Actor input contains flipper state as intended. The limited data and one seed per terrain in train are plausible causes, but this pilot does not isolate them.

## Decision and next data needs

- **Existing-data fit: Pass** for the small overfit probe.
- **Held-out paired Path effect: Pilot Pass** by the predeclared contrast RMSE criterion; effect-size accuracy is incomplete, especially Reverse +30° and `random_rough`.
- **Path input: supported** by the No Path comparison. **DEM, Actor and Front/Rear interaction: Inconclusive** with four held-out seeds and two training initializations.
- **Formal MI-B Gate: undefined.** This report does not promote the pilot result to a formal MI-B Pass or authorize MI-C.

Before Phase C, collect additional **independent terrain seeds within each terrain×level cell**, so level is not identical to seed and each split has more than one independent condition per terrain. Preserve paired Direct/Arc execution for both postures and all four env tiles, and add varied pre-Path body/contact states, including terrain engagement and less-flat DEMs. Include reverse arcs and hard cases like `random_rough` and K-rail. A staged target for planning is at least three independent seeds per terrain×level (36 seeds total, 1,728 six-Path trials including both postures and four tiles), followed by a learning curve before setting a full collection size. This is a proposal, not an approved collection. Wider Path angles require their separate feasibility gate. DEMO remains Front-folded fixed; future collection posture randomization remains at StateGroup granularity.
