# Evaluator multi-head teacher and ranking-loss design freeze

## Decision and scope

**Decision:** retain distinct observed locomotion reward, direct-relative reward, navigation task, pairwise preference, conditional terminal Critic, and Critic residual targets. The 63 sealed train traces support an offline label pipeline smoke, not a trained selector or a generalization claim. No rollout, candidate generation, NoMaD inference, Evaluator training, checkpoint, formal validation, collection, or real-robot control occurred.

The received instruction file ends mid-list in M10 (after an opening backtick). This review implements and verifies the complete visible M0–M10 requirements. Any later requirements in the missing tail need reconciliation before declaring the full instruction set complete.

## M0: source-confirmed provenance

The dedicated branch `exp/eatnav-evaluator-multihead` begins at the clean G40 boundary commit `51d9bf201f76fd8e3b54d3d9496ca4c3429cbb3a`. The unchanged source audit revalidated 7 train StateGroups and 63 protocol-valid 120-step traces, their SHA values, Path SHA/seed/repeat identity, the duplicate gate, exact stamps, corrected map-frame geometry, frozen checkpoint and reference config. The G40 SHA ledger and its input/teacher/Safety contracts were rechecked. The 20 Hz, 40-transition, γ=0.99 analysis config was frozen before this label-generation pass. [M0 provenance](m0-provenance.json), [config](multihead-analysis-config.yaml).

All 63 observed `Rγ40` labels were recomputed from `trace[0:40].reward`, which already includes one `dt` factor. Each exactly equals both prior G40 artifacts. No step-40, post-reset, component-level synthetic reward, or 2–6 s Outcome enters the primary teacher. All 63 are finite, complete, nonterminated, and uncensored. [Audit](label-generation-audit.json).

## M1–M3: observed and relative targets

`discounted_reward_2s = Σ(k=0…39) 0.99^k r_k` is the observed locomotion teacher. The `direct_relative_reward_2s` target subtracts the mean of the three direct repeats **within the same StateGroup**, including for each direct repeat itself. The three repeats have zero sample variance for reward, goal reduction, and endpoint error in all seven groups. This is evidence of duplicate reproducibility; it is not a population noise estimate. Relative reward removes group offset, not absolute safety risk.

Corrected map-frame task heads are goal-distance reduction at 2 s, endpoint error at 2 s, endpoint encounter within the production follower's 0.20 m radius during 0–2 s, Path arc-length progress fraction, and cross-track P95 over 0–2 s. The endpoint radius is diagnostic, not an unsafe/success label. Goal reduction and endpoint error produce identical pairwise ordering here because each group's fixed start/endpoint make them affine transforms; this redundancy needs prospective review before assigning equal training weight. The follower tracks geometry, not waypoint timestamps.

Unique candidate summaries use the median over direct repeats, with one sample per other candidate. The frozen pair universe is **21 unique candidate pairs per StateGroup**, 147 descriptive pairs total; direct repeats are used for baseline/noise, not as three independent candidate identities. Pair labels are reward, goal reduction, and endpoint error separately. A higher reward/goal reduction is better; a lower endpoint error is better. `i_better`, `j_better`, `tie_within_duplicate_noise`, and `invalid_or_censored` are explicit. The global margin rule uses only the seven groups' direct-repeat pairwise absolute differences plus a frozen `1e-6` float-comparison floor. All direct differences were zero, so all three margins are `1e-6`; no observed candidate result was used to tune them. No pair was tied or invalid at that margin. Reward/goal preference conflicts occur in **66/147** descriptive pairs, across all seven groups. Pair counts are not independent-sample counts. [Labels](multihead-labels.jsonl.gz), [pair table](pairwise-labels.json).

## M4–M6: Critic and output semantics

`terminal_value_hold` and `terminal_value_canonical` remain separate auxiliary conditional Critic targets. They query the unchanged observed terminal body state under the last candidate command or the common `(0.3,0,0)` command. They are **offline Critic counterfactuals**, not outcomes of continuing that command in physics. Same-group direct-reference command is optional. The recorded boundary remains a diagnostic; literal-zero Critic value exists for 57 traces and is **NA for the six nonzero recorded boundaries**. No imputation was made.

`critic_residual_hold/canonical = Rγ40 + γ^40 V40(condition) − V0` measure error relative to the fixed locomotion Critic's start prediction. Positive does not prove an absolutely good Path and negative does not prove failure. A common StateGroup `V0` leaves the corresponding G40 ranking unchanged; candidate-specific `V0` partly reflects differing initial command conditions. The terminal command dependency survives subtraction. Residuals are auxiliary calibration/hard-state-mining diagnostics, never the primary online selector.

The [output contract](model-output-contract.yaml) separates primary observed reward, direct-relative reward, task regression, reward/goal preference; auxiliary task and Critic heads; predictive uncertainty and missing/out-of-support flags. `p_unsafe` is explicitly disabled for training and inference with zero loss weight. Stack proxies, termination proxies, infrastructure errors and censored timeouts are not positive Safety labels.

## M7–M9: input, loss, selection

The [portable input contract](portable-evaluator-input-contract.json) is byte-identical to the preceding G40 contract. It requires pre-Path 4-frame DEM height/valid-mask history, relative goal, 8×2 candidate Path, deterministic geometry/profile/coverage, projected gravity, base linear/angular velocity, ordered joint position/velocity, previous six-dimensional ActionManager action, masked pre-Path controller state, and availability masks. Raw effort and future/post-Path fields are excluded. The current ROS2 bridge lacks verified base linear velocity, full ordered 52-joint vectors, previous action, and physical DEM/IMU/TF parity; a missing critical field blocks primary inference rather than becoming zero. A fixed-checkpoint, anchor-only Actor latent remains a [separate ablation](optional-latent-input-contract.json).

[Loss and normalization contract](loss-and-normalization-contract.yaml): Phase A uses Huber regression on train-StateGroup-scaled continuous targets plus strict-pair Bradley–Terry logistic loss, reduced within each StateGroup and then across groups. Ties and invalid/censored labels have zero ranking mask. The included numeric fixture checks loss masking. Six equal weights are *smoke defaults only*; none were tuned on these 63 outcomes, and formal weights require additional independent validation. BCE for endpoint encounter is disabled pending class support. Phase B adds terminal/residual heads only as independently compared ablations; a drop in primary ranking rejects the auxiliary variant. Splits keep all candidates/repeats of a StateGroup, source trajectory/environment instance and terrain seed together. Scalers fit train groups only; current validation and held-out were not used.

The [selection contract](inference-selection-contract.yaml) is prospective: verified Type A invalid removal, missing/out-of-support abstention, uncertainty fallback, task-positive preference, reward comparison, then validated terminal-value tie breaking. Conflict resolution and thresholds await independent validation. The old 0.50 progress, 0.20 m reduction/tolerance, cross-track and 0.52 rad tilt thresholds are not immediate hard exclusions. No single weighted score or Critic residual selector is approved.

## M10: label support and limits

All primary continuous heads have 63 valid labels across 7 groups/7 terrains, zero censored records, nonzero candidate spread, and zero observed duplicate noise. Reward is 13.513–14.647; relative reward is −0.297–0.149; goal reduction is 0.374–0.552 m; endpoint error is 0.047–0.225 m. Their `supported_for_pipeline_smoke` status does **not** imply formal generalization support from seven train groups. Duplicate-effect ratios are NA because the denominator is zero. The complete per-head range, median/IQR, spread, sign, valid/missing count and support are in [data sufficiency](data-sufficiency-review.json).

Endpoint encounter is 57 positive / 6 negative; negatives occur in only two groups. It is `insufficient_label_variation`, so BCE remains disabled. Literal-zero terminal value has only 57 valid labels, six NA. The current traces have zero failure-positive support; Safety training remains disabled. Direct-relative reward has 25 negative, 17 positive and 21 zero per-run labels; the zeros are the direct repeats. There are no reward/goal/endpoint pair ties at the pre-fixed numeric floor; this does not establish a nonzero real-world indifference region.

The figures show [label distributions](fig-01-observed-label-distributions.svg), [separate pair labels](fig-02-pairwise-labels.svg), [reward/goal conflicts](fig-03-reward-goal-conflicts.svg), and [head coverage](fig-04-head-support.svg).

## Next approved design question

Prepare real input-source/parity capture and prospective collection of additional independent StateGroups with reward components, command-boundary timing, full body state, verified failures and diverse endpoint encounters. Then define split/scaler and head-weight selection on independent validation. The present contract does not authorize training or hard-state rollout. [SHA ledger](artifact-sha-ledger.json) records all small artifacts.
