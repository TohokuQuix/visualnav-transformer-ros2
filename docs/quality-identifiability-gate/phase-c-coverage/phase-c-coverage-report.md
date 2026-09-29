# Phase C candidate coverage diagnostic

## Decision

**C: sampling-only と global train PCA residual proposal の両方が offline coverage gate 不合格。** Formal Phase C、Phase D/E、モデル再学習、full collection、実機試験は実施していない。held-out は一切使用していない。少数 state serial replay は、offline gate 不合格のため計画どおり実行していない。

## 固定条件と実行

[config](coverage-config.yaml)、[state IDs](diagnostic-state-ids.json)、[入力/source SHA](source-provenance.json)を結果確認前に保存した。Reference NoMaD best EMA SHA は `191d6a03…b4e0`、complete-only train 49,598、validation 6,171。診断 state は train 32 と validation 32、seed 1001–1032 を固定し、K=8 は K=32 の先頭 8 seed とした。Phase B GT Evaluator 3 seed と実 `simple_trajectory_follower` を使用した。

Stage 1 は 15:03–15:46、DDPM trace と sampling は 15:46–16:01、structured proposal は 16:01–16:04 JST に実行した。途中の bridge lifetime と NumPy index のシリアライズ問題は結果の確定前に修正し、[source amendments](source-amendments.json) に旧/新 SHA と原因を記録した。全 stage は完了し、[status](status.json) は `complete`、[出力 SHA](output-artifact-hashes.json)を保存した。

## Train teacher 条件付き多峰性

994 anchor のうち、事前固定した DEM history・relative goal・cross-episode 条件で 3 window 以上の近傍が作れたのは **781**。実 follower command が 2 mode 以上の近傍は **124/781 = 15.9%**で、固定した 20% 閾値に届かなかった。分類は `data_unimodal`。ただし 124 近傍に別 command mode があるため、完全な単峰分布とは主張しない。

Path cluster が複数でも follower command は 1 cluster の近傍は **70.4%**。この結果は、教師軌跡の形状差が初期行動差にならないことを示す。terrain・goal strata、軌跡形状、episode concentration、mode 間 raw Quality 差は [詳細データ](teacher-multimodality-details.json) に保存した。

## DDPM と sampling-only

64 state の 10-step DDPM trace で、waypoint 分散の初期値に対する最終値の比は中央値 **0.054**。effective rank の中央値は初期 **1.55**、最終 **1.27**。途中の noisy action、予測 noise、更新 sample、denormalized delta、累積 waypoint、Evaluator ensemble spread は [trace](</tmp/eatnav-quality-identifiability-gate/phase-c-coverage/denoising-variance-trace.json>) と state 別 NPZ に保存した。計測上、denoising 後に候補差が大幅に縮小した。

| 条件 | 2 command cluster 以上 | 3 Path cluster 以上 | train P95 OOD 超過率 | 判定 |
|---|---:|---:|---:|---|
| baseline K=8 | 1/64 (1.6%) | 0/64 | 0% | fail |
| baseline K=32 | 3/64 (4.7%) | 4/64 (6.3%) | 0.39% | fail |
| 初期 noise scale 0.5 / 1.5 / 2.0、各 K=8 | 各 1/64 | 各 0/64 | 各 0% | 診断のみ、fail |

必要条件は 2 command cluster 以上が 75% state、3 Path cluster 以上が 50% state。全条件で非有限候補は 0。K=32 と初期 noise の変更で follower-distinct coverage は回復しなかった。diffusers DDPM `step` に独立した stochasticity 引数はないため、scheduler 数式・重みは変更していない。各条件の全値は [sampling 判定](sampling-coverage-decision.json)にある。

## Global train PCA residual proposal

Complete-only train trajectory だけから endpoint 固定 residual の PCA 3 成分を構築した。尺度は train coefficient 絶対値の事前固定 75% quantile。K=8 は原 NoMaD seed 1001、最も離れた raw seed、±3 PCA 成分で構成し、違反候補の置換は行っていない。

3 Path cluster 以上は **43/64 state (67.2%)**、OOD 超過率は **0.98%**だった。一方、**2 follower command cluster 以上は 0/64 state**で、複数 terrain の行動 coverage を満たさない。よって structured gate は明確に不合格。raw NoMaD の軌跡差を増やしても、今回の follower の初期 command 差にはつながらなかった。

Constraint 記録には注意が必要。`endpoint_changed` 64 件はすべて **候補 1 の「最も離れた raw Diffusion sample」**に付いた。raw sample が候補 0 と異なる endpoint を持つのは構成上あり、structured perturbation の endpoint 保存違反として数えるのは診断実装の過大計上である。gate はこれを除いても command coverage **0/64**で不合格のまま。self-intersection は 39 件、curvature 上限超過は 1 件記録した。凍結した実行結果・判定ファイルは書き換えていない。[structured 判定](structured-coverage-decision.json)と [全候補の記録](</tmp/eatnav-quality-identifiability-gate/phase-c-coverage/structured-proposal-results.json>)を参照。

## 後段の扱い

通過した offline 候補集合がないため、Phase B Evaluator の候補間 score spread 診断と 8–16 state serial replay は `not_run`。DDPM 各 step の Evaluator 予測分散は trace に記録済みだが、offline gate を通った候補への最終 Stage 5 評価とは区別する。formal Phase C の再提案は現時点で行わない。

次段階を検討する場合は、counterfactual trajectory 収集、goal/subgoal 多様化、maneuver mode conditioning、別 candidate generator などの設計判断が必要。今回の結果を根拠にモデル構造・loss・scheduler を自動変更していない。

## 図

- [Teacher mode count](figures/teacher-mode-count.png)
- [Teacher Path と command cluster](figures/teacher-path-vs-command.png)
- [DDPM 分散](figures/ddpm-step-variance.png)
- [Denoising effective rank](figures/denoising-effective-rank.png)
- [K=8 と K=32](figures/k8-vs-k32.png)
- [Sampling coverage と OOD](figures/sampling-coverage-ood.png)
- [Residual PCA basis](figures/residual-pca-basis.png)
- [Raw と structured proposal](figures/raw-vs-structured.png)
- [Path と follower command cluster](figures/path-vs-command-cluster.png)
- [Evaluator score spread: not run](figures/evaluator-score-spread.png)
- [Duplicate と candidate effect: not run](figures/duplicate-vs-candidate-effect.png)
