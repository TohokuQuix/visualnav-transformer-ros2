# EAT-Nav NoMaD Path追従性・候補縮退箇所診断

## 対象と整合性

7 train stateの112 complete live rolloutを解析した。21 nominal、56 NoMaD、35 bankで、7 anchorをoutcomeから除外した。各runは120 step、20 Hz、6秒。8 future pointのPathは0.25秒間隔、2秒horizonである。既存のPath・trace SHA、command応答、step時刻、座標変換を全件で照合した。map Pathはsimulator環境ローカル、root poseはworldであり、env originを復元して比較した。入力は変更していない。

## Path追従と終点

| generator | run | 2秒まで到達 | 6秒まで到達 | 到達後逸脱 | 0–2秒cross-track P95中央値 |
|---|---:|---:|---:|---:|---:|
| nomad | 56 | 28 | 56 | 22 | 0.0229 m |
| bank | 35 | 11 | 35 | 14 | 0.0453 m |
| nominal | 21 | 6 | 21 | 12 | 0.0266 m |

「Pathを追従目標として走行した」は全112件で成り立つ。全件が終点0.2 m圏に一度入ったが、到達後の逸脱もあり、Path形状を高精度に追ったことや6秒間終点に留まったこととは異なる。6秒間は全runが非terminalで、failure-positive outcomeはない。0–2秒の幾何学的追従と0–6秒の終点到達を別集計した。

## 多様性の消失

| state | terrain | NoMaD P/C/R mode | bank P/C/R mode | NoMaD分類 |
|---|---|---|---|---|
| readiness-000 | boxes | 1/4/1 | 1/2/1 | generator_collapse |
| readiness-001 | hf_pyramid_slope | 2/4/1 | 1/1/1 | realization_collapse |
| readiness-002 | k_rail_field | 1/3/2 | 1/5/2 | generator_collapse |
| readiness-003 | k_rail_step_field | 1/5/1 | 1/2/1 | generator_collapse |
| readiness-004 | pyramid_stairs | 1/4/1 | 1/1/1 | generator_collapse |
| readiness-005 | pyramid_stairs_inv | 1/4/1 | 1/3/1 | generator_collapse |
| readiness-006 | random_rough | 1/2/1 | 1/2/1 | generator_collapse |

NoMaDのPは6/7 stateで1 mode。Cは2–5 modeだが、Rは6/7 stateで1 modeだった。P段階の候補生成が最初の主なボトルネックで、C→Rでも差が縮む。followerがP→Cを潰すことが主因という証拠はこの7 stateにはない。距離分布と各候補のP/C/R対応はJSONに保存した。cluster閾値は既存値を固定した。

## Outcomeと凍結GT(P)

cross-track P95とΔVのstate内Spearman平均は-0.238、progress ratioとΔVは0.188。leave-one-state-out値は`tracking-outcome-analysis.json`に記録した。追従指標だけでoutcome順位を説明できるとは言えない。全件到達のためreached/not reached比較はNAである。

凍結GT(P)のall-candidate top-1は0/7、top-3は1/7、state-macro pairwise accuracyは0.411、平均oracle regretは0.244。層別表には支持不足をNAとして残し、selectorとしての再承認はしない。

## Evaluator入力と設計判断

DEM history、relative goal、実際のpost-delay Actor観測、latent、root/joint state、previous action、Path、初手follower commandは112件で取得可能。raw joint effortは0件で、Actor observationに含むscaled/clipped/noisy joint effort with track sumから正確なraw値は逆算できない。同stepのオンラインjoint effort取得とfailure-positive教師を確認するまで、execution-aware Evaluatorの大規模再学習やscaled collectionは提案しない。

**優先順:** N1 NoMaD候補多様性を改善。C→R差の縮小をfollower/Actor/車体応答に切り分ける。N3の入力と教師を整備する。N4 scaled counterfactual collectionは保留。Safety headはfailure-positive 0件のため未ready。

## 8つの問いへの回答

1. No: planned Path one cluster in 6/7 states; two in 1/7 under frozen threshold.
2. 28/56 by 2 s and 56/56 by 6 s entered 0.2 m; 22/56 later left; this does not establish shape tracking.
3. NoMaD command modes range 2–5; only the sole multi-Path state retained multiple command modes. No dominant follower collapse.
4. NoMaD realized modes: [1, 1, 2, 1, 1, 1, 1]; multiple realized modes in only 1/7 states despite multiple command modes in all seven.
5. Planned Path is one mode in 6/7 states. C-to-R compression is additional: NoMaD realized trajectories have multiple modes in only one state.
6. Within-state Spearman cross-track P95 vs ΔV macro -0.238; progress ratio vs ΔV 0.188. Descriptive, weak association; it does not explain ranking causally.
7. Improve NoMaD planned candidate diversity first; investigate command-to-realization compression and same-step embodiment input before Evaluator redesign or more collection.
8. Raw same-step joint effort is absent in 112/112; scaled/noisy Actor observation is not invertible to raw effort. Need verified online joint effort and candidate-specific, failure-positive counterfactual teachers with state-level splits.

## 図と再現性

Path重ね描きはNoMaD・bank・nominalで分図した。終点、progress、cross-track、P/C/R mode、遷移、ΔV、GT(P)誤順位の図も同ディレクトリにある。raw seriesは`/tmp/eatnav-candidate-coverage/path-tracking-audit/tracking-series.jsonl.gz`。SHA台帳、実装コード、テストを同branchに保存する。

適用範囲はこの7 train stateに限る。validation、held-out、実機へ一般化しない。新規simulator rollout、学習、candidate再生成、閾値変更はしていない。
