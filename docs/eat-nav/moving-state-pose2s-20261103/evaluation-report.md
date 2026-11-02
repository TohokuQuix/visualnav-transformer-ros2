# 走行中2秒位置・姿勢予測：実行結果

仕様承認後に実装、既存データ再加工、headless追加収集、3モデルの学習、固定分割での評価、14候補CPU推論を実施した。実機走行とplanner変更は実施していない。

既存データで初期区間限定M_stopと走行中データを加えたM_moveを学習し、追加後のM_finalを同じ旧encoder初期値・seed123・学習recipeから別途学習した。旧checkpointは上書きしていない。M_finalは57epoch、best37epoch、CPU8threadで35.77秒。追加前M_moveは113epoch、best93epochだった。

**追加学習で一律改善したとは言えない。** 同じ既存audit807区間ではM_moveの平均3D誤差7.61cmに対しM_finalは8.91cm。新規audit168区間ではM_move9.29cmに対しM_final9.76cm、回転は0.13646radから0.13232radへわずかに低下したが、その差の親bootstrap95%CIは0を含む。combined validationでも位置・回転の平均誤差がM_moveを改善しなかったため、既定モデルはM_moveのままとし、M_finalも比較・研究用に配布する。audit結果から構造・loss・重み・閾値を再調整していない。

## 追加収集・有効件数

7状態×10親×14候補=980枝の縮小計画を事前固定。67/70親が資格を満たし、938枝の実行結果を保存。降坂1親はsettle中終了、低速2親はprefix中終了／20秒以内に資格不成立で、別seedへの置換をしていない。境界打切り70枝を主datasetから除外し、868枝すべてに完全2秒poseが得られた。実行中の自然終端は0。追加868枝のイベント陽性はtip0、near2、直線11、旋回10。陽性windowは独立した事故件数ではない。

| 状態 | 資格成立親 | 観測枝 | 採用枝 | 全14候補の完全2秒poseがある親 |
|---|---:|---:|---:|---:|
| 平坦移動 | 10 | 140 | 140 | 10 |
| 旋回 | 10 | 140 | 140 | 10 |
| 登坂直前 | 10 | 140 | 70 | 0 |
| 登坂途中 | 10 | 140 | 140 | 10 |
| 段差直前 | 10 | 140 | 140 | 10 |
| 降坂 | 9 | 126 | 126 | 9 |
| 停止・低速 | 8 | 112 | 112 | 8 |

登坂直前は有効DEM履歴を確保する地形外縁から内向きに開始したため、後退7候補が境界で打切られた。この方向の完全2秒教師が不足している。全候補を実行開始したことと、全候補の結果が揃うことを区別する。詳細はfresh-coverage.json、collection/collection-final-audit.json、collection/postcollection-readiness.jsonを参照。

snapshot復元では状態・結果の差を検出したため、その失敗記録を採用せず、候補ごとに新しいシミュレータープロセスで同一prefixを連続実行する方式へ変更した。GUIはOFF、実測10並列。初回候補の重複と最後の再実行、全候補の可視開始状態を照合した。平坦移動のみ逆順全14候補の診断も完了。残り6状態は時間予算を守るため逆順全候補診断を省略し、counterfactual_complete=falseを維持している。完全な14候補対照実験と呼ばない。地形の登坂直前・降坂初期化位置の修正は結果を見る前の幾何検査に基づき、元planは保持、実行amendmentに記録した。

| dataset | train | validation | audit_eval | 全件 | 完全2秒pose |
|---|---:|---:|---:|---:|---:|
| 既存 | 5299 | 791 | 807 | 6897 | 6880 |
| 追加 | 532 | 168 | 168 | 868 | 868 |
| 統合 | 5831 | 959 | 975 | 7765 | 7748 |

同じ親・初期状態のsplit跨ぎ、転移元の学習親と評価親の重複、完全NN入力のsplit跨ぎは0。K-Railとstepfieldには地形hashの共有が残る。主結果は既知地形を含む独立走行への評価であり、未知地形の性能ではない。地形hash非重複subsetは既存641区間・6親、新規140区間・11親を別報告した。同一地形クラスに対する結果であり未知クラスへの汎化ではない。

## 使える範囲・残る制約

2秒終点のXYZ・相対回転・重力基準の将来傾斜・2秒以内4イベントを14候補で取得できる。40点の途中GT軌跡はdatasetに保存したが、NNによる途中軌跡・最大傾斜の予測は実装していない。poseは完全2秒を観測できた区間のみで学習し、早期終端後を補完していない。

tip/nearの既存・新規audit陽性はともに0で、検出性能を評価できない。M_finalの閾値0.5における直線・旋回スタックrecallは既存・新規auditとも0。分類headは学習され出力できるが、有効なリスク回避器として検証できたとはしない。閾値をauditに合わせて下げていない。確率は未校正。回転量の過小予測も残る。位置変位の平均振幅はGTに近く、単なる不動予測への崩壊ではないが、候補ごとの正確さを保証しない。

接触時刻は必須条件にせず、接触前だけの性能は未評価。同一状態から14候補比較による登坂性能改善、closed-loop走行改善、実機PCの処理時間、実機状態のscale/フリッパー原点変換は未検証。実機plannerへ自動接続しない。

CPU4thread、14候補一括の全predict APIはmedian2.27ms／P95 2.30ms（入力検証修正後、warmup20、200回。同一サーバー、ファイルIO・モデルload・JSON出力を除く）。M_finalを新しいプロセスでCPUロードし、K=1との一致・候補順序置換・別PyTorch/NumPy環境との一致も確認した。

集計表示でcollection_stateが抜ける不具合を最終整理時に修正し、同じ保存予測から状態別表だけを再生成した。重み・教師・正規化・集計全体値は変えていない。元評価はローカルに保存し、公開版のsource hashと差分理由はreporting-correction.jsonに記録した。収集freezeは当時のsourceを示し、その後の表示用2行修正とは区別する。

## ローカルに残るデータ

`/home/ubuntu/eatnav-moving-state-pose2s-data-20261103/`

- collection/production/*/sim-output/：938枝の入力・trace・raw DEMとprefix/役割の対応。
- collection/fresh-process-runtime/、runtime-stage-archives/：診断・失敗記録と復元用コードarchive（collection配下）。
- reused-dataset/、fresh-dataset/、combined-dataset/：テンソル、40点GT、索引、分割、監査。
- models/：3モデル、optimizer再開用ローカル状態、学習history。
- evaluation/：保存予測を含む元評価。Git公開版は小型metrics・表・レポートのみ。

GitHubから取得できるものはコード、設定、仕様、モデル3個、最小現在入力例、SHA/索引、評価・監査metadata。生記録・巨大テンソル・optimizer状態はGitへ追加しない。再学習するPCへは上記ローカルデータを別途転送する必要がある。

保全していたR6評価は1925ファイルのSHAを再照合して既存PIDだけ再開した。R6評価そのものの完了を今回の完了範囲に含めない。

以下は完了済みJSONから生成した詳細表。層別のcollection_stateは公開evaluations/*/state-stratified.csv・metrics.jsonを参照。

| Evaluation | Role | Model | Windows | Parents | Pose count | Mean 3D m | Median 3D m | P95 3D m | Mean SO(3) rad |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| M_move-existing-audit | audit_eval | M_move | 807 | 8 | 807 | 0.0761422 | 0.059348 | 0.204501 | 0.122921 |
| M_move-existing-audit | audit_eval | no_motion | 807 | 8 | 807 | 0.233285 | 0.203695 | 0.467363 | 0.175563 |
| M_move-existing-audit | audit_eval | constant_body_twist | 807 | 8 | 807 | 0.26629 | 0.239974 | 0.557717 | 0.900761 |
| M_final-combined-validation | validation | M_final | 959 | 21 | 956 | 0.0907678 | 0.0708986 | 0.246837 | 0.13947 |
| M_final-combined-validation | validation | no_motion | 959 | 21 | 956 | 0.238513 | 0.205365 | 0.506542 | 0.17277 |
| M_final-combined-validation | validation | constant_body_twist | 959 | 21 | 956 | 0.350259 | 0.271945 | 0.894529 | 1.02632 |
| M_final-existing-audit | audit_eval | M_final | 807 | 8 | 807 | 0.0891033 | 0.0691493 | 0.234073 | 0.136661 |
| M_final-existing-audit | audit_eval | no_motion | 807 | 8 | 807 | 0.233285 | 0.203695 | 0.467363 | 0.175563 |
| M_final-existing-audit | audit_eval | constant_body_twist | 807 | 8 | 807 | 0.26629 | 0.239974 | 0.557717 | 0.900761 |
| M_final-fresh-audit | audit_eval | M_final | 168 | 13 | 168 | 0.0975842 | 0.0701632 | 0.278805 | 0.132318 |
| M_final-fresh-audit | audit_eval | no_motion | 168 | 13 | 168 | 0.275626 | 0.238538 | 0.555986 | 0.183755 |
| M_final-fresh-audit | audit_eval | constant_body_twist | 168 | 13 | 168 | 0.482554 | 0.423677 | 1.11835 | 1.31765 |

## Same-holdout model differences

| Evaluation | Scope | Reference | Windows | Parents | Metric | Current mean | Reference mean | Current − reference | Parent bootstrap 95% CI |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| M_move-existing-audit | moving_only | M_stop | 590 | 6 | position_3d_m | 0.075452 | 0.0901733 | -0.0147213 | [-0.0209673, -0.00836523] |
| M_move-existing-audit | moving_only | M_stop | 590 | 6 | so3_rad | 0.124775 | 0.142057 | -0.0172813 | [-0.0219445, -0.013129] |
| M_move-existing-audit | all_windows | M_stop | 807 | 8 | position_3d_m | 0.0761422 | 0.0892959 | -0.0131537 | [-0.0192687, -0.00739318] |
| M_move-existing-audit | all_windows | M_stop | 807 | 8 | so3_rad | 0.122921 | 0.13955 | -0.0166291 | [-0.019219, -0.0142168] |
| M_final-combined-validation | moving_only | M_move | 746 | 19 | position_3d_m | 0.0871298 | 0.0822213 | 0.00490847 | [-0.00284079, 0.0106385] |
| M_final-combined-validation | moving_only | M_move | 746 | 19 | so3_rad | 0.134531 | 0.127021 | 0.00751038 | [-0.000154957, 0.0152777] |
| M_final-combined-validation | all_windows | M_move | 959 | 21 | position_3d_m | 0.0907678 | 0.083567 | 0.00720076 | [-0.0027379, 0.0153363] |
| M_final-combined-validation | all_windows | M_move | 959 | 21 | so3_rad | 0.13947 | 0.132328 | 0.00714213 | [0.00131842, 0.0132122] |
| M_final-existing-audit | moving_only | M_move | 590 | 6 | position_3d_m | 0.0889037 | 0.075452 | 0.0134518 | [0.00841719, 0.0218897] |
| M_final-existing-audit | moving_only | M_move | 590 | 6 | so3_rad | 0.136364 | 0.124775 | 0.0115889 | [0.00592525, 0.0183461] |
| M_final-existing-audit | all_windows | M_move | 807 | 8 | position_3d_m | 0.0891033 | 0.0761422 | 0.0129611 | [0.00736064, 0.0203651] |
| M_final-existing-audit | all_windows | M_move | 807 | 8 | so3_rad | 0.136661 | 0.122921 | 0.0137402 | [0.00896168, 0.0199328] |
| M_final-fresh-audit | moving_only | M_move | 168 | 13 | position_3d_m | 0.0975842 | 0.0928727 | 0.00471152 | [-0.0108512, 0.0218149] |
| M_final-fresh-audit | moving_only | M_move | 168 | 13 | so3_rad | 0.132318 | 0.136461 | -0.00414334 | [-0.011152, 0.00511146] |
| M_final-fresh-audit | all_windows | M_move | 168 | 13 | position_3d_m | 0.0975842 | 0.0928727 | 0.00471152 | [-0.0108512, 0.0218149] |
| M_final-fresh-audit | all_windows | M_move | 168 | 13 | so3_rad | 0.132318 | 0.136461 | -0.00414334 | [-0.011152, 0.00511146] |
| M_final-fresh-audit | moving_only | M_stop | 168 | 13 | position_3d_m | 0.0975842 | 0.116429 | -0.0188444 | [-0.0339494, -0.00556114] |
| M_final-fresh-audit | moving_only | M_stop | 168 | 13 | so3_rad | 0.132318 | 0.187211 | -0.0548929 | [-0.106564, -0.0225006] |
| M_final-fresh-audit | all_windows | M_stop | 168 | 13 | position_3d_m | 0.0975842 | 0.116429 | -0.0188444 | [-0.0339494, -0.00556114] |
| M_final-fresh-audit | all_windows | M_stop | 168 | 13 | so3_rad | 0.132318 | 0.187211 | -0.0548929 | [-0.106564, -0.0225006] |

Moving-only means subsequent decision windows (`is_initial=false`), including low-speed/stalled decisions. These paired comparisons isolate the holdout cohort; data volume, normalization and validation-selected early stopping can differ between models. More windows do not imply more independent runs.

## Motion magnitude diagnostics

| Evaluation | Scope | Pose count | Mean displacement pred / GT m | Ratio of means | Mean rotation pred / GT rad | Ratio of means |
|---|---|---:|---|---:|---|---:|
| M_move-existing-audit | all | 807 | 0.226897 / 0.233285 | 0.972619 | 0.121878 / 0.175563 | 0.694214 |
| M_move-existing-audit | moving | 590 | 0.211166 / 0.216043 | 0.977428 | 0.124808 / 0.177875 | 0.701665 |
| M_move-existing-audit | speed:<0.05 | 133 | 0.19109 / 0.179148 | 1.06666 | 0.0834354 / 0.118775 | 0.702464 |
| M_move-existing-audit | speed:0.05-0.2 | 489 | 0.217519 / 0.229719 | 0.946894 | 0.119628 / 0.172952 | 0.691683 |
| M_move-existing-audit | speed:>=0.2 | 185 | 0.277427 / 0.28163 | 0.985079 | 0.155463 / 0.22329 | 0.69624 |
| M_final-combined-validation | all | 956 | 0.233861 / 0.238513 | 0.980495 | 0.120673 / 0.17277 | 0.698458 |
| M_final-combined-validation | moving | 743 | 0.229345 / 0.227427 | 1.00844 | 0.122496 / 0.165007 | 0.742371 |
| M_final-combined-validation | speed:<0.05 | 148 | 0.194106 / 0.186727 | 1.03952 | 0.0999014 / 0.141011 | 0.708464 |
| M_final-combined-validation | speed:0.05-0.2 | 452 | 0.216237 / 0.220232 | 0.981859 | 0.117901 / 0.158428 | 0.744198 |
| M_final-combined-validation | speed:>=0.2 | 356 | 0.272765 / 0.283254 | 0.962972 | 0.132827 / 0.204183 | 0.650526 |
| M_final-existing-audit | all | 807 | 0.225691 / 0.233285 | 0.96745 | 0.113932 / 0.175563 | 0.648953 |
| M_final-existing-audit | moving | 590 | 0.211537 / 0.216043 | 0.979145 | 0.112559 / 0.177875 | 0.632797 |
| M_final-existing-audit | speed:<0.05 | 133 | 0.191827 / 0.179148 | 1.07078 | 0.094049 / 0.118775 | 0.791822 |
| M_final-existing-audit | speed:0.05-0.2 | 489 | 0.219516 / 0.229719 | 0.955583 | 0.112331 / 0.172952 | 0.649496 |
| M_final-existing-audit | speed:>=0.2 | 185 | 0.266361 / 0.28163 | 0.945783 | 0.132457 / 0.22329 | 0.593205 |
| M_final-fresh-audit | all | 168 | 0.271554 / 0.275626 | 0.985227 | 0.133175 / 0.183755 | 0.72474 |
| M_final-fresh-audit | moving | 168 | 0.271554 / 0.275626 | 0.985227 | 0.133175 / 0.183755 | 0.72474 |
| M_final-fresh-audit | speed:<0.05 | 28 | 0.264762 / 0.309582 | 0.855225 | 0.1272 / 0.273732 | 0.464689 |
| M_final-fresh-audit | speed:0.05-0.2 | 42 | 0.264125 / 0.230893 | 1.14393 | 0.127666 / 0.191831 | 0.665512 |
| M_final-fresh-audit | speed:>=0.2 | 98 | 0.276678 / 0.285095 | 0.970476 | 0.137242 / 0.154586 | 0.887805 |

These are ratios of cohort means, not mean per-window ratios or a motion classifier. A ratio near one does not establish accurate direction or per-example motion. Rotation magnitude underprediction must remain visible even when endpoint error improves.

## Events and threshold support

| Evaluation | Head | Positive | Negative | Unknown | Positive parents | AP | AUROC | Brier | BCE | recall@0.5 | FPR@0.5 | F1@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M_move-existing-audit | tip_over | 0 | 807 | 0 | 0 | N/A | N/A | 2.21144e-06 | 0.000402795 | N/A | 0 | N/A |
| M_move-existing-audit | near_fall | 0 | 807 | 0 | 0 | N/A | N/A | 3.16403e-06 | 0.00060096 | N/A | 0 | N/A |
| M_move-existing-audit | linear_stall | 19 | 788 | 0 | 3 | 0.192828 | 0.888124 | 0.0218783 | 0.0886736 | 0.0526316 | 0.00253807 | 0.0909091 |
| M_move-existing-audit | turn_stall | 11 | 796 | 0 | 4 | 0.0621886 | 0.833029 | 0.013123 | 0.0616386 | 0 | 0 | 0 |
| M_final-combined-validation | tip_over | 3 | 956 | 0 | 2 | 0.0109881 | 0.801953 | 0.00313014 | 0.0211644 | 0 | 0 | 0 |
| M_final-combined-validation | near_fall | 5 | 954 | 0 | 3 | 0.0130528 | 0.71174 | 0.00542853 | 0.0352913 | 0 | 0 | 0 |
| M_final-combined-validation | linear_stall | 30 | 926 | 3 | 3 | 0.0359185 | 0.575126 | 0.0338198 | 0.151581 | 0 | 0 | 0 |
| M_final-combined-validation | turn_stall | 4 | 952 | 3 | 3 | 0.0266202 | 0.845063 | 0.00449317 | 0.031334 | 0 | 0 | 0 |
| M_final-existing-audit | tip_over | 0 | 807 | 0 | 0 | N/A | N/A | 9.67395e-06 | 0.00144898 | N/A | 0 | N/A |
| M_final-existing-audit | near_fall | 0 | 807 | 0 | 0 | N/A | N/A | 2.95161e-05 | 0.00248772 | N/A | 0 | N/A |
| M_final-existing-audit | linear_stall | 19 | 788 | 0 | 3 | 0.0403037 | 0.48417 | 0.0249061 | 0.125444 | 0 | 0 | 0 |
| M_final-existing-audit | turn_stall | 11 | 796 | 0 | 4 | 0.0444676 | 0.77958 | 0.0132961 | 0.0656428 | 0 | 0 | 0 |
| M_final-fresh-audit | tip_over | 0 | 168 | 0 | 0 | N/A | N/A | 1.55913e-07 | 0.000267858 | N/A | 0 | N/A |
| M_final-fresh-audit | near_fall | 0 | 168 | 0 | 0 | N/A | N/A | 5.8905e-07 | 0.000563022 | N/A | 0 | N/A |
| M_final-fresh-audit | linear_stall | 3 | 165 | 0 | 1 | 0.0169775 | 0.329293 | 0.0182195 | 0.0998112 | 0 | 0 | 0 |
| M_final-fresh-audit | turn_stall | 1 | 167 | 0 | 1 | 0.0166667 | 0.646707 | 0.00621146 | 0.0398129 | 0 | 0 | 0 |

AP/AUROC require both classes. A zero-positive cohort cannot verify event detection, even with a low Brier/BCE or no false alarms. Probabilities are uncalibrated unweighted-BCE estimates, and 0.5 is the frozen evaluation threshold, not a robot STOP setting. The per-head masks, support, 10-bin reliability and parent intervals remain in the source metrics.

## Scope and unresolved limits

- Pose metrics condition on observing a full 2s endpoint; early natural termination can retain event labels while pose is missing. Missing/invalid counts must be reported alongside error.
- Reused audit is a parent-separated development holdout, not the reserved formal test. Some train/evaluation geometry hashes overlap. Even hash-nonoverlap subsets share terrain classes.
- Fresh candidate branches share parents. Reduced order-repeat diagnostics are not proof of full counterfactual completeness; inspect final collection/coverage records before claiming paired 14-branch reproducibility.
- Candidate-arc comparisons cover initial-yaw horizontal XY/yaw only; no terrain z/roll/pitch claim. No Euler trajectory ADE or intermediate NN trajectory is implemented.
- CPU benchmark rows are host measurements. Separate NN, input-transform-plus-NN and full API timings; do not relabel them as robot-PC measurements.
- No planner selection/STOP changes, ROS commands, real-robot validation, unseen-class generalization or pre-contact guarantee is established.
- Preserve the initial/final freeze records. This report must not trigger recipe, architecture, threshold or collection-choice tuning from audit outcomes.

## 新規auditの資格付き状態別pose誤差

| 状態 | 区間 | 独立親 | 平均3D m | P95 3D m | 平均SO(3) rad |
|---|---:|---:|---:|---:|---:|
| descent | 14 | 1 | 0.261623 | 0.412726 | 0.206293 |
| moving_flat | 28 | 2 | 0.076114 | 0.146446 | 0.117526 |
| slow | 28 | 2 | 0.081114 | 0.171529 | 0.184775 |
| step_before | 28 | 2 | 0.065061 | 0.155318 | 0.106615 |
| turning | 28 | 2 | 0.092262 | 0.221852 | 0.141856 |
| uphill_before | 14 | 2 | 0.068222 | 0.109711 | 0.097924 |
| uphill_mid | 28 | 2 | 0.106031 | 0.275292 | 0.091025 |

軸別XYZの符号付き誤差・MAE・RMSE・中央値・P95、wrapped roll/pitch/yaw、重力傾斜、候補arc baselineのXY/yawは各metrics.jsonのmodel.pose/candidate_arcに保存。角度単位はrad。

入力shape・全raw値の有限性・binary maskを変換前に検査し、DEMの最新観測が250ms以上古い場合を拒否する。最終レビューの修正はinput-contract-correction.jsonを参照。各モデルの固定train IDと保存済み統計から計算した分位点・標準化後範囲はinput-diagnostics/*.jsonに保存し、統計の再fitはしていない。再作成は `PYTHONPATH=src:scripts python scripts/eat_nav_moving_pose2s_input_diagnostics.py --dataset <bound dataset> --checkpoint <checkpoint> --output <fresh JSON>`。
