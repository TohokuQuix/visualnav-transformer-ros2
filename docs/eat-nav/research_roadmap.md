# EAT-Nav 学生向け研究ロードマップと既知の限界

## 研究の最終目標と現在の証拠

最終目標は、地形のDEM、ロボットの身体状態、候補Pathから、各Pathを実行したときの運動・負荷・進捗を予測し、実機でも使える局所Path選択につなげることである。現時点で確認したのは、**候補Pathによって実際の運動が変わること**と、限られたデータでその運動差をある程度予測できることまでである。予測器を使ったPath選択の改善、未知地形への一般化、実機での性能は未検証である。

MI-A v3.1正式評価では、12 terrain seeds、4 terrain types、Front-folded／Rear-foldedの2初期姿勢を含む **192/192 process、576/576 candidate trials** がprotocol-validだった。同じ初期条件でPath候補を変えると2秒後のbody-frame XY変位が変わり、初期フリッパ姿勢によってPathへの運動応答も変わった。事前凍結した探索的Gateで**MI-A Pass**。ただし、terrain type・level・seedは結び付いており、未知地形への一般化を実証した結果ではない。[MI-A正式評価報告](motion-predictor-vnext/mi_a_v31_formal_result_report.md)とsealed evidenceの `mi_a_v31_motion_results.json` を参照。

MI-B Pilotでは、上記の576 trialを用いてMotion Predictorを学習した。少数例へのsmall overfitはPass。4 held-out terrain seedsでの主要指標、Arc−Directの2秒後XY変位差RMSEは、Fullモデルの2初期化で **0.0713／0.0724 m**、train-only Direction平均baselineで **0.0834 m**（約13～15%改善）だった。No Pathはbaselineから改善せず、Path入力の寄与が確認できた。一方、No DEMはFullと同程度で、一方の初期化ではFullを上回る。No Bodyとの差も小さく、DEMとActor stateの寄与はまだ明確でない。Front/RearによるPath応答差の予測も十分ではなく、Reverse +30°と `random_rough` などに誤差が残る。**MI-B Pilotは事前定義した探索的基準でPassしたが、正式MI-B Gateは未定義であり、正式Passではない。** [MI-B Pilot報告](motion-predictor-vnext/mi_b_pilot_report.md)とsealed evidenceの `pilot-results.json`、`pilot-subgroups-v2.json` を参照。

上記のMotion Predictorはoffline研究用であり、GUIに組み込まれた旧Baseline Evaluatorとは別物である。GUI demoはROS2 goalから候補score、C++ PureFollower command、RL Actor action、Isaac motionまでのソフトウェア経路を示す。旧Baseline Evaluator checkpointは**1 StateGroup×1 seed、12 trialのtrain-set smoke test**で、独立したvalidation/testがない。35 terrain条件は生成とorigin選択を確認したもので、35条件のnavigation benchmarkではない。GUI制御にはROS-free C++ follower coreを使い、production ROS followerは実行しない。実機転移は未検証である。

GUIの現行bankはDirect、Left +20°、Right −20°で、各8点。global RViz goalは0.75 m以内のlocal subgoalに変換される。回転・スケーリング後の入力は旧smoke trainingの分布外になりうる。MI-A／MI-Bの前進・後退Path実験とGUI bankの結果を同じ評価結果として混ぜないこと。DEMOの初期姿勢はFront-folded固定。将来の学習データ収集でFront/RearをStateGroup単位でランダム化する方針は維持するが、Phase C収集は未承認である。

## 研究課題と比較の観点

**中心課題：DEMや身体状態を入力しているにもかかわらず、それらが運動予測精度の改善に十分つながっていないのはなぜか？** 原因はまだ特定されていない。入力情報の表現方法、DEM・Path・身体状態の関係、ネットワーク構造、データ量と多様性、学習目標・損失関数・評価方法は、仮説を立てるための観点であり、確定した原因や推奨実装ではない。

| 問い | 確認済みの事実 | 未解決の点 |
| --- | --- | --- |
| R1: 教師信号 | MI-Bは2秒後のbody-frame XY変位とArc−Direct差を学習した。旧Evaluatorにはmotion residual、common-goal progress、観測reward、Painの別ラベル定義がある。 | どの教師信号が実行品質や選択結果に適するかは未検証。異なる量を一つのスカラーへ先に混ぜない。Painには検証済みの工学的許容値が必要。 |
| R2: 入力 | MI-BのNo PathはArc−Direct差を予測できず、Path寄与が見えた。No DEMとNo Bodyの結果から各入力の寄与は確定できない。 | DEM・Path・身体状態の関係がモデルにどう伝わり、どの条件で有効になるか。推論時に得られない情報を入力に加えない。 |
| R3: 表現・ネットワーク | MI-B Pilotは小規模なCNN DEM encoderとActor／Path入力処理を使用した。少数例のoverfitは成立した。 | 構造・表現・データ・目標のどれがheld-outでの弱い入力寄与に関係するか。特定の変更を解決策と決めていない。 |
| R4: Path bank | MI-Aの既存候補は運動応答に差を生んだ。GUI bankとは幾何と評価目的が異なる。 | Bank拡張は今回の学生課題に含めない。新しい幾何を使う場合は別途feasibilityと分布変更の検証が必要。 |
| R5: 一般化 | MI-Bはterrain seed単位でtrain／validation／testを分けた。held-outは4 seedで、terrain typeごとに1 seedだけ。 | Level・seed交絡、姿勢応答、未知地形、独立StateGroupへの一般化。env tileを独立seedとして数えない。 |
| R6: 停止挙動 | Goal／Cancel後のzero follower commandと、Actor action・root motionの継続は両立しうる。 | 実機の安全停止は未検証。[残留移動解析](cancel-residual-report-20261013.md)は測定事実と未解明点を分けている。 |

## 次回の打ち合わせまでの学生課題

1. 中心課題について、互いに区別できる仮説を**2～3個**立てる。
2. 各仮説の根拠を、MI-Bのablation・失敗例・ネットワークと入力契約のどこに見いだしたか説明する。観測事実と推測を分ける。
3. 各仮説を検証する実験を提案する。**既存データだけで実施できる方法を優先**し、比較条件、独立単位、評価指標を示す。
4. どの結果なら支持し、どの結果なら棄却または保留するかを事前に明記する。
5. AIを文献調査、仮説比較、実験設計の検討に使ってよい。ただし、AI案をそのまま採用せず、**なぜその実験で仮説を検証できるかを本人が説明できること**を求める。

次回までに新モデルの実装や大規模学習は依頼しない。Phase Cの追加収集、MI-C、実機接続、Path Bank拡張も今回の課題に含まない。

## 今後の実験の順序と承認境界

まず、上記の仮説と既存evidenceで検証できる実験案をレビューする。その結果を踏まえてから、必要な追加検証やデータを別途判断する。現在のDEMO、MI-A／MI-Bのevidence、split、checkpointは変更しない。MI-B Pilotの探索的Passを正式MI-B PassやPhase C開始の承認と読み替えない。

将来、新たな学習や評価を承認した場合にも、独立seed／StateGroupでsplitし、normalizationはtrainのみで作り、検証でcheckpointを選び、testは最後の評価だけに用いる。candidate間比較では同一pre-Path条件を保ち、失敗・censor・terminationを保存する。ここに記したのは再現性の条件であり、現在の学生作業として新規収集や学習を指示するものではない。

## 学生が最初に把握すべき未解決事項

- MI-BのPath入力効果は見えたが、DEM・Actor stateの効果、Front/Rear応答差の予測は未確定。Reverse +30°と `random_rough` に残る誤差も、原因を断定できない。
- MI-B held-outは4独立seedで、各terrainに1 seedのみ。terrain levelとseed／splitが交絡するため、未知地形への一般化を主張しない。
- 旧GUI Baseline Evaluatorのtrain-set scoreとMI-B held-out Motion Predictorの指標は別系列であり、GUIでPath選択が改善した証拠ではない。
- `k_rail_field` level 2の短いGUI demoではgoal progressが低かったが、1 runだけでは地形、bank、modelのどれが原因か分からない。
- GUI候補はRViz goalへ向けて回転・スケーリングされるため、凍結済みP1 teacher bankと入力分布が一致するとは限らない。
- ROS2 clock／TF／goal transportはローカルで試験したが、production follower、hardware latency、実機DEM整合性はGUI demoでは検証されていない。

技術的な入力形状、source、再現手順は[実験パイプライン](experiment_pipeline.md)、[MI-A正式評価](motion-predictor-vnext/mi_a_v31_formal_result_report.md)、[MI-B Pilot](motion-predictor-vnext/mi_b_pilot_report.md)を参照。環境起動は[environment setup](environment_setup.md)を参照。
