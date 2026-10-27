# K-Rail（k_rail_field）継続走行・転倒スタック学習・Evaluator比較 v0.1

2026-10-23 JST。**レビュー用仕様案。新しいK-Rail収集・学習・実験は未実施。**
利用者からの教授の助言を受け、実施中のstep field比較と並行して作成した。
本案の承認前に、現在実行中の比較の地形・モデル・制御ルールは変更しない。
新しい`k_rail_field`の実装・収集・学習・評価は、現在のstep fieldの9走行と結果監査が完了し、かつ本案の承認後に開始する。並行して行うのは本仕様案の作成・レビューだけとする。

## 1. 対象と目的

以後、本研究でいうK-Railは **`k_rail_field`**（斜面上にレール状の梁を配置したフィールド）とする。
**`k_rail_step_field`はstep field（段差ブロック型）として明確に区別する。**
過去のsealed成果物を改名・書換えせず、terrain IDを基準に読む。

目的は、短い局所経路の到達で走行が実質停止することを避け、K-Rail上で複数の地形・姿勢を経験し、その結果を教師化して学習し、学習済みEvaluatorのリスク回避効果を比較することである。
時間を延ばしただけでは学習が進んだとは扱わない。走行、入力/結果保存、教師化、重み更新、閉ループ比較を区別する。

関連資料:

- [現在のstep field比較仕様](krail-risk-ablation-spec-v0.1.md)
- [既存学習結果](command-free-learning-20261021/learning-report.md)
- [教師v2と入力の定義](fall-stall-learning-spec-v0.1.md)

## 2. 現状と短いpathの問題

現行bankの候補は弧長0.75m・8点である。PureFollowerは終点との距離が0.2m未満、または終端接線方向で終点を越えた場合に局所到達状態となり、速度指令を0にする。
直線なら約0.55m進むと到達扱いになり得るため、「約60cmの経路」という指摘と整合する。
現在のcollection/GUIも2秒周期でpathを更新しており、1本の短いpathだけを90秒追い続けているわけではない。ただし、2秒以内に局所到達すると次の更新まで停止が残り得る。

継続走行によって転倒・スタックが必ず増える保証はない。短いpathの局所到達による停止を避け、経験する位置・姿勢を増やすことを狙う。

本案では次の3つを分離する。

1. **候補preview**: Evaluatorが採点する0.75m・8点。既存入力schemaと候補形状を維持する。
2. **追従用path**: 終端へ近づくと先へ継ぎ足す。局所終端への接近だけでepisodeを終了しない。
3. **全体目標**: 順序付きwaypoint列。到達したら次の目標へ進み、同じ物理走行を継続する。

転倒の少なさがRLの踏ん張りによるものかは仮説である。転倒・near-fall・並進停滞・旋回応答不足・通過成功を別々に記録する。
旋回専用候補やActorの改造は初回に加えない。旋回スタックの希少性は不足として報告し、全件収集後も学習開始を延期する理由にしない。

## 3. 初回の範囲と区切り（提案値）

| 段階 | 初回の上限 |
|---|---|
| K-Railデータ収集 | 新規8 base seed（train6 / validation2）、anchor2時刻、4候補、計64 branch走行 |
| anchor | settling後0秒 / 10秒。10秒anchorは同じ決定的prefix走行を再現して到達する |
| 1 branch | anchor後最大10秒。10秒anchorのprocessはprefix込み最大20秒＋settling |
| 学習 | 新規rail＋既存train、既存28次元architecture、optimization seed123、最大300 epochの1 run |
| 評価 | 収集と非重複の新規development3 seed、4条件、各最大60秒、計12走行 |
| 再試行 | infrastructure不具合のみ各論理走行1回まで。物理失敗・陽性不足を理由に再試行/seed追加しない |

収集64走行と評価12走行をそれぞれ終えた時点で報告し、陽性seed数が揃うまで自動的に追加しない。
これは探索的な1回目の学習ループであり、正式な統計的十分性や安全性を保証する数ではない。
予算・tile寸法・path継足し・学習混合比は本案での提案値であり、承認後にmanifestへ固定する。

## 4. 地形・身体条件・走行可能範囲

- terrain IDは厳密に `k_rail_field`。`MeshKRailFieldTerrainCfg` / `k_rail_field_terrain`を使用することを実行時に検証する。
- 初回はlevel4、front-folded、1env、固定RL Actor、20Hz。地形の梁高さ・梁幅・傾斜等は現行OnixのK-Rail設定を継承し、実際の値・生成関数・source SHAを保存する。
- tile面積は**16m×16m**を初回案とする。小さいtileで走行延長が境界落下へ置き換わることを避けるための変更であり、既存8m等の条件との同一実験とは扱わない。実行時の元寸法と変更寸法を両方記録する。
- 選択tileのworld範囲、レールが配置された範囲、origin、difficulty band、mesh identityを保存する。scene.env_originsはspawn patchの位置で、tile幾何中心とは一致しない場合がある。境界と目標の基準はgeneratorの配置transformおよびpallet/梁の実配置範囲から定め、spawn origin±size/2で代用しない。別levelや別tileへ渡る走行をK-Rail level4の結果へ混ぜない。
- tileと実際のレール配置範囲の共通領域から、全方向2mを控えた領域をroot走行範囲とする。ロボットの平面外形・目標pathの余裕もこの内側で検証する。2mが身体寸法に対して不足する場合は起動不成立として報告し、勝手に余裕を縮めない。
- 範囲逸脱は`boundary_censor`として区別し、その後の落下をK-Rail形状による転倒教師へ加えない。逸脱前に確認済みのeventは保持し、未観測区間はunknownとする。
- 外力、故意の転倒姿勢、結果を見たspawn変更・難化は行わない。自然terminationのtilt_overとfinite検査は維持する。
- 初回の実走行画面にレール状梁があることを、設定/mesh情報と併せて確認・保存する。名前だけを根拠に地形同一としない。

## 5. pathの継足しと目標の連続化

### 5.1 共通path更新

候補順はforward直進、forward−30度、forward＋30度、reverse直進。候補preview・方向encodingは既存のままにする。
通常の候補再評価とbody起点のpath再構成は40 step（2秒）周期を維持する。

20Hzで追従用pathの残arc長と終点距離を監視し、いずれかが0.30m以下なら、followerへ指令を問い合わせる**前に**同じ候補の継続segmentを末尾へ追加する。
追加segmentは弧長0.75m、現在の末端位置・解析的な末端body headingを起点とする同じ曲率/方向のbank形状とする。重複する先頭点は除く。
各segmentの解析的な末端body headingは、そのsegmentの開始body heading＋候補angleとする（reverseも同じ加算）。次segmentは`endpoint + R(terminal_body_heading) * bank_points(direction, angle)`で作る。reverseでは進行接線のπ反転をbody headingへ誤加算せず、最終chordの近似角を解析的headingの代わりにしない。

- 継足しは候補IDの切替ではない。guardの保持時間・前後進確認回数をリセットしない。
- 元の0.75m previewは保存し、継足し後の実追従path・末端heading・event時刻・path IDを別に保存する。継足した長いpathを8点へ無断で再標本化してモデル入力へ入れない。
- 既存PureFollowerにはpath更新APIがないため、初回は継足し後の全polylineで追従器を作り直す方式に固定する。内部のmotion/progress/recovery/clock履歴は破棄し、branch開始・2秒更新・継足しで同じreset契約を使う。これは状態を保持したpath延長とは異なる制御変更であり、全収集・全評価条件に共通適用して記録する。selector/guardの保持時間・方向確認、観測履歴、Actor状態、物理状態は継足しでresetしない。
- 2秒境界ではその時点のbody poseからpreviewと追従pathを作り直す。継足しでmacro cycleを新設したり、教師windowの時計をリセットしたりしない。
- 1 cycleの継足しは最大4回とし、超過は`control_error`として0指令を出してその走行を終了し、観測済み区間と失敗を保存する。無制限な継足しで不正状態を隠さない。
- STOP中は新たな走行指令を出さずfollowerを進めない。再開時は現在poseから再構成する。

### 5.2 順序付き目標

初回の目標列はtile中心相対座標で `(0,-2) → (2,-2) → (2,2) → (-2,2) → (-2,-2) → (0,0)` とし、評価時間内は順に周回する。
全点と必要な身体余裕が節4の走行範囲に入ることを実行前に検証する。入らない場合は別の目標を自動選択せず起動不成立とする。

post-stateで目標半径0.25m以内を確認したら、まず幾何学的進入を記録する。正常waypoint到達として加算し次の目標indexへ進めるのは、同stepにtilt45度未満、適用中の保護/予測guardがSTOPでない、自然failure/境界censorなしを満たす場合だけとする。これは操作的な到達基準であり安全保証ではない。条件を満たさなければ現在目標を維持し、episode/resetせず物理観測を続ける。
次の2秒判定で新しい全体目標へ候補選択を更新し、それまでも追従pathの継足しは続ける。
現step field比較では、目標へ入り傾斜83.56度の時点で物理停止し、85度転倒・60度×5 sampleのnear-fallが未確認となる例があった。このようなgoalによる打切りを安全成功と扱わないため、今回のwaypointでは物理を停止しない。60秒等の本来の観測終了時にも、最終傾斜・未確定near-fallの連続sample数・打切り理由を保存する。
単なる局所path終点を全体目標への到達と数えない。最後の目標後も時間内は次のlapへ進む。

評価は60秒、自然termination、境界censor、GUI close、実行エラーのいずれかで終了する。
最後のstepでの目標到達も記録し、同stepに転倒等があれば正常到達成功とは分ける。

## 6. 収集: 動的状態からの候補継続outcome

現行GUIの候補切替traceを、そのまま固定candidateの10秒教師へ追加しない。
同じ状態から候補を継続したoutcomeを取るため、**prefixを再現してから分岐する**方式を使う。Isaacの任意snapshot復元機能の新規開発は初回の前提にしない。

1. 各base seedについてanchor0秒と10秒を事前割当てする。
2. anchor10秒のprefixでは、節5の目標列を、riskを使わない決定的な幾何goal指向selectorで追う。scoreは候補endpoint方向へ0.75m進むと仮定したgoal距離減少、同点は固定順とする。候補選択は2秒周期、継足しは節5と共通。
3. 同じprefixを4つの独立processで再現し、anchorで全4候補をそれぞれ10秒継続する。branch中は候補IDを変えず、2秒ごとのbody起点更新・同一候補の継足しだけを行う。
4. anchorの4-frame DEM、raw/実Actor入力、body・joint状態、遅延buffer、prefixのselector/follower状態・経路履歴と乱数状態を保存する。floatはatol1e-6/rtol0、整数・mask・RNG状態は完全一致で4候補間を照合する。Actor観測辞書はキーごとの数値配列へ展開し、object dtype/pickleを使わない。Python/NumPy/TorchのRNGは数値配列・JSONで記録する。branch開始時はprefix followerを閉じ、候補ごとの新しいfollowerを同じゼロ履歴から作る。
5. anchorの照合不成立は`anchor_mismatch`として全attemptと差分を保持する。その4候補StateGroupを学習dataset/pair lossから除外し、共通入力を捏造しない。代替seed・都合のよい再試行を加えない。
6. prefix中の自然失敗でanchorへ到達できなければ`anchor_unavailable`とする。存在しない初期状態を複製せず、prefix失敗自体は別の観測記録へ残す。

branch収集では、自然な転倒・スタックを観察するため、学習済みriskによる候補除外・予測STOPおよび実測45度STOPを使用しない。
固定Actor/追従器・自然tilt_over終了・境界censor・finite検査を使う。この収集policyと、節8の保護付きEvaluator policyを混同しない。
既存収集と同様に実機操作ではなくシミュレーションで行う。転倒を起こすための外力や物理の変更は加えない。

新規seedは既存pilot・全追加batch・step field比較・既知runtime registryとbase/派生seedの非重複をmetadataだけで確認する。
train6 / validation2のrole、anchor、候補順、実行順、source、全終了条件を収集前に固定し、結果を見てroleやseedを入れ替えない。
全64論理走行を実施・失敗含めて報告したら、この収集batchは終了する。

## 7. 教師・dataset・学習

### 7.1 教師

branchのactivationを時刻0とし、1/2/5/10秒のevent・progressとmaskを生成する。prefix未来をbranch教師へ混ぜない。
転倒85度/tilt_over、near-fall60度×5 sample、並進/旋回40 intervalの教師v2定義を維持する。
並進進捗は各2秒cycle開始時の固定previewへの投影で算出し、継足しイベントで進捗を加算・リセットしない。継足し後の任意path長をそのまま進捗とみなさない。各cycleのpreview投影は最大0.75mでclampされるため、教師のprogressと物理的な累積走行距離は別々に保存する。継足しだけを理由とした教師v3への変更は行わない。

実際の継続policy（2秒更新＋継足し）を新しいcontroller revisionとしてdataset manifestへ記録する。
定義と保存信号の対応を検証し、既存teacher v2をそのまま適用できない信号形式の場合はadapterを新設する。ラベル定義を結果に合わせて緩めない。
reset前のroot/角速度等を使い、未観測・boundary censor・prefix不成立を陰性へ補完しない。
同じbase seedのprefix・anchor・候補・途中windowは同じsplitへ置く。途中frame数を独立seed数として数えない。

### 7.2 学習の初回案

既存28次元architectureを維持し、新規K-Rail trainと既存step/階段等のtrainを使用して**新規モデルをseed123から学習する**。
初回は既存学習条件（Adam lr0.001、CPU1thread、full batch、最大300 epoch）を基準とする。既存checkpointへ上書きしない。
旧モデルを固定baselineとして残し、古い正規化のまま新しい身体状態を入力することを学習時の当然の前提にしない。

- 正規化・pos_weight等は両sourceのtrainのみでfitする。rail validation、既存validation、評価seed、reserved testはfitへ使わない。
- lossは既存の`event + 0.1*progress + 0.1*pair`（10秒event主head、既存mask/ペア成立条件）を維持し、source別の有効mask内平均lossに対して `2/3 * rail + 1/3 * existing` を初回の混合重みとする。source内に有効targetがない項は対応を明記して除外し、NaNや架空陰性で埋めない。
- 全既存trainと全新規trainの会員・maskを保存する。既存trainを混ぜる目的は過去地形の情報保持であり、保持が保証されたとは扱わない。
- checkpointは `2/3 * rail validation BCE + 1/3 * existing validation BCE` が最小のepochで選ぶ。同点は早いepoch。railに有効eventが全くない等で基準が算出不能なら事前規則としてexisting validation BCEを使い、その限界を表示する。
- source別・head別の陽性/陰性/unknown、陽性seed数、分類・progress指標を分ける。希少eventの正式性能を主張しない。
- head supportはpooled trainに既知陽性>0かつ既知陰性>0があることとし、既存training_support契約を使う。rail単独のcoverage不足とpooled supportを区別する。新データの転倒・旋回スタックが少なくても、観測可能なtargetと既存trainで初回学習を実施する。未対応headはunsupportedとして明示し、出力を無断で0リスクとして扱わない。

既存validationの悪化も報告し、K-Railだけの指標改善を全体改善と呼ばない。
railデータそのものが不成立で学習に使えるsampleが0の場合は、学習済みと装わず原因と全件結果を報告する。

## 8. K-RailでのEvaluator実験

収集seedと非重複のdevelopment3 seedを評価前に固定し、同じK-Rail・身体・目標列・path継足しで以下を比較する。

| 条件 | モデルとrisk機能 |
|---|---|
| R0 | 旧固定モデル、転倒＋スタックrisk有効（新規学習前の基準） |
| R1 | 新モデル、risk項なし（progress予測は使用） |
| R2 | 新モデル、スタックriskのみ |
| R3 | 新モデル、転倒＋スタックrisk |

初回は3 seed×4条件＝12走行、各最大60秒。条件順はcaseごとに循環させてmanifestへ固定する。
R1/R2/R3が既存の3条件比較に相当し、R0はK-Rail追加学習前後の比較用である。
R0−R3にはデータ・正規化・学習の変更全体が含まれ、特定headだけの効果とは呼ばない。

旧モデルR0と新モデルの比較可能headをそれぞれの保存済みtrain supportから走行前に固定する。unsupportedのheadを必要とする条件はN/Aとし、黙って別条件へ置き換えない。対応済み条件は実施し、不足だけを理由に全比較を無期限延期しない。

採点重み、候補除外、通常切替、共通45度保護と35度以下の再開条件は[step field比較仕様](krail-risk-ablation-spec-v0.1.md)のA0/A1/A2に対応させる。
全条件で共通のpath継足しと目標列を使い、候補の保持・前後進確認を継足しでリセットしない。
実測保護によって転倒が抑えられ得るため、停止・再開を別評価する。0指令で身体が静止するとは仮定しない。

主要指標は順序どおりの正常waypoint到達数、lap完了、規定route上の進捗、最初のlap完了時間、動作中stall、姿勢、保護/予測STOP、boundary censorとする。
最初のlegの始点は各seedのsettling後spawn XYとし、初回lapはそこから最初のwaypointへ進む。以後のlap始点は最後のwaypoint `(0,0)` とする。
route進捗は正常到達済みlegの規定長の累積（lap offsetを含む）＋現在legへのclamped投影arcで計算する。投影arcは`clip(dot(root-start,end-start)/||end-start||^2,0,1)*||end-start||`。1 control stepで正常到達を加算するwaypointは最大1つ、列末尾の正常到達でlapを1増やす。転倒等と同stepの幾何学的進入は別記し、正常到達数へ加算しない。waypointを飛ばした近道や、目標切替直後の距離増大を誤って成功/悪化へ数えない。
単一goalの4m残距離を使う過去step field比較と、数値をそのまま横並びの改善率にしない。

全条件の開始状態・Actor実入力・履歴を数値配列で照合し、状態が違う組は比較不成立とする。
学習の同一candidate継続と、評価の2秒ごとの候補切替には条件差が残る。
同じ入力上のscore差は解析できるが、未選択候補の実outcomeを観測したことにはならない。

## 9. GUI・保存・確認

- 初回から実シミュレーションGUIを表示し、terrain ID、rail geometry、seed/role/anchor/branchまたは評価条件、sim時刻、現在waypoint、候補previewと追従用path、risk、STOP理由を表示する。
- host起動コマンドと実際のDocker/Isaac起動コマンドを表示・保存する。実行前のruntime/Actor/config/source/model SHAと、全終了理由を記録する。
- DEM/raw rays/4-frame stamps、pre-noise/実Actor入力、body/joint/遅延状態、goal、候補、継足し、指令、実移動、reset前状態を保存する。入力NPZはobject dtypeを禁止する。
- prefix再現、branch初期照合、数値配列保存、継足し前後のpreview不変、局所goal停止の回避、reverse継続、40 interval時計、waypoint更新、境界censor、STOP全3軸0をテストする。
- 旧source・sealed成果物・reserved testを保持し、新しいnamespace/output/schema revisionを使う。既存testのraw/ラベル/coverageは開封しない。
- 失敗attemptとunsupported/unknownを含め全件報告し、保存入力から予測・選択・教師を再現する。

## 10. 終了条件と次の判断

初回64 branchの収集報告、利用可能データでの1回の学習、対応可能条件の12評価予定に対する結果・N/A・失敗と監査報告を揃えたら一区切りとする。
陽性不足・都合の悪い結果を理由に、同じbatchの収集数や学習探索回数を自動追加しない。

- R2/R3でroute進捗と正常到達が改善するなら、Evaluatorの候補選択への正式統合を進める材料とする。
- STOPだけ増えて進まない場合は、予測・閾値・候補・追従のどこが制約かを分ける。
- 新規データが必要なら、失敗した地形・進入方向・姿勢を特定して次の小規模batchを計画する。候補に迂回方向がなければ、データ量だけを原因にしない。
- 転倒しないが進めない例と、通過できる例を対照としてreviewする。Actorによる転倒抑制の因果関係や実機安全性はこの比較だけで結論しない。

実装順は、K-Rail形状/領域確認とpath継足し → 64 branch収集・教師化 → 既存データを併用した学習 → 12走行のEvaluator比較 → 監査・報告とする。
本案は仕様作成までの成果物であり、承認後に実装・収集を開始する。
