# Forward / Reverse Unified Path Bank: Phase B追加gate

## 判定

**B-R0は限定的にPass、B-R1の±30°段はFail。Phase Cはblockedのまま。** 後退DirectはPureFollowerから負速度指令を出せ、Isaac Labの4 envで実際に後退した。一方、独立terrain seedでのReverse +30°は初期角速度指令が正でも、2秒間の平均root yaw変化が負だった。事前凍結した「後退Arcの平均yaw符号がarc角の符号に合う」条件を満たさない。±60°/±90°のphysics rollout、データ収集、学習は実施していない。

## 実装と保全

- 旧13 Arc Bank、旧MotionPredictor、旧Evaluator checkpoint、既存Phase A/B成果物は変更していない。作業基点は `946f5cd6d4b6360b369f6f4667ed141cf291a9c9`、branch `exp/eatnav-motion-predictor-vnext`。
- 新しい独立bank `eatnav_forward_reverse_path_bank_v1` は8点・0.75 m等弧長、前進/後退それぞれDirect・±30°・±60°・±90°の14候補を保持する。角度は移動方向に沿った接線の旋回角。後退で正角はbody-frame右側へ曲がるが、幾何上の車体yaw変化は正角。候補の `direction` はbank/model/validationに明示し、既存C++ followerへはPathのみを与える。Followerが推定した `travel_direction` と照合する。
- Direction `[B,K,1]` をPath特徴と同じfusion headへ入れる単一のXY MotionPredictorを別schemaで追加した。4-frame DEM CNNとActor state encoderは再利用した。モデル学習は未実施で、directionが学習されることは未確認。
- Dataset準備schemaにStateGroup、terrain seed、signed direction、XY/yaw label、valid maskを定義した。後退Xを正値に変換しない。splitはStateGroup単位、normalizationはtrainのみという既存規則を維持する。今回の検証結果を学習datasetには昇格していない。
- 後退Directと14候補の実C++初期指令・幾何は [offline audit](forward-reverse-bank-audit.json)、候補一覧と状態は [provisional manifest](forward-reverse-bank-manifest.json)、per-env実測とseal SHAは [gate data](forward-reverse-gates.json) に保存した。

## Followerと座標系

PureFollowerはPathの0.5 m接線から前後を判定し、後退時は制御yawを車体yaw+πへ移す。heading errorが小さいとき `active_direction * 0.3` m/sを出す。後退Directの原点初期指令は `(-0.3, 0, 0)`、`travel_direction=-1` だった。Goalは終点から0.2 m以内または終端延長線で成立し、その後はzero command。直線と±30/±60/±90のoffline初期指令はすべて意図したdirectionと角速度符号に一致した。実行時はbody-frame Pathをrobot poseとtile originからtile-localへ変換し、初期poseでC++ coreを起動する。

`direction` は現状ROS `nav_msgs/Path` やPureFollower ABIに直接渡らない。Pathの接線から推定したdirectionと明示値が将来の全stepで一致するかをruntimeで確認する必要がある。今回のReverse Directと±30°では、全4 env・全40 stepのdiagnostic `travel_direction` は−1だった。

## B-R0：後退Direct

`random_rough` level 2、seed `2026101101`、4 env、40×50 ms。Forward Direct、Reverse Direct、zero commandの3 fresh processはすべて40/40 step、finite label、同一pre-Path比較、native exit 0、inner/outer SHA sealを通過した。

| 条件 | 2秒平均body X変位 | 初期linear command |
| --- | ---: | ---: |
| Forward Direct | +0.441 m | +0.300 m/s |
| Reverse Direct | −0.217 m | −0.300 m/s |
| Zero command | +0.007 m | 0 m/s |

Reverse−zeroのpaired X差はenv 0–3で `−0.257, −0.235, −0.221, −0.181 m`、平均 `−0.223 m`。事前固定した平均≤−0.05 mかつ3/4 envで負というR0条件を満たした。別seed `2026101102` のReverse Directも平均−0.197 mだった。ただし、同一seed・同一envのfresh-process反復による分散は測っていないため、数値ラベルの厳密な再現性は未確認。

全env・全stepから集計した平均body X速度はForward `+0.226`、Reverse `−0.115`、Zero `+0.006 m/s`。16本のsprocket joint平均角速度は同順で `+3.830`、`−3.125`、`+0.370 rad/s`。6次元Actor actionの初期値・平均値、env別body速度・yaw・sprocket角速度は [Actor audit](forward-reverse-r0-actor-audit.json) に残した。これにより、負速度指令だけでなく車体とsprocketの後退応答も確認した。

## B-R1：独立seedの±30°段

`random_rough` level 2、seed `2026101102`。Forward/Reverse DirectとForward/Reverse ±30°の6 fresh process、24 trialはすべてprotocol-valid。pre-Path root/joint/Actor/DEM/terrain/material/seed gate、40-step trace、terminal 0、SHA sealを通過した。

| 条件 | 平均body X変位 | 平均root yaw変化 | 平均積分yaw command |
| --- | ---: | ---: | ---: |
| Reverse Direct | −0.197 m | −0.159 rad | +0.090 rad |
| Reverse −30° | −0.173 m | −0.160 rad | −0.144 rad |
| Reverse +30° | −0.220 m | **−0.128 rad** | +0.344 rad |

Reverse +30°の初期角速度は全envで約+0.088 rad/s、積分指令も正だが、絶対yaw変化は負。Reverse Directとの差は+0.031 radで、候補由来の正方向の変化を示す診断値ではある。しかし、凍結gateは絶対yaw符号を要求しており、この結果を理由に閾値や定義を変更しない。Reverse −30°とDirectの差は約−0.001 radで、候補固有の旋回応答は弱い。地形接触、RL Actorの角速度追従、共通yawドリフトが原因候補だが、この測定だけで原因は特定できない。

R1の最初のattempt rootはcontainer stage名の衝突で無効。R0出力をR1 Directの最初の2件へ誤ってコピーした。3件目はseed不一致をpre-Pathで検出しPath公開前に停止した。元root・sealを保存し、host stageをfull attempt pathのdigestで一意化、simulator input manifestとhost manifestのseed/direction/angle/source/checkpointを照合し、failure markerを拒否する修正を加えた。新rootで6件を再測定した。旧rootの2件はR1結果に含めていない。

## 資源、provenance、残る判断

R0のGPU device-total peakは5,858 MiB、R1有効6件は5,776 MiBで、固定cap 6,656 MiB内。R0合計wall 61.1秒・保存量3.24 MB、R1有効6件は140.7秒・6.88 MB。最低host available RAMは約80.5 GiB。各runのnative exit 0、cleanup、inner/outer sealを確認した。raw evidenceは `/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/forward-reverse-r0-v1/` と `forward-reverse-r1-v2/`、無効attemptは `forward-reverse-r1-v1/` に保全。

新bankは**provisional**であり、14候補の採用仕様は未確定。Reverse ±60°/±90°はoffline初期指令のみ確認し、physics feasibilityは未検証。次は独立StateGroupと左右ミラー条件で共通yawドリフトと候補固有のyaw応答を分ける実験設計をレビューする。凍結B-R1 gateを改訂する場合は新protocolに分離する。Phase Cで必要なsplit/normalization、direction入力、XY/yaw labelの実測検証、候補集合のfreezeは未完了である。

**学生向け更新案:** 新bankを現行ナビゲーションの既定候補として紹介せず、後退Directの指令・走行は確認済み、後退Arcの2秒旋回応答とMotionPredictorの学習可能性は未確定と記す。Goalはモデル入力ではなく予測XYからの進捗計算にのみ使い、local Path追従とGlobal Goal進捗を区別する。
