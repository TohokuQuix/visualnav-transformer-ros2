# 既存データ・実機条件の調査（2026-11-03）

調査のみ。新規の収集・学習・モデル実装は行っていない。参照コードは `reference-index.json`、元ファイルの凍結コピーは `references/`。以下の実測は10/29の約26秒bagに限る。長時間・他地形・他負荷の実機分布を代表するとはしない。

## 1. コード・成果物

- visualnav-transformer-ros2参照commit：`43bb7b69fde5d29095b6add17965aacd48ae58fa`、元branch `exp/eatnav-moving-state-pose2s-20261103`。
- 既存M_move SHA：`75fa9659d6924b4f6bae421498a45f485fb737d8e6b86e1a9a91a245eb9457b1`。
- 追加後M_final SHA：`3f64a6617702a5d50ac704ccc099cb18fc9e171aabf427e51b2865bd800d272b`。
- 既存入力はDEM `[B,4,2,20,20]`、19 command-free actor＋body9、候補8点と方向。新CLIは取得間隔250msを厳密要求。2秒XYZ・6D相対回転・2秒以内4イベントを出す。
- 既存807区間でM_move平均3D7.61cm、M_final8.91cm。新規168区間では9.29cmと9.76cm。既存公開結果は比較可能なsim評価であり、実機bag代理評価の15–21cm／14–17度と同じ母集団ではない。
- 旧学習設定 `config/eat_nav.yaml:2` と `demo_isaac_env_cfg.py:252` がDEM更新0.25秒。4Hzを実機比較で最適化した証拠はない。

## 2. 実機bagの実ファイル

`/home/ubuntu/eat_nav_planner_20261029_210926/bag/bag_0.mcap` と同ディレクトリのmetadata.yaml。bag本体SHA：`f2ba42fdc16c8a86853fd79e508eb98b0ca64638ef527a75cca941baa7127ba3`。

| topic | 件数 | 注記 |
|---|---:|---|
| /filtered_elevation_map | 87 | header時刻・記録時刻を別々に確認 |
| /bno055/imu | 2605 | orientation/gyro/acceleration値あり、covariance先頭−1を確認 |
| /joint_states | 521 | 16 joint、position/velocity/effort各16成分 |
| /diff_drive_controller/odom | 261 | 全件z=0、quaternion x=y=0 |
| /tf | 616メッセージ | map→chassis/bodyは261変換 |
| /flipper_fold_mode | 0 | 未記録。mode=0とはみなせない |
| 選択path・候補・goal等 | 0 | 実行候補を確定できない |

このbagにはbridgeの採用速度出力、flipper目標出力、操作モード状態、電圧・電流の確定ログがない。実機にtopicが存在しないことを意味しない。

### DEMの時刻

- header間隔：中央値300ms、最小200.160ms、最大399.845ms。重複stampなし。
- bag記録時刻の間隔：中央値300.725ms、最小157.385ms、最大438.730ms。
- DEM87件の先頭〜末尾25.577152363秒、記録時刻からの平均3.36237587Hz。
- 記録時刻−headerの中央値：DEM **−319.915ms**、IMU **−517.270ms**、joint **−516.680ms**、odom **−516.801ms**。

負の差を負の伝送遅延として使わない。時計domain差、記録時刻の意味、header生成点が未確定。中央値を単に引いて「同期済み」と扱うことも禁止。headerは実センサー取得時刻かmap更新時刻か未確定。bag記録時刻もNNノードの実受信時刻とは限らない。

### DEMの中身

map座標、0.1m、2m四方、20×20、grid回転identity、basic_layers空。

| layer | 実測 |
|---|---|
| elevation | 有限212–230/400セル、中央値223 |
| elevation_inpainted / elevation_smooth | 全87frameで400/400有限 |
| variance | 全87frameで有限値0セル（ゼロ分散という意味ではない） |
| horizontal variance | 有限244–252セル。垂直方向の信頼度として代用不可 |
| time / dynamic_time | 上流の意味・単位・更新条件は未確定 |

inpaintedの有限性だけで「観測済み／高信頼」と判断しない。現行handoffのinputs.py:96–99は選択height layerの有限性をvalidへ使うため、inpainted使用時には実観測マスクと区別できない。新仕様では分ける。

### 実機の位置・姿勢教師

wheel odomはplanar。map→chassis/bodyのzは−0.209397〜−0.166372mでroll/pitch成分もあるが、生成publisher・センサー融合・精度・輪滑りからの独立性はbagだけでは確定しない。map→odom／odom→bodyの記録edgeもない。

したがって、このbagのTFは測定poseの代理値としては使用できるが、独立した10cm精度の3D正解ではない。wheel odom由来速度とsprocket速度が同じ輪速に依存する場合、その比率だけで真の滑りを識別できるとも言えない。

## 3. 候補と制御

`bidirectional_bank.py:12–14,27–54`：前後各7本、角度0/−30/+30/−60/+60/−90/+90、解析arc長0.75m、始点を含む8点、7区間。角度は接線の総回転で、終点の方位角ではない。後退はXYを反転し方向−1を別に保持。

旧sim follower `p1_follower_core.cpp:204–243` はlookahead0.5m、最大0.30m/s、heading比例減速、|heading|≥0.6radでは並進0、angular cap1.6rad/s、goal tolerance0.2m、20秒のrecovery履歴を持つ。

ユーザーが実機側で確認した新しい対象は **eat_nav_cycle_follower**。lookahead **0.8m**、linear_speed0.30、angular_gain0.5、max angular1.6、20Hz、goal tolerance0.2。実機launchがtopic/frame等を上書きし、follower自身のsubcrawler出力はOFF。完全source/config/runtime parameter dump・実行中binaryの識別は未受領。旧simple followerで代用しない。angular_direction_sign等の未提示値から実効gainを推測しない。

0.75m候補は2秒の実軌跡ではない。最大速度でも2秒で0.60m、実際には減速・加速・旋回・滑り・終端処理で変化する。

## 4. フリッパー

旧sim collectorは開始姿勢rr,rl,fr,fl=`0,0,π,π`へ設定するが、fold modeは0、毎stepActorを実行する。前180度・後0度を保持する制御ではない（mi_fi_v1.py:14–34、collector:206,340–354）。

ユーザー確認の実機方針は動的：6actionのpolicy出力へ方向clip・干渉制約等を適用。foldを観測するpolicyのmode0、foldを観測しないpolicyの速度由来方向推定、ゼロ指令時の角度保持、2action policyでの実測角度保持は別分岐。使用Actor checkpoint／観測構成／起動設定が未確定なので、いずれの分岐も勝手に選ばない。

実機既定のflipper出力順FL,FR,RL,RRと、NNのRR,RL,FR,FLは異なる。position/velocityモードも分ける。velocityモード用kp3.0・limit1.5rad/s・deadband0.02rad、およびenv max_velocity2.0をpositionモードの実速度とみなさない。

## 5. 既存生データの再利用可能性

ローカル `/home/ubuntu/eatnav-moving-state-pose2s-data-20261103/`。既存dataset合計7,765、完全pose7,748。trainは再利用5,299＋前回追加532=**5,831**。全trainの保存DEM時刻は250ms間隔。

| field / level | R4 train | common train | 前回追加train |
|---|---:|---:|---:|
| boxes / 4 | 708 | 0 | 0 |
| boxes / 1 | 0 | 0 | 84 |
| hf_pyramid_slope / 4 | 600 | 0 | 126 |
| k_rail_field / 4 | 579 | 0 | 70 |
| k_rail_step_field / 4 | 767 | 96 | 0 |
| k_rail_step_field / 1 | 0 | 96 | 0 |
| pyramid_stairs / 4 | 673 | 96 | 84 |
| pyramid_stairs / 1 | 0 | 96 | 0 |
| pyramid_stairs_inv / 4 | 787 | 0 | 84 |
| random_rough / 4 | 801 | 0 | 0 |
| random_rough / 1 | 0 | 0 | 84 |
| 合計 | 4915 | 384 | 532 |

- R4代表raw-dem-0080.npzは現時点400ray。代表directoryのraw-demは0000/0040/0080/0120/0160で約2秒ごとのsnapshot。20Hzでroot/commandがあっても300ms時点の新しいrayはない。joint履歴もない。
- common代表rawは4env×4frame×400ray×3。4取得時刻だけ。traceにjoint/flipper/torque等あり。
- 前回追加のinitial-inputは4DEM。anchor-visible-stateは現在ray cacheであり、過去の密なray列ではない。traceは52jointのpos/vel/torque等を保存。
- scannerのupdate_period自体が0.25秒。隠れた高Hzの実観測があるとは確認できない。

**新契約（真の300–333ms観測・補間マスク・校正済み時刻・実機制御）を全て満たすと証明できた件数は現在0。** これは既存2秒教師が全部無価値という意味ではない。旧時刻のままlegacy profileで事前学習・ablationに利用できる。joint履歴を追加監査できるtrain poolはcommon384＋前回追加532=916枝が上限で、有効認定数ではない。250/500msの実frame選択を333ms取得と偽らない。

## 6. 資源・変更範囲

調査時GPU RTX4090 Laptop16GiB、使用約7.4GiB、RAM62GiB中available約53GiB、disk約250GiB空き。既存R6処理等を無断終了していない。承認後のbenchmark時に再測定する。

新branchは `exp/eatnav-real-conditions-pose2s-20261103`。今回の変更は仕様・調査・参照snapshotのみ。実機用コード、モデル、収集runner、元データは変更していない。originは古いローカル一時pathを向いていたため共有設定を勝手に変更せず、URL確認済みgithub-pose2s remoteへ仕様書のみ共有する。
