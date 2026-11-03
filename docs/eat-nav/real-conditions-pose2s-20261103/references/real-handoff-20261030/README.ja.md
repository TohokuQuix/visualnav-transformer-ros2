# EAT-Nav R5 実機ノード

`random_rough/seed123` の新しい14候補チェックポイントと `R5FourteenGuard`を既存 `simple_trajectory_follower` と `onix_policy_bridge` に接続する。既定は可視化のみ。自動モードへの切替や実機の起動は行わない。

## ビルド

Jazzy の Python 3.12、NumPy、PyYAML、PyTorch が必要。GridMap の ROS 2 メッセージパッケージも必要。ロボットからtopicが見えていた端末と同じROS環境を使う。

```bash
source /opt/ros/jazzy/setup.bash
source ~/onix/install/setup.bash
cd ~/visualnav-transformer-ros2
colcon --log-base build/real_robot/log build \
  --base-paths ros2/eat_nav_real_robot --packages-select eat_nav_real_robot \
  --build-base build/real_robot/build --install-base build/real_robot/install
source build/real_robot/install/setup.bash
```

`grid_map_msgs` がない場合、ROS2 Jazzy用の `ros-jazzy-grid-map-msgs` を導入するか、既存のROS2版GridMapワークスペースをsourceする。ROS1のメッセージは使用できない。モデルはリポジトリから読むため、このクローンを残す。

## 接続とモデル

| 入力 | 型 | 用途 |
|---|---|---|
| `/filtered_elevation_map` | grid_map_msgs/msg/GridMap | 高さ地図。既定layerは `elevation` |
| `/bno055/imu` | sensor_msgs/msg/Imu | 姿勢・角速度。実値を使用し、有限性とquaternionを検査 |
| `/joint_states` | sensor_msgs/msg/JointState | 位置・速度・effort。関節名で取得 |
| `/flipper_fold_mode` | std_msgs/msg/Int8 | -1/0/1。未受信時は既存bridgeと同じ0。受信後はその値を保持。不正値は拒否 |
| `/diff_drive_controller/odom` | nav_msgs/msg/Odometry | 実測速度。`odom_topic` で `/Odometry` 等へ変更可能 |
| `/tf`, `/tf_static` | tf2_msgs/msg/TFMessage | 地図・機体・IMU・Odometry child frame の変換 |
| `/goal_pose` | geometry_msgs/msg/PoseStamped | 目標位置 |
| `/eat_nav/cancel` | std_msgs/msg/Empty | 目標キャンセル |

関節名は `chassis/{rr,rl,fr,fl}_flipper_joint` と `chassis/{fr,rr,fl,rl}_sprocket_axle`。JointStateはフリッパーposition、スプロケットvelocity、8関節のeffortを欠けずに供給する。配列順は任意。欠損値をゼロで補完しない。

地図は列優先または行優先レイアウトラベル、循環開始位置、任意の地図サイズ/解像度、水平yaw回転に対応する。傾いたGridMap poseは拒否する。高さ値はheader座標系の高さとして扱う。機体水平座標で0.1m・20×20へ再標本化し、0.3〜0.8m環状領域の高さ中央値を引く。valid率25%未満は拒否する。元のGridMapについては[公式定義](https://github.com/ANYbotics/grid_map/tree/ros2/grid_map_msgs/msg)と[配列格納仕様](https://github.com/ANYbotics/grid_map/blob/ros2/grid_map_ros/include/grid_map_ros/GridMapMsgHelpers.hpp)に従う。

3Hzの配信を目標に、4枚の異なる実観測を約333ms間隔で選択する。欠測を考慮し、各目標時刻から±1秒以内で時系列順に組み合わせる。時間窓が重なっても同じ観測を2回使わず、誤差が最小になる組合せを選ぶ。最新DEMは基準時刻から±1秒まで許容する。stampの書換えや観測の複製はしない。履歴が不足する間は待機し、後続の実データで再開する。単発の遅延・不正パケットで有効履歴を全消去しない。

学習時は250ms間隔であり、現在の約333ms・±1秒の許容条件とは異なる。この条件での実機予測精度は未検証。`dem_max_age_sec`、`history_interval_sec`、`history_tolerance_sec` で変更できる。

IMU・関節・Odometryは、DEM時刻と現在のROS時刻から±1秒以内の観測をキャッシュから選ぶ。最新TFは現在時刻から±1秒、時刻指定TFはその指定時刻から±1秒を許容する。指定時刻のTFがまだない場合は、時間差が範囲内の実際の最新TFを使う。到着待ちのDEMは時刻条件を満たす間だけ保留して再試行する。静的TFのstamp=0は期限検査の対象外。

情報不足や1秒を超える欠測では待機する。有限値、関節名、有効セル率25%などデータ内容のチェックは残す。ゲートの速度指令の受信期限など、制御系の短い期限はこのセンサー許容とは別に維持する。`/flipper_fold_mode` の初期値は0。不正な受信値を0に置換しない。

Odometryのtwistは `child_frame_id` の座標系と解釈する。機体原点へ速度を変換し、IMUもTFで機体へ変換する。BNO055のorientation/角速度は既存bridgeと同様にメッセージの実値を使用し、covariance先頭-1だけでは無効と判定しない。値自体の有限性・quaternionは検査する。


## 可視化のみで起動

Zenohルータは別端末で起動し、ロボットと同じRMW/ROS_DOMAIN_IDを使う。下の環境変数はロボットと接続していた端末に合わせる。

```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 launch eat_nav_real_robot real_robot.launch.py
```

system Python にPyTorchがなく、既存のPython 3.12仮想環境にある場合は、そのsite-packagesを追加指定できる。システム側のNumPyなどを優先してから追加する。今回のPCでは以下の仮想環境を使って起動を確認した。

```bash
ros2 launch eat_nav_real_robot real_robot.launch.py \
  torch_site_packages:=$HOME/venvs/isaaclab_env/lib/python3.12/site-packages
```

`/eat_nav/selected_path`、`/eat_nav/candidates` をRVizで表示する。Fixed Frameは `map`。RVizの2D Goal Poseを `/goal_pose` に送る。状態は次で確認する。

```bash
ros2 topic echo /eat_nav/status
ros2 topic echo /eat_nav/gate_status
```

ロボット切断中は入力待ちになる。previewでは `/eat_nav/path`、`/cmd_vel_auto` のpublisherを作らず、followerも起動しない。

## 指定rosbagでの再生確認

このPCではビルド済み。次の1コマンドで、記録センサーからR5推論とPathの可視化を確認できる。

```bash
source /opt/ros/jazzy/setup.bash
source ~/visualnav-transformer-ros2/build/real_robot/install/setup.bash

ros2 launch eat_nav_real_robot bag_replay.launch.py \
  bag:="/media/yasuda-haruki/Extreme SSD/rosbag2_2026_10_27-16_10_47" \
  torch_site_packages:=$HOME/venvs/isaaclab_env/lib/python3.12/site-packages
```

再生launchはROS_DOMAIN_ID=217、localhost限定Fast DDSを設定する。bagから再生するのはTF、DEM、IMU、関節、選択したOdometry、記録があればfold modeとgoalだけ。記録された速度指令・他のplannerの経路・ROSログを再配信しない。実機向けのPath/速度publisher、gate、follower、bridgeは起動しない。

`/clock` を配信して `use_sim_time=true` で動作する。実際のPC時刻を変更する必要はない。plannerの再計画・履歴期限もROS時刻を使い、再生を一時停止するとその時間基準も止まる。`rate:=0.5` 等で速度を変更できる。

指定bagは約16.6秒で、`/goal_pose` が0件。そのため再生launchは**検証用に機体前方2mの目標を1回生成**する。記録された目標や正解ラベルではない。`goal_forward_m:=0.0` で自動目標を無効にし、RViz等から自分で目標を与えられる。目標が記録されている別bagならその記録を使う。

別ターミナルでの確認:

```bash
source /opt/ros/jazzy/setup.bash
source ~/visualnav-transformer-ros2/build/real_robot/install/setup.bash
export ROS_DOMAIN_ID=217
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_STATIC_PEERS
ros2 topic echo /eat_nav/status
```

RVizも同じ環境で起動し、`use_sim_time=true`、Fixed Frame=`map`、Path=`/eat_nav/selected_path`、MarkerArray=`/eat_nav/candidates` を設定する。

このbagのDEMは16枚で、途中に1.2〜3.5秒の欠測がある。±1秒を超える不足期間も含まれるため、全区間で連続推論するとは限らない。データが揃えば推論し、欠測中は待機する。再生終了時はこのlaunchのプロセスを終了する。


## 新チェックポイントとRVizからの目標入力

既定は `navigation_mode:=r5_14`。チェックポイントは
`docs/eat-nav/r4-star-fourteen-training-six-evaluation-20261026/models/random_rough/seed123/checkpoint.pt`、
SHA-256は `c29bef5a56947db03f91ae2261dd8d259762479ebd9d38553017f3d5bd3c0421`。
新インデックスのSHAと14候補の学習設定を照合してから読み込み、前進7・後退7候補を評価する。
起動ログの `R5 checkpoint`、`candidates=14`、`replan=2.000s ROS time` で確認できる。

先に別端末でRVizを起動する:

```bash
source /opt/ros/jazzy/setup.bash
source ~/visualnav-transformer-ros2/build/real_robot/install/setup.bash
export ROS_DOMAIN_ID=217
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_STATIC_PEERS
rviz2 --ros-args -p use_sim_time:=true
```

Fixed Frameを `map`、Path表示を `/eat_nav/selected_path`、MarkerArray表示を `/eat_nav/candidates` にする。
Tool Propertiesで **2D Goal Pose** のTopicを `/goal_pose` にする。再生開始前のTFエラーはデータ待ち。

再生端末で:

```bash
source /opt/ros/jazzy/setup.bash
source ~/visualnav-transformer-ros2/build/real_robot/install/setup.bash
ros2 launch eat_nav_real_robot bag_replay.launch.py \
  bag:="/media/yasuda-haruki/Extreme SSD/rosbag2_2026_10_27-16_10_47" \
  navigation_mode:=r5_14 \
  replan_period_sec:=2.0 \
  goal_forward_m:=0.0 \
  loop:=true \
  torch_site_packages:=$HOME/venvs/isaaclab_env/lib/python3.12/site-packages
```

再生が始まったらRVizで **2D Goal Pose** を選び、地図上の目標をクリック・ドラッグする。
`goal_forward_m:=0.0` のため、自動目標は作らない。新しい目標で最初の評価を開始し、その後はROS時刻で2秒ごとに再評価する。
情報不足の間は待機するため、Path配信間隔が2秒を超える場合がある。今回のbagには最大約3.5秒のDEM欠測があり、全区間で0.5Hzの配信はできない。
`rate:=0.5` ならROS時刻の2秒は実時間の4秒になる。

このbagは約16.6秒で、`loop:=true` は繰り返し再生。巻き戻り時は目標も破棄するので、ループごとに目標を入れ直す。
1回だけなら `loop:=false`。Ctrl+Cで終了。

RVizと同じ環境の別端末で更新stampを確認する:

```bash
ros2 topic echo /eat_nav/selected_path --field header.stamp
```

経路形状が同じでも再評価でstampは更新される。plannerの `R5 Path update` ログにも評価時刻と選択候補を出す。
`/eat_nav/status` に `navigation_mode`、`candidate_count`、`checkpoint_sha256`、`inference_count`、`evaluation_stamp_ns` を出す。

## 実機への経路出力を有効にする

先に可視化でDEMの軸、機体姿勢、経路方向、予測値を確認する。既存 `onix_policy_bridge` は動かしたまま使用する。既存の通常followerは停止し、このlaunchから専用followerを起動する。ゲートは自動モードへ切り替えない。手動優先とモード選択は既存bridgeに残る。

```bash
ros2 launch eat_nav_real_robot real_robot.launch.py \
  enable_robot:=true \
  height_layer:=elevation_inpainted \
  torch_site_packages:=$HOME/venvs/isaaclab_env/lib/python3.12/site-packages
```

実機用経路は既存の `/teaching_playback/path` と分けた `/eat_nav/path` へRELIABLE・TRANSIENT_LOCAL（depth=1）で送り、専用follower `/eat_nav_follower` が `/eat_nav/follower_cmd` に TwistStamped を出す。別プロセスのゲートだけが `/cmd_vel_auto` へ中継する。既存bridgeから `/diff_drive_controller/cmd_vel` とフリッパー指令が出る。

既存の `/onix_global_planner_adapter` と `/path_server` は `/teaching_playback/path` の配信を継続できる。専用followerが読むのは `/eat_nav/path`。ゲートは専用経路の配信元がEAT-Nav plannerだけ1つであることを確認する。

ゲートは既存bridge `/onix_policy_bridge` のパラメータを定期確認する。`cmd_timeout` が正の有限値で0.5秒以下、入力が `/cmd_vel_auto` のTwistStampedである必要がある。専用followerの出力・Path・recovery経路設定も確認する。別の自動速度配信元や競合するPath配信元がある場合は許可しない。通常followerを残して二重起動しない。

自分でfollowerを起動する場合は `start_follower:=false` とし、ノード名 `/eat_nav_follower`、`cmd_vel_topic:=/eat_nav/follower_cmd`、`cmd_vel_use_stamped:=true`、`path_topic:=/eat_nav/path`、`recovery_path_topic:=/eat_nav/unused_recovery_path`、`publish_subcrawler_commands:=false`、`robot_frame:=chassis/body` を設定する。既存のrecovery経路からEAT-Navの選択を上書きしない構成とする。

```bash
ros2 topic pub --once /eat_nav/cancel std_msgs/msg/Empty '{}'
```

入力途絶・計画失効・キャンセル時はゲートがゼロ速度を継続送信する。通常の終了はCtrl+C。ROS時刻が巻き戻った場合は停止し、目標・履歴を破棄する。ゲートの以前の時刻基準も残るため、システム時刻を安定させてからこのlaunch全体を再起動する。ゲート自体が停止した場合はbridgeの指令期限に依存する。bridgeがフリーズした場合やハードウェア側の停止まで保証するものではない。

## スタック候補除外の比較（true / false）

`navigation_mode:=r5_14` で起動引数 `stall_risk_only` を指定する。既定はfalse。

| 引数 | 除外条件 |
|---|---|
| `stall_risk_only:=false` | 従来R5：直線または旋回スタック出力が0.95以上、かつ10秒予測進捗が0.05m以下 |
| `stall_risk_only:=true` | 直線または旋回スタック出力が0.95以上なら、予測進捗に関係なく除外 |

新条件で起動:

```bash
ros2 launch eat_nav_real_robot real_robot.launch.py \
  navigation_mode:=r5_14 \
  stall_risk_only:=true \
  replan_period_sec:=2.0 \
  height_layer:=elevation_inpainted \
  enable_robot:=true \
  torch_site_packages:=$HOME/venvs/isaaclab_env/lib/python3.12/site-packages
```

従来条件と比較する場合はEAT-Nav launchをCtrl+Cで終了し、上の `true` を `false` に変えて再起動、goalを再指定する。起動時専用のread-onlyパラメータなので、実行中の `ros2 param set` では切り替えない。trueはR5十四候補モードのみ対応。

0.95ちょうども除外する。現在の候補が除外された場合は通常の4秒保持・方向確認を待たず、除外されていない候補へ切り替える。全14候補が除外されたら候補はSTOP、走行許可を取消しgateが速度ゼロを出す。再開は従来と同じ最低停止時間・連続確認に従う。

スコア式、checkpoint、進捗予測、2秒周期、候補保持・再開規則は比較で共通。研究用の元のR5実装も変更しない。既存の転倒/実測傾斜STOPの無効設定は今回の変更対象に含めない。また、判定は再評価時であり、高さからの独立した衝突停止機能を追加したものではない。

使用条件を確認:

```bash
ros2 param get /eat_nav_planner stall_risk_only
ros2 topic echo /eat_nav/status --once --full-length
```

statusの `stall_risk_only`、`stall_exclusion_mode`、`stall_risk_threshold` と、推論後の `excluded_candidate_mask`、`candidate_ids`、`candidate`、`guard_reason`、`guard_stopped` を確認する。STOP時は前のPath表示がRVizに残ることがあるが、新しい実機Pathを出さず走行許可を取り消す。STOPはstatusとgateの速度出力で確認する。

既存の録画スクリプトはstatusとplanner_params.yamlにこの条件を保存するため、記録トピックの追加は不要。bag再生launchでも同じ `stall_risk_only` 引数と `height_layer` 引数を指定できる。

## 設定と限界

`navigation_mode`、`replan_period_sec`、`repository`、`checkpoint`、`checkpoint_sha256`、`odom_topic`、`height_layer` はlaunch引数で変更可能。インデックスにないcheckpointはSHA-256の明示指定が必要。command-free v2以外のcheckpointは拒否する。詳細は `config/real_robot.yaml` をコピーし `params_file:=/absolute/path.yaml` で渡す。launch引数の値が同名yaml値に優先する。設定変更は再起動で反映する。

既定のR5はlinear-stallとangular-stallを重み1.0/0.5で使用し、転倒角によるSTOPは無効。`navigation_mode:=r4` で旧4候補モデルへ切替可能。`checkpoint` の省略時は各モードの既定モデルを選ぶ。予測値は実機で精度検証されていない。シミュレータの片側24関節と実機の2スプロケットでは速度・effort集約が異なり、encoder符号・ゼロ点・effort単位にも差があり得る。既存RLモデルの交換や、この差の自動補正は行わない。

既存followerのTwistStampedには計画IDがない。計画切替時は150ms待機して古い指令キャッシュを破棄するが、新経路をfollowerが採用したという厳密な応答確認にはならない。外部の `/chassis/direction` 反転を利用する場合は、DEM・goal・IMU・関節と制御方向の整合を別途確認する。本ノードは `chassis/body` の固定した機体前方で計画する。

## テスト

```bash
PYTHONPATH=ros2/eat_nav_real_robot:src:scripts:$PYTHONPATH \
  python3 -m pytest ros2/eat_nav_real_robot/test/test_inputs.py ros2/eat_nav_real_robot/test/test_gate.py -q
```

モデル試験にはPyTorchが必要。ROS輸送試験にはビルド済みのROS2 `grid_map_msgs` と隔離環境が必要で、`ROS_DOMAIN_ID=217`、`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`、`RMW_IMPLEMENTATION=rmw_fastrtps_cpp` で実行する。実機のdomainで実行しない。
