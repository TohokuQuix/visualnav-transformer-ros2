# NoMaDのロボット接続仕様：圧縮画像入力・map座標Path出力

## 1. 目的・範囲

RobotecAI/visualnav-transformer-ros2 の既存NoMaD推論を、自前ロボットのカメラとsimple_trajectory_followerへ接続する。
ネットワーク構造・学習・モデルの入力テンソル構造を変更せず、ROS入出力、画像履歴管理、距離単位の復元、座標変換、停止通知だけを実装する。

対象リポジトリ: `/home/ubuntu/visualnav-transformer-ros2`。
追従側は `/home/ubuntu/Downloads/onix_automation` の `feat/nomad-path-following`、コミット `39bd4b2` に対応する。
この文書は仕様の作成であり、推論側のコード変更はまだ行っていない。

## 2. 変更しない部分

- Vision encoder、距離予測ヘッド、ConditionalUnet1Dの構造とパラメータ。
- 既存のDDPMスケジューラ、推論ステップ数、get_actionによる出力復元。
- 保存済みトポマップ画像列から近傍画像を評価し、当面のgoal画像を選ぶ方式。
- モデル設定で定まる画像サイズ・context_size・予測点数。
- 複数サンプルを生成した場合は既存実装と同じ先頭サンプルを採用する。

分類ヘッドの追加、Normal/switchbackの自動判定、状態分類と拡散モデルの接続、追加学習は範囲外。
今回の接続モードにはstate_sourceパラメータを追加しない。

### チェックポイントの互換性

モデル定義と一致するチェックポイントだけを使う。不足・余分なキーやshape不一致を黙って無視しない。
以前指定された `krail_directional_camera_switchback_conditioned_2026_09_06_17_30_01` の重みには
switchback_state_embedding / switchback_state_projection が存在する。
クローン直後の標準NoMaDにはこれらの層がないため、このチェックポイントを標準モデルに部分ロードして動作確認済みと扱わない。
今回の初期接続対象は、既存標準モデルと一致するNoMaDチェックポイントとする。
条件付け済みモデルの利用を継続する場合は、そのモデルを変更せず提供する別工程が必要。

## 3. 全体構成

```text
/front_camera/image_raw/compressed
  -> JPEGデコード・RGB化
  -> 時刻付き画像履歴
  -> 既存NoMaD推論 <- 保存済みgoal画像列
  -> 先頭サンプルの全予測点を取得
  -> メートル単位へ復元
  -> 観測時刻の車体位置・yawでmapへ変換
  -> /teaching_playback/path
  -> simple_trajectory_follower (rolling)
  -> /cmd_vel_auto
```

このノードから速度指令を配信しない。scripts/publish_cmd.pyは併用しない。

## 4. 画像入力

- Topic: `/front_camera/image_raw/compressed`。
- Type: `sensor_msgs/msg/CompressedImage`。
- QoS: Best Effort / Volatile / KeepLast(1)。Reliableな画像Publisherにも接続できる購読設定とする。
- bagで確認したformatは `rgb8; jpeg compressed bgr8`、frame_idは `camera`。
- 圧縮データをOpenCV等でデコードし、BGRからRGBへ明示的に変換して既存transform_imagesへ渡す。
- デコード失敗、空画像、無効な時刻は推論に使わない。失敗が継続した場合も受信期限で停止する。
- カメラframeのcamera/front_camera差異は画像推論には使わない。モデル軌跡の基準frameは別途robot_frameで指定する。

### 画像履歴と時刻

モデルが要求するcontext_size+1枚を、古い順に与える。
context_interval_secは学習時の画像間隔に合わせて起動時に指定する必須値とする。
最初の画像を採用し、以降は最後に採用した画像からこの間隔以上経過した最初のフレームを採用する。
間隔が1.5倍を超えた場合は履歴を破棄して作り直す。欠落した画像を複製して補わない。
同一時刻画像は重複推論しない。時刻が巻き戻った場合は進行中の推論を無効化して履歴を作り直す。
最新の履歴画像のstampを推論基準時刻t_obsとし、推論終了時刻に置き換えない。

## 5. goal画像

既存のtopomaps/images/<名前>/内の画像列を利用する。--dir、--goal-node、--radius、--close-thresholdに相当する設定を維持する。
座標列やROS画像トピックからgoalを与える新機能は追加しない。
目標画像はRGBに統一し、既存と同じサイズ・正規化を適用する。
空ディレクトリ、画像番号の不整合、読み込み失敗、範囲外のgoal-nodeは起動エラーとする。
最終ゴールへの到達判定方式は既存navigate.pyを維持し、到達後に軌跡を送り続けない。

## 6. 距離と座標系

### 距離の復元

get_actionは正規化された差分を復元し累積する既存処理である。接続層で再度累積しない。
モデル設定normalize=trueの場合、その結果にaction_scale_mを一度だけ乗じてメートル単位に戻す。
action_scale_mは正の有限値として起動設定で必須とする。学習時の正規化に一致する値を指定する。
K-Railの確認済みコードではmetric_waypoint_spacing * waypoint_spacingが係数となる。
0.05という値を他のチェックポイントに一律適用せず、MAX_V / RATEからも自動推定しない。
normalize=falseの場合はメートル単位として扱い、action_scale_m=1.0を要求する。

### mapへの変換

モデルの教師座標が車体の位置とyawに基づく平面座標であることをチェックポイントごとに確認する。
robot_frameは `chassis/body`、global_frameは `map` とする。
t_obsのTFを取得し、位置(tx,ty,tz)とyaw ψを使う。

```text
x_map = tx + cos(ψ) * x_local - sin(ψ) * y_local
 y_map = ty + sin(ψ) * x_local + cos(ψ) * y_local
z_map = tz
```

roll/pitchを含む3次元回転を一律に掛けず、学習の平面座標変換を逆にする。
最新TFで過去の観測を変換しない。該当時刻のTFがなければその推論結果を破棄して停止通知する。
TF取得待ちがROS時刻停止中に停止処理を妨げないよう、非ブロッキングlookupを使う。

## 7. Path出力

- Topic: `/teaching_playback/path`。
- Type: `nav_msgs/msg/Path`。
- QoS: Reliable / Transient Local / KeepLast(1)。追従側の購読QoSと一致させる。
- Path.header.frame_idと各Pose.header.frame_id: `map`。
- Pathと各Poseのstamp: t_obs。将来時刻の速度付き軌跡を表すものではない。
- 点列: 観測時点の車体位置を先頭に置き、選択した1サンプルの全予測点を時間順に追加する。
- 予測長が8の場合は通常9点。モデル設定による予測長を固定値8で上書きしない。
- orientation: 有効な単位quaternion (0,0,0,1)。姿勢予測値ではなく、追従側では使用しない。
- NaN/Infを含む出力や、非零長区間が存在しない軌跡は配信せず、空Pathで停止通知する。
- 切り返し点での区間制限は追従側の責務とし、送信側は全点列を保持する。

## 8. 並行処理・停止

画像購読・TF更新・監視タイマーは推論計算から分離する。推論ワーカーは1つで、モデルへの同時呼び出しをしない。
待ち行列を増やさず、次に処理する観測は最新の履歴スナップショットだけ保持する。
時計巻き戻り・停止・画像履歴リセットには世代番号を使い、それ以前に開始した推論結果を配信しない。

- observation_timeout_sec: 0.8秒を初期値とし、t_obsからのROS時間経過で結果の古さを判定する。
- image_receive_timeout_sec: 0.8秒を初期値とし、有効画像の受信途絶を単調時計で監視する。
- freshness_check_period_sec: 0.05秒。推論ワーカーとは独立して監視する。
- 追従側path_timeout_sec=1.0秒を初期設定とし、上記期限・推論周期と整合させる。
- 期限切れ・TF欠落・不正軌跡・時刻巻き戻り・最終ゴール到達では、map frameの空Pathを送る。
- 空Path送信後は古い結果を配信しない。復帰時は新しい有効な画像履歴とTFが揃ってから再開する。
- 最終ゴール到達はその起動中の停止状態として保持し、自動再開しない。
- 正常終了時にも空Pathを送る。異常終了や通信断時は追従側の期限切れ停止に委ねる。

## 9. 設定・実装単位

起動設定は固定値とし、動作中のモデル・frame・スケール・周期切り替えは行わない。

```yaml
image_topic: /front_camera/image_raw/compressed
path_topic: /teaching_playback/path
robot_frame: chassis/body
global_frame: map
observation_timeout_sec: 0.8
image_receive_timeout_sec: 0.8
freshness_check_period_sec: 0.05
```

context_interval_sec、action_scale_m、チェックポイント・モデル設定・topomapは必須設定。
値が未指定なら走行を開始しない。画像間隔・座標基準・スケールの実データとの一致は受入条件に含める。

予定ファイル:
- deployment/src/navigate_robot.py: ROSノード、既存推論との接続、停止制御。
- deployment/src/robot_interface.py: 圧縮画像復号、画像履歴、平面座標変換、Path組み立て。
- config/robot_interface.yaml: 接続設定。モデル依存の必須値は起動時検証する。
- tests/: 変換・復号・履歴・ROS接続の検証。
- README.md: 起動方法、モデル互換性、追従側との設定対応。

既存navigate.pyを参照し、モデル生成・推論アルゴリズムの共通化が必要な場合も計算内容は維持する。
新規のcolconパッケージ化は行わず、既存uv/Pythonスクリプト起動方式を使う。

## 10. 検証と受入条件

1. 赤青の既知画像をJPEG圧縮して入力し、RGB順序とデコード失敗処理を確認する。
2. 同一画像列・goal画像・乱数条件で、接続前後のモデル出力が一致することを確認する。
3. yaw=0、±90度、180度、並進ありの既知TFでmap座標を数値検証する。
4. 前進・後退・折り返し点の順序、メートル復元の一回適用、原点追加、frame/stampを検証する。
5. 履歴不足、重複時刻、画像欠落、時刻巻き戻り、TF欠落、NaN/Inf、推論遅延で古い経路を配信しないことを確認する。
6. ROS時刻停止中も受信タイムアウトが働き、推論中でも空Pathが配信されることを確認する。
7. 追従ノードと接続し、QoS接続、空Path停止、期限切れ停止、新しい観測による再開を確認する。
8. 指定bagを使う際は記録済み同名Pathを除外またはリマップし、--clockとuse_sim_timeを揃える。
9. RVizでmap上の位置・向き・スケールを確認する。単体試験とbag試験と実機試験の結果は区別して記録する。

実機への配信前に、採用する互換チェックポイント、学習時の画像間隔・教師座標・スケール、goal画像列を確定する。
前後カメラを使い分けて学習したモデルを前方カメラ固定で利用する場合の精度は、通信互換性とは別に評価する。
