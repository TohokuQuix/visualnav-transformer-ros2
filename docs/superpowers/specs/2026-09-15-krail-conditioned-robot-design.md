# K-Rail条件付きNoMaD：最新モデルの実機接続仕様

作成日: 2026-09-15
状態: レビュー用。今回は仕様書のみ。モデル移植・実機試験は未実施。

## 1. 目的と対象

これまで指定された最新モデル `krail_directional_camera_switchback_conditioned_2026_09_06_17_30_01` を、起動時に指定するgoal画像1枚と前方カメラを用いて実機へ接続する。「最新」はこのrunを指し、日時だけで別の重みへ自動変更しない。

学習済みの条件付きネットワークをそのままROS2側へ移植する。新しい層、分類器、再学習を追加しない。baselineで整備した画像履歴・map Path変換・期限監視・単一goal方式を再利用する。

初期評価は前方カメラ固定・前進を基本とし、Normal指定とswitchback指定それぞれの予測を検証する。指定した状態はモデルへの条件であり、前進・後退の速度指令ではない。

## 2. ブランチと共通処理

リポジトリ: `/home/ubuntu/visualnav-transformer-ros2`

- `feat/krail-baseline-robot`: 標準モデル。今回のモデル変更を加えない。
- `feat/krail-conditioned-robot`: 本仕様書と最新モデル移植の作業先。
- `feat/nomad-robot-interface`: 共通接続の元となるブランチ。

条件付きブランチは共通接続コミット `35594bf` から作成されているため、現時点ではbaselineの単一goal対応 `6be35f4` を含まない。実装開始時に `feat/krail-baseline-robot` の `6be35f4` までを条件付きブランチへ取り込み、動作検証済みの接続コードを共通の土台にする。baseline側へ逆方向にモデル変更をマージしない。取り込みに伴いbaseline設定も存在してよいが、起動設定で使用する重み・モデル種別を明示する。

追従側は `/home/ubuntu/Downloads/onix_automation` の `feat/nomad-path-following`、`39bd4b2`。今回そのコードは変更しない。

## 3. 重みとネットワークの確認結果

学習リポジトリ: `/home/ubuntu/visualnav_docker/visualnav-transformer/train`

- 設定: `config/nomad_krail_directional_camera_switchback_conditioned.yaml`
- 重み: `logs/nomad_krail/krail_directional_camera_switchback_conditioned_2026_09_06_17_30_01/latest.pth`
- SHA256: `0c9e456b0f7dd48f0bd4c9245f6bfe6bb9856dfb6d40ff74655a6c4d03c69778`
- 保存state_dict: 647キー。

| 構成 | 確認内容 |
| --- | --- |
| vision_encoder | 既存NoMaD_ViNT、encoding_size=256 |
| noise_pred_net | ConditionalUnet1D、down_dims=[64,128,256] |
| dist_pred_net | 既存距離ヘッド |
| switchback_state_embedding | Embedding(2,16)、重みshape=(2,16) |
| switchback_state_projection | Linear(16,256,bias=False)、重みshape=(256,16) |
| mode_pred_net | この重みに存在しない |

設定には `use_switchback_state_condition: true`、`switchback_state_embedding_dim: 16`、`lambda_mode: 0.0` がある。従ってこのチェックポイントに学習済みのNormal/switchback自動分類器が含まれるとは扱わない。

条件付けは学習元 `vint_train/models/nomad/nomad.py` と同じ演算を行う。

```text
c = vision_encoder(observation_history, goal_image)
s = 0 (Normal) または 1 (switchback)
c_diffusion = c + s * projection(embedding(s))
actions = DDPM(noise_pred_net, global_cond=c_diffusion)
distance = dist_pred_net(c)
```

距離ヘッドには条件付け前のcを入力する。Normalでは条件への加算がゼロになるが、encoderやdiffusion等の重み自体は追加学習されているため、9月4日baselineの出力と一致する保証はない。

全647キーをstrictロードする。余剰キー無視、欠落キー補完、標準モデルへの部分ロードを禁止する。既存モデルを切り貼りして新しいネットワークを作るのではなく、学習時の構造と演算を再現する。

## 4. 状態の指定（レビューで確定）

推奨する初期仕様は、起動時のパラメータで状態を固定する方式。

```yaml
state_source: fixed
switchback_state: normal  # normal または switchback
```

- `normal`を0、`switchback`を1へ変換する。未指定・未知の文字列・未対応state_sourceは起動エラー。
- 指定状態を推論サンプル数に合わせて複製し、画像特徴と同じdeviceで条件付けする。
- 起動ログに重み・state_source・状態・goal画像を表示し、評価結果を区別できるようにする。
- 動作中の変更は受け付けない。状態変更時は停止・ノード再起動で切り替える。
- この方式は状態条件の効果を比較する初期実機接続であり、実環境に応じた自動切り返し判断は行わない。

選択肢として外部ROSトピックによる状態指定も可能だが、観測との時刻対応・状態期限・進行中推論の無効化の仕様が必要になるため、初期案には含めない。画像による自動分類には別の学習済み分類器が必要であり、この重みだけでは提供しない。

ユーザーへ状態の指定方法を確認中。本仕様書では固定方式を提案しており、この項目は未承認。

## 5. 画像入力・goal・Path

```text
/front_camera/image_raw/compressed + 過去3枚
                      + 起動時goal画像1枚
  → 学習済みvision encoder
  → 指定状態で条件付け → 既存DDPM → 先頭サンプルの予測8点
  → メートル復元 → 観測時刻のmap変換
  → /teaching_playback/path → rolling追従 → /cmd_vel_auto
```

| 項目 | 設定 |
| --- | --- |
| 入力 | `/front_camera/image_raw/compressed`、CompressedImage |
| 画像QoS | Best Effort / Volatile / KeepLast(1) |
| 履歴 | context_size=3、現在を含む4枚、0.25秒間隔 |
| モデル画像サイズ | [96,96]、既存RGB前処理 |
| goal | 起動時に指定する画像ファイル1枚 |
| diffusion | 10ステップ、8サンプル生成、先頭サンプル採用 |
| 予測点 | 8点、learn_angle=false |
| 距離復元 | normalize=true、action_scale_m=0.05、一回のみ乗算 |
| TF | 観測時刻の `map ← chassis/body`、平面yawと並進 |
| 出力 | `/teaching_playback/path`、nav_msgs/Path、map座標 |
| 出力QoS | Reliable / Transient Local / KeepLast(1) |
| Path点列 | 車体原点＋予測8点、通常9点 |

Pathと全Poseのstampは観測時刻、orientationは単位quaternion。速度指令をNoMaDノードから直接配信しない。起動時goalのRGB読み込み・不正画像検証はbaseline方式を使用する。topomap探索は行わない。

条件付き学習も `krail_directional_camera_master` を使用する。復路rear画像の教師yawは180度変更されている。Normal/switchback状態はfront/rearの選択とは別であり、switchbackを指定しただけでカメラや座標を180度回転しない。

## 6. 到達・停止と追従側の範囲

- 条件付け前の特徴による予測距離が `goal_distance_threshold` 未満で到達とする。初期値3.0、メートル単位ではない。最新モデルでもbagで調整する。
- 到達時は空Pathを配信し、再起動まで停止する。
- 観測期限0.8秒、画像受信期限0.8秒、監視間隔0.05秒を継承。推論中も監視する。
- TF欠落・画像途絶・非有限出力・時刻巻き戻り・期限切れでは古い予測を破棄し、空Pathで停止する。
- 追従側はrolling、Path期限1.0秒、自律リカバリー無効。

追従側は予測列の最初の折り返しまでの区間を採用し、その先は次の予測更新で判断する。したがって全予測点を配信できることと、一回のPathで切り返し全体を実行できることは別である。最新モデルが前進→後退などを出した場合、更新されたPathとの組み合わせで意図した運動になるか実機前に確認する。現状の追従仕様で不足する場合は別の変更として提案する。

## 7. 実装予定ファイルと設定

- `config/krail_conditioned_model.yaml`: 学習時の構造・条件付け設定。
- `config/krail_conditioned_robot.yaml`: 単一goal・状態・入出力・frame・スケール。
- `deployment/src/utils.py` とモデル定義: 条件付き構造の生成とstrictロード。標準モデルは引き続き使用可能にする。
- `deployment/src/robot_inference.py`: diffusion特徴だけへの条件付け、単一goal推論。
- `deployment/src/navigate_robot.py`: 起動時状態の検証・受け渡し。
- `tests/`: 条件演算・推論一致・入力不正・既存baseline回帰の検証。
- `docs/krail-conditioned-robot.md`: 重みハッシュ、起動・状態別評価・未確認事項。

パスのdeploymentは `src/visualnav_transformer/deployment`、モデル定義は `src/visualnav_transformer/train/vint_train/models/nomad/nomad.py` を指す。

```yaml
model_config: krail_conditioned_model.yaml
checkpoint: ../model_weights/krail_conditioned_20260906_latest.pth
goal_image: ""  # 目的地の画像を明示指定
state_source: fixed
switchback_state: normal
goal_distance_threshold: 3.0
context_interval_sec: 0.25
action_scale_m: 0.05
image_topic: /front_camera/image_raw/compressed
path_topic: /teaching_playback/path
robot_frame: chassis/body
global_frame: map
```

これは予定設定であり、まだ実装していない。重みは固定名でコピーしSHA256を検証、Gitには含めない。

## 8. 検証・受入条件

1. 保存重み647キーと学習設定の一致を確認し、ROS2側条件付きモデルへstrictロードできる。
2. Normalでc_diffusion=c、switchbackで学習済みprojection(embedding(1))が加算される。距離ヘッドへは元のcを渡す。
3. 同じ重み・観測4枚・goal・状態・初期ノイズ・scheduler乱数で、学習元のgoal-conditioned推論とROS2実装の8点を比較する。学習元が先に無条件サンプルを生成する場合は乱数消費を揃える。CPUでまず検証し、GPU差は許容誤差を記録する。
4. 分類ヘッドのない重みに自動分類を要求した場合、黙ってNormal扱いにしない。
5. 標準baselineと単一goalの既存テストが引き続き成功する。
6. 指定bagでNormal固定・switchback固定を別々に実行し、Path形状・観測年齢・停止を記録する。元bagの同名Pathと速度指令は再生しない。
7. 教師位置yawと観測時刻TFの対応を照合する。baseline段階から残る同期対応の未確認を解消する。
8. 折り返しを含む予測とrolling追従の区間選択を検証する。前進・後退の符号や途中停止の妥当性を記録する。
9. 実機計算機で推論周期が期限を満たすことを測定する。実機goalを確定し、速度制限と停止手段を確認して試走する。

本仕様作成時点で確認済みなのは、保存重みの構成・shape・ハッシュ、学習設定、条件演算、分類器重みの不存在。ROS2へのstrictロード・実重み推論一致・bag走行・実機試験は未実施。
