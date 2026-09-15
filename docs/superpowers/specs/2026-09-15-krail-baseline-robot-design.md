# K-Rail baseline NoMaD：実機接続仕様

作成日: 2026-09-15
状態: 承認済み。baseline専用設定と重み配置を実装済み。検証結果・未確認項目は [起動手順](../../krail-baseline-robot.md) を参照。実機試験は未実施。

## 1. 目的と対象

既存NoMaDをK-Railデータで追加学習したbaselineを、前方カメラと既存の経路追従ノードへ接続する。ネットワーク構造、学習済みパラメータ、推論アルゴリズムを変更せず、モデル設定とロボット接続設定を整備する。

Normal/switchback分類ヘッド、状態embedding、条件付きdiffusionへの変更、再学習、カメラ切り替えは対象外。最新のネットワーク変更版は別ブランチで後日扱う。

共通の入出力契約は [NoMaDロボット接続仕様](2026-09-15-nomad-robot-interface-design.md) に従う。同仕様書は作成時点の記述を含むが、共通接続実装はコミット `35594bf` に存在する。本仕様書はbaselineのモデル選択・設定・受入条件を具体化する。

## 2. ブランチ構成

対象リポジトリ: `/home/ubuntu/visualnav-transformer-ros2`

```text
main
  └─ feat/nomad-robot-interface  (共通ロボット接続、35594bf)
       ├─ feat/krail-baseline-robot     (今回の対象)
       └─ feat/krail-conditioned-robot (将来のネットワーク変更版)
```

両ブランチは `35594bf` から作成済み。現在の作業ブランチは `feat/krail-baseline-robot`。将来用ブランチにはモデル変更をまだ加えていない。mainへのマージは今回行わない。

baselineの設定・検証記録はbaselineブランチに置く。分類・条件付けなどのモデル変更は将来用ブランチに置き、baselineへ混入させない。共通接続の修正が必要になった場合は共通ブランチで修正し、両ブランチへ取り込む。

追従側は別リポジトリ `/home/ubuntu/Downloads/onix_automation` の `feat/nomad-path-following`、コミット `39bd4b2` を使用する。

## 3. 採用するbaseline

### 採用モデル

最新の条件付きモデルとのデータ条件を近づけるため、次を推奨対象とする。2026-09-15に9月4日版の採用が承認された。

学習リポジトリのルート:
`/home/ubuntu/visualnav_docker/visualnav-transformer/train`

- 学習設定: `config/nomad_krail_directional_camera_baseline.yaml`
- 重み: `logs/nomad_krail/krail_directional_camera_baseline_2026_09_04_16_13_16/latest.pth`
- SHA256: `c07ac8491190e5237c29f35e43ce3b30b80b1ca5b85c424ef28eb0c2df813e3b`
- データセット: `krail_directional_camera_master`

確認済み事項:

- 保存重みのトップレベル構成は `vision_encoder`、`noise_pred_net`、`dist_pred_net` のみ。
- 既存ROS2側NoMaDにCPU上でstrictロードできた。state_dictは645キー。
- この確認はモデル互換性の確認であり、実機走行性能の確認ではない。

8月21日の `krail_clean_finetune_2026_08_21_16_21_30/latest.pth` は別baselineである。採用変更時は設定・データ条件・ハッシュを差し替え、対応する検証を行う。2つの重みを同じ名称で管理しない。

### 固定するモデル設定

```yaml
model_type: nomad
vision_encoder: nomad_vint
encoding_size: 256
mha_num_attention_heads: 4
mha_num_attention_layers: 4
mha_ff_dim_factor: 4
down_dims: [64, 128, 256]
cond_predict_scale: false
num_diffusion_iters: 10
normalize: true
context_size: 3
image_size: [96, 96]
len_traj_pred: 8
learn_angle: false
```

学習設定を出典としてbaseline専用モデル設定を作る。重みは名前を固定したコピーとして配置し、SHA256と出典を記録する。重み本体はGitへ追加しない。採用したのは上記 `latest.pth` であり、別のEMA重み等へ暗黙に差し替えない。strictロード失敗時は起動を中止する。

## 4. ロボットとの接続

```text
/front_camera/image_raw/compressed
  → RGB画像・時刻付き履歴
  → 標準NoMaD + K-Rail baseline重み ← 保存済みgoal画像列
  → メートル復元 → 観測時刻のTFでmapへ変換
  → /teaching_playback/path
  → simple_trajectory_follower (rolling)
  → /cmd_vel_auto
```

| 項目 | 設定 |
| --- | --- |
| 画像入力 | `/front_camera/image_raw/compressed`、CompressedImage |
| 画像QoS | Best Effort / Volatile / KeepLast(1) |
| 経路出力 | `/teaching_playback/path`、nav_msgs/Path |
| 経路QoS | Reliable / Transient Local / KeepLast(1) |
| 車体frame | `chassis/body` |
| 出力frame | `map` |
| 画像履歴間隔 | `context_interval_sec: 0.25` |
| 距離復元係数 | `action_scale_m: 0.05` |
| 観測・画像受信期限 | 各0.8秒 |
| 監視間隔 | 0.05秒 |
| 追従側Path期限 | 1.0秒 |

画像は現在と過去3枚の合計4枚。通常の履歴時間幅は0.75秒。学習データの連続予測8点は公称2秒分に相当するが、Pathは時刻付き速度指令ではない。

0.25秒はdirectional-cameraデータの `sync_debug.csv` の連続target時刻差から確認した。0.05は `krail.metric_waypoint_spacing=0.05` と `waypoint_spacing=1` に基づく。0.05m間隔の等間隔経路を生成するという意味ではない。

既存 `get_action` が復元・累積した局所XYへ0.05を一度だけ掛ける。観測時刻の `map ← chassis/body` の並進とyawで変換し、先頭に車体原点を加える。通常は合計9点。全PoseとPathに観測時刻とmap frameを設定する。最新TFによる代用や追加の累積は行わない。

教師座標の基準・時間対応を確認する必要がある。bagのTFと教師CSVの簡易照合では同一値まで確認できていないため、教師位置がそのまま観測時刻の `chassis/body` と一致するとは現時点で断定しない。データ生成時の時刻基準・同期方法・位置yawの出典を照合し、対応を記録してから実機へ進む。誤差を距離係数の調整で隠さない。

## 5. goal画像と最初の走行範囲

goalは既存方式の保存済みトポマップ画像列で与える。Pathをgoal入力にしない。`0.jpg` から連番のRGB画像列など、共通ローダーが受理する形式を使用し、最終画像を初期のgoalとする。近傍探索と到達判定は既存方式を維持する。

bag接続確認の候補:

- bag: `/media/ubuntu/Extreme SSD/20260803_experiment/2026_08_03-10_31_05`
- 画像列候補: `/home/ubuntu/visualnav_docker/visualnav-transformer/datasets/krail_directional_camera_master/2026_08_03-10_31_05/`
- データ一覧でforward/front、193フレームを確認済み。

この画像列はbagでの接続検証候補であり、実機の走行先として確定したものではない。実機では走行場所・方向・目的地点に一致する前方カメラの画像列を選び、設定に明記する。学習データ由来の画像列による確認を、未知経路への汎化性能評価として扱わない。

学習データにはforward/frontが21系列、return/rearが13系列含まれる。9月4日baselineの設定自体がこのdirectional-cameraデータを参照しており、rear利用は最新の条件付きモデルだけの変更ではない。既存krail_cleanとのtraj_data.pkl比較では、往路21系列のyawは同一、復路13系列のyawは180度差（角度周期を考慮した最大誤差4.5e-16 rad未満）であることを確認した。これは教師座標のyaw変更であり、画像画素を180度回転したという意味ではない。前方カメラ固定で始める今回の初期評価は前進・前方画像の条件に限定する。後方カメラで学習した帰路や切り返しの性能は別途評価する。

## 6. 実装予定

共通 `navigate_robot` を使用し、別の推論ループやモデルクラスを作らない。

1. `config/krail_baseline_model.yaml`: 上記の学習設定に対応する推論モデル設定。
2. `config/krail_baseline_robot.yaml`: 画像・Path・frame・0.25秒・0.05係数・重み・goal画像ディレクトリを指定する設定。
3. READMEまたは専用手順書: 重みの配置・ハッシュ確認、goalの選択、起動、bag再生、検証記録の方法。
4. 必要な検証: 実重みとの互換性、既存推論との一致、設定の読み込みと経路配信の確認。

将来の起動形式:

```bash
uv run python -m visualnav_transformer.deployment.src.navigate_robot \
  --interface-config config/krail_baseline_robot.yaml
ros2 launch simple_trajectory_follower nomad_follower.launch.py
```

これらのbaseline専用設定は作成済み。goal未確定・重み不足では起動エラーとし、仮のgoalで走行を開始しない。ROS環境はROS2のみをsourceする。既存 `scripts/publish_cmd.py` は併用しない。

追従側は `rolling`、自律リカバリー無効のNoMaD設定を使う。空Path、経路期限切れ、不正経路で停止する。接続側も画像途絶、TF欠落、古い推論結果、時刻巻き戻り、goal到達時に空Pathを配信する。詳細は共通仕様を継承する。

## 7. 検証順序と受入条件

### A. モデル・データ

- 採用重みとSHA256を固定し、strictロード成功を記録する。9月4日候補のCPUロードは確認済み。
- 固定した観測画像・goal画像・乱数条件で、元のbaseline推論と接続側推論の局所XYを比較する。実重みでの比較は未実施。
- RGB変換、画像前処理、履歴間隔、距離係数が学習条件と一致する。
- 教師座標とbag TFの位置・yaw・時刻基準の対応を確認する。

### B. bag・経路配信

- 記録済み `/teaching_playback/path` は再生対象から除外または別名にし、今回の出力と混在させない。
- bagはclock付きで再生し、推論・追従・可視化の `use_sim_time` を揃える。
- RVizでPathの始点・向き・距離単位を確認する。mapの実経路との重ね合わせを記録する。
- 画像停止、TF欠落、推論遅延、再生時刻巻き戻りで停止し、古いPathで走り続けない。
- 実機で使用する計算機で推論時間とPathの観測年齢を測定する。0.8秒以内に有効経路を継続配信できない場合は原因を解決し、期限変更が必要なら追従側も含めて再検証する。

### C. 実機

- 走行先とgoal画像列を確定し、前方カメラ固定の前進区間で確認する。
- ロボット側の速度制限と停止手段を確認して試走し、画像、TF、出力Path、速度指令を記録する。
- 経路更新への追従、goal到達停止、画像途絶時停止を確認する。
- 単体試験・bag試験・実機試験の結果を分けて記録する。今回の仕様作成時点では実機動作は未確認。

## 8. レビューで確定する事項

- 9月4日のdirectional-camera baselineを使用する（承認済み）。
- 実機試験の走行場所・方向に対応するgoal画像列。

教師座標とTFの同期対応、および実機計算機の処理時間は、実装・検証工程で解決する項目とする。最新の条件付きモデルの対応は、このbaseline接続と検証の後に別仕様として扱う。

### 2026-09-15 追加確認：標準モデルと条件付きモデルの区別

9月4日baselineは標準NoMaD構造を維持し、K-Rail追加学習と方向別カメラ・教師yawのデータ条件を使用する。一方、9月6日の設定には `use_switchback_state_condition: true`、`switchback_state_embedding_dim: 16` があり、重みにも状態embedding/projectionが存在する。従って最新の条件付きモデルの変更をカメラと180度の教師yaw変更だけとは扱わない。同設定の `lambda_mode: 0.0` から分類ヘッドが学習済みであるとは断定しない。
