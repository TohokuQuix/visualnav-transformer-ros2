# K-Rail baselineの実機接続

対象ブランチ: `feat/krail-baseline-robot`。9月4日directional-camera baselineを使用する。
標準NoMaDのモデル・推論処理は共通接続ブランチから変更していない。

## 1. 重みと設定

| ファイル | 用途 |
| --- | --- |
| `config/krail_baseline_model.yaml` | 学習時に対応する標準NoMaD構造 |
| `config/krail_baseline_robot.yaml` | front画像入力・map Path出力 |
| `model_weights/krail_baseline_20260904_latest.pth` | 実際の学習済み重み（Git管理対象外） |

重みの出典:
`/home/ubuntu/visualnav_docker/visualnav-transformer/train/logs/nomad_krail/krail_directional_camera_baseline_2026_09_04_16_13_16/latest.pth`

このPCにはコピー済み。別PCでは同じファイルを上記配置先にコピーする。EMAや条件付きモデルへ差し替えない。
リポジトリ直下で確認:

```bash
sha256sum model_weights/krail_baseline_20260904_latest.pth
```

期待値: `c07ac8491190e5237c29f35e43ce3b30b80b1ca5b85c424ef28eb0c2df813e3b`

設定は画像履歴0.25秒、context_size=3（合計4枚）、距離復元係数0.05、8予測点。
モデルが出す累積局所XYに係数を一度だけ掛け、車体原点を加えた通常9点を配信する。

## 2. goal画像を1枚設定する

`config/krail_baseline_robot.yaml` の `goal_image: ""` を、目的地の画像ファイルの絶対パスへ変更する。
JPEG/PNGなどPillowが読み込める画像を指定する。起動時にRGBで読み込み、毎回同じgoal画像を使用する。実行中の切り替えは行わない。

```yaml
goal_image: /absolute/path/to/goal.jpg
goal_distance_threshold: 3.0
```

未指定、読めない画像、topomapとの同時指定は起動エラーになる。相対パスは設定YAMLの位置を基準に解決する。
画像履歴は現在と過去3枚を使い、ネットワーク構造・重み・DDPM処理は変更しない。画像列の探索やサブゴール切り替えは行わない。

既存距離ヘッドの予測値が `goal_distance_threshold` 未満になった観測で到達と判定し、空Pathを送り、その起動中は停止を維持する。これは**メートル単位ではない**。初期値3.0は既存の近接閾値を参考にした値で、単一goalの到達判定として実機検証済みではない。bagで早すぎる停止・停止不足を確認して調整する。厳密な位置への到達保証ではない。

bag検証用の候補画像（実機goalとして自動設定しない）:
`/home/ubuntu/visualnav_docker/visualnav-transformer/datasets/krail_directional_camera_master/2026_08_03-10_31_05/192.jpg`

この画像は学習データ由来なので未知経路での性能評価には使わない。初回の実機評価は前方カメラ固定の前進区間で行う。学習データの復路にはrear画像と180度変更した教師yawが含まれるが、この設定はrear入力や復路座標切り替えを実装しない。
単一goal方式は途中の通過点を指定しないため、遠い・見た目が異なるgoalへの長距離走行性能は別途検証する。

## 3. 起動

依存環境はリポジトリREADMEのInstallationとロボット接続手順に従う。ROS1をsourceしていない端末を使い、ROS2 Jazzyの場合は次を実行する。別PCではチェックアウト先へ移動する。

```bash
source /opt/ros/jazzy/setup.bash
cd /home/ubuntu/visualnav-transformer-ros2
uv run python -m visualnav_transformer.deployment.src.navigate_robot \
  --interface-config config/krail_baseline_robot.yaml
```

- 入力: `/front_camera/image_raw/compressed`
- 出力: `/teaching_playback/path`（map座標）
- 必須TF: 画像stamp時点の `map ← chassis/body`
- `device: auto` はCUDA利用可能時にGPU、それ以外はCPU。CPUで起動可能でも実時間性能を保証しない。

追従側のビルド済みROS2 workspaceをsourceした別端末:

```bash
ros2 launch simple_trajectory_follower nomad_follower.launch.py
```

追従側はONIXリポジトリの `feat/nomad-path-following`（`39bd4b2`）を使用する。出力 `/cmd_vel_auto` をロボット側へ接続する。`scripts/publish_cmd.py` は併用しない。

## 4. bag確認

記録済みPathや速度指令を同時配信しないよう、画像とTFだけを再生する。

```bash
ros2 bag play '/media/ubuntu/Extreme SSD/20260803_experiment/2026_08_03-10_31_05' \
  --clock --topics /front_camera/image_raw/compressed /tf /tf_static
```

推論側は次のようにシミュレーション時刻を有効にする:

```bash
uv run python -m visualnav_transformer.deployment.src.navigate_robot \
  --interface-config config/krail_baseline_robot.yaml \
  --ros-args -p use_sim_time:=true
```

追従側も `nomad_follower.launch.py use_sim_time:=true` で起動し、RVizにも同じ時刻設定を使う。再生検証中は実機の速度指令入力へ接続しない。
Pathの始点・向き・単位をTFと重ねて確認する。画像途絶、TF欠落、再生時刻巻き戻りで空Pathと追従停止を確認する。

## 5. 検証記録と実機前の確認項目

2026-09-15、このPCで確認:

- 配置した重みのSHA256一致、標準NoMaDへのstrictロード成功。
- topomap方式の実装時には実重み、実画像4枚、193枚のgoal列、seed42、8サンプル、10 DDPMステップを使用。既存 `navigate.py` のNoMaD分岐と接続側の局所8点が完全一致。
- 単一goal方式でも実重み・実画像4枚・goal画像192.jpgで、同じgoalを与えた従来推論との局所8点の完全一致を確認。到達判定はFalseとなり、1枚指定だけで即到達しない。
- 実設定・重み・単一goal画像によるROSノードの構築と終了。
- 単一goal対応後はテスト31件成功。PyTorch Transformerのnested-tensorに関する警告4件あり。
- 検証環境: ROS2 Jazzy、Python3.12、CPU、Torch2.14.0+cpu、torchvision0.29、diffusers0.40。検証用環境は `/tmp/nomad-interface-venv`。`uv sync`による本番環境全体の検証とは区別する。

既存テストの再実行（ROS2とPython依存環境を準備して実行）:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  ROS_DOMAIN_ID=188 uv run python -m pytest -q tests
```

未確認:

- 教師位置・yawの生成時刻とbag TFの同期対応。サンプルの単純照合で完全一致は未確認。
- 全bag再生でのPath位置・向き・到達動作と追従ノード接続。
- 実機で使うgoal画像、GPU環境と推論時間、実機走行。

実機では有効Pathの観測年齢が0.8秒の期限内に収まることを測定する。期限変更時は追従側の1.0秒期限との関係も再検証する。上記の座標同期とbag検証を終えてから、速度制限と停止手段を確認して前進試走する。
