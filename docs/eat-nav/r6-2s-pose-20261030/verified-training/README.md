# R6 学習済みモデルと学習証跡（評価前）

これは学習・load検証済みのモデルを保存するパッケージです。本番閉ループ比較、上位3設定の凍結、未使用test、実機bagへのモデル適用、およびGit pushの完了を意味しません。最終結果は別途作成する評価報告を参照してください。

## 保存内容

- `checkpoints/eat_nav/r6_2s_pose_20261030/model_seed123.pt`: DEM＋機体状態の新しい2秒イベント・未来軌跡モデル。
- `checkpoints/eat_nav/r6_2s_pose_20261030/state_only_seed123.pt`: 同じ分割・学習seedで学習した、DEMを使わない比較モデル。
- 同ディレクトリの`index.json`: モデルのSHA256とdatasetのSHA256、学習supportの制約。
- このディレクトリのmetrics/training/production-profiles/calibration JSON: 学習精度、学習設定、validation由来の閾値候補、フリッパのシミュレーション校正。
- `checkpoint-load-verification.json`: このパッケージから実際にloadし、全出力の有限値・shape・全4headの学習supportを確認した結果。

入力はDEM履歴4×2×20×20、状態28次元、14候補（各8点XYと前後進区分）です。DEMは2m四方・0.1m解像度です。イベント出力は候補ごとの今後2秒以内の4種類で、旧10秒モデルとは意味が異なります。未来出力は0.1～2.0秒の20時刻×9成分です。成分は初期yawに合わせた重力整列相対XYZと、yaw差・重力基準roll・pitchそれぞれのsin/cosです。

学習は1プロセスのCUDA mini-batch、300epoch、seed123です。979組（train519、validation460）、13,706本の実行済み候補走行を使用しました。14候補は同一初期状態を共有するため、独立した13,706事故例を意味しません。

主モデルのvalidation ADEは約0.0416m、FDEは約0.0694mです。state-only比較からDEMによる精度改善は確立していません。validationに転倒・near-fallの陽性例がないため、その再現率・PR-AUCは未確立です。イベントsigmoid出力を校正済みの実発生確率と断定しないでください。

## 大容量データの保管場所

元PC上の成果物ルート:

`/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030`

- `collection/`: 元の80収集ジョブ（70のfield/levelと10の補足scene）。
- `wall60x100-collection/`: 幅0.6m・高さ1mのケース相当の追加8ジョブ。
- `reused-dataset/`: 既存の実走行を真の2秒教師へ再ラベルした105組。
- `final-merged-dataset/`: 最終学習dataset。samples.npzのSHA256は`955a5ab0b5e2bfc3b49c620ab33c24c0f7f9824485e793ff5951e536f229fc67`。
- `final-dataset-audit/`: split・mask・候補幾何・未来点数の監査と20Hz姿勢統計。
- `final-model/`、`state-only-diagnostic/`: 学習ログ・設定・予測・重み。
- `production-profiles/`: 本番比較前に固定した12閾値候補。10設定を比較予定。
- `bag-intermediate-observed/`: 指定bagの因果的入力抽出（一次評価用）。inpaintedだけの未観測セルを安全な観測として扱わない。

生データはGitには含めません。別PCへdatasetを移す場合は、元PCにSSH接続できるホスト名を指定して以下を実行します。

```bash
rsync -a --info=progress2 SOURCE_HOST:/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/final-merged-dataset/ ./r6-final-merged-dataset/
sha256sum ./r6-final-merged-dataset/samples.npz
```

`SOURCE_HOST`は実際の元PCのSSH接続先へ置き換えてください。SHA256が上記の値と一致することを確認します。生の候補走行も必要なら`collection/`と`wall60x100-collection/`を同様に移してください。`all-collection-sources/`は元PCのフォルダを指すsymlinkなので、それだけをコピーしても生データは移りません。

## Pythonからのload

このリポジトリのrootで、PyTorchとNumPyが利用可能なPythonから実行します。学習で使用したのはPython3.10とPyTorch2.7 CUDA12.8です。ROS bag抽出用Python3.12とは分離しています。

```bash
PYTHONPATH=src python - <<'PY'
from visualnav_transformer.eat_nav.r6_model import load_checkpoint
model, payload = load_checkpoint('checkpoints/eat_nav/r6_2s_pose_20261030/model_seed123.pt')
print(payload['contract'])
print(payload['support'])
PY
```

このモデルはROS-freeの推論部品です。旧R5のcheckpointパスだけを差し替えて使うことはできません。新しい契約のloaderとR6 plannerを使用してください。実ロボットへの駆動指令は今回の作業では送信していません。
