# R6 コード・学習済み重みの中間保存（評価未完了）

このブランチはコードと学習済み重みを別PCへ取得するための中間保存です。全研究の完了、実機への導入検証、安全性の確認を意味しません。公開時点の比較評価はvalidation計画481ジョブ中16ジョブ（480条件走行）完了、次の1ジョブが実行中です。全比較後の上位3設定凍結、未使用test、指定bagへのモデル適用、最終結果レビューは未完了です。実ロボットへの指令は送っていません。

## 別PCで取得

既存checkoutで未保存変更があれば、先に保存してください。以下はリポジトリrootで実行します。

```bash
git fetch origin exp/eatnav-r6-2s-pose-thresholds-20261030
git switch exp/eatnav-r6-2s-pose-thresholds-20261030
git pull --ff-only origin exp/eatnav-r6-2s-pose-thresholds-20261030
sha256sum checkpoints/eat_nav/r6_2s_pose_20261030/*.pt
```

新規cloneの場合:

```bash
git clone --branch exp/eatnav-r6-2s-pose-thresholds-20261030 https://github.com/TohokuQuix/visualnav-transformer-ros2.git
cd visualnav-transformer-ros2
```

重みは通常のGit blobに含まれます（各739,701 bytes）。期待SHA256:

| ファイル | SHA256 |
|---|---|
| `model_seed123.pt`（DEM＋状態） | `3b9491b03bf81cd8bb642468f8dab7d6292063e45c52d19a5fd2dba7f2339070` |
| `state_only_seed123.pt`（状態のみの診断比較） | `f73c665080c3f95f48b5c1de9dd1891546094bdba24c9aa09c6fc0103ce09aa9` |

モデル単体のCPU loadにはPyTorch・NumPyと本repoのコードを使用します。検証環境はPython3.10.13、PyTorch2.7.0+cu128、NumPy1.26.4です。既存のROS環境を変更せず分離したPython環境を使用してください。

```bash
PYTHONPATH=src python - <<'PY'
from visualnav_transformer.eat_nav.r6_model import load_checkpoint
model, payload = load_checkpoint('checkpoints/eat_nav/r6_2s_pose_20261030/model_seed123.pt', device='cpu')
print(payload['schema'])
print(payload['contract'])
print(payload['support'])
PY
```

旧R4*/R5へcheckpointパスだけを差し替える方式では利用できません。R6は14候補別の真の2秒イベント4種と、0.1～2.0秒の20点の相対XYZ・yaw/roll/pitch sin/cosを出力する新schemaです。ブランチ切替だけでは稼働中の実機ノードの重みは更新されません。この手順では実機ノードを起動しません。

## 学習データと結果

- 収集88ジョブと再利用した実branchから、979組・13,706候補走行のdatasetを作成しました。候補は初期状態を共有しており、13,706独立事故例ではありません。
- CUDA mini-batch、seed123、300epochで主モデルとstate-only比較を学習済みです。主モデルvalidation ADE約0.0416m、FDE約0.0694m。DEMによる精度優位は確立していません。
- 学習datasetのvalidationには転倒・near-fall陽性がなく、両headのvalidation再現率は未確立です。閉ループで観測した転倒数とは区別します。
- 閾値10設定＋旧方式・イベント方式・併用方式の全7field×level0～4比較が進行中です。途中の0転倒や停止増加を安全性・有効性の証明とはしません。
- [学習証跡・データ転送手順](verified-training/README.md)、[中間比較](validation-progress/first-cell-13-condition-interim.md)、[実行環境](execution-records/environment-reproduction.json)を参照してください。

大容量データは元PCの `/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030` に保存し、Gitへ生bag・巨大datasetを追加していません。最終datasetは `final-merged-dataset/samples.npz`、SHA256は `955a5ab0b5e2bfc3b49c620ab33c24c0f7f9824485e793ff5951e536f229fc67` です。別PCへのrsync例は学習証跡READMEにあります。

公開前検証ではGit内の両重みのstrict load・全head support・有限値・shapeを確認し、R6テスト106件がpassしました。報告用copy検証器のP2を修正し、独立レビューで完了16バッチの28,660 originalファイル・480解凍traceが元copy証跡と一致することを確認しました。旧モデル・凍結runtime・評価設定は変更していません。

GitHubへのpushの事実とremote commitは、実際のpush後の記録で確認します。本書だけではpush成功の証拠になりません。最終成果は後続commitで更新します。
