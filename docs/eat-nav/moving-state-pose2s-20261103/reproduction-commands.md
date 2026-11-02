# 実行設定と再現コマンド

実行rootは `/home/ubuntu/eatnav-moving-state-pose2s-20261103`、ローカル成果rootは `/home/ubuntu/eatnav-moving-state-pose2s-data-20261103`。環境は `/home/ubuntu/venvs/isaaclab_py310/bin/python`。CPU学習8thread、評価・推論4thread。GUI OFFで収集し、学習もheadless。RTX4090 Laptop 16GBで収集10envを実測選択。学習はCPUを使いGPUと排他的な取り合いを避けた。

以下の出力先は実行済み。再実行では元成果を上書きせず、新しい出力先を使う。生記録とdatasetはローカル専用で、GitHubのcloneだけでは再学習できない。

```bash
export PYTHONPATH=src:scripts
PY=/home/ubuntu/venvs/isaaclab_py310/bin/python
DATA=/home/ubuntu/eatnav-moving-state-pose2s-data-20261103

# 実際の収集はprototypeのsnapshot方式ではなくfresh-process方式。
# 途中まで完了したjobはledgerに従って再利用。独立した別実験は新plan/outputで行う。
"$PY" scripts/eat_nav_moving_pose2s_collect.py --plan "$DATA/collection/plan.json" --fresh-process --execution-amendment "$DATA/collection/execution-amendment-v4.json" --output "$DATA/collection/production"

"$PY" scripts/eat_nav_moving_pose2s_dataset.py --fresh-collection "$DATA/collection" --output "$DATA/fresh-dataset"
"$PY" scripts/eat_nav_moving_pose2s_audit.py --dataset "$DATA/fresh-dataset"
"$PY" scripts/eat_nav_moving_pose2s_dataset.py --merge "$DATA/reused-dataset" "$DATA/fresh-dataset" --output "$DATA/combined-dataset"
"$PY" scripts/eat_nav_moving_pose2s_audit.py --dataset "$DATA/combined-dataset"

"$PY" scripts/eat_nav_moving_pose2s_train.py --dataset "$DATA/combined-dataset" --output "$DATA/models/M_final" --variant final --device cpu --threads 8 --epochs 150 --seed 123 --time-budget-sec 7200
"$PY" scripts/eat_nav_moving_pose2s_evaluate.py --dataset "$DATA/reused-dataset" --checkpoint "$DATA/models/M_final/checkpoint.pt" --output "$DATA/evaluation/M_final-existing-audit" --role audit_eval --threads 4 --compare-checkpoint "$DATA/models/M_move/checkpoint.pt"
"$PY" scripts/eat_nav_moving_pose2s_evaluate.py --dataset "$DATA/fresh-dataset" --checkpoint "$DATA/models/M_final/checkpoint.pt" --output "$DATA/evaluation/M_final-fresh-audit" --role audit_eval --threads 4 --compare-checkpoint "$DATA/models/M_move/checkpoint.pt" --compare-checkpoint "$DATA/models/M_stop/checkpoint.pt"
"$PY" scripts/eat_nav_moving_pose2s_evaluate.py --dataset "$DATA/combined-dataset" --checkpoint "$DATA/models/M_final/checkpoint.pt" --output "$DATA/evaluation/M_final-combined-validation" --role validation --threads 4 --compare-checkpoint "$DATA/models/M_move/checkpoint.pt"
```

元encoderへの転移パスはconfig.jsonに保存し、SHAは `c29bef5a56947db03f91ae2261dd8d259762479ebd9d38553017f3d5bd3c0421`。別PCでは既存モデルpathを `--transfer-checkpoint` に明示できる。学習時に元checkpointのSHAを検証する。既存datasetからM_stopを作る際は `--variant initial`、M_moveは `--variant moving`。

実際の実行は独立監査のfreeze成功後にdataset→audit→merge→audit→fit→freeze→evaluateの順で進めた。公開版ではcollection_stateの層別表示だけを修正し、保存予測から同じ集計全体値を再現した。元評価ファイルはローカルに残している。

入力・座標・損失の完全な定義は承認仕様、学習に使った各checkpointのconfig.json／dataset-binding.json／training.json、collection/の固定plan・amendmentを参照。checkpointの取得と推論だけならREADMEのコマンドでよく、Isaac Simとローカルdatasetは不要。
