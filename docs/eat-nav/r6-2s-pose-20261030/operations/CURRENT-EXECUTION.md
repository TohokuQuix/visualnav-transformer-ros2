# R6 最新の実行・再開状態（2026-11-01 17:15 JST）

ユーザーがgoalをactiveへ戻したため、前回の電源断準備後の停止状態から評価を再開しました。停止・保存について書かれたPOWER-OFF-HANDOFF.mdは、その時点の履歴です。今の電源断可否は必ず実プロセスを確認してください。

- 保存済み180条件走行（6完了ジョブ）は保持。
- 中断したpose_q2_0だけを固定条件で再試行。997 partialファイルは以前のsnapshotに保持し、SHAを再検証済み。
- 全859 runtime source・モデル・条件・cohort・capacity30がfrozen-bindingと一致。
- 復旧証拠: 成果物ルート operations/latest-poweroff-resume.json とそのarchive。
- 独立tmux: socket eatnav-r6、session evaluation。起動後のpipeline PID20746、simulator PID20966を実確認（PIDは過去の観測値なので、そのまま生存とみなさない）。
- exitマーカーは旧証拠をarchive後に除き、現実行が終了した時に新規保存する構成。旧exit1を現在の失敗と混同しない。
- collector/dataset/GPU学習のやり直しは不要。次は全validation比較を継続し、上位3凍結→unusedtest→指定bag→最終レビュー→新branch push/remoteSHA確認まで元のscopeを保持する。

確認コマンド:

```bash
bash /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/ssh-status.sh
```

CLI/SSH終了では評価は止まりません。電源断・Docker停止では評価も止まります。実機指令は禁止です。
