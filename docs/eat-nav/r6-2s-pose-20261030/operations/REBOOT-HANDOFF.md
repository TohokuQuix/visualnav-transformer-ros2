# R6 最新引継ぎ

最新の実行状態は **CURRENT-EXECUTION.md**、実際の復旧記録は **latest-poweroff-resume.json**、現在の台帳は成果物ルート evaluation-pipeline.json / validation-evaluation/progress.json を確認してください。

2026-11-01 17:15 JST、ユーザーの再開指示で180条件走行を保持してpose_q2_0を再試行しました。以前のPOWER-OFF-HANDOFF.mdは停止時点の履歴です。実プロセスが動いていれば重複起動せず、以前の180走行時点の台帳へ復元しないでください。

会話を開くコマンド:

```bash
bash /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/resume-codex.sh
```

実機指令は禁止です。
