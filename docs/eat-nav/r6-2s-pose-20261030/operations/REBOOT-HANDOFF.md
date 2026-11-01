# 最新のR6再開引継ぎ（2026-11-01）

現在はユーザーの電源断希望で停止済みです。完了済み6ジョブ180走行を保持し、pose_q2_0だけを意図的に中断・SHA付き退避しました。

最新の手順は同ディレクトリの **POWER-OFF-HANDOFF.md** と **latest-poweroff-save.json** を読んでください。以前の3ジョブ90走行時点の引継ぎや復旧記録を、現在の台帳へ復元しないでください。

再開コマンド:

```bash
bash /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/resume-codex.sh
```

会話を開いたら、POWER-OFF-HANDOFF.md記載の再開指示を送信してください。実評価の重複起動は禁止、実機指令は禁止です。
