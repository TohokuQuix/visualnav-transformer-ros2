# R6 電源断前の保存（2026-11-01 18:43 JST）

ユーザーの指示で評価と監督プロセスを停止しました。再開指示があるまで起動しません。

- 完了済み10ジョブ・300条件走行。全てk_rail_field level0、同じ30初期条件を条件間で共有。
- 中断: pose_q0_2-k_rail_field-l0-batch00。未完了のため集計対象外。
- 途中出力1809ファイル、台帳、ログ、モデルSHAを保存: operations/poweroff-save-1793526175/。
- 最新の主記録: operations/latest-poweroff-save.json。
- validation_rolloutsは意図的中断でfailed。完了10件を保持し、中断ジョブのみ原因記録付きで再実行する。
- 収集・dataset化・GPU学習は完了済み。全validation、top3凍結、test、指定bagモデル評価、GUI再現、最終レビュー、GitHub push/remote SHA確認は未完了。
- 新規報告用20Hzピーク監査は1768window（完全1749、部分19）。専用テスト3pass・独立レビューP1/P2なし。実験の凍結コード・モデル・閾値は変更なし。
- rawピークの最終監査は全queue完了後にreporting/raw_peak_metrics.pyを--interimなしでvalidation/testへ実行。部分windowをMAEに混ぜない。

## 再開

```bash
bash /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/resume-codex.sh
```

会話が開いたら「R6を再開してください。operations/CURRENT-EXECUTION.mdとlatest-poweroff-save.jsonを確認し、完了10ジョブを保持して中断q0_2のみ回復し、最終評価・GitHub pushまで続けてください」と送信。

## 再開時の確認

1. host/container実PIDを確認。ライブ処理があれば二重起動しない。
2. 全runtime source・モデル・profile/cohort/capacityの凍結bindingとsnapshot SHAを照合。
3. 古いexit/台帳をバックアップし、中断q0_2のrunning/failedとvalidation_rolloutsのfailedだけを調査記録付きでpendingへ戻す。完了10件やbenchmark/leadtimeを削除しない。
4. 独立tmuxのrun-evaluation-independent.pyで再開し、実PIDとtrace書込みを確認。resume-codex.sh自体は評価を自動起動しない。
5. 元の全scopeを継続。実機指令禁止。途中成果を完成扱いしない。
