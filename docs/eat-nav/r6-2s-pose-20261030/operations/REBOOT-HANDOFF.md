# R6 再起動後の引継ぎ

ユーザーが元PCの再起動を希望したため、R6の自動思考goalはpaused、OS上の評価プログラムも停止しました。システムの再起動そのものはエージェントから実施していません。

コード・Git・checkpointの自己完結した永続コピーは restart-backup/latest.json のworktreeにあります。.git/objects/info/alternatesの一時ディレクトリ参照はrepackで解消し、Git fsckで検証しました。SHA一覧はbackup worktree親のsha256.jsonです。/tmpの旧コードが再起動後も残るとは仮定しないでください。元のdirty作業コピーを上書きしてはいけません。

成果物ルート:
/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030

停止時のvalidation評価:
- old_reference-k_rail_field-l0-batch00: completed、30走行
- event_2s-k_rail_field-l0-batch00: completed、30走行
- pose_q3_2-k_rail_field-l0-batch00: completed、30走行
- pose_q3_1-k_rail_field-l0-batch00: 再起動に備えた意図的SIGINTで中断、failed/incomplete

未完了jobはcomplete marker不足の検証gateで拒否されました。モデル学習のやり直しは不要です。完了済み90走行を再実行せず、中断の実証記録を読み、partial出力を別フォルダへ保持した上で、同じ固定assignmentの中断jobだけを再試行できる状態へ回復してください。ledgerのrunning/failedを原因確認せず削除してはいけません。元状態・出力SHA・意図的中断・再試行理由を記録してください。

operations/intentional-interruption-before-reboot.json と validation-evaluation/progress.json、evaluation-pipeline.json を実物で確認してください。再起動後はDockerコンテナが動いているかも確認が必要です。実プロセスが残っていれば重複起動しないでください。frozen-bindingのrelative source SHAは永続worktreeでも同一である必要があります。

常駐化の未承認試作drain_to_tmux.pyは実行していません。独立レビューで問題を発見したため/tmp/r6-unapproved-handoff-prototypeへ退避済みです。再起動後の復元対象ではなく、使用しないでください。

正常な再開はtmux等の独立した実行セッションで行い、Codex終了に依存しない構成を検討してください。モデル・閾値・seed・初期状態・走行時間は変更しません。最終目標の全field/level・10以上の閾値比較・top3凍結・unusedtest・指定bag・レビュー・結果表/図・新branch push/remoteSHA確認は未完了であり、縮小しないでください。実機指令は禁止です。

CLIから再開する操作:
bash /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/resume-codex.sh

会話が開いたら次を送ってください:
「成果物ルートのoperations/REBOOT-HANDOFF.mdを読み、R6の作業を再開してください。実行中プロセスを確認し、完了済み試行を守り、中断jobだけを記録付きで回復してください。元の全仕様に従い評価・最終確認・GitHub pushまで続けてください。」
