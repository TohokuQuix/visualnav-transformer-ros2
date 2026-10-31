# R6 SSH引継ぎ（旅行中の確認・Codex再開）

評価プログラムは1回の起動でvalidation→上位3設定凍結→test→bag評価を順に進めます。各走行のたびにCodexへの依頼は不要です。OS上のPython/Docker実行と、Codexの推論利用枠は別です。ただし実行中のアプリや親プロセスを終了した場合の継続は保証しないため、現在のPC・Docker・Codexアプリは起動したままにしてください。tmuxへの現在の評価プロセスの移行はまだ実施していません。

## SSHから進捗確認

bash /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/ssh-status.sh

またはログだけを見る場合:

tail -f /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/evaluation-production.log

ログ監視をCtrl+Cで終了しても、評価を停止する操作ではありません。

## Codexが停止し、利用枠が回復した後にSSHで再開

同時に二重のCodex作業や評価を起動しないでください。現在の会話のIDは
01a1541e-50e0-78f1-9938-d41a6a5e74f7
です。ローカルのsession保存ファイルの存在とCLI resumeコマンドを確認済みです。

tmux new -A -s r6-codex
cd /tmp/eatnav-r6-20261030
/usr/lib/chatgpt/resources/codex --no-daemon resume 01a1541e-50e0-78f1-9938-d41a6a5e74f7

アプリ同梱CLIで「this CLI has no complete local package」と表示された場合に備え、--no-daemonを明示しています。このオプションの存在は実機上のresume --helpで確認済みです。対話的な同時再開は既存作業との競合を避けるため実行していません。必ず別PCからSSHで元PCへ接続した状態で実行してください。

再開後の指示:
「docs/eat-nav/r6-2s-pose-20261030/operations/SSH-HANDOFF.mdを読み、既存プロセスと成果物を確認してください。動いている評価は再起動せず監視し、失敗時は原因と証拠を確認して安全な境界から再開してください。元の引継ぎ仕様の全範囲を保ち、最終比較・上位3設定凍結・未使用test・指定bag評価・最終報告・必要なレビュー・新ブランチpushとremote SHA確認まで継続してください。実機指令は禁止です。」

SSHを切る前にCtrl+Bを押してからDを押すとtmuxをdetachできます。接続し直したらtmux attach -t r6-codexで戻ります。同じアカウントのCodex CLIにも利用上限が適用されるため、上限の回避手段ではありません。

## 評価が止まっていた場合

progress.jsonのrunning表示だけでは生存確認になりません。実プロセスを確認してください。停止済みでも、原因を見ずにpipelineを重複起動したり、running/failed記録を削除したりしないでください。現在のrunnerは未調査のrunning/failed stageを自動再実行せず拒否します。成功済みjobの出力は保持し、実装・モデル・設定・cohortのSHAが変わらないことを確認してから明示的に再開します。

作業コード: /tmp/eatnav-r6-20261030
成果物: /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030
branch: exp/eatnav-r6-2s-pose-thresholds-20261030
元の依頼: /home/ubuntu/eat_nav_retraining_handoff_20261030/追加学習依頼プロンプト.txt
元repoのdirty作業・旧checkpoint・旧testは保護し、新実装を元の作業ツリーへ上書きしないでください。

既存の独立レビューでP1/P2を修正済み、R6テスト87pass。モデル学習完了。anchor leadtime集計完了。閉ループ比較はまだ進行中で、test/bag/pushは未完了です。実測と計画を混同しないでください。

参考公式資料:
https://learn.chatgpt.com/docs/codex/cli
https://learn.chatgpt.com/docs/pricing

## デスクトップからCLIへ切り替える確認

直前の状態は成果物ルート operations/before-cli-switch.json に保存しています。デスクトップの応答を停止してアプリを終了した後、SSHでCLIを再開してください。評価が残って動いている可能性も、停止している可能性もあるため、実プロセスと新しいtrace書込みを先に確認します。完了済みjobは保護し、未完了jobは成功へ変換しません。

同時接続のロック表示が残る場合は、元のデスクトップ側が終了したことを確認してrで再試行します。fでforkしないでください。
