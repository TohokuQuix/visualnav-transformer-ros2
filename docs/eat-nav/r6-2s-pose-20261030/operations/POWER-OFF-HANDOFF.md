# R6 電源断前の保存・再開引継ぎ（2026-11-01）

ユーザーの「あと10分ほどで電源を切りたい」という依頼に従い、評価をSIGINTで停止しました。実評価プロセスの消滅、pipeline exit1、途中ジョブの完了gate拒否を確認済みです。システムの電源断自体は実行していません。

## 保存された状態

- 完了済み: 6ジョブ・180条件走行。全てK-Rail level0、同じ30parentを条件間で共有するため180独立地形ではありません。
- 完了条件: old_reference、event_2s、pose_q3_2、pose_q3_1、pose_q3_0、pose_q2_2。
- 意図的中断: pose_q2_0-k_rail_field-l0-batch00。未完了であり成功数に含めません。
- 中断出力997ファイルをSHA確認後に退避済み: `/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/poweroff-save-1793502720/partial-job`。
- 同snapshotにprogress/summary/frozen-binding、pipeline台帳、全監督ログ、launch/exit証拠、partial SHA manifest、saved-state.jsonを保存しました。
- 主記録: 成果物ルート `operations/latest-poweroff-save.json` と `operations/user-poweroff-interruption-20261101.json`。
- 永続コード: `/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree`。/tmpのコードの存在を仮定しません。Gitはself-contained、connectivity fsck成功。元repoのdirtyコードは上書きしていません。
- branch: exp/eatnav-r6-2s-pose-thresholds-20261030。
- main model SHA: 3b9491b03bf81cd8bb642468f8dab7d6292063e45c52d19a5fd2dba7f2339070。
- state-only SHA: f73c665080c3f95f48b5c1de9dd1891546094bdba24c9aa09c6fc0103ce09aa9。
- 859 runtime sourceの固定binding一致を確認済み。収集・dataset化・GPU学習は完了しており、電源断を理由とするやり直しは不要です。

## ユーザーの再開操作

元PCを起動し、SSH接続した後に実行:

```bash
bash /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/resume-codex.sh
```

会話が開いたら次を送信:

> R6の作業を再開してください。成果物ルートのoperations/POWER-OFF-HANDOFF.mdとlatest-poweroff-save.jsonを読み、実行中プロセスを確認してください。完了済み6ジョブ180走行を保持し、意図的に中断したpose_q2_0だけを記録付きで回復してください。固定仕様を保ち、全比較評価・上位3設定凍結・未使用test・指定bag評価・最終レビュー・GitHub pushとremote SHA確認まで続けてください。実機指令は禁止です。

このスクリプトは会話を開きます。評価の二重起動を避けるため、評価を無条件に起動するスクリプトにはしていません。再開指示後にエージェントが復旧します。

## エージェントの再開手順

1. 原依頼 `/home/ubuntu/eat_nav_retraining_handoff_20261030/追加学習依頼プロンプト.txt`、本書、latest-poweroff-save.json、user-poweroff-interruption-20261101.json、実progress/pipeline台帳を確認。ユーザーが再開するまでgoalはpaused。
2. host/containerの実PIDを調べ、ライブ評価があれば重複起動しない。監視用 /tmp/r6_watch_current_job.py は評価本体ではない。
3. Docker isaac-lab-onixの状態を確認。再起動後に /run/user/1000/gdm が消えてbind元不在で起動できない場合、所有者・mountを確認して必要な空ディレクトリのみ復元する。GUI認証や表示成功とは解釈しない。
4. 全runtime source・checkpoint・conditions・cohort・capacity30とfrozen-bindingの一致を再検証。partial SHAと6完了記録を保全。古い3完了時点の引継ぎへ戻さない。
5. q2_0の出力は既にsnapshotへ退避済みなので、重複して元jobフォルダを探して移動しない。active ledgerは意図的中断のfailedを残している。原因と旧台帳のバックアップを記録した後、中断q2_0の1件だけをledgerから除いてpendingへ戻し、failed validation_rollouts phaseだけを復旧する。完了6件や既存runtime benchmark/leadtime phaseは保持。
6. operations/independent-evaluation-exit.jsonを含む前回終了証拠を保存してから、新しいlaunch記録で独立tmuxセッションにpipelineを起動する。現在のrun-evaluation-independent.pyは永続worktreeを指す。CLI/SSH終了に評価寿命を依存させない。起動後はPIDとtrace書込みを確認。
7. 全7field×5level、10pose閾値＋比較3方式のvalidationを完了してからtop3凍結→未使用test→指定bag。途中表や0転倒を安全性の証明としない。実行していない結果・GUI表示・pushを成功扱いしない。
8. 最終報告・図表・必要な独立レビュー・回帰/strictload・新branch push・remote SHA照合まで元の全scopeを継続する。

## 未完了

validationは計画481ジョブ中6完了。top3選定、test、モデルbag評価、代表GUI再現、最終全成果レビュー、GitHub push/remote SHA確認は未完了です。既存の報告用コード・統計・学習成果の独立レビューは保存済みですが、最終全結果レビューの代わりにはしません。
