# R6 電源断前保存・再開（2026-11-02、36ジョブ時点）

ユーザーの明示依頼により評価を一時停止。実PIDを検証後pipeline49204→simulator87518の順でSIGINT。両プロセス消滅、pipeline exit -2を確認。PCの電源断操作はしていない。

- 完了36/481 validationジョブ、1080条件結果を保持。34実シミュレーションバッチ＝1020走行、2方策一致再利用バッチ＝60条件結果。1080独立走行ではない。
- 中断ジョブ `pose_q0_2-k_rail_field-l2-batch00` はfailed/KeyboardInterrupt。部分走行を完了・成功・新教師として数えない。
- snapshot: `/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/poweroff-save-1793575169`
- 途中814ファイルを永続退避。Docker停止後原本と全SHA一致、コピーSHA再照合済み、complete.jsonなし。`poweroff-verification.json` 参照。
- 固定runtime859ファイル・checkpoint SHAを再照合済み。収集・dataset・学習済みモデルは保存済み。
- 最新中間push `7c67be5580d16d099dc1cc9546234a559faec386` はremote SHA一致済み。新旧コード・モデルを保持。全validation比較・top3凍結・未使用test・bag・最終レビュー・最終pushは未完了。
- `latest-poweroff-save.json` とsnapshot内saved-state.jsonが今回の再開基準。過去16ジョブのsnapshotを現在状態として使わない。

## 再開

元PC起動後、SSH端末で:

```bash
bash /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/resume-codex.sh
```

会話を開き、以下を送信:

> R6を再開してください。operations/POWER-OFF-HANDOFF.mdとlatest-poweroff-save.jsonを読み、実プロセス確認後、完了36ジョブ1080条件結果（実走行1020＋再利用60）を保持してください。snapshotの814途中ファイルSHA・859 runtime binding・モデル・profile/cohort/capacityを照合してください。中断したpose_q0_2-k_rail_field-l2-batch00のfailedエントリーと出力、failed validation_rollouts phase、旧exit証跡を退避してから、そのバッチだけをpendingへ戻し、固定された30環境バッチを最初から再実行してください。途中時刻から物理状態を再現できるとは仮定しないでください。独立tmuxで二重起動せず、全評価・未使用test・bag・最終レビュー・GitHub最終push/remoteSHA確認まで進めてください。実ロボットへの指令は禁止です。

## 再開時の注意

- container /tmpやホスト/tmpの残存を前提にしない。再開時のruntime stageは固定ソースから再作成する。
- complete36件、benchmark、anchor-leadtimeを保持し、故意中断の当該1件とfailed phaseだけを証拠付きで復旧する。
- 過去の/tmp/r6_resume*.pyはHEAD/件数/ジョブ固定が古いため、そのまま実行しない。
- 一致再利用を新規の独立物理走行や独立推論処理時間として扱わない。
- 前回引継ぎに残っていたcopy証跡P2は既に修正・全16バッチ再監査・独立レビュー済み。今回の新規完了バッチも保持original＋解凍traceのSHAを検証済み。同修正を繰り返さない。
- 未使用testとbagで閾値を調整しない。モデル/入力契約/固定設定は途中変更しない。
