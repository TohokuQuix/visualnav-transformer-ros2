# R6 電源断前保存・再開（2026-11-02）

ユーザー依頼により評価を中断しました。pipelineとsimulatorへSIGINTを送り、両プロセス消滅とpipeline exit -2を確認済み。電源断自体は実行していません。

- 完了済み16ジョブ・480条件走行を保持。validation計画481ジョブの途中です。
- 中断ジョブ: `pose_q3_1-k_rail_field-l1-batch00`。台帳はfailed、完了には含めません。
- 途中893ファイルを永続退避し、コピー後SHA再照合済み。complete.jsonはありません。
- snapshot: `/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/poweroff-save-1793549172`
- worktree: `/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree`
- 保存前HEAD: `b7b7e4e50bbf6a07678d7bf0c3d7cc337dbf3389`
- データ収集・dataset・モデル学習は保存済み。全比較・top3固定・test・bag・最終レビュー・GitHub pushは未完了です。
- GitHubへのINTERIM保全pushもまだ実行していません。

## 再開

元PCを起動してSSH接続後:

```bash
bash /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/operations/resume-codex.sh
```

会話を開き、次を送信:

> R6を再開してください。成果物ルートoperations/POWER-OFF-HANDOFF.mdとlatest-poweroff-save.jsonを読み、実プロセス確認後、完了済み16ジョブ480条件走行を保持してください。893ファイルの途中保存SHA・859 runtime binding・モデル・profile/cohort/capacityを照合し、中断したpose_q3_1-k_rail_field-l1-batch00だけを証拠付きでpendingへ復旧してください。failed validation phaseと前回exitを退避してから独立tmuxで再開し、二重起動しないでください。監査P2修正、全評価・test・bag・最終レビュー・GitHub push/remote SHA確認まで進めてください。実機指令は禁止です。

## エージェント向け

- 途中出力はsnapshotへコピー済み。元containerの/tmpは電源断後に存在する前提にしない。
- 中断ジョブのfailed台帳とfailed validation_rolloutsだけをバックアップして復旧する。完了16件、benchmark、anchor-leadtimeを保持。
- runtime 859files、モデル、profile、cohort、capacityの凍結は変更しない。
- Docker再起動時のbind元欠落は実mountを確認して対処する。GUI表示成功と解釈しない。
- 古い/tmp/r6_resume_verified.pyはHEAD固定が古いためそのまま使わない。
- 独立レビューで報告専用 `execution-records/completed-batch-verifier.py` にP2: trace以外の保持JSON/NPZのSHAを元copy証跡と照合せずcopy_sha_verified=Trueと記録する。TDDで全保持original＋解凍trace SHA検証を追加し、16完了ジョブを再検証・独立レビューする。凍結runtimeは変更しない。
- 最新独立レビュー: R6 tests95件pass、両checkpoint strictload/SHA/finite/shapes確認済み、旧checkpoint変更なし。上記P2以外に追加P1/P2なし。最終全研究レビューは別途必要。
- INTERIM保全pushを行う場合も評価未完了と明記し、全研究完了扱いにしない。最終push/remote SHA照合まで継続。
