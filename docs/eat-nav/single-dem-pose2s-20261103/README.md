# 1フレームDEM・2秒状態予測：暫定モデルと条件B評価

[仕様書](../single-dem-pose2s-spec-v0.1.md)／[実機側確認票](real-pc-checklist.md)／[調査](investigation.md)／[参照commit・SHA](reference-index.json)／[段階status](stage-status.json)。

`reference-onix/`は参照commitと一致する外部コードの文書添付snapshot。新しい実装や実機install用ファイルではない。既存の実機planner・旧checkpoint・生記録は上書きしていない。新しい予測モデルと専用評価コードを追加した。巨大な生データ・dataset本体は添付しない。

本bundleで`sha256sum -c SHA256SUMS`を実行すると仕様書を含む整合性を確認できる。

## 現在の成果

- [実機側PCでの取得・最小推論手順](REAL-PC-QUICKSTART.txt)
- [モデル索引・SHA256](model-index.json)と[学習済みcheckpoint](models/seed123/checkpoint.pt)
- [単一入力から14候補を予測した例](inference-example/predictions.json)と[CPU推論時間](inference-example/benchmark-cpu.json)
- [全490組の予測精度CSV](prediction-evaluation/candidate_field_level.csv)、[評価report](prediction-evaluation/report.json)
- [dataset件数・漏洩監査](dataset-index/audit.json)、[manifest](dataset-index/manifest.json)、[分割](dataset-index/split.json)
- [旧Actorと予測NNの左右差調査](left-right-turn-audit/root-cause-followup.txt)
- [ユーザー指示による暫定モデル・条件B先行への移行](provisional-selector-transition.txt)

全体平均は許容値内だが、元の各組最大誤差判定は323/490組合格で未達。暫定モデルでの探索的評価をユーザーが指定した。旧Actorの左右非対称は確認できたが、予測誤差の主要原因と証明したわけではない。実機互換性・実機安全性は未検証。

本番は条件Bのみ350走行。Aは後日同じseedで追加できる。A/Bの効果比較を実施したとは扱わない。rosbagはユーザーの実機側PCで確認する。今回の実機bagはodomがFAST-LIO以外との報告があり、入力分布の相違を明記する。

## 条件Bの結果

条件Bの350走行（7field×5level×10seed）が完了。[集計](selector-b-evaluation/summary.txt)／[35条件表](selector-b-evaluation/field-level.csv)／[350走行表](selector-b-evaluation/per-run.csv)／[代表trace](selector-b-evaluation/representative-traces/)。Aは未実施なので、傾斜除外による改善効果を比較したとは主張しない。
