# 独立コードレビューと修正

レビュー対象commit dea4eb8。reviewerはread-onlyで確認しCritical0、Important2を報告。

1. LiDARがenv origin高さをmesh配置と混同：local originを別保存してmesh translationを修正。raised1.5mのCPU回帰と、origin3.411mの実階段64rayの幾何一致(最大差5.73e-7m)、5区間の実FAST-LIO+教師生成を確認。
2. 再開でruntime-stage索引を上書きし版混在が検出できない：不変stage記録、実import/config/Actor/late sensor fileのhash、canonical source identity、parent/batchリンクを保存。次batch・datasetで未承認混在を拒否。

追加の検査としてcheckpointとdataset manifest/auditのhash一致を評価開始前に確認。重複workerによるOOM後、GPU collectorのglobal lockと既存worker検出を追加。IMU生成は同一基準点のroot_link_state_wへ統一。

reviewerが独立認証していない範囲：全GPU物理/センサのparity、外部Actor全変換の完全同一性、本収集完了、学習モデル品質・精度達成。これらをレビュー済みと読み替えない。修正後は29 CPU tests pass、raised-originの実入力変換5/5区間を確認。全datasetと最終モデルは未完了。
