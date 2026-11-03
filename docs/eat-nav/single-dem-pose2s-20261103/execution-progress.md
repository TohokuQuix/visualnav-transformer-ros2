# 実行状況：実FAST-LIO・最大30台へ移行

## 最新状態

ユーザーが実機互換性未確認でのシミュレーション先行を承認し、本格収集gateの旧拒否理由は解決済み。その後、ユーザーが実FAST-LIOの使用と最大30台収集を明示選択した。旧wheel-odom収集を区切りで保存・停止し、新profileの本収集を開始した。旧R6は中断状態で保全している。

- 新profile SHA256: `e0f7f59133a0669f390bda2c5df164dab1bdeebb74f83ade8314d445de170e20`
- 本収集root: `/home/ubuntu/eatnav-single-dem-pose2s-data-20261103/actual-fastlio-v2/production/`
- train350、validation490、final490親試行、予定6650区間。各cellの38親を30+8台へ編成する。role・seed・候補列・分割は変更しない。
- 物理dt0.005s、制御20Hz、DEM4Hzで1frame。NNへ点群を入力しない。実FAST-LIO内部処理のための模擬LiDAR10Hz、模擬IMU20Hzを保存する。
- 収集後、実FAST-LIOを親ごとに独立実行し `/Odometry` を固有topicへremapして記録する。twist未代入を実ソースで確認済みのため、過去pose差分から車体速度・角速度を作る。未来poseや車輪速度への代用はしない。
- 本datasetは未完成、学習は未開始、最終精度評価・経路選択・新成果物pushも未実施。

## 試作検証

- 実FAST-LIOのROS通信検証で初期のZenohルーター未起動によるtimeoutを確認し、作業用domain77だけCycloneDDSへ変更して解決した。PointCloud2のpoint_step24byte整合も修正し、実センサreplayが完了した。
- 30台LiDAR/IMU試作は約180.9秒、145/150有効区間。実FAST-LIOは30/30親で成功、4並列CPU処理約83.3秒。因果入力と教師を監査して145区間を確認、5区間は終端欠測。試作はbenchmark roleで本学習には入れない。
- 現在の回帰テスト29件pass。旧followerとのC++parity検証、因果時刻、2秒境界、raw/初期状態/seed漏洩、最大値合否、FAST-LIO入力上書き、旧入力schema拒否を含む。
- `prototype/input-fourteen.npz` は実FAST-LIO出力から作った新schemaの入力例へ更新した。14候補の入力を持つが、教師は実行1候補のみ。

## 残る工程

本収集→実FAST-LIO処理→dataset化・分割/欠測/条件別監査→dataset完成時の通知→train/validationで学習→独立最終評価。精度未達なら経路選択へ進まない。dataset本体と大量センサ記録はGitHubへ入れず、manifest・保存場所・hash・再現方法を共有する。

詳細: `actual-fastlio-thirty-amendment.md`、`actual-fastlio-prototype-audit.json`。実機LiDAR形状・ノイズ・遅延・extrinsicの一致は未確認であり、今回の理想模擬センサを実機一致と主張しない。

レビューの2 Important（地形高さ原点・runtime provenance）を修正し、raised-originの実シミュレーションと実FAST-LIOで5/5区間を再検証。本収集を新rootで再開。1度の重複GPU起動OOMは失敗出力を保存・隔離、排他制御で対策済み。詳細はactual-fastlio-thirty-amendment.md。
