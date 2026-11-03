# 承認後の明示的な変更・実装判断

## 地形の分割：ユーザー承認済み

2026-11-03、K-Rail系はdifficultyを10個の離散indexへ丸め、level2のdifficulty .4〜.6ではindex4/5の2種類しかないことを実コードで確認した。noise=0、位相固定の既存設定を維持すると、3roleのmesh完全分離は不可能。

ユーザーは「既知地形の独立走行評価として進める」を明示承認した。よって本仕様第9節のmesh完全分離条件を変更し、同一meshのrole間重複を許容・全件集計する。親試行、走行seed、初期状態、生記録、同じ10秒内の5windowはroleを分離する。未知地形への性能とは主張しない。7field/level/候補/精度閾値は変えない。

## 初期配置の地形貫通修正

診断benchmarkで、旧resetのbody z=.35mに対しK-Railの局所上面が約.37mとなり、初期接触で打ち上がる事例を確認した。試行開始時のみ、生成済みmeshの局所支持面最大高さ＋旧clearance.35mへ配置する。XY/yaw/初速/Actor/foldは承認仕様を維持。10秒内のreset・停止は追加しない。修正前診断データは学習に使わない。修正後profile名はterrain_support_clearance_v1。

## R6資源の一時停止

ユーザー指示でR6 coordinator/simulator/watchdogを一時停止。物理実行状態と既存成果物を保持し、GPU計算を今回へ割り当てる。R6のVRAMは保持される。wallclock timeoutがあるため長時間後の再開は単純CONTではなくwatchdog/queue保全処理が必要。R6を完了したとは扱わない。
