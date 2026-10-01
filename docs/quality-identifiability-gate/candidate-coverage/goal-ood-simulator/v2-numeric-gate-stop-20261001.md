# Cost-aware v2 pilot: DEM数値同等性gate適用と停止記録

## 判定

`v2_pilot_incomplete`。4 stateのanchor gateはすべて通過した。最初のnominal live runはcontrol step 39の途中で外部resource monitorにより停止した。nominal duplicate、candidate outcome、GS1–GS4は未評価。停止後のretryと次runは行っていない。

## DEM history gate

`config-120.yaml`で既に固定されていた各frameのDEM RMS ≤ 0.03 m、mask IoU ≥ 0.95を適用した。生float配列のSHA一致は判定に使用せず、診断値として保存した。各frameのepisode identity、global step、20 Hz post-step論理timestamp、shape `[2,20,20]`、座標軸、resolution 0.1 m、finite性を照合した。source frame `map`と変換後frame `chassis/body`は固定reference configとSHA一致したreplay変換規則から検証した。シミュレータの`SourceGrid.stamp_ns`は0のplaceholderであり、ここでいうtimestampは独立したセンサstampではなく、記録と再生の共通制御時計による値である。

| state | terrain | 各frameのRMS (m) | 各frameのmask IoU | history raw SHA |
| --- | --- | --- | --- | --- |
| v2-train-000 | k_rail_step_field | 6.721484e-8, 0, 0, 0 | 1, 1, 1, 1 | 不一致（診断のみ） |
| v2-train-001 | pyramid_stairs | 0, 0, 0, 0 | 1, 1, 1, 1 | 一致 |
| v2-train-002 | pyramid_stairs_inv | 0, 0, 0, 0 | 1, 1, 1, 1 | 一致 |
| v2-train-003 | boxes | 0, 0, 0, 0 | 1, 1, 1, 1 | 一致 |

`v2-train-000`の保存済みanchorについては、`started.json`の全入力SHA、simulator markerの出力SHA、`run-complete.json`の出力SHA、simulator source/reference config/manifest/selected inputのSHAを再照合してからオフラインgateを適用した。他3 anchorにも同じgateとSHA照合を適用した。旧failure成果物は保持した。

## Live runの停止

`v2-train-000-nominal-00`の外部monitorが約94.7秒時点で `probe output reached or is projected above storage cap` を発報した。monitor観測時の実出力は21,272,558 bytesで、800,000,000 bytes上限より小さい。一方、60秒以降の実出力率をtiming probe用の3600秒へ外挿した予測は808,506,380 bytesとなり、上限を超えた。120-step live runの予測時間は約数分のため、この停止は実測容量超過ではなくmonitor予測horizonの不一致による。GPU 5550 MiB、anon＋shmem 3,972,685,824 bytes、進捗停止0秒、OOM eventなし。simulator log上でcontrol step 38までのreturnがあり、step 39のreturnはない。

失敗はstep 0以降なので、凍結済みinfrastructure retry条件に該当しない。部分runを再利用・retryせず、残りのrunを開始していない。follower、bridge、モデル、Path、seed、state、制御則、reset/dynamics、閾値には変更を加えていない。

## 証拠

- [停止判定JSON](/tmp/eatnav-candidate-coverage/goal-ood-simulator-v2/final-decision-v2-numeric-gate.json)
- [最初のanchorオフライン照合](/tmp/eatnav-candidate-coverage/goal-ood-simulator-v2/v7-first-anchor-offline-numeric-gate-v2.json)
- [4 anchor結果](/tmp/eatnav-candidate-coverage/goal-ood-simulator-v2/anchor-results-v2-numeric-gate.json)
- [resource stop snapshot](/tmp/eatnav-candidate-coverage/goal-ood-simulator-v2/runs/v2-train-000-nominal-00/resource-stop.failure.json)
- [runner log](/tmp/eatnav-candidate-coverage/goal-ood-simulator-v2/v8-numeric-gate-runner.log)
- 固定schedule SHA: `4ebc78fd13cf1e69af754368d8a47d9327e54b188244199e9de7f10833695b86`
- 停止判定JSON SHA: `9f6b199b2fc739f4e56b8ba2c87e6e2e51305e0bebdda0febc27cc8fcbf69024`
- Tests: `PYTHONPATH=src:scripts /usr/bin/python3 -m pytest -q tests/eat_nav/test_goal_subgoal_dem_gate.py tests/eat_nav/test_goal_subgoal_v2_select.py tests/eat_nav/test_goal_subgoal_v2_schedule.py` → 15 passed。

次に進むには、monitorの予測horizonを120-step live runの時間に合わせる変更と、step開始後に停止したnominal runの扱いを先に決める必要がある。現時点ではその変更を行っていない。
