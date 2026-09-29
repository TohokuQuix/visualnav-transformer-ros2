# EAT-Nav feasible-set duplicate recovery

## 判定

- Fresh-process duplicate: **0.000000 m XY RMS**, gate 0.0105 m: **pass**。
- 旧snapshot restore失敗: 報告値0.023661はx/y/yaw混合。訂正XY RMSは**0.016937 m**で、gate不合格は変わらない。
- 3 runでanchorのcritical fieldと固定Path、40-step command・観測・Actor・物理・接触・DEM・Critic・rewardが一致。first-divergenceは検出されなかった。
- recorded-command controlは条件未成立のため実施していない。
- 旧snapshot失敗traceではpre-noise Actor observationがstep 0、commandがstep 1で初めて分岐。fresh-processでは全signalで分岐なし。差異の領域は同一processの不完全なsnapshot restoreに絞られるが、個別の隠れ状態は未特定。
- 旧runのfull anchor snapshotとmap-frame Path JSONは元runで保存されておらず復元不可。元trace・base Path・source/logは凍結済み。
- 収集器のcommand rowは再サンプル時にpost-step値となる。state 13の照合をpost-stepへ修正し、記録command/DEM/root一致を確認した。
- state 42のnominal 3回目は初回follower command取得でfresh=false。8 followerは生存・全Path subscriber=1だが2秒間command sequenceは全0。候補は未実行で測定無効。
- 56 train + 20 validationのformal rolloutは開始していない。

## 固定3-state maneuver smoke

| train state | terrain | duplicate XY RMS m | max candidate XY effect RMS m | first command divergence | max rho |
|---:|---|---:|---:|---:|---:|
| 19 | k_rail_field | 0.000000 | 0.024624 | 0 | 0.0 |
| 13 | hf_pyramid_slope | 0.000000 | 0.020906 | 0 | 0.0 |
| 42 | pyramid_stairs_inv | NA (2成功+1失敗) | NA | NA | NA |

state 19/13はnominal 3回と固定5 Pathをfresh processで測定。state 42はnominal 2回成功（相互差0）、3回目が初回command timeoutとなり候補を開始していない。ΔV、A_H、D、R_recover、termination、progress、body tilt、contact、proposal OOD、利用可能な40-step commandとXY軌跡は[JSON](three-state-maneuver-smoke.json)に保存。
候補間の統計的優位性や他terrainへの一般化はこの少数stateから主張しない。

## 成果物

- [凍結失敗case](failure-baseline.json)、[既知良好との差分](known-good-comparison.json)、[RMS定義監査](baseline-metric-audit.json)
- [anchor比較](prefix-replay-anchor-comparison.json)、[duplicate](prefix-replay-duplicates.json)、[first-divergence](first-divergence-report.json)
- [gate](duplicate-gate.json)、[maneuver smoke](three-state-maneuver-smoke.json)、[state 42 interruption](state42-interruption.json)、[hashes](artifact-hashes.json)
- [図](plots/)：prior/current RMS、first-divergence、snapshot/fresh、live/recorded、duplicate/candidate、3-state commands

## 次の判断

state 42のfollower first-command timeoutを未解決として停止。formal 76 state rolloutも開始していない。再診断・続行方法は結果レビュー後に判断する。

## Stop conditions

- state 42: follower first-command timeout; third nominal repeat and candidates unavailable
