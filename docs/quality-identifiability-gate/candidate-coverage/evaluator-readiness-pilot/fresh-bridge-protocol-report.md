# Fresh-bridge readiness pilot: 最終検証

## 判定

| 項目 | 結果 | 根拠 |
|---|---|---|
| Primary failure mechanism | `inconclusive` | 旧bridge内部の詳細機序を復元できるログがない。根因修正とは判断しない。 |
| Failure domain | follower publish return後、bridge adoption前 | 失敗stampで8 followerが有限commandをpublishしてreturnし、旧bridgeは採用していない。 |
| Measurement protocol | 変更して継続 | 各live runでfresh bridgeと8 fresh followerを起動・終了。新19 process中の1件はcommandを使わないanchor確認。 |
| Protocol equivalence | **pass** | outcomeを見ずSHA順で選んだnominal `readiness-002`、NoMaD `readiness-026`、bank `readiness-084`を各120 step再生。入力SHA一致、比較対象の観測・command・stamp・接触・軌跡差はすべて0。replayはoutcomeから除外。 |
| Cleanup robustness | **operational pass** | 1回の故障注入でfresh-command timeout 3.011秒、atomic failure markerとpartial trace保存、complete markerなし。8 followerとbridgeがgraceful shutdownし、残留processなし。bridge shutdownはsimulator close entryから0.841秒。 |
| Pilot completeness | **119/119** | 旧100件とfresh19件のidentity、marker、出力SHAを再照合。112 live rollout、7 anchor。state 6のnominal 3 repeatはDEM SHA一致、Actor・command・Critic・rewardのRMS差0。 |
| Candidate coverage | **insufficient** | NoMaD K=8のfeasible-union recallは7 state中6、NoMaD valid precisionは各state 1.0。ただしrealized trajectoryが複数modeとなったのは1/7 stateのみ。 |
| Relative-outcome dataset | **ready_for_scaled_collection** | ΔV range、advantage/drawdown、NoMaD順位差の支持条件が7/7 train stateで成立。 |
| Safety-head readiness | **insufficient_failure_coverage** | stack、excessive tilt、terminationのpositive stateが0。infrastructure failureをsafety labelに変換していない。 |
| Evaluator input schema | **incomplete** | finite・同stepのDEM/goal/Actor latent/previous actionは確認。joint effortが別fieldとして取得されていない。frozen Phase B GT(P)の6秒all-candidate top1は1/7、macro pairwise accuracyは0.414で、副次診断に限る。 |
| Next action | `measurement_remediation` | joint effort取得、failure-positive stateの支持、候補の実現軌跡多様性を設計面で検討する。追加収集・再学習・formal validation・実機試験は開始していない。 |

## Protocol gate と資源

R3～R7は順に通過した。R5の連続3 replayでは残留ROS node・process・zombieは0、FD最大値64、socket最大値14で単調増加はなく、reset countは各run 1、SIGKILLは0。新しいlive runでもlifecycle gateは全件通過し、GPU peakは5551 MiB（約5.42 GiB）だった。R6の保存量予測は約1.225 GB、最終実測は約1.088 GBで上限1.5 GB以内。

fresh19件の測定時間は約55.8分。R0監査開始からR9完了までの追加作業は約100分で、4.5時間の上限以内。自動生成された`runtime-storage.json`の5.81時間は、以前に完了した旧100件の4.88時間と今回のfresh19件を合算したprocess時間であり、今回の追加wall-clockではない。

故障注入では`simulation_app.close()`のentryとnative process exit、Docker exec returnを確認した。Isaacのnative shutdownがPython interpreterを終了するため、close直後のPython行からreturn eventは出ない。このAPI上の観測限界を残したまま、実運用上のcleanup完了をpassとした。

## 統計上の範囲

7 terrainの各1 train stateを記述した結果であり、candidate pairを独立標本と数えない。validation／held-outや実機への一般化は主張しない。診断、fault injection、旧failed attempt-0、3件の等価性replayは112件のlive outcomeへ含めていない。Phase B Evaluator、NoMaD、RL、production follower制御則、loss、scheduler、候補・Path・seed・順序は変更していない。

raw trace、圧縮callback log、atomic marker、解析結果は `/tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol/` に保存。全119件のSHA台帳は `artifact-sha-ledger.json`、gateと出所のSHAは `fresh-bridge-gate-ledger.json`、詳細指標は `final-decision.json` と `final-report.md` を参照。
