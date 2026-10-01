# Goal-conditioned local subgoal simulator pilot v2: anchor gate stop

## 判定

`v2_anchor_gate_fail`。V0–V6は通過したが、V7の最初のanchorで実装済みの4-frame DEM history生SHA一致条件が不成立となった。4 anchorのうち1件だけを実行し、残り3件、nominal duplicate、candidate live outcome、GS1–GS4は開始していない。元pilotのstate、goal、Path、SHA台帳、停止結果は変更していない。

## 固定・実行済み範囲

- 専用worktree `/tmp/eatnav-vnt-goal-ood-simulator-v2`、branch `exp/eatnav-goal-ood-simulator-v2`、出力root `/tmp/eatnav-candidate-coverage/goal-ood-simulator-v2/`。V0でmanifest/split、complete-only train identity、NoMaD/RL/Evaluator checkpoint、follower source/binary/config、bridge/runner、最新parity、resource timing、goal supportを照合した。Evaluator GT-11/29/47のhost側旧pathは欠けていたが、Docker内の同じpathに3件とも残り、過去の固定SHAに一致した。旧pilotのcandidate outcomeは0件、開始前の残留processは0件。
- V1/V2のoutcome-blind SHA順位で、元pilotと同じterrain順に4 stateを選んだ。episodeは全件0、global/local step一致、prefixは61/61/36/36 control step。条件を満たす候補数は順に82/112/123/31。complete-only train 49,598 windowsの固定identityと照合し、episode非重複、goal距離0.30 m以上、報酬/Critic/outcome非依存のテストを通過した。選択後のstate差し替えはしていない。
- V3で同じgoal slot 4種×4 state＝16 goalをtrain-only support規則で生成し、geometry screenは16/16 valid。同じNoMaD checkpointとseed 1001/1002で32 Pathを1回だけ推論した。保存時に`model.sample()`の累積waypointを増分としてもう一度累積する後処理ミスを検出した。元の32 model sampleとそのSHAをアーカイブし、**再サンプリングせず**増分、累積waypoint、endpoint誤差の解釈だけを修正した。修正後のendpoint誤差は0.00468–0.15230 m。元sample 32件のfloat32配列と修正後waypointは全件完全一致。
- V4で4 anchor＋12 nominal duplicate＋32 candidateの48-process順序、100-env記録action prefix/suffix、goal metadataを固定した。V5の予測はpoint 2.229 h、P95 step 2.250 h、P95＋20% 2.700 h、worst observed 2.381 hで4 h以内。保存量予測239,907,072 bytes、GPU推定5550 MiB、anon＋shmem推定3,916,652,544 bytesも固定上限内。これらは`train-003`のstep時間を新しい4 terrainへ外挿した値であり、44 runの実測ではない。
- v2 state IDを診断assertionで認める1行だけsimulator sourceを変更した。V6では最新sourceで既知の`readiness-002`を120 step再実行し、command、exact-stamp、Actor/Critic、root/joint、contact、DEM、raw effort、実現軌道を含む30 fieldで旧traceと差0。raw effortとActor pre-noiseの最大差は`3.33786e-6`。この後はsimulator/bridge/follower/NoMaD/RL制御経路を変更していない。

## V7で最初に外れたsignal

`v2-train-000`（`k_rail_step_field`、episode key `[0,60,0]`、global step 60）のfresh-process anchorはsimulator exit 0、wall 73.64秒、OOMなし、GPU peak5550 MiB、残留process 0で完了した。root位置差0 m、姿勢差0.000269 rad、速度差0、current DEM RMS 0 m、current mask IoU 1.0、prefix command差0。logged actionと`action_manager_action`は6成分完全一致した。旧logged anchorと今回の別時点の`actor_action`の最大差は0.774、Critic value差は0.243で、同じ観測量としての完全一致を主張しない。joint/Actor observation・latent/Critic observation/previous action/raw effortの旧logged参照値はなく、過去値との比較はNA。

4-frame DEM historyでは、global step 45由来の最初のframeのみ高さfloat32値が188セルで異なった。高さ最大絶対差`2.682209e-7 m`、既知セル上のRMS `6.721484e-8 m`、mask IoU 1.0。global step 50/55/60由来の残り3 frameは完全一致した。したがって4-frame配列の生SHAは異なる。差の数値規模は固定のDEM RMS許容差0.03 mより十分小さいが、現在のV7 runnerは生SHA一致も必須にしており、`anchor pose/DEM/mask/command gate failed`で停止した。差の根本原因は未確定で、早いframeのraycast/浮動小数点再現差が候補である。anchorの記録値を使って判定を緩和していない。

許容差ベースのhistory比較と既存anchorの再利用へコードを変更する試みは、自動承認レビューから「失敗後のgate緩和に当たり、停止して原因報告すべき」として拒否された。拒否された変更は適用していない。再試行0回、候補outcome 0件。専用routerとtmux sessionを終了し、simulator/bridge/follower/monitorの残留は0。

## 成果物と次の判断

`v2-source-provenance.json`、`v2-selection-candidates.jsonl.gz`、`selected-states-v2.json`、`goals-v2.json`、`paths-v2.jsonl.gz`、`run-schedule-v2.json`、`runtime-forecast-v2.json`、`v6-parity/parity/parity-result.json`、`anchor-results-v2.json`、`final-decision-v2.json`、`sha-ledger-v2.json`を新しい出力rootへ保存した。raw traceとlogも同rootに保持した。V8以降の結果はNAであり、local subgoal方式の走行性能について結論は出せない。

継続には、4-frame DEM historyの**生SHA一致を必須とするか**、設計書にあるDEM RMS・mask IoU許容差を4-frameすべてに適用して生SHAは診断値とするか、gate定義の明示的な決定が必要。後者を承認する場合も、今回完了したanchorの入力・出力SHAを再照合したうえで、そのanchorを再実行せずに残り3 anchorへ進める案である。state、goal、Path、seed、順序、モデル、follower制御則、dynamics、評価閾値の変更は提案しない。
