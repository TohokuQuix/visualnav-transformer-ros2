# EAT-Nav Phase C remediation report

**判定（2026-09-29）: formal Phase Cは開始しない。** Gate C0は通過したが、自然な4-frame DEM historyで生成した候補のcoverageが不足した。32 train-only state中、29 stateは8候補のPathが1 cluster、31 stateは実`simple_trajectory_follower`の初期commandが1 clusterになった。1 anchorのserial candidate smokeでは効果がduplicate variationを上回ったが、episode/state bootstrap CIを作れるstate数ではない。Phase B Evaluatorの再学習、NoMaD architecture/loss/scheduler/trajectory表現の変更、formal 50–100 state rollout、Phase D/E、full collection、実機試験は実施していない。

## 実行順とprovenance

[固定config](remediation-config.yaml)、[source provenance](source-provenance.json)、[追加診断source SHA](diagnostic-source.json)、[artifact hashes](artifact-hashes.json)を保存した。git基点は`ab2408f`で、追加実行前のuncommitted diff有無・SHA、Docker image ID、Python 3.11.13、PyTorch 2.7.0+cu128、CUDA 12.8、Isaac Sim 5.1.0-rc.19、diffusers 0.35.1を記録した。元epoch 100 latest checkpoint SHAは`42bb7bb2…db794`。dataset manifest SHAは`55d9e2d8…1475`、complete-only IDの**gzip展開内容**SHAは`ff8a9ee8…bbf50`（train 49,598、validation 6,171）。正規化はcomplete-only trainの396,784点だけから計算し、held-out 6,208件は学習・選択・thresholdに使っていない。元のPhase C成果物・checkpointは上書きしていない。

## Gate C0: 同一runの学習継続

元のepoch 100 latestからmodel、EMA、optimizer、Python/NumPy/CPU/CUDA RNG、sampler state、global stepと連続historyを復元した。Quality-disabled、batch 64、LR 1e-4、schedulerなし、seed 123、patience 10、min delta 0、total max epoch 300を固定し、best EMAは**全validationのaggregate diffusion loss**だけで選んだ。[130 epochの連続履歴](reference-resume-history.json)と[結果](reference-resume-result.json)を保存した。

epoch 130でpatience 10によりearly stop。bestはepoch 120、validation loss **0.19695670**。元epoch 100の0.19798414より改善したためGate C0は**pass**。resume追加30 epochの合計epoch時間は313.0秒、PyTorch GPU peak allocated 206,456,832 bytes、CPU process peak RSS 3,571,320 KiB。best EMA SHAは`191d6a03…b4e0`、latest SHAは`0aac7fec…6f9f`。last 10/20/50 epochのloss改善量は順に−0.0000923、+0.0001450、+0.003064。[学習曲線](figures/reference-learning-curve.png)。bestはDocker/hostの`/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt`に保存した。

## First divergenceと対照実験 A–D

同一Pathを8 branchへ与え、Path、実follower command、command manager、pre-noise／post-noise-pre-delay／post-delay Actor input、latent、action、processed action／joint target、root／joint、contact、raycast／DEM、Critic、rewardを40 stepで記録した。[全signalの差分](first-divergence-report.json)には最大絶対差、RMS、最初の超過step、直前stepのRMS、branch IDとの相関を保存した。[step図](figures/first-divergence-by-signal.png)。最初の差はstep 0のpost-noise Actor observationとCritic observationに出た。pre-noise Actor observationとroot stateはstep 1から分岐し、contact sensor/raycast差はstep 4に現れた。contact sensorの更新stepは物理接触がstep 4から始まったことを意味しない。

| 実験 | command RMS | Actor action RMS | XY trajectory RMS |
|---|---:|---:|---:|
| 同一Path baseline | 0.01842 | 0.31203 | 0.05550 m |
| A: recorded follower command | 0 | 0.30100 | 0.05533 m |
| B: Aに加えてrecorded post-delay Actor input | 0 | 0 | 0.03493 m |
| C: recorded 6-D action | 0 | 0 | 0.03493 m |
| D: 同一branch ID・同一command/action履歴を独立processで3回 | 0 | 0 | **0 m**（branch 0のrun間） |

[各対照結果](recorded-command-control.json)、[B](recorded-observation-control.json)、[C](recorded-action-control.json)、[D](fixed-branch-repeat.json)、[比較図](figures/controls-a-to-d-duplicate-rms.png)。Aではcommand差を消してもXY差が残る。BではActor input/latent/actionが決定論的に一致したがXY差が残り、Cも同じ。Cでprocessed actionはstep 11から分岐した一方、root stateはstep 1から分岐した。Dではbranch 0の全traceが3回完全一致し、同一runのparallel branch間XY RMSは0.03493 mだった。**確認済み**なのはparallel env index固有の残差であり、PhysX contact cacheとterrain tile geometryのどちらかまでは特定していない。[原因の分類](cause-classification.json)に候補と確認済みを分けた。

## 同一env serial replayとbridge startup race

同じenv index 0を各runでresetし、15 control stepのcommand/action prefixを再生した。step 0/5/10/15のDEMから同一episode内0.25秒間隔の**実際に異なる4 frame**を作り、anchor後の同一Pathを40 step追従した。anchor root/joint、Actor input/latent、Critic observationとDEM history SHAは5回完全一致。最初の探索5回では生成Pathに最大約6 µmの数値差があったため、[探索結果](serial-replay-validation-path-unfixed.json)を正式duplicate推定に使わず、Path JSONを数値的に固定した別実験を行った。

診断用ROS bridgeではPath subscriber接続前にpublishされるstartup raceを確認した。失敗runのreset直後、branch 7のsubscriber数が0で、その後1になってもbranch 7のcommand sequenceは0、他7 branchは35～36だった。bridgeだけを修正し、8 subscriber接続を待って**同じPathを再送**した。follower本体、RL dynamics、reset event、noise、latency、PhysX設定は変更していない。修正版bridgeで5回（run 3–7）を取得した[正式serial duplicate結果](serial-replay-validation.json)ではPath、anchor state、DEM historyが完全一致し、40-step XY RMSは**0.0035008 m**、command RMSは0.000735、Actor action RMSは0.08228。対照として同じprefix後のrecorded commandを2回再生するとcommand／Actor input／action／processed action／XY／contactのrun間差はすべて0だった。残るlive serial差はROS側のcommand sequence/timing領域に属するが、timer時刻とTFのどちらが微小command差を作るかは未分離である。[parallelとserialの図](figures/parallel-vs-serial.png)。

## 自然DEM historyのcandidate coverage

結果を見る前に[32 state ID](candidate-states.json)を固定した。各stateは異なるepisodeで、train splitの7 terrainから4～5 stateずつ選び、held-outは不使用。実followerのlookahead 0.5 m、goal tolerance 0.2 m、turn-in-place閾値0.6 radと、2秒に識別可能な旋回・横変位およびtrain-only軌跡分布から[閾値](candidate-thresholds.json)を固定した。各stateで同じdiffusion seed 1001～1008のK=8候補を生成し、trajectory RMS、endpoint距離/heading、path length、curvature、lateral、forward/reverse、train軌跡への正規化最近傍距離を[保存](candidate-diversity-offline.json)した。候補Pathと[実follower command](candidate-diversity.json)を比較した。[分布図](figures/natural-history-diversity.png)、[Path対command図](figures/path-vs-follower-command.png)、[terrain別collapse図](figures/terrain-state-collapse.png)。

候補ペアのtrajectory RMS中央値は**0.02203 m**、endpoint距離中央値は**0.00661 m**。実followerのangular command差中央値は**0.00140 rad/s**、linear差中央値は**0.000259 m/s**で、事前閾値0.05 rad/s／0.03 m/sを大きく下回る。Path 1 clusterは**29/32 state**、command 1 clusterは**31/32 state**。train-only P95 OOD閾値を超えた候補は0/256。terrain別supportは各4～5 stateに限られ、terrain一般化の強い主張はできない。これはcandidate coverage不足である。

同一simulator anchorの**1 stateだけ**で8候補をserial replayした探索比較では、候補Path pairwise RMS中央値0.02256 m、候補による観測XY RMS 0.02435 m、5回serial duplicate RMS 0.00350 m、記述的なρ=**0.144**。[比較値](duplicate-vs-candidate.json)と[図](figures/duplicate-vs-candidate.png)。候補観測差にはROS timing/noiseも含まれ、1 stateではepisode/state bootstrap CIの下限を算出できない。したがってρが0.25未満という単独結果をformal gate通過とは扱わない。

## 総合判定と再開条件

Gate C0はpass。first-divergenceの初期noise寄与とparallel env固有残差は切り分けた。同一env serial replayはanchorとPathを一致させられ、duplicate RMSを小さくできた。ただし**自然履歴で31/32 stateのfollower commandが同一cluster**になり、candidateが行動選択として十分に分かれていない。episode/state bootstrap CIも承認された少数state範囲では得られない。分類は**candidate coverage不足**、加えてparallel branch/physicsの具体的機序とlive ROS timingの微小残差は未解決。formal Phase Cの開始条件を満たさない。

実測の40-step serial runは概ね14～16秒/候補（scene/model初期化とbridge resetを含む）。50～100 state×8候補だけで約1.6～3.6時間、圧縮traceは1候補約1.5 MBで約0.6～1.2 GB、実際にはstate選択・prefix・retries等の追加時間と容量が要る。これらは実測単位時間からの見積もりであり、formal rolloutは開始しない。関連unit testはDockerで**15 passed**。全repo suiteのgreenを今回新たに主張しない。

SSH切断中もtmux `eatnav-phase-c-remediation`で実行し、stageごとにatomic marker・status、reference checkpointはtempからatomic renameした。長時間計算終了後はtmux sessionが自然に終了する。

```bash
tmux ls
tmux capture-pane -pt eatnav-phase-c-remediation
tmux attach -t eatnav-phase-c-remediation
cat /tmp/eatnav-quality-identifiability-gate/phase-c-remediation/remediation-status.json
tail -n 100 /tmp/eatnav-quality-identifiability-gate/phase-c-remediation/runner.log
```
