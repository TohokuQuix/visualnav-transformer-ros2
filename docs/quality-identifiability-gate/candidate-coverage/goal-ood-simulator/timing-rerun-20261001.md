# Goal-conditioned local subgoal timing rerun: resource stop

## Decision

`runtime_gate_inconclusive`。最新sourceの`readiness-002` parityは通過したが、1回だけ許可された`train-003` full-prefix timing probeはglobal step 9で事前固定のcgroup memory上限に達し、anchorまで完走しなかった。anchor確認0件、nominal duplicate 0件、candidate 0件。したがって4時間予測、GS1–GS4、local subgoal方式の性能はNA。固定した4 state、16 goal、32 Path、seed、run順序を維持し、候補runは開始していない。

## T0 provenance

新規出力rootは`/tmp/eatnav-candidate-coverage/goal-ood-timing-20261001/`。`timing-source-provenance.json`にbranch/HEAD/worktree、manifest/split、NoMaD/RL checkpoint、reference config、4 state、44 ordered run、Path/metadata/input/source SHA、旧報告と失敗log SHA、開始前process状態を保存した。T0はpass。provenance SHA256は`6c01fa67083c2510e0ff4157a919b474307571a7bce16a5278c212630d958632`。

## T1 latest-source parity

assertion修正後source SHA256 `593bb0cc83a89e54cbd0220c17aaf75ffd3e379779cc68c99e782bd51dc2f20f`で、`readiness-002`の120 control stepをdetached tmuxから1回実行した。simulatorは正常終了し、実測185.41秒、GPU peak 5550 MiB。host側runnerの後処理はNumPyのないPython環境で失敗したが、120-step trace、clock sidecar、simulator markerは保存された。保存済みtraceを既存の正常trace（SHA256 `4647b2ff0b723f491cf2a0d0392c6a87bb1961ff7f50dd5b018bda4518e4b6bf`）と比較し、30 fieldすべてでfirst differenceなし、same-step TF/command/clock/DEM stamp一致、非有限値なし。raw `robot.data.applied_torque`は52 joint、4 flipper＋左右track sumは6-D、clip `[-100,100]`後0.01倍とActor pre-noise effortの最大差は`3.337860107421875e-06`。production exact-stamp followerとRL checkpoint SHAはT0から不変。parity結果SHA256は`6883710a4ff625309915b4325d291a39eb5df867e48f216f7bbe638297d15d3a`。

## T2/T3 heartbeat timing probe

計測のみを加えたsourceを新しいcontainer pathへコピーし、元のfresh-process full-prefix replayを使用した。外部monitorはrunnerとは別processで10秒周期のatomic heartbeatを出し、Docker・simulator・router生存、step、episode、logical time、CPU/RSS/cgroup/GPU、出力量を記録した。事前configはstep停止120秒、heartbeat停止90秒、GPU 6656 MiB、cgroup memory 8 GiB、run timeout 3600秒。SSH/VS Code接続への依存はない。

`train-003`のtiming probeは15:30:25 JST頃に開始。Isaac開始からscene構築完了まで12.78秒、initial reset完了までさらに0.26秒。global step 9まで進み、20.31秒時点のheartbeatはcgroup memory `10,098,774,016` bytes、simulator RSS `5,694,177,280` bytes、GPU `5550` MiBを示した。外部monitorが凍結済みcgroup上限`8,589,934,592` bytes超過を検出し、failure marker、process tree、GPU process、FD/thread/socket、log末尾を保存した。runnerはsimulatorを停止し、final statusの残留processは0。timing heartbeat SHA256 `317ce22d64965d038be536626c55d8cb727121aa8cae6921fd99d4bf56cbe5a5`、failure marker SHA256 `642ed0a2caa15c50ebe926b6bdec4b2bb226a0d44ed4937dc82c90d2a36a6104`。

この上限は今回のmonitor側の事前設定値であり、Dockerの実際の`memory.max`は`max`。停止後のcgroup memoryは約6.17 GB、そのうちfile cache約5.88 GB、anonymous約0.005 GBだった。running時のhost runner記録ではanonymous peak約2.49 GB。総cgroup memoryには大きいfile cacheが含まれ、上限超過だけで実際のOOMやGPU異常とは結論できない。観測されたstepは継続しており、120秒stallも検出されていない。monitor上限の選び方がT3を早期停止させた可能性が高いが、結果確認後に上限を変更して再試行していない。

## Runtime and scope

1621-step anchorが未完了のため、step durationの全分布、4 stateのanchor時間、44 live runのpoint/conservative/margin付き予測は計算不能。前回の6.3時間外挿がSSH切断や停滞を含んだかも判定不能。T4は`inconclusive`、T5–T7のcandidate実行とGS解析は未実施。元の出力root、失敗log、NA判定JSONは上書きしていない。専用routerは終了し、tmux server、simulator、bridge、follower、monitorの残留は0。retryは0回。次の計測には、起動前のcgroup baselineとfile cacheを考慮した資源上限を**新たに固定**する必要がある。今回の許可された1回のprobeは消費済みのため、自動再実行はしない。
