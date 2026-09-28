# Phase C 第2 preflight report — 第1段階

**判定（2026-09-29）:** 第1段階の実装・診断を完了した。**formal 50–100 state best-of-K rolloutは開始しない。** Gate C0は`inconclusive`（100 epoch時点のbest更新）、同一Path duplicateのbranch差はcandidate差と同程度である。固定設定を変更した再学習、Phase B Evaluator再学習、Phase D/E、full collection、実機試験は実施していない。

## Complete-only監査とPhase C-0

[固定config](phase-c-config.yaml)、[学習config](reference-eat.yaml)、[全train/validation window ID](complete-window-ids.jsonl.gz)、[data audit](reference-data-audit.json)を保存した。既存の同一manifest・split seed 123によるcomplete監査61,977件（train 49,598、validation 6,171、held-out 6,208）とID集合を完全照合した。既存監査では全context・goal・教師8点についてepisode跨ぎ0件、terminal/reset跨ぎ0件である。今回のtrain/validation viewは全8点mask=true、finiteなDEM・goal・教師を要求し、partial/zero-valid混入は0。held-outは学習・checkpoint選択に使用していない。action normalizationはcomplete-only trainの396,784点からのみ計算した（mean [0.02811686, −0.00004291] m、std [0.08537876, 0.01502491] m）。

1 epochの[短時間timing probe](timing-probe-result.json)を**学習前**に実施した。epoch 10.42秒、PyTorch GPU peak 206 MB（allocated）、CPU process peak RSS約3.31 GiB、latest checkpoint 114.15 MB。lossによる学習条件変更は行っていない。固定seed 123、batch 64、LR 1e-4、max epoch 100、patience 10、min delta 0、LR schedulerなし、scratch・Quality-disabledでreference学習を開始した。

[100 epoch履歴](reference-history.json)の合計epoch時間は1,039.6秒（17.33分）。validation lossはepoch 1の0.26983からepoch 100の0.197984へ改善し、**best epochは100**。明確なplateauがないため、[Gate C0結果](reference-result.json)は`inconclusive`。best EMA weightsのSHA-256は`9201d912532e84c9e34721c3b48736fed1604c8c9e7865a80b9b7e7d85fe7688`で、既存`weights_only=True` loaderで読み込み確認した。このcheckpointは**smoke専用**として扱い、正式なcandidate coverage判定には採用しない。latestにはmodel、EMA、optimizer、epoch 100、global step 77,500、train-only normalization、Python/NumPy/CPU/CUDA RNG、sampler state、software version、split/source provenanceが入る。総学習中GPU peak allocated 206 MB、CPU process peak RSS約3.32 GiB。

window IDのgzipヘッダ時刻が再実行時のファイルSHAを変える不具合を発見し、mtime=0・空filenameへ修正した。[同一展開内容の照合](identity-equivalence.json)後にlatestのresume SHAと監査ファイルのみ更新した。smokeで使用済みbest weights/checkpoint SHAは保持した。隔離コピーのlatestからのresumeはexit 0、同一epoch 100・global step 77,500・best lossで終了し、追加学習は0 epoch。sampler順序のRNG restoreも単体テストで一致した。best側の旧gzip SHAとlatest側のcanonical gzip SHAは異なるが、展開したwindow ID内容のSHAは同一である。 学習時にrunnerが未コミットだったため、latestのbase git commitは実行スクリプトそのもののSHAを証明しない。Python/PyTorch/CUDA等のversionと入力SHAは保存されているが、厳密な実行コードprovenanceにはこの制約がある。正式referenceを再学習する場合はrunner sourceを実行前にcommit/SHA固定する。

## Goal–endpoint shortcut診断

[全結果](shortcut-result.json)と[追加GTE/GTS学習履歴](shortcut-training-summary.json)を保存した。Phase BのG/GTチェックポイントとsplit・標準化統計は変更せず、GTE/GTSだけを同じarchitecture容量、seed 11/29/47、optimizer、validation選択条件で学習した。GTSは各時点から終点への線形補間分を差し引き、残差increment和の最大絶対値は1.4e-7 mで、endpointを直接復元できない。

| Seed | G MSE | GTE MSE | GTS MSE | GT MSE | GT−GTE gain [episode 95% CI] | GTS−G gain [episode 95% CI] |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.608 | 0.606 | 0.431 | 0.440 | 0.166 [0.142, 0.192] | 0.177 [0.151, 0.202] |
| 29 | 0.616 | 0.621 | 0.443 | 0.447 | 0.174 [0.149, 0.202] | 0.173 [0.150, 0.199] |
| 47 | 0.612 | 0.609 | 0.463 | 0.469 | 0.140 [0.119, 0.165] | 0.149 [0.126, 0.172] |

同terrain内の中間点を入れ替えてendpointを固定すると、GT aggregate MSEはseed順に0.313、0.344、0.242悪化した（すべてepisode bootstrap CI下限>0）。各primary headのR²、Spearman、MSEとpaired CIは全結果に保存した。このlogged trajectory診断ではGTの利益をendpointだけで説明できない。ただしgoalがlogged endpointから構成された既存データ上の結果であり、新しいcounterfactual候補への一般化を証明しない。held-out supportは実在する5 terrainまでで、全7 terrainへの主張はしない。

## 実follower bridge、state、observation

[ROS2 bridge](../../../scripts/eat_nav_phase_c_ros_bridge.py)は実ビルド済み`simple_trajectory_follower`をbranchごとに8プロセス起動し、`nav_msgs/Path`を送り、`map → branch_i/chassis/body` TFを更新し、各`TwistStamped`の新しいsequenceを待つ。旧transient-local Path混入を防ぐためPathを新controller起動前に保持した。実follower単体smokeでは同一Pathに対して8 branch全てからfreshな0.3 m/s commandを受けた。Isaac Python 3.11とROS Jazzy Python 3.12のABI差があるため、ROSはhost process、Isaacは同一host network上のTCP clientで接続した。

simulator command managerの`base_velocity`へfollower commandを入れ、physics後・次のobservation計算前に更新した。Criticのcommand sliceと実commandは[40-step trace](smoke/duplicate-third.npz)で最大誤差0。Actorのpre-noise command sliceも最大誤差0。Actorに実際に渡したpost-delay commandは初期に以前のcommandを含むが、5 step以降の平均絶対誤差は約0.0007。これはcheckpoint設定のpolicy observation latencyを保持した結果である。`ObservationManager`を**1 control stepにつき1回だけ**通し、pre-noise、post-noise/pre-delay、post-delay Actor入力を別々に保存した。[live term配置](smoke/observation-layout.json)と全traceにはActor 22-D、Actor head直前latent 256-D、Critic 267-D、action 6-D、Critic値、command、報酬、contact、DEMが入る。Critic値はduplicate中に37.49–38.83で変動した。

[clone実装](../../../src/visualnav_transformer/eat_nav/phase_c_sim.py)は同じ`random_rough` mesh/difficultyを8タイルへ配置し、root（env origin相対）、joint、latency値、policy delay前回値、ActionManager last action、flipper previous target、commandとsampler stateをコピーした。複製時、root local state、joint position/velocity、commandの最大差は**全て0**、初期pre-noise Actor観測差も0。DelayBuffer ringは最初のaction前には未生成で、生成済みの場合はring、push数、lagをコピーする。PhysX contact warm-start cache、contact sensor/raycaster履歴、per-env化できないglobal RNGは複製できない状態として明示した。controller stateは8つのfresh processで初期化した。代表branchでcontact forceは最大約2.2–3.3 kN、DEMとrewardは時間変化した。

## Duplicate controlと少数state smoke

8 branchへ**同一Path**を与えるduplicateを3回、各40 control step実行した。[3回のsummaryとtrace](smoke/)に初期観測、Actor入力・latent・action、follower command、trajectory、reward、Critic、contact、DEMを保存した。duplicateのbranch間XY trajectory RMSは**0.077、0.068、0.054 m**。同じ初期local poseで8つの異なるgenerated Pathを与える少数stateの[candidate smoke](smoke/candidate-smoke.json)も40 step完走し、branch間XY RMSは**0.055 m**。candidate Path自体のbranch間RMSは約0.021 m。候補のsimulator差はduplicate変動の範囲内なので、candidate効果と物理/ノイズ変動を識別できない。これはformal rollout前の**state/duplicate gate不合格**であり、原因候補は未複製のPhysX/contact/raycast履歴、global noise RNG、および候補集合の小さい多様性である。これらを勝手に修正・補正していない。

このsmokeのcandidate入力は同一時点DEMを4 frameに複製したもので、履歴収集を伴う正式評価ではない。duplicateとcandidateは別Isaac processで実行し、初期local poseは完全一致した。40-step rollout実測は約7.0–7.4秒/8 branch、1 caseのmodel/scene初期化を含む所要は約12–14秒。代表caseの全GPU使用量は終了時4,989 MiB、PyTorch peak allocated約45 MB、CPU process peak RSS約5.6 GiB。全GPU値は他プロセスを含むdevice総量であり、PyTorch値はPhysX allocationを含まない。

## 検証・次の判断

新規Phase Cと既存dataset/trainingを含むDocker検証は**96 passed**、repoのEAT全suiteはfixture配置を揃えて**131 passed、7 subtests passed**。Docker setupの2テストはhost Pythonで通過した。ROS関連の既存3テストファイルはIsaac Python 3.11とJazzy Python 3.12のABI差、およびhost Jazzy環境のPyTorch欠如により同一interpreterではcollectできず、今回のfull-repo suiteはgreenと主張しない。実followerのtopic・TF・commandは上記live smokeで確認した。live試験では同一Path 3回、candidate Path 1回、follower command、state一致、observation/action/latent/critic shape、Critic command入力、接触とterrain反応を確認した。resumeはstate-level marker/source照合とlatestからのepoch境界再開を確認した。markerはatomic JSON、checkpointはtempからrename、runnerはdetached `tmux eatnav-phase-c`で実行した。成果物はhost `/tmp/eatnav-phase-c`にも保管した。

**停止判断:** Gate C0は100 epoch時点でbest更新中のため`inconclusive`。duplicate variationはcandidate差と同程度。どちらも正式rolloutの前提を満たさない。第2承認は現状では求めず、fixed recipeを変える再学習やbranch残差の扱いについて、結果レビュー後に別途方針を決める。Phase B Evaluator重み変更、formal rollout、Phase D/E、full collection、実機試験は未実施。
