# EAT-Nav Goal-conditioned Local Subgoal simulator pilot: runtime gate stop

## 判定

**44 live runは開始せず停止した。** GS1–GS4とfar-OODの実走行性能はNAであり、local subgoal方式のpass/failは判定しない。offlineで固定した4 state、16 goal、32 Path、seed、順序は変更していない。NoMaD、Evaluator、RL、follower制御則、reset event、dynamicsも変更していない。

## S0とlogging parity

元のoffline proposal SHA、manifest/split、NoMaD/RL checkpoint、32 Pathのidentity/SHAを照合し、44 runを事前固定した。既知の`readiness-002`をlogging追加前後で120 step比較し、command、Actor pre/post-delay・latent・action、Critic、reward、root/joint、contact、DEM、realized trajectoryが完全一致した。raw joint effortは`robot.data.applied_torque`から52 jointで取得でき、4 flipper＋左右track sumの6 termへ変換した。rawを[-100,100]でclipしてから0.01倍するActor pre-noise termとの最大差は3.34e-6。実機onlineで同等の順序付きeffortを取得できるかは未確認。

最初のparity smokeはZenoh router不在でPath subscriberが0となり、Path投入前に失敗した。router起動後の同一run parityは通過した。その後に過去episode resetの診断assertionだけを修正したため、最新source全体のparity再試験は未実施である。pilot再開時には再試験が必要。pilot専用routerは停止し、残留simulator/bridge/follower/ROS processは0。

## 停止理由と資源

選択stateはepisode-local prefixが20/20/35/40 stepだが、現行の確立済みfresh-process replayは過去episodeを含むglobal full-prefixを再生する。global prefixは`{'train-003': 1621, 'train-004': 421, 'train-019': 436, 'train-021': 3241}`で、44 live run合計62,909 stepとなる。

最初のtrain-003 anchorは、episode 4より前の正常resetを旧診断assertionが拒否し、344.9秒で停止した。診断assertionだけを修正し同じstateで再試行したが、258.7秒時点でruntime forecastにより終了した。anchorの一致は**未確認**であり、duplicateとcandidateは各0件。前回試行のログは保持した。

最初のresetは遅くとも400 stepまでに生じる。測定344.9秒から起動固定費を過大な200秒控除しても、44 live runのprefixだけで**約6.3時間**となる。120-step suffix、follower、残る4 anchorはこの値に含まない。固定上限4時間を超えるため、新規runを開始しなかった。

GPU peak 5550 MiB、cgroup anonymous CPU memory peak 2.49 GB、raw保存量 27.5 MB。二度目のanchorはoperator SIGTERMを受け、Isaac native closeがrecursion abortでexit 1となったが、processは終了し残留0。これは候補outcomeではない。

## 次の設計課題

同じ固定state・Pathを使用するには、対象episode開始状態からのepisode-local replayがfull-prefixと一致することを別gateで立証する必要がある。過去の同一process snapshot restoreや未検証の短縮方式へ切り替えて44 runを続行しない。短縮方式が成立しなければ、4時間上限と固定stateの組合せは実行不能である。新しいstate、goal、seed、Pathをこの結果を見て選び直していない。

成果物のNA JSONは未実行を明示する。performance図はデータがないため作成していない。raw log・failure marker・statusは`/tmp/eatnav-candidate-coverage/goal-ood-simulator/`に保存した。
