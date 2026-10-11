# P1 G3 接触・物理再現性の診断計画（2026-10-11）

## 判断と範囲

G3 direct duplicate attempt-0 の凍結済み post-Path gate は **Fail** のままとする。次に必要なのは Path の優劣判定ではなく、同一プロセス内の env 差と独立プロセス再実行差を分け、最初の物理 step の差がどこで生じるかを測ることである。この文書は既存証拠の読み取り専用監査と、未実行の最小 runtime 実験案である。新しい rollout、接触状態の書換え、曲線 Path、学習、full collection は行っていない。

元の [attempt-0 報告](p1-g3-direct-control-attempt-0-20261011.md) と `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-direct-control-v1-attempt-0/batch-0/` は変更しない。再照合した host seal は valid、291 ファイル、SHA `5bda703cc76c464664143067a22c6592c20fbf4e35873412eebd294431a36409`。

## 既存証拠の再解析

pre-Path の tile-local root 13値、52 joint position/velocity、applied torque、Actor input/action、follower command は direct 2 branch で完全一致した。paired observation noise も全89 control stepで一致した。接触センサ force は pre-Path で最大 **176.835 N** 異なり、Critic observation は最大1.0、Critic valueは0.107334異なった。step 0 の command/action は一致し、その最初の physics step 後に tile-local XY が **1.075 mm** 分岐した。step 1 の Actor input/action と follower command の最大差はそれぞれ0.007604、0.025566、0.000361だった。step 39 のXY距離は0.11609 mとなった。到達はstep 48対40で0.40 s差、2 sのcommon-goal progressは0.44637対0.56246 mで0.11609 m差となり、既定gateの0.10 s／0.025 mを超えた。tracking spread 0.00548 mとrecovery分類はgate内だった。

重要な訂正として、現在の `contact_forces.history_length` は1である。[rough_env_cfg.py](/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/config/onix/rough_env_cfg.py:444) と [ContactSensor](/home/kojima/lib/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py:373) の実装どおり、attempt-0 の `contact_history[:,0]` は `contact_current` と**完全一致**した。したがって「force差」と「history差」は独立した2つの証拠ではない。forceは直前の物理計算から得た接触応答の読出し値であり、それ自体が solver に入力される状態だとは確認できない。[Critic contact_binary](/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/mdp/observations.py:71) はforceから計算され、[Critic設定](/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/config/onix/rough_env_cfg.py:515) にある。Critic値はActorの入力ではない。よってCritic差を運動差の原因とは扱わない。

## 静的監査：複製できるものと未解決のもの

| 対象 | ソースで確認できた事実 | 未確認・解釈上の限界 |
| --- | --- | --- |
| Visible state | [clone_internal_branch_state](../../src/visualnav_transformer/eat_nav/phase_c_sim.py) はroot/joint、command、ActionManagerの一部、latency ring、observation historyを複製し、`write_data_to_sim()`と`sim.forward()`を呼ぶ。接触・raycaster履歴とPhysX warm-start cacheは未複製と明示される。 | `sim.forward()`が接触manifold／solver cacheを等しくする保証は見つからない。初期値の一致はsolver内部の一致を意味しない。 |
| ContactSensor | [ContactSensor.reset](/home/kojima/lib/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py:144) は表示用force/history bufferをゼロ化する。force更新は`RigidContactView.get_net_contact_forces`の読出しである。 | Python tensorのhistoryだけなら複製できるが、PhysX接触状態の複製にはならず、Critic/rewardの読出しも変える。今回の実験では書き換えない。 |
| PhysX contact / cache | 使用中の[Omni Physics tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)はcontact force、pair、point等の取得を公開する。PhysXは接触・constraintの処理順と内部状態を持つ。 | このIsaac Lab／tensor APIでcontact manifold、warm-start impulse、solver cacheをenv間で読み書きして完全cloneする検証済み手段は確認できない。「不可能」と一般化はしない。 |
| Reset / RNG | [ManagerBasedEnv.reset](/home/kojima/lib/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:341) は任意seed設定、scene/manager reset、`sim.forward()`を行うがterrain生成は再実行しない。[RL reset](/home/kojima/lib/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:349) はcurriculum、reset event、各managerを順序付きでリセットする。[seed](/home/kojima/lib/IsaacLab/source/isaaclab/isaaclab/utils/seed.py:15) はPython、NumPy、Torch、CUDA、Warpを設定する。P1の[paired noise](../../src/visualnav_transformer/eat_nav/p1_paired_noise.py)はgroup／seed／step／termから決まる。 | 同一processでの`env.reset()`は新しいPhysX sceneを構築しない。reset event、sensor／actuator内部、他envの状態、GPU実行順を含む同一性を保証しない。fresh process同士でもscene構築順、実際の有効config、checkpoint SHAを照合する必要がある。 |
| GPU PhysX | 使用中cfgは0.005 s×10 substep、stabilization有効、GPU contact/patch/heap bufferを指定する。[scene replication](/home/kojima/lib/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene_cfg.py:87)は既定で有効。`enable_enhanced_determinism`は[Isaac Lab既定false](/home/kojima/lib/IsaacLab/source/isaaclab/isaaclab/sim/simulation_cfg.py:136)で、P1 sourceにはoverrideがない。[Isaac Lab再現性説明](/home/kojima/lib/IsaacLab/docs/source/features/reproducibility.rst:4)は同一hardware／version／seedでの再現性と、GPU schedulingによる微小差・長期増幅の可能性を述べる。 | 有効なlive PhysX scene属性、solver iteration、collision pair順、GPU partitionをattempt-0原本は網羅していない。新規診断では読み取り専用で記録する。設定を変更して原因を推定しない。 |

[PhysX公式API文書](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/API.html)は、同一platform／buildでの厳密再現に**新しいsceneを同じAPI呼出し順で作ること**が必要で、全actorを除いて再投入するだけでは一般に十分でないとする。また非相互作用actorの追加でもconstraint処理順が変わり得るため、enhanced determinismはその影響を減らすが万能ではない。したがって「同一env fresh process」と「異なるenvのclone」は異なる再現性命題である。最小診断では現在の設定を維持し、enhanced determinismの変更は別設計とする。

## 比較する3条件

最小構成は**同一seedの独立fresh process 3本、各4-env固定scene**とする。各processでenv 0を基準、env 1と3を20-step settling後にvisible clone、env 2をcloneなしとする。env 0／1はattempt-0と同じdirect pairの役割を維持する。terrainのcanonical mesh、material、relative初期pose、episode、checkpoint、seed、scene生成順、physics設定を3本で固定する。cloneなしenv 2も初回配置時には同じ相対初期条件にする。processごとの出力root、source/config/input SHA、GPU UUID、Isaac/PhysX build、live scene属性を記録する。環境配置・clone対象の変更によりattempt-0との絶対的な数値同一性は要求せず、この診断内で同一process間の条件を固定する。

| 条件 | 比較単位 | 識別できること／制限 |
| --- | --- | --- |
| C1：同一env・同一seed・独立process | 各processのenv 0同士。**process**が独立標本（n=3）。 | scene再構築と全入力が同じ場合の再実行ばらつきの下限。差があればenv間clone差だけを原因とする推論を止める。 |
| C2：異なるenv・cloneなし | 各process内のenv 0対env 2。20-step settling前後を別々に比較。 | 同じrelative配置から、env ID／world位置／並列解法／settlingが作る差を示す。settling後のvisible state差が交絡するため、Pathやcontactの純粋な因果効果とは言えない。 |
| C3：異なるenv・visible cloneあり | 各process内のenv 0対env 1（env 3は補助複製）。**process**が独立標本（n=3）；env 1/3をn=6と数えない。 | visible stateを一致させても残るcontact／hidden physics／env位置／並列処理差を測る。force差とmotion差の共変動だけではcontact force自体の因果性を証明しない。 |

## 測定境界と解析

1. **R0** scene生成後のresetとfixture配置直後・初回physics前：terrain/collision/material SHA、env origin、root/joint、実効body mass/inertia・drive・collision filter・live PhysX scene属性、actuator target/torque、command、RNG state、contact viewとsensor buffer、sensor update stamp、Actor／Critic observationを保存する。元cfgにはmassやreset poseのrandomizationがあるため、設定値だけでなく実効値を照合する。
2. **R1** 20-step settlingの各control境界、特にstep 4/9/14/19：root/joint、contact、DEM native/shared、Actor pre-noise・post-noise/pre-delay・post-delay、action、Critic、commandを記録する。必要な初期接触の時刻特定にはsettlingの各0.005 s physics substepでroot velocityとraw contactを記録する。
3. **R2** clone直前とclone＋`sim.forward()`直後（追加physicsなし）：C2の自然なvisible差とC3のclone後のvisible一致を**別gate**で検証し、clone対象外のenv 2が変化していないことも確認する。contact sensorの`current/history`、raw contact pair/point/normal/impulse、bodyごとのforce、Critic binary/value、actuator state、sensor actual update stampを保存する。raw pair詳細が現在のview条件で取得できなければ欠損を明示し、その仮説判定を止める。履歴長1を独立時系列と数えない。
4. **R3** 全envへenv 0から同じcommandと同じ`act_inference` actionを配布し、**1 control step＝10 physics substep**を追加する。これはPathなし・followerなしの物理切り分けであり、Actor本来の各env出力も配布前に別途記録する。既存`env.step`内の順序を変えず、各substepのpre/post root/joint/velocity、actuator target/torque、raw contact pair/normal/point/impulse、sensor forceと更新時刻を受動的に採る。ログ取得のための追加physicsや接触状態変更はしない。
5. **R4（R3が解釈可能な場合のみ、別途承認後）** direct Pathを各branchへ1回だけ公開し、既存exact-stamp follower、paired noise、shared DEM、20 Hz、最大600／640 stepを使う。Path公開時と各stepでfollower command、Actor 3段階・action、root、contact、Critic、arrival／tracking／recovery／body loadを比較する。C1は同一env fresh-process、C2/C3は同一process内で集計し、全条件に既存のarrival 0.10 s、tracking 0.02 m、2 s progress 0.025 m（strict）、recovery分類のgateを**変更せず**適用する。C2が初期state gateを満たさない場合、post-Pathの値は記述統計に限る。

各境界で「時刻の意味」を固定する。physics substepは5 ms、control stepは50 ms。raw PhysX contact、ContactSensor buffer、Critic observationはそれぞれ取得／更新時刻を別に残す。first divergenceはsignalごとに、(a) byte/bitwise差の最初、(b)既存manifestの報告用数値許容差を超える最初、を両方報告する。比較順は **pre-step visible/actuator/command/action → raw contact／solver response → post-step motion → sensor／Critic → 次control stepのActor → follower command** とし、同時刻の読出しを因果の前後関係へ勝手に並べない。

contact差との対応は、各processのC2/C3で「R2 body別force・pair/normal/point差」と「R3最初の10 substepのroot速度／XY／joint差」を対応づけて図示する。特に接触の有無・body・pair IDと初回速度差の時間順を示す。n=3の記述相関は仮説生成に留める。sensor forceをコピーして運動が変わるかという介入は実施しない。sensor forceが異なりraw contactと最初の運動が同じなら、force差は古い読出し／履歴の指標である可能性が高い。raw contactと最初の運動が同時に異なるなら、接触機構の候補は強まるが、同一の隠れた状態差が両者を生んだ可能性は残る。

## 最小runtime gateと分岐

今回提案する最初のruntime申請は**3 fresh process × 4 env × 20 settling control step＋post-clone 1 control step**のcandidate-free診断だけである。独立したprocessごとに新しい永続root、atomic marker、逐次trace、resource log、入力・出力SHA、host sealを使用する。既存GPU attribution gap ≤256 MiB、device-total ≤7680 MiB、RAM／storage／heartbeat／wall cap、same-episode／nonfinite／cleanup条件を維持する。3本の起動は逐次とし、各seal／資源gateを通した後に次を始める。正確なresource上限値とsource SHAは実行前manifestで旧G3から照合・固定する。

- **停止**：C1のprephysics入力・scene SHAが不一致、ログで最初のsignal時刻が特定不能、raw contactが欠損／overflow、出力seal・resource gate不合格、または追加loggerが物理step／状態setterを呼ぶことが静的監査で判明した場合。設定や閾値を観測後に変更しない。
- **C1は一致、C3は最初のsubstepから差**：within-scene／clone後に残る状態・接触・配置を優先調査。post-Path再試行はまだ行わない。
- **C1も差**：fresh scene再構築、reset/RNG、asset・material、GPU実行順を優先調査。env間cloneだけを責めない。
- **C3のopen-loop物理は一致し、後のPath結果だけが差**：Actor latency／観測履歴／follower state／TF同期の閉ループ差を優先する。R4を別途レビューする。
- **C2が先に差、C3は一致**：cloneはvisible差を除くのに有効だった可能性がある。ただしC2は未複製状態とenv配置の交絡を含み、汎用的なclone再現性は主張しない。

この3本でcontact forceの「原因／指標」を確定できるとは限らない。contact manifold／warm-start／solver cacheの検証済みsnapshot APIが見つからなければ、最終的な因果主張は「一致入力下でcontact関連の差が先行した」までに留める。判断としては、まずこのcandidate-free一歩目を別途承認して測り、その結果に応じて (i) contact/solver内部診断、(ii) 同一env fresh-processの統計的Path比較、(iii) 現行cloneを使わない設計、のどれへ進むか決める。現時点で曲線G3、Evaluator学習、full collectionへ進む根拠はない。
