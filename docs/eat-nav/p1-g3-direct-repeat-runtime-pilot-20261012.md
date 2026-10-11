# P1 G3 direct Path反復：1-process runtime pilot（2026-10-12）

## 判定

`first-differential / batch 0 / repeat 0` の1 process・4 trialは **protocol-valid**。凍結済み `eatnav_p1_longhorizon_g3_direct_repeat_v1` の正式48 trialのうち4 trialとして算入する。direct duplicate gateは **0/2 pair Pass、2/2 pair Fail**。これは有効な同一Pathばらつき観測であり、protocol failureではない。残り11 process、曲線Path、Evaluator学習、full collectionは実行していない。

保存先：`/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-direct-repeat-v1-attempt-0/batch-first-differential-0-repeat-0/`。12-process入力は同じattempt rootへstageしたが、runtime出力があるのはこのbatchだけ。Gitの実装commitは`68895516ae617b1b0b77e5e95f97dae952686edc`、凍結manifest SHA-256は`39b19cec76e03d0a15daa0e297229ef9c9a9e1325462d786c9d7e2248dd00874`。実行source SHAは`7e1eb3f6617c0028be4497677457878fb5059f69d48dd229004674b2664bba39`、checkpoint SHAは`b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d`、direct Path SHAは`4ec126da6c09e7ebc02f2febba35666ef57a334cec9745301554a13aa1d0155e`。batch seedは`2026101101`、terrain seedは`1134960613`。4 tileのterrainは`boxes`で、canonical mesh SHAとmaterial SHAが一致した。

## Protocol gate

- 20-step settling後のvisible physical stateのroot・joint等の最大spreadは0。shared DEM v2の4 frameは全branchでbyte/SHA一致、高さRMS/max 0、mask IoU 1.0。pre-Path Actor差はmax abs 0（閾値`1e-4`）。Path公開前の公開回数0。
- Native DEM historyは独立診断で **不一致**。shared DEMをnative sensor履歴とは表示していない。これはshared-input gate Passで隠していない。
- 4 followerのPath subscriber、TF、status、clockがready。pre-Path command sequence/rejectは各0。direct Pathは各branchへ1回公開。既存bridgeのempty Path初期化後、followerの`path_seq`は全traceで2に固定され、Path stampは`2,000,000,000 ns`から変化しなかった。
- 各envの90 trace rowはcontrol step 0–89、論理時刻`2.000–6.450 s`、間隔50 ms。clock/TF/command/bridge acceptance/physicsのstampと順序は全360 branch-stepで一致し、accepted command 360、reject 0。最終command sequenceは各90。40秒episode capは800 stepとして有効で、episode counterは20→110。reset、termination、truncation、非finite値なし。
- 最後のenvがstop holdを完了するまでprocessを継続したため、早く到達したenvにもzero command rowが追加された。各trialのOutcomeは到達前曝露のみを使い、stop holdやその後のrowをtracking/body-load平均から除外する。到達後の初回rowと後続40 stepでzero commandを確認した。

## 4 trialのOutcome

| env | arrival | censor | 2秒progress | tracking平均 | recovery | 接触force平均 | 絶対torque平均 |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.40 s | なし | 0.44637 m | 0.02764 m | 0 | 7.59 N | 178.40 |
| 1 | 2.00 s | なし | 0.56246 m | 0.02336 m | 0 | 9.32 N | 248.43 |
| 2 | 2.00 s | なし | 0.56079 m | 0.02737 m | 0 | 7.88 N | 157.86 |
| 3 | 2.45 s | なし | 0.43780 m | 0.02779 m | 0 | 6.74 N | 161.49 |

Tracking/contact/torqueは到達前の40–49 control stepで計算した記述統計。Contact forceは記録された各bodyのnet force normのstep内平均、torqueは各jointの絶対値のstep内平均。最大・p95、raw traceはseal内に保持する。2秒progressはstep 0のpre-physics rootからstep 39のpost-physics rootまで、Path終点への距離の減少量。

**Localとcommon-goalの照合：** このprotocolの`two_second_progress_m`と`common_goal_progress_2s_m`は、いずれも同じPath終点までの距離減少を使う。root world XYから各tile originを引いたbranch-local座標で計算し、ROS followerへ渡した`map` Pathも同じbranch-local座標である。source contractの共通ゴールは初期base frameの`[0.75, 0.0] m`。初期root/yawでこれをbranch-localへ変換した点とdirect Path終点は4 envとも計算上完全一致し、開始距離は0.75 m。したがって今回の両progress欄は数値が一致する。同一の計算を別名で保存した値であり、独立なnavigation qualityの裏付けではない。

## 同一Path duplicate差

| env pair | 到達時間差 | 2秒progress差 | 凍結gateのtracking差 | 判定 |
| :--- | ---: | ---: | ---: | :--- |
| 0–1 | 0.40 s | 0.11609 m | 0.00548 m | Fail：到達時間、2秒progress |
| 2–3 | 0.45 s | 0.12299 m | 0.00017 m | Fail：到達時間、2秒progress |

閾値は到達時間差`>0.10 s`、2秒progress差`>=0.025 m`、tracking差`>0.02 m`で凍結済み。全trialでrecovery modeは`default`、linear/angular recoveryは0。4 envは1 process内の相関する測定であり、独立4標本として分散や検出保証を推定しない。

## Resource・保存

Host wall time **52.30 s**、native exit **0**。65 GPU bracket中61件がowned GPU sample。GPU device-total peak **5,195.25 MiB**（cap 7,680）、attribution gap peak **108.85 MiB**（cap 256）、container anon+shmem peak **3,514,466,304 B**（cap 6,442,450,944）、host available RAM min **89,220,648,960 B**（floor 10,095,421,850）、active root storage peak **11,871,633 B**（cap 1,000,000,000）。final root file容量は**11,971,161 B**。monitor failureなし。owned simulator/follower/bridge residual 0、全writer終了後にseal済み。

`host-seal.json`のSHA-256は`2c16bd99fa5575f017db06e1e68472c05e4b370905596f669c04f767cc29cc92`。seal検証は297ファイルでPass。4 trace、pre-Path DEM配列とledger、complete markerの参照SHA、および必要なhost/container SHA照合はPass。外部routerは起動前後で同一PID/start tick/ZID/endpointだった。既存G1/G2、G3 attempt-0、旧direct duplicate Failのseal・原本は変更していない。

## 旧direct control attempt-0との関係と次の判断

旧`g3-direct-control-v1-attempt-0/batch-0`のenv 0–1は、到達step 48/40、2秒progress 0.44637/0.56246 m、凍結gateのtracking値とFail理由が今回の0–1 pairと一致する。ただし旧実験は2 branch・89 rowで、duplicate Failをhost failureとして記録した別protocolであり、今回の独立なfresh-process repeatや正式48 trialへ加算しない。今回の4-env protocol-validな結果は、**同一Pathでもenv間の2秒progress差が約0.12 m出る**ことを示す1 processの記述結果。seed間・fresh-process間のばらつき、将来のPath差の検出可能性は残り11 processなしには評価できない。残り11 processの開始は今回の承認範囲外。
