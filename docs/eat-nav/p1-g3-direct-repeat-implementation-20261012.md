# P1 G3 direct Path反復：実装・static/synthetic verification（2026-10-12）

## 判定と実行境界

`eatnav_p1_longhorizon_g3_direct_repeat_v1`の**実装とoffline検証は完了**。Isaac runtime、Path公開、48試行は未実行。最初のruntime processをpilot兼resource calibrationとし、protocol-validなら正式48試行の4 trialに算入する。pilot後の自動continuationは無効で、実測をレビューしてから残り11 processを判断する。条件変更が必要な場合、pilotは旧versionの診断として分離し、元の結果に混ぜない。

既存のG1/G2、G3 attempt-0、direct duplicate Fail attempt-0の原本・sealと既存runner/production followerは変更していない。G1/G2/direct Failのhost sealはread-only再照合し、それぞれ251/273/291ファイルでvalid。後者のFail判定は保持する。

## 実装差分

| ファイル | 役割 |
| --- | --- |
| `src/visualnav_transformer/eat_nav/p1_g3_direct_repeat.py` | G2の3 seed/groupから12-process schedule生成、4 follower readiness/final検証、trial別Outcome、direct差とfresh-process差の記述統計。 |
| `scripts/eat_nav_p1_g3_direct_repeat_sim.py` | G3 v2から派生した独立runner。4 env全てに同一direct Pathを1回公開し、shared DEM v2・paired noise・`act_inference`・20 Hz exact-stamp制御を維持。各envのtraceとprovenanceを保存。 |
| `scripts/eat_nav_p1_g3_direct_repeat_host.py` | 4 follower bridge、G1方式のPID/GPU監視、全4 traceのSHA・same-episode・stamp照合、2組の凍結済みduplicate gate、writer終了後のseal。duplicate Failは測定結果として保存し、protocol failureと分離。 |
| `scripts/eat_nav_p1_g3_direct_repeat_stage.py` | G1/G2 sealと旧G3 direct Path SHAを確認し、既存データを変更せず、新rootへ12個の独立batchと入力SHA台帳を作る。 |
| `scripts/eat_nav_p1_g3_direct_repeat_analyze.py` | protocol-valid sealのみを読み、48 trialのlocal arrival/censor、2秒progress、tracking、recovery、contact/torque負荷、duplicate Pass/Failとterrain supportを集約。 |
| `docs/eat-nav/p1-g3-direct-repeat-v1-manifest.json` | source/input/config/Path/checkpoint SHA、seed・順序・閾値・resource capを凍結。 |

G2統計pilotの`assignment.json`が3 seed/group、旧G3 assignmentは2 block/groupだけを含む。両者のdirect Path SHAが最初の2 blockで一致することをstaging時に検証し、第3 seedのdirect Pathは同じG2 A条件から取る。6 seedはscene生成とpaired observation-noiseの両方へ渡るため、seed変更に伴う差は総合効果であり、noiseだけの効果とは称しない。terrain mesh/materialがStateGroup内で変われば別条件として集計し、fresh-process対応付けもしない。

4 followerの起動前readinessは各branchの生存、Path subscriber、TF、status、command sequence 0、reject 0、sim clockを照合する。各stepは4 branchのaccepted event、TF・command・Pathのstamp、bridge acceptance→physics順、全envのepisode連続性を検証する。終了時は4 branchのcommand sequenceとclockを確認する。Pathのstampが途中で変わればFail。実測native DEMはshared実験入力とは別の診断値のまま残す。

新しい主要Outcomeのtracking/body-load平均は、Goal到達を初めて検出したstepより前を対象とし、40-step stop holdを含めない。凍結済みduplicate gateのtracking計算・閾値は変更せず、その値を別に保存する。最初のstepで到達したtrialはpre-arrival負荷の曝露0としてNAを出す。

## SHAとoffline検証

| 凍結項目 | SHA-256 |
| --- | --- |
| Manifest | `39b19cec76e03d0a15daa0e297229ef9c9a9e1325462d786c9d7e2248dd00874` |
| 12-process assignment | `cd64acf309d6dd073cb8cca26eab46f1c24877de231bd4ab53146c572f3ced16` |
| Simulator source | `7e1eb3f6617c0028be4497677457878fb5059f69d48dd229004674b2664bba39` |
| Host source | `b8c53a832966cdaa4e1f256a7b44a113f76ceff8774d39e1635055293131dee0` |
| Stage source | `84250ba24ec999b1148e8fd06977d94cebe4d6da68acdd82fd7656526b52c469` |
| Analysis source | `fa1e7d02543b9ff7ee6b4921fb1da886bc2d0b7946a15042862b5de0dce07e8f` |
| Direct-repeat module | `630cb5f802dd577b3932c7638cb1327a4dc0e74e2f5eb50500979377c17ddba8` |

新規および既存G3/GPU/paired-noise関連の**81 test pass**（`PYTHONPATH=src:scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`tinyvla`のPythonでtestのみ実行）。実データを使うoffline stagingでは12 batchと12入力SHA台帳が一致し、一時検証rootは終了時に削除した。`native exit 0`でもsimulatorの`diagnostic-error.json`をhost failureに変えるsynthetic checkが通過。新ファイルの`py_compile`とmanifest source SHA再照合も通過。

repository全testの収集は現在のtest用`tinyvla`環境で12件のimport error（主に`diffusion_policy`、`rclpy`、`geometry_msgs`不足）となった。この結果は新しいG3のtest failureではない。Isaac Lab/ROS/GPUを使うruntime動作は未検証。

## Pilot実行手順と停止条件（未実行）

1. 新しい永続root `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-direct-repeat-v1-attempt-0/`を用意し、`scripts/eat_nav_p1_g3_direct_repeat_stage.py --root <root>`で12 batchをstagingする。stageは既存原本をread-onlyで参照し、入力SHA・G1/G2 seal・旧G3 direct Pathを確認する。Docker側の対応pathは`/workspace/isaaclab_onix/.eatnav-p1-long-horizon-v1/g3-direct-repeat-v1-attempt-0/`とする。
2. 外部Zenoh routerの現在のPID/start tick、ZID、endpointとログを再監査する。routerやfollower binary、driverは変更しない。`/home/kojima/anaconda3/envs/tinyvla/bin/python`で新host runnerのimport-only checkを行う。simulatorはhost Pythonではなく既存`isaac-lab-onix`の`/isaac-sim/python.sh`で起動する。
3. 最初の`batch-first-differential-0-repeat-0`だけを新host runnerへ`--anchor-id first-differential --batch-index 0 --repeat-index 0`で渡し、対応するpersistent `--root`とDocker `--remote`を指定する。SSH切断に依存しないtmuxで実行し、完了まで他のbatchを開始しない。
4. pre-Path physical/shared DEM/Actor gate、4 follower readiness、全4 envのexact-stamp・episode・trace、native exit、GPU ownership、cleanup、host/container SHA、host sealを確認する。duplicate gateのPass/Failはpilot Outcomeとして保存する。protocol/resource failureなら停止し、Path公開後の自動retryをしない。
5. pilotがprotocol-validなら4/48 trialとして算入し、実測wall/GPU/RAM/storageとterrain fingerprintをレビューする。変更が不要と判断され、残り11 processの実行承認を得た場合だけ、frozen assignment順に逐次実行する。

既存hard cap：1 process wall **600 s**、GPU device-total **7680 MiB**、owned attribution gap **256 MiB**、host available RAM **10,095,421,850 B以上**、container anon+shmem **6,442,450,944 B以下**、process root storage **1,000,000,000 B以下**、heartbeat stale **90 s**、実行前の永続ディスク空き **2,000,000,000 B以上**。設計時見積もりは12 process約1.5–2時間（最初の4-env/4-follower実測前）、最終保存約0.9 GB。最初のpilotもこれらのcapを変更せず測る。

同一Pathのpaired差の中央値・p90・最大値は記述的なrepeatability benchmarkとして出す。p90を検出限界や95%検出保証には使わない。将来のPath効果の検出可能性は、独立条件単位のpaired effect分散と反復数から別途設計する。
