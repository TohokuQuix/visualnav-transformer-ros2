# 同一状態・長時間候補比較 pilot 仕様 v0.1

- 作成日: 2026-10-19
- 状態: ドラフト。シミュレータ起動・データ収集は未実施
- 目的: 既存の2秒経路応答データを、同一状態から複数経路を比較する長時間リスク評価データへ拡張するための最小pilotを定義する

## 1. 要約

データの基本単位を、同じ地形・初期ロボット状態から複数の候補経路を実行する `StateGroup` とする。各候補をイベント発生または時間上限まで走らせ、進捗・near-fall・転倒・スタック・復帰を時系列で記録する。

- C（同一状態からの候補比較）を実験単位とする。
- A（長時間実行と時系列記録）を試行方法とする。
- B（地形・難度の境界探索）は、固定pilotで収集不足が判明した後の追加割当てに使う。

このpilotはEvaluatorのデータ生成可能性を確認するもの。少数データで未知地形への一般化や実機転用を結論しない。経路生成ネットワークの教師データを完成させることも本pilotの範囲外とする。

## 2. 既存データとの関係

共有済みMotion Predictorデータは、4地形種・3難度・36 seedで、前後進と7種類の旋回角を組み合わせた14候補を比較している。各候補は20 Hzで40 control step、すなわち2秒間実行され、2秒後のbody-frame XY変位と補助yaw変位を記録する。traceにはroot state、追従指令、Actor action、接触力、関節状態・トルク等が含まれる。

既存protocolでは低進捗やstallは有効な観測結果だが、転倒・リセットで40 stepのラベルを作れない試行はprotocol failureとなる。従って既存データは短期応答の基準として保持し、新規長時間データとはschemaとラベル定義を分ける。既存のsealed evidenceには追記・上書きしない。

## 3. 実行環境監査

### 3.1 確認できたこと

| 項目 | 確認結果 | 含意 |
|---|---|---|
| Git branch | `exp/eatnav-ros2-gui-demo-terrain` | 共有されたGitHub handoffと一致 |
| ホストOS/ROS | Linux、ROS 2 JazzyのCLIあり | GUI handoff記載のdistroと一致。ROS環境のsource状態とtopic通信は未試験 |
| Docker/GPU | Docker、NVIDIA Container Runtime、RTX 4090 16 GBが見える | GPUを使える基盤はある。Isaacシミュレーション起動成功を意味しない |
| Isaacイメージ | `isaac-lab-onix:latest`、Isaac Sim 5.1関連イメージが存在 | 新規ダウンロードなしで試せる可能性が高い |
| ONIX bind mount | `/home/ubuntu/Projects/onix_isaac_lab` → `/workspace/isaaclab_onix` | ソースディレクトリは存在し、停止中コンテナの設定にbind済み |
| コンテナ状態 | `isaac-lab-onix` は停止中、exit 137、`OOMKilled=false` | 終了理由はinspect情報だけでは特定できない。起動・GUI・GPUアクセス確認が必要 |
| GUI用checkpoint | `checkpoints/eat_nav/motion14/training/full-seed123/best.pt` が存在 | 現行GUIの推論用。走行データを生成するRL Actorとは別物 |
| RL Actor | 共有フォルダに `rl_actor_model_10000.pt` と `rl_actor_env.yaml` が存在 | コンテナの既定参照先に同じActorは見つからない。重み・環境設定の対応とSHA確認が必要 |
| コンテナ内Python | handoffでは `/isaac-sim/python.sh` を使用 | Isaac/ONIX moduleを使う収集処理はホストのpyenv Pythonでなくコンテナ内Pythonで試す |

### 3.2 未確認・ブロッカー

- 停止中のコンテナを現在起動できるか、Isaac Sim/PhysXとONIX taskのimportが通るか。
- コンテナ内でGPUが見えるか、GUI/X11またはheadless実行が成立するか。
- 共有RL Actor checkpointと `rl_actor_env.yaml` が、bind mount中のOnix code・actor architecture・observation定義に一致するか。
- 現在の `/home/ubuntu/Projects/onix_isaac_lab` 内にpilot専用の出力先を作れるか、空き容量が十分か。
- GUIの既定topic・ゴール管理を介さず、候補ごとに同一初期条件を作るROS-freeの収集ループを現branchで再利用できるか。

収集開始条件は、コンテナ内Python、GPU、Onix環境・Actorのロード、1試行のresetとtrace保存をそれぞれ確認すること。exit 137だけをOOMと断定しない。

## 4. 研究質問と仮説

**研究質問:** 同一の局所DEM・身体状態から実行する候補経路に応じて、進捗、転倒/スタックリスク、復帰可能性はどのように変わるか。その差を実機で取得可能な入力から予測できるか。

**作業仮説（未検証）:** 候補経路とDEM・身体状態を同時にEvaluatorへ与えると、状態だけを用いるモデルより候補間のリスク・進捗順位を捉えやすい。同じ初期条件のpaired comparisonが、異なる初期状態を単純に混ぜるより経路選択の教師信号を明確にする。

このpilotでは仮説を確証しない。まずpaired trialの再現性、イベント発生、判定可能性、データschemaを検証する。

## 5. Pilotデザイン案

### 5.1 実験単位

- `StateGroup`: terrain type/level/seed、初期root pose、joint/flipper状態、settling後の状態を識別する単位。
- `Trial`: 1つのStateGroupから1候補だけを実行した記録。
- `Branch`: 同一StateGroupの候補別Trialを結び付けるID。
- 同一seedだけで状態一致を仮定しない。候補公開前のroot state、joint state、DEM、Actor observation、material/noise設定を比較し、一致基準は既存のpre-Path gateを確認してから適用する。一致しないgroupはpaired解析から除外し、理由を記録する。

### 5.2 最初のpilot条件（提案値、未承認）

- 地形: `k_rail_step_field` を主対象とし、対照に `hf_pyramid_slope` または `boxes` のどちらか1種を選ぶ。まず地形2種に限定し、全地形へ広げない。
- 難度: 低・中の2水準から開始。levelの幾何学的意味は現branchのterrain設定とログを照合して固定する。
- seed/初期状態: 各地形×難度から独立seedを3つずつ選ぶ案。計12 StateGroup。これは実行系・ラベルのpilot規模であり、統計的に十分な標本数とは主張しない。
- 初期姿勢: まず一つの標準flipper poseに固定する。front/rear条件はpilot後の追加因子とする。
- 候補: 既存Arc Bankの `forward_direct`、`forward_-30`、`forward_+30`、`reverse_direct` の4つを暫定候補とする。候補bankの実装・ID・軌跡が現branchと一致するか確認してからfreezeする。
- 総試行: 12 StateGroup × 4候補 = 48 Trialの提案。物理条件とActor checkpointは同じpilot内で固定する。
- 収集順: 各group内の候補順を事前に固定またはseed付きで均衡化する。失敗結果を見て候補定義を変更しない。

この規模の意図は最初の分岐差・イベント例の確認である。失敗が出ない場合は失敗を捏造せず、次段階で段差・摩擦・進入位置等を境界探索して追加する。

### 5.3 実行時間

- 既存の2秒label windowは維持して短期応答を算出する。
- pilot候補はまず最長10秒まで継続し、成功/イベント/上限到達で終了する案とする。
- イベントがほとんど発生しない場合は、trace・資源・ゴール/指令期限を確認後に20秒へ延長する。無制限走行はしない。
- 既存demoの「最大step数」はGUI session全体の上限であり、1候補の継続時間とは同一でない。handoff記載ではSupervisorのgoal deadlineはsimulation time 30秒。GUI上で時間上限を延ばしただけで、同一候補のteacher episodeが長くなったとは扱わない。
- paired data生成にはGUI選択器の選択結果だけを使わず、各候補を単独で指定できる収集runnerを用いる。GUIは可視化・通信確認用であり、現状topicだけでは候補別実測未来ラベルを自動生成しない。

### 5.4 行動主体の固定

pilot第1段階では、同一のRL Actor checkpoint、同一のC++ PureFollower、同一のphysics/material/noise設定を使用する。経路候補以外の要因を変えない。

サブクローラ姿勢変更、操舵noise、RL方策学習、敵対的地形探索は第1段階に同時導入しない。経路効果が測れた後、別因子として一つずつ追加する。操作を変える比較では、path/action commandをEvaluator入力に含め、操作条件付きの結果として記録する。

## 6. イベント定義と記録

### 6.1 ラベル

各Trialに次の独立した属性を持たせ、一つの成功/失敗bitへ過度に縮約しない。

- `success`: 経路進捗/終端条件を満たし、終了まで転倒・スタックなし。
- `near_fall`: 姿勢・角速度・支持/接触信号が危険域へ入り、転倒定義には至らない状態。
- `tip_over`: 姿勢だけでなく、車体接触、支持状態、環境termination/reset等の複数証拠が成立した状態。
- `stuck`: 有効な移動指令が継続している間に、一定windowの経路進捗がほぼなく、指令停止や通常旋回だけでは説明できない状態。
- `turn_failure`: yaw進捗が継続して指令に追従しない状態。
- `entangled/contact_block`: 接触・関節/クローラ動作制限と低進捗が同時に続く状態。
- `recovered`: near-fall/stuck等の後、後退・旋回・flipper操作等で進捗を再開した状態。
- `timeout`, `sim_reset`, `numeric_failure`, `protocol_failure`: 物理的な失敗と実験系の失敗を区別する終了理由。

### 6.2 判定の原則

- Roll/Pitch単独、低速度単独ではラベルを確定しない。
- 転倒は姿勢・角速度、root height、車体/地形接触、termination/reset reason等を組み合わせ、継続時間と回復可能性を確認する。
- スタックは指令の有効性、経路方向進捗、body/joint速度、追従誤差、接触/effortの持続を組み合わせる。停止指令中や通常の旋回ではスタックとしない。
- 初期閾値は、ONIX寸法・sim signalの符号/単位・既存GUI stall detectorとの定義差を調べた後に設定する。v0.1では未決定とし、raw signalと判定理由を残す。
- event時刻、最初の予兆時刻、最大重症度、終了時刻を保存する。曖昧例には `uncertain` を付け、確定ラベルへ強制変換しない。

### 6.3 時系列と学習ターゲット

- 状態/指令traceはsim control rate（既存系では20 Hz）を基本とする。
- DEMは既存仕様に合わせ、4 frame/1秒履歴、frame間隔250 msを基本案とする。pilot中は同時刻stampとsensor actual update stampを保存する。
- カメラは、Visual入力をpilot比較する場合に限り追加し、DEM/状態とのsimulation timestampを保持する。
- 各観測時刻から1秒・2秒先のイベント有無と進捗量を補助ターゲット候補とし、4秒先はイベント数とデータ量を見て追加する。
- Trial終端が予測horizonより短い場合は右打ち切り情報を保存し、観測されなかった将来を「安全」と誤ラベルしない。
- 失敗前2秒程度とイベント後/復帰完了までのtraceを必ず保持する。全エピソードtraceを原本とし、window化は後処理で行う。

## 7. 入力・記録信号

### 7.1 学習入力候補

- 候補Path: robot-base-yaw座標、単位m、候補ID/方向。
- DEM: `[4,2,20,20]` 相当のheight+valid mask、robot-relative frame、各stamp。
- 身体状態: 実機で取得できる姿勢/角速度、速度、flipper joint state、必要ならクローラ/関節速度・effort。
- カメラ: 実験目的に含めるablation段階で追加。
- 候補操作: follower command/target pathと、必要ならflipper action。将来結果は操作条件付きであることを明示する。

シミュレータ内の正確なCOM/support polygon、contact manifold、friction ground truth等は教師ラベル・分析用のprivileged signalとし、実機転用の入力へ無検証で混ぜない。

### 7.2 ROS 2およびsim記録例

GitHub handoffのROS 2 topicはGUI/可視化のI/O契約であり、すべてがraw training dataではない。利用可能なtopic・message型は実行時に確認する。

- handoffに記載の候補: `/clock`, `/tf`, `/odom`, `/goal_pose`, `/eat_nav/dem`, `/eat_nav/actor22`, `/eat_nav/bank`, `/eat_nav/evaluation`, `/eat_nav/mock/selection`, `/eat_nav/mock/*`。
- 学習用に必要な候補: camera image(+camera info)、`/joint_states`、IMU、速度/経路指令、制御器出力、simulation root state、contact/torque、terrain/seed provenance。
- `/eat_nav/evaluation`は現モデルの予測結果、`/eat_nav/selected_path`やdisplay Pathは選択/可視化であり、実測未来運動ラベルではない。
- ROS topicだけでは得られないsim内部信号は、runner traceへ同じsimulation timestampで保存する。

### 7.3 1 Trialの保存schema案

```text
StateGroup:
  state_group_id, terrain_type, level, seed, tile_sha, initial_state_sha
  actor_checkpoint_sha, sim/config/source/protocol_sha, split_group
Trial:
  trial_id, state_group_id, branch_id, candidate_id, path_xy, command/action
  time_series: sim_time, root_pose/twist, IMU, joint_state/effort, DEM, image_ref
              follower/actor commands, contact, termination/reset, resources
  events: type, start_time, end_time, severity, evidence, confidence
  outcomes: path_progress, endpoint_error, yaw_change, event_free_duration, status
```

Raw artifactsは新規run rootに保存し、dataset tensor/windowは再生成可能な別階層に置く。各trialはmanifest、source/config/checkpoint SHA、trial status、失敗理由を持つ。途中で止まった試行も消さずに記録する。

## 8. データセット化と学習

### Evaluator（pilot主対象）

- Input: 初期観測、DEM、身体状態、候補Path/操作。
- Target候補: 1/2秒進捗回帰、1/2秒内near-fall/転倒/スタック、試行終了までのevent-free時間、最終progress。
- Same-StateGroup内でpairwise rankingを作り、「より安全」「より進捗が大きい」を比較する。ラベルが確かな組だけ使う。
- 評価: held-out seedでevent recall/precision、危険候補の見逃し率、候補順位、progress回帰を報告する。小標本では信頼区間と各seed結果を併記する。

### 経路生成ネットワーク（後段）

- 成功し、かつ目標方向に有効な経路/サブゴールを教師候補とする。
- 失敗経路は危険性ラベルとして保持し、成功経路の模倣教師へ混ぜない。
- pilotにgoal条件と複数成功経路が含まれない場合、経路生成モデルの学習用data readinessを満たしたとは見なさない。

### 分割と偏り

- 同一StateGroupの全候補、全windowを同一splitに置く。
- splitは地形seed単位で行い、testはモデル/閾値選択に使わない。
- 自然発生率を保った評価セットと、訓練時のevent-balanced samplingを分ける。過剰samplingした場合は元の頻度と重みを保存する。
- Pilot 48試行は収集系の検証規模であり、正式なモデル性能を主張するためのサンプル量ではない。正式規模はpilotでのイベント率、seed間変動、信頼区間、学習曲線から決める。

## 9. 実行フローと判定gate

### Gate 0: runtime readiness

1. Docker inspectでbind source、image、env、GPU設定を確認。
2. コンテナを起動し、`/isaac-sim/python.sh`、Isaac Lab、Onix task、torch/CUDAを確認。
3. 共有Actor checkpointを新規staging場所へ複製し、SHA、state-dict keys、env yaml、observation dimensionを照合。
4. 既存GUIの起動確認と収集runnerの起動確認を区別する。
5. 出力先の空き容量、GPU/host RAM、X11/headless条件を記録。

### Gate 1: 1 StateGroup × 2 candidate smoke

1. 固定seed/terrainをresetしsettlingする。
2. candidate A/Bのpre-Path状態・DEM・Actor stateの一致を確認。
3. 各candidateを単独実行し、2秒以降のtraceが連続保存されることを確認。
4. 意図したtermination/event/timeoutとprotocol failureが分離されることを確認。
5. 原本traceから同じlabels/windowsが再生成できることを確認。

Gate 1不通過時はpilot本収集を始めない。

### Gate 2: fixed pilot

上記の提案値（12 StateGroup × 4 candidate）を固定manifestにし、全trialを実行する。問題修正後に同一manifestを黙って再利用せず、schema/protocol versionを更新する。

### Gate 3: pilot review

- candidate間で初期入力一致が成立した割合。
- terrain/candidate別の成功、near-fall、転倒、スタック、復帰、timeout件数。
- eventラベルの自動判定とtrace確認の不一致。
- runtime、trial時間、メモリ、保存容量、失敗したprotocol割合。
- held-out seedでの簡易Evaluator baselineと、進捗・危険見逃し。

結果に基づいて、B（境界探索）で次のterrain/level/entry poseを選ぶ。結果を見てpilotの既存splitを変更しない。

## 10. 安全・再現性・実機転用

- simulator reset/terminationを成功や通常の物理失敗と混同しない。
- 長時間試行に最大sim時間、wall-clock/resource上限、numeric failure stopを設定する。
- 失敗データを作ることと危険行動を実機で実行することを区別する。実機検証は低速・監視下・非常停止系確認後に別計画で行う。
- 実機入力に使えないcontact/COM等でのみ分類できる場合、そのlabelはsim privileged labelと明示する。
- ROSのゼロ速度指令は物理停止保証ではない。GUI handoffもmock command経路であり、実機cmd_velを発行しない。
- run rootは新しい一意の名前とし、共有のsealed archive、既存checkpoint、ユーザーの未コミット変更を上書きしない。

## 11. 次の作業

1. `isaac-lab-onix`を起動する前に、利用者とGPU/GUIを使う時間を調整する。
2. コンテナ内でIsaac/ONIX import、GPU、Actor/Env設定一致を確認する。
3. Gate 1の実行runnerと厳密なpre-Path比較項目をソース監査し、現branchで実装可能か判定する。
4. 候補数、地形/level、時間上限、near-fall/stuck閾値を確定し、仕様v0.2をfreezeする。
5. freeze後にのみ、新規rootでpilot収集実装・実行へ進む。

## 参考

- [ROS 2 + Isaac GUI demo handoff](ros2-isaac-gui-demo-student.md)
- [既存14候補収集protocol](motion-predictor-vnext/motion_14_collection_protocol.md)
- [実験pipelineと既存Evaluatorの範囲](experiment_pipeline.md)
- [GitHub handoff branch](https://github.com/TohokuQuix/visualnav-transformer-ros2/blob/exp/eatnav-ros2-gui-demo-terrain/docs/eat-nav/ros2-isaac-gui-demo-student.md)
