# 同一状態・長時間候補比較 pilot 仕様 v0.2

- 作成日: 2026-10-19
- 状態: 受動観察pilot・教師v2監査・dataset/split固定・baseline比較学習と探索的holdout評価を完了。追加収集protocol固定と新24 seedの収集・blind検証も完了。追加データでの本学習・正式test・実機評価・独立した人間による全件annotationは未実施
- v0.1からの主変更: 4環境batchを正しく数え直し、10秒のreceding-horizon試行、候補・地形・閾値を固定。smokeで未実証の転倒/スタック有効性を主張しない

## 1. 目的

同一の地形・初期ロボット状態から候補経路を別々に実行し、経路条件付きの進捗・危険予兆・転倒/スタック・回復を記録する。主目的はEvaluator向けpaired dataの収集パイプラインを成立させ、後段で危険候補を識別する学習に使えるデータを得ること。

- 実験単位は同一初期状態の候補比較。
- 各candidateを10秒まで走らせ、2秒ごとに同じcandidate IDを再計画する。
- 候補経路以外のActor、物理、material、noise、flipper初期姿勢は固定する。
- 転倒・スタックを意図的に起こす操作や初期化は行わない。通常の候補経路を実行した結果として自然に発生する条件を観測する。
- 経路生成ネットワークの教師データ完成、未知地形一般化、実機安全性の証明は本pilotの結論に含めない。

## 2. 実行環境と確認結果

### 確認済み

| 要素 | 結果 |
|---|---|
| GUI handoff | 利用者より `scripts/eat_nav_phase3_gui_launch.py` の指定コマンドでdemo実行確認済みとの報告あり。手順は `ros2-isaac-gui-demo-student.md`。今回そのGUIを再起動していない |
| Branch | `exp/eatnav-ros2-gui-demo-terrain` |
| Isaac/ONIX | `isaac-lab-onix` container、Isaac Sim 5.1、Onix task importを確認 |
| GPU | RTX 4090 Laptop GPU、16,376 MiBがIsaac Sim / torchから認識。torch CUDA available |
| ONIX bind | `/home/ubuntu/Projects/onix_isaac_lab` → `/workspace/isaaclab_onix` |
| RL Actor | 共有 `rl_actor_model_10000.pt` とActor env YAMLをcontainer内一時領域へcopy。checkpoint SHA-256は `b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d`。本pilot manifestのenv YAML SHAは `14fc2441d928a92b1ba45c06ccd8603eb0cf71aebb5d216efc4005f09696545c`。旧smokeの記載値は `c2b7e9b7959a231c5391ed714a4de4be958fe5812cf03c10ab60f5636598233b` であり、両者の同一性は未確認。OnPolicyRunnerでActor stateをloadし、pre-Path runを完了 |
| Actor observation config | `use_last_action_obs=false`, `use_joint_effort_obs=true`, `use_height_scan_obs=false`。同toggleのOnix env YAMLを選択 |
| Paired short smoke | K-Rail level 1でforward Direct/+30°、同seed・同姿勢・同config。各process 4 env×40 step、全pre-Path gate pass、2秒XY/yaw labelsとsealを生成 |
| Long paired smoke | 同じStateGroupでDirect/+30°を各200 step（10秒）、40 step（2秒）ごとに同candidateを再計画。各trialで4 env×200 rows、pre-Path gate pass、cycle 0–4、ゼロ指令なし、seal complete |
| 高難度条件の観察 | K-Rail level 4のDirectを4 env×10秒実行。85°tilt terminationは発生せず。sampled effective tilt maxはenvごと約15–29° |
| 受動event-label smoke | 通常10秒走行で4 env×200 windowsを後処理。near-fall/tip-over/stallは0件、1/2/5秒labelを生成。trial末端の未観測horizonは`event_free=null`/censoredとして保存し、成功扱いにしない |
| Host経由paired smoke | 新host runnerでpre-Path baseline、Direct、+30°を実行。両candidateのdecision・inner/outer seal・pre-Path gate pass、各4 env×200 rows、1/2/5秒labelsとcensor生成を確認。eventは0件 |
| 本pilot | 60/60 process、うち48 candidate-process、192 candidate trajectory。全60 verification/outer seal pass、failed attempt 0。実行時間約65分、artifact約281 MB |
| 有効観測量 | 200 step完走172 trajectory、自然termination partial 20 trajectory。合計37,452 observation windows。自然termination env 5、同じvector batchのparallel censor env 15 |
| 暫定event count | `linear_stall` 36 window、`near_fall` 6 window、`tip_over` 5 window、`turn_stall` 2 window。eventを含むtrajectoryは27（複数event種別の重複を含む） |
| 保存root | `/home/ubuntu/Projects/onix_isaac_lab/.eatnav-paired-long-horizon-pilot-v0.2-20261019-01`。pilot manifestのfrozen protocol SHA-256は `1df876cc86e749dbff56b9b81ab1fd10e2ad69790f576bbda08db0c82844d9a7` |

既存の `eat_nav_motion_14_collection_host.py` にActor/env YAML、variable-length trace、event/censor labelへの対応を追加し、host経由smokeで検証した。

### 確認できていないこと

- 外部test seedでのevent定義・閾値の妥当性とEvaluator性能。
- 実機センサとの入力parity、およびカメラ画像の同期データ。
- より多いseedでの発生率・候補順位の再現性。

## 3. Smokeからの実装上の結論

1. 旧collectorは固定40 step、4 env、旧Actor env YAML絶対path、Isaac RayCasterの非公開 `_ray_starts_w` に依存していた。
2. collectorに `--actor-env-yaml` を追加し、同branchのGUIが使う `scanner.ray_starts + scanner.data.pos_w` 方式に合わせたところ、pre-Path DEM/raw ray captureが完了した。
3. 旧collectorに `--control-steps` と `--replan-every-steps` を追加。既定40 stepは維持し、10秒/2秒replanのDirect/+30° paired smokeが完了した。
4. 固定0.75 m Arcだけを10秒実行するとDirectは約3.15秒でpath goalに到達し、その後commandがzeroになる。長時間比較では2秒ごとに同じcandidate IDの新Pathを現在poseから発行する。
5. 既存互換の `motion-labels.json` は最初の40 step、つまり2秒XY/yaw labelのみ。別artifact `long-horizon-event-labels.json` が10秒traceから1/2/5秒event/censor labelsを生成する。
6. 既存Onix tilt terminationはbody up-axisとworld upの合成傾斜角85°。現GUI reset policyはこのterminationを有効にし、timeoutを無効化している。pilotでは最大sim時間で管理する。
7. smokeで転倒/near-fall/stuckは発生しなかった。従ってsmokeは環境・Actor・DEM・paired runner・10秒trace・censor-aware label生成の成立確認であり、通常走行でのevent発生頻度やrisk learningの有効性確認ではない。失敗を起こすための介入は行っていない。

## 3.1 本pilotの中間結果（2026-10-19）

### 条件別event window数

| Terrain | Level | Linear stall | Near-fall | Tip-over | Turn stall |
|---|---:|---:|---:|---:|---:|
| `k_rail_step_field` | 1 | 7 | 0 | 0 | 0 |
| `k_rail_step_field` | 4 | 29 | 3 | 2 | 0 |
| `pyramid_stairs` | 1 | 0 | 0 | 0 | 1 |
| `pyramid_stairs` | 4 | 0 | 3 | 3 | 1 |
| **Total** | | **36** | **6** | **5** | **2** |

### Candidate別event window数

| Candidate | Linear stall | Near-fall | Tip-over | Turn stall |
|---|---:|---:|---:|---:|
| `forward_direct` | 8 | 0 | 0 | 0 |
| `forward_-30` | 13 | 2 | 2 | 0 |
| `forward_+30` | 12 | 2 | 2 | 2 |
| `reverse_direct` | 3 | 2 | 1 | 0 |

### 暫定的に読めること

- 通常candidate走行だけでlinear-stall、near-fall、tip-over、turn-stallが観測され、失敗誘発操作を加えなくてもイベント時系列とreset直前snapshotを取得できた。
- この割当てではstall windowが主にK-Rail、near-fall/tip-overがlevel 4に集まった。これは本pilot内の観測差であり、terrain難度との一般的な因果関係や発生率を確定しない。
- 同一seedのcandidate比較では同じ初期StateGroupから異なるevent有無を比較できた。K-Rail level4 seed `1387731893`ではforward Direct/−30°にstall window、forward +30°/reverse Directにnear-fallからのtip-overが記録され、同一group内でoutcomeが分かれた。
- near-fall後にtip-over terminationせず10秒traceが完了したのはK-Rail level4のseed `2788660503`、`reverse_direct`、env2。数値trace監査ではonset後5秒以内の経路投影progress増分約0.815 m、tip-overなしを確認した。旧記載の階段level1 seed `3064563552`、forward +30°はnear-fallではなくenv1のturn-stallであり、訂正した。`recovered`の新しい教師は別revisionで定義する。
- `linear_stall`の件数はwindow数であり、独立した失敗件数ではない。同一candidate trajectory内のwindowは相関するため、event-window数をサンプル数/独立事故数として扱わない。
- 後続監査で全192 trajectoryの保存v1ラベルを再生成・照合し、イベントを含む27 trajectoryと対照16 trajectoryの時系列図を確認した。Codexによる数値監査と図の確認であり、独立した人間による全件annotationや映像確認ではない。

本pilotは条件探索の初期観察であり、モデル学習・held-out評価・実機評価は行っていない。イベント定義や閾値の有効性を確定せず、次段階ではevent trajectory単位のtrace review、seed単位分割、簡易Evaluator baselineで追加検証する。

## 3.2 後続trace監査と学習仕様（2026-10-19）

詳細は[trace監査報告](paired-long-horizon-review-20261019/trace-review-report.md)、次の設計は[転倒・スタック学習仕様 v0.1](fall-stall-learning-spec-v0.1.md)に保存した。原本と以下のfreeze条件は変更していない。

- 60 processの内外seal・decision・pre-Path gateを再検証し全pass。全192 trajectoryの保存v1ラベルも現builderから再生成して一致。
- 転倒5例はreset前snapshotと一致。linear-stallは36 window / 19 trajectory / 5 base seed、near-fallは6 / 6 / 4、tip-overは5 / 5 / 3、turn-stallは2 / 2 / 2。独立事故数へ読み替えない。
- 全転倒は開始5.45–9.50秒。初期入力から1/2/5秒先の転倒陽性は0であるため、次の学習仕様は初期入力→10秒outcomeを主対象とする。途中時刻のDEMを持たない本pilotで、初期DEMへ各時刻windowの教師を付けてsample数を増やさない。
- 現linear-stall builderは`remaining_path_distance`の始終差を使用し、freeze仕様の`best_remaining_path_distance`と一致していない。diagnostic best条件だけへ置換すると5 window追加となるが、followerのbestは0.10 m刻みの更新であり、正式な新ラベル数とはしない。40 intervalの距離積算・連続best・欠損unknownを新教師revisionで揃える。
- turn-stallの2例は並進できており、旋回応答不足としてlinear-stallと分ける。終端rowのroot以外の信号にはreset後値が混ざり得るため、新教師はreset前snapshotの信号を優先する。
- base seedを守る[3分割rotation案](paired-long-horizon-review-20261019/split-proposal.json)を作成した。正例がないhead/foldはN/Aにし、全データ確認後の分割を正式な未閲覧testとは呼ばない。
- 関連4ファイルの今回の再実行は8 passed。turn-stall専用testが別関数内にネストしており独立収集されないため、以前の9件passという報告とは区別する。テスト・ラベル実装はこの監査で変更していない。

### 承認後の教師v2実装・差分監査

[教師v2差分監査報告](fall-stall-targets-v2-audit-20261019/audit-report.md)と[派生教師manifest](fall-stall-targets-v2-audit-20261019/artifacts/dataset-manifest.json)を追加した。原本と旧v1 builderは変更していない。

- 全192 trajectory / 60 processを検証。初期DEMはnative world rayから再構成してframe・mask・高さを照合した。
- v2でもlinear-stall 36、near-fall 6、tip-over 5、turn-stall 2 window。完全な40 interval積算と連続bestで判定し、diagnostic bestなら追加される5 windowは陰性と確認した。
- near-fall 6例はonsetから4 step後の確認時刻へ移動。20 partial cycle×2 stall headをunknownとし、初期1/2/5/10秒教師の正負/unknownは新旧で一致した。
- 回復候補はconfirmation 1.25秒後から5秒の観測でprogress最大増分約0.914 m、tip-overなし。旧onset起点の約0.815 mとは区間が異なる。
- turn-stall専用testのネストを解消し、中断指令・大きい応答で陰性になることも確認した。最新の検証結果は[verification.txt](fall-stall-targets-v2-audit-20261019/verification.txt)を参照。学習・held-out推論は未実施。

### 初期入力datasetとseed分割の固定

[dataset生成報告](fall-stall-dataset-freeze-20261019/dataset-report.md)と[固定dataset manifest](fall-stall-dataset-freeze-20261019/artifacts/dataset-manifest.json)を追加した。48 StateGroup×4候補を初期DEM・pre-noise Actor/body状態・Path・directionへ対応させ、未知教師のmaskを保持して保存した。replicateによる3通りの分割は各role 4 seed / 16 StateGroup / 64候補で、原proposalのseed集合を変更していない。全データを確認済みの探索的holdoutであり、正式な未閲覧testではない。normalization fit・モデル学習・held-out推論は未実施。

## 4. Pilot条件（freeze）

### 4.1 Terrain・seed・StateGroup

- terrain type: `k_rail_step_field` と `pyramid_stairs`。
- terrain level: 1と4。Onix generator difficulty range `(0,1)` に対し、level 1 band `[0.2,0.4)`、level 4 band `[0.8,1.0)`。これは通常走行での地形条件の範囲を観察する設定であり、転倒を起こす目的で初期姿勢や物理条件を操作するものではない。
- terrain×levelごとに3つの独立base seed、計12 base-seed assignments。
- 初期flipper condition: `front_folded` に固定。後部姿勢比較はpilotの追加実験に回す。
- 既存Onix root reset: x/yは各±0.1 m、yawは `[-pi, pi]`、roll/pitchと初期twistは0 range。seed・env indexから決まる初期pose drawをmanifestに保存。
- **StateGroupはenv tile単位**。各base seed/processの4 env tileは異なるterrain/initial sub-seedを持つため、4状態を1つの独立StateGroupとは数えない。同じenv indexをcandidate間でpairする。
- 12 base seeds × 4 env = 48 paired StateGroups。3 seed/cellはpilotの確認用であり、正式なgeneralization testに十分とはみなさない。

### 4.2 Candidateと試行数

同じcandidate bankの4 pathを固定順で実行する。

1. `forward_direct`
2. `forward_-30`
3. `forward_+30`
4. `reverse_direct`

1 base seed×candidateが1 process、各processに4 env trajectory。総数は48 candidate-process、192 candidate-trajectory、48 StateGroup×4候補。各base seed内のcandidate全てでpre-Path状態/DEM/Actor input gateを通す。不一致はpairとして使わず理由を残す。

各base seed groupではcandidate順をfixed/均衡化し、結果を見て順序や候補を変更しない。学習前のpilot集計はdescriptiveに限定し、信頼区間やseed数から得られない統計的主張をしない。

### 4.3 動作・時間・recording

- control rate 20 Hz、最大200 step = 10秒。
- 同じcandidate ID/path classを40 step = 2秒ごとに現在のbody poseで再計画し、follower stateを新規化する。指令のhold、経路追従Actor、physics、material、noiseは既存protocolの固定値を保つ。
- trial終了は10秒、Onix tilt termination、numeric failure、または安全resource stop。
- 失敗後に自動resetしたstateを同一episodeの次時刻として扱わない。termination直前stateと終了理由を保存して終了する。
- traceは毎control step、DEM/Actor inputは初期StateGroupで保存。candidate ID、cycle ID、activation時刻、各再計画pose/path、root state/twist、follower/Actor command/action、joint state/effort、contact、DEM stamps、termination、資源を記録する。
- 予測horizonは1/2/5秒。全trace原本を保存し、複数horizon window/labelは後処理で作る。イベント前2秒・イベント時・後の状態を維持する。
- この固定candidate反復は「同じ局所候補選択を維持した将来」のcounterfactual。毎cycle別candidateへ切替えるpolicyの性能とは別に扱う。

### 4.4 イベント閾値（pilot v0.2）

傾斜角はroot quaternionからbody +Zとworld +Zの角度として計算する。閾値はv0.2のpilot label定義であり、実機安全限界を意味しない。

- `near_fall`: effective tilt `>=60°` が20 Hzで連続5 sample（250 ms）成立。発生開始・最大角・回復有無を記録。
- `tip_over`: effective tilt `>=85°` またはOnixの既存85° `tilt_over` termination。reset後の姿勢ではなくreset前のstate、termination mask/reasonを取得する。
- `linear_stall`: candidate cycleの約2秒windowで、`commanded_distance >=0.20 m`、`actual_distance / commanded_distance <=0.25`、かつ `best_remaining_path_distance` の改善が `<=0.05 m` を全て満たす。いずれかの信号が欠けるwindowは `unknown` とする。
- `turn_stall`: `abs(commanded_yaw_rate) >=0.01 rad/s` が継続する2秒間で、観測yaw-rate responseが指令絶対値の50%以下。50%/2秒は現行turn-stall bridge ruleと整合。
- `recovered`: near-fallまたはstallのあと5秒以内に、経路方向progressが0.10 m以上増え、tip-over terminationせず走行を再開。recovery操作/候補の変更があれば別フィールドに記録。
- `event_free`: 完全な10秒を観測し、上記eventがないtrialのみ。termination前の未観測時間は安全例とせず右打ち切りにする。
- path progress、最大tilt、event duration/severity、event-free durationを保持し、成功/失敗1bitには潰さない。

## 5. Dataset・学習対象

### Evaluator（pilot主対象）

- input: initial 4-frame DEM height+valid mask、pre-noise Actor state、candidate Path/direction、initial robot body state。
- target: 1/2/5秒progress、各horizon内near-fall/tip-over/linear-stall/turn-stall、event-free time、severity/recovery。terminal short traceはcensored label。
- 同一StateGroup内で安全性・progressのpairwise rankingを作る。
- 同一base seed由来の4env tileと全candidate/windowは常に同一data splitへ置く。
- pilot規模では正式test性能を宣言しない。まずevent counts、pair consistency、自動ラベル対trace review、極小baselineが学習可能かの診断に限る。

### 経路生成（後段）

成功・目標向きの経路とサブゴールを教師にする。危険候補はEvaluator用ラベルとして保持し、成功経路模倣へ混ぜない。v0.2にはglobal goal到達ラベルを定義していないため、このpilot単独でgoal-conditioned path generatorを学習したとはしない。

## 6. 次段階

このpilotと後続のbaseline比較学習・探索的holdout評価は完了。未使用seedでの正式test・実機評価は未実施。失敗誘発は行わず、取得した自然eventとpaired outcomeを次の順で検証する。

1. **数値trace監査済み。** 5 tip-over、6 near-fall、2 turn-stall、36 linear-stall windowの根拠と保存v1ラベルを照合し、27 event trajectoryと対照16本の図を確認した。独立した人間による全件annotationは未実施。stall windowは独立事故として数えない。
2. **学習仕様承認・教師v2実装済み。** [転倒・スタック学習仕様v0.1](fall-stall-learning-spec-v0.1.md)に入力、10秒教師、censor、seed分割、モデル、loss、評価、追加収集条件を定義した。
3. **教師v2差分監査・初期入力dataset生成・seed split固定済み。** 原本と既存教師を保持し、新しいdatasetへ48 StateGroupと192候補を保存した。
4. **baseline比較完了（2026-10-20）。** 固定datasetのtrain-only正規化と4方式の比較を実行した。18学習runと6定数baseline、checkpointと全指標は[比較報告](fall-stall-baseline-comparison-20261020/comparison-report.md)を参照。unsupported/N/Aを保持し、正式test性能は宣言しない。
5. **追加収集protocol固定済み（2026-10-20）。** [独立seed追加収集protocol v0.1](additional-passive-collection-protocol-v0.1.md)に24新seed、train/validation/reserved test、120 process、通常条件と非開封規則を固定した。manifest対応runnerとblind verifierを実装し、[追加収集完了報告](additional-passive-collection-20261020/collection-report.md)のとおり全120/120 process・384候補を収集済み。test意味情報は未開封。2026-10-21に[train/validation教師v2・datasetとcoverage監査](additional-development-dataset-20261021/dataset-report.md)を完了した。batch01時点では全headで開発側coverage未達のため正式比較はN/A。[batch02の事前freeze](fall-stall-additional-batch02-freeze-20261021/freeze-report.md)も完了した。[batch02実行配線とbinding固定](batch02-runtime-binding-20261021/runtime-binding-report.md)も完了した。[固定bindingによるbatch02受動収集](additional-passive-batch02-collection-20261021/collection-report.md)も120/120 process・384候補で完了し、全sealと別プロセスblind再検証がpassした。[batch02教師化と累積dataset・coverage監査](additional-development-batch02-cumulative-20261021/dataset-report.md)も完了した。累積128 StateGroup・512候補でlinear-stallは開発側coverageがpass、他3 headは未達。test coverageはpendingで、本学習は未実施。[batch03のprotocol/manifest事前固定](fall-stall-additional-batch03-freeze-20261021/freeze-report.md)も完了し、次はruntime/binding準備。未使用seedによる正式test、未知地形・実機評価は別に行う。

## 7. Stop/coverage rule

- container/resource/numeric/protocol failureは物理eventと別コードで記録し、そのtrialを隠さない。
- event labelsが自動生成されない、termination pre-stateが失われる、candidate間初期状態がgate不一致なら本収集を停止する。
- 追加収集が必要な場合も、独立seedと自然な地形条件を通常操作で観察する。地形/level/entry position/flipper conditionを追加する場合は、研究対象の運用範囲内で系統的に記録し、極端な初期化・物理変更・故意の危険操作でeventを誘発しない。変更は新protocol revisionでfreezeする。
- eventが少ないまま危険例を重複oversampleして独立データのように数えない。
- pilot event yieldが十分でも、未知地形/実機への一般化は別のseed-held-out/実機評価が必要。

## 8. 変更・実行履歴

- v0.1: ブレインストーミング後の初期仕様。
- v0.2: 2026-10-19にpilot条件をfreeze。追加したsim runner optionは `--actor-env-yaml`, `--control-steps`, `--replan-every-steps`。既定40-step protocolは維持。収集条件の凍結後に結果を追記したため、pilot manifestは上記frozen protocol SHAを参照し、現在のこの文書SHAとは異なる。
- RayCasterはIsaac Sim 5.1の公開 `ray_starts` と `data.pos_w` を用いるようcollectorを修正。同branch GUIと同じray origin式。
- Smoke outputはcontainer内 `/tmp/eatnav-pilot-preflight/`、本pilot evidenceは上記Onix bind mount下の新規rootに置いた。共有archive/sealed evidenceやOnix学習runは変更していない。
- v0.2は「失敗を起こす」のでなく「通常走行で失敗条件を知る」受動観察仕様。event label builder、reset前snapshot、host verifier、60process本pilotは完了。
- 2026-10-19後続作業: 全trace数値監査・図の確認、seed分割案、転倒・スタック学習仕様v0.1を追加。回復候補の誤記とpilot Actor env SHAを訂正し、stall定義との差・テスト収集件数を記録した。独立した人間による全件annotation、教師v2実装、学習、held-out統計評価、実機検証は未実施。
- 2026-10-19仕様承認後: 教師v2を別moduleへ実装し、全192 trajectoryで旧v1との差分を監査。イベント数36/6/5/2は一致し、6 confirmation時刻と40 partial-cycle unknownを記録。初期入力dataset化・学習・held-out統計評価・実機検証は未実施。
- 2026-10-19次段階: 初期入力dataset 48 StateGroup×4候補と3通りのseed rotationを新規artifactへ固定。全60 processと教師v2を再検証し、未知ラベルmaskを保持した。normalization fit・学習・held-out推論・実機検証は未実施。

## 参考

- [ROS 2 + Isaac GUI demo handoff](ros2-isaac-gui-demo-student.md)
- [既存14候補収集protocol](motion-predictor-vnext/motion_14_collection_protocol.md)
- [初期pilot仕様 v0.1](paired-long-horizon-pilot-spec-v0.1.md)
- [GitHub handoff branch](https://github.com/TohokuQuix/visualnav-transformer-ros2/blob/exp/eatnav-ros2-gui-demo-terrain/docs/eat-nav/ros2-isaac-gui-demo-student.md)

- 2026-10-20後続作業: 仕様task 3のbaseline比較を完了。探索的linear-stall APはState 0.278 / DEM 0.410 / prior 0.111（fold・最適化seedの記述平均）。転倒の評価可能な陽性は1候補・1 seedで、追加独立seedのcoverageが必要。原本・教師・固定datasetは維持。追加収集・正式test・実機評価は未実施。

- 2026-10-20 task 4: 独立seed追加収集protocol v0.1と収集前manifestを固定。既存48 seed/576 sub-seedと重複のない24新seed/288 sub-seedを生成し、train12 / validation4 / reserved test8へ割当てた。原本・既存artifactは維持し、追加収集は開始していない。

- 2026-10-20追加収集実行: runner・test非開封verifierとrole-aware adapterを実装し、固定24新seedから120/120 process・384候補を受動収集。全stage parity・試行/batch seal・別プロセスblind再検証がpass。関連156 tests passed、全体suiteは既知依存不足で28 collection error。testの意味情報、追加教師化、coverage監査、本学習、正式test・実機評価は未実施。

- 2026-10-21追加データ教師化: train/validationのみ64 StateGroup・256候補を教師v2と固定role datasetへ変換。陽性base seed数はtrain/validationでtip-over 4/1、near-fall 4/1、linear-stall 5/2、turn-stall 1/0。全headで最低coverage未達、正式比較はN/A、test coverageは未開封pending。関連69 tests、全horizon教師/mask対応、独立再exportのbyte一致がpass。本学習は未実施。詳細は[開発dataset・coverage報告](additional-development-dataset-20261021/dataset-report.md)を参照。

- 2026-10-21 batch02固定: [protocol v0.2](additional-passive-collection-protocol-v0.2.md)・[manifest](fall-stall-additional-batch02-freeze-20261021/artifacts/collection-manifest.json)へ24新seed、train12/validation4/reserved test8、120 process・384候補を固定。既知72 seed/864 sub-seedとの非重複と全9 artifactの別directory再生成byte一致を確認。batch01との同role累積pool規則も固定し、test意味情報は未開封。batch02対応runtimeと実行bindingは未準備でready_to_collect=false、収集・本学習は未実施。詳細は[固定報告](fall-stall-additional-batch02-freeze-20261021/freeze-report.md)を参照。

- 2026-10-21 batch02実行準備: 歴史的なbatch01コードのbyteを保持してbatch02用runner・CLI・開発専用adapterを追加。実containerのversion/CUDA/GPU、Onix43 source、staged267 source/config/Actorの一致と別プロセス再検証がpass。新規25＋既存69の計94 tests passed。実行binding SHAは17f77664625cca5501acd251ddf62f49cbd2fabf200983fda8e77be0e2b2cfe5。収集前manifestはsnapshotのまま保持し、今回のpreparationで実行準備済みと記録。起動時のruntime/資源/source再検証を必要とし、今回の収集・本学習・test意味情報開封は未実施。詳細は[実行準備報告](batch02-runtime-binding-20261021/runtime-binding-report.md)を参照。

- 2026-10-21 batch02収集完了: 固定binding（17f77664625cca5501acd251ddf62f49cbd2fabf200983fda8e77be0e2b2cfe5）のまま24新seed・120/120 process・384候補を受動収集。全120 stage parity・inner/outer receipt・batch seal、別プロセスblind verificationがpass。protocol failure・自動retryは0件。raw rootは`.eatnav-fall-stall-additional-v0.2-batch02`、約128.97分・563.4 MB。testの意味情報は未開封で、batch02教師化・dataset結合・coverage監査・本学習・正式test/実機評価は未実施。詳細は[batch02収集完了報告](additional-passive-batch02-collection-20261021/collection-report.md)を参照。

- 2026-10-21 batch02開発dataset・累積監査: train/validationのみ64 StateGroup・256候補を教師v2化し、batch01と同roleで結合して128 StateGroup・512候補を固定。累積陽性base seed（train/validation）はtip-over 10/1、near-fall 7/1、linear-stall 8/4、turn-stall 2/1。linear-stallのみ開発側coverageがpassし、testは全head未開封pending。既存mask・入力・教師・出典を保持し、97 testsと別生成byte一致・独立レビューを確認。本学習・正式test・実機評価は未実施。詳細は[累積dataset・coverage報告](additional-development-batch02-cumulative-20261021/dataset-report.md)を参照。

- 2026-10-21 batch03事前固定: [protocol v0.3](additional-passive-collection-protocol-v0.3.md)・[manifest](fall-stall-additional-batch03-freeze-20261021/artifacts/collection-manifest.json)へ24新seed、train12/validation4/reserved test8、120 process・384候補を固定。既知96 seed/1152 sub-seedとの非重複と全9 artifactの別生成byte一致がpass。走行条件・教師・cell別3/1/2とtest非開封を維持し、3 batchの累積pool規則を記録。batch03 runtime/binding・収集・本学習は未実施。詳細は[固定報告](fall-stall-additional-batch03-freeze-20261021/freeze-report.md)を参照。
