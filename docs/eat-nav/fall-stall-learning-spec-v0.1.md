# 転倒・スタック学習仕様 v0.1

- 作成日: 2026-10-19
- 状態: 承認済み。教師v2実装・差分監査、48 StateGroupの入力dataset化、seed分割固定を完了。学習実装・学習・held-out推論・実機評価は未実施
- 対象: 受動観察で収集した同一初期状態の候補比較によるEvaluator学習
- 収集元: [paired long-horizon pilot v0.2](paired-long-horizon-pilot-spec-v0.2.md)
- 根拠: [全trace数値監査と時系列図の確認結果](paired-long-horizon-review-20261019/trace-review-report.md)
- 新しい派生教師schema: `eatnav_fall_stall_targets_v2`（実装・pilot差分監査済み）
- 新しい学習入力schema: `eatnav_fall_stall_input_v1`（dataset化・検証済み）

## 1. 目的と今回の決定

同じ初期StateGroupから4つの通常候補経路を実行した結果を使い、**初期状態と候補経路から、その候補IDを10秒継続した際の転倒・並進停滞・旋回応答不足を予測するEvaluator**を作る。危険候補を区別できるかを、定数予測・state-only・DEMを含むモデルの比較で診断する。

ユーザーの意図は、受動観察pilotの結果を検証して転倒・スタック学習へ進める仕様を完成させること。転倒を起こす操作、初期姿勢変更、物理変更、危険イベントの水増しを追加しない。本書作成までに行ったことは原本再検証、数値trace監査、図の確認、分割案作成である。モデルやラベルbuilderの実装変更はこの作業に含めていない。

その後、利用者の仕様承認と実装依頼により、教師v2の実装と差分監査まで完了した。結果は[教師v2差分監査報告](fall-stall-targets-v2-audit-20261019/audit-report.md)を参照。旧v1 builderと原本は保持し、別moduleと新規派生directoryで実装した。

さらに利用者の依頼により、[初期入力dataset生成とseed分割固定](fall-stall-dataset-freeze-20261019/dataset-report.md)を完了した。未正規化の48 StateGroup×4候補と教師maskを保存し、3通りの探索的rotationを固定した。モデル学習は未実施。

検討した方式と決定は次のとおり。

| 方式 | 得られることと制約 | 決定 |
|---|---|---|
| 初期入力→10秒のcandidate outcome | 保存済み初期DEMと全trajectoryが対応し、5転倒を陽性にできる。先の局所地形が初期DEM外に出る制約がある | **最初の学習対象** |
| 各時刻入力→1/2/5秒先のevent | 直前予兆の学習に向くが、現pilotには各時刻対応のDEM履歴がない | 追加収集の後段 |
| recovery Actor・危険例生成 | 別の介入実験・policy学習が必要 | 本仕様の対象外 |

予測対象は「2秒ごとに同じcandidate classを現在poseから再計画する固定継続policy」のoutcomeである。2秒後に別candidateへ変更するpolicyの性能ではない。Actorの再学習、経路生成器、global goal到達、安全な復帰動作の生成は扱わない。

## 2. Pilotから確認できたこと

原本rootは`/home/ubuntu/Projects/onix_isaac_lab/.eatnav-paired-long-horizon-pilot-v0.2-20261019-01`。60 processの内外seal・decision・pre-Path gateが再検証を通過した。192 trajectory、48 StateGroup、12 base seed、37,452 observation windowである。172本完走、5本tilt termination、15本parallel censor。

| 保存v1イベント | Window | Trajectory | Base seed |
|---|---:|---:|---:|
| linear-stall | 36 | 19 | 5 |
| near-fall | 6 | 6 | 4 |
| tip-over | 5 | 5 | 3 |
| turn-stall | 2 | 2 | 2 |

転倒5本は全て開始5秒より後（5.45–9.50秒）。初期時刻から1/2/5秒先の転倒陽性は0本である。途中windowを37,452個作っても、初期入力に対する独立した転倒教師が増えるわけではない。

near-fall後に完走したのはK-Rail L4 / seed `2788660503` / reverse Direct / env2。数値traceではonset後5秒以内の経路投影progress増分約0.815 m、tip-overなし、傾斜回復と並進再開を確認した。回復候補1本だけでrecovery分類headを学習しない。

保存v1に限った10秒教師のsupportは次のとおり。新v2教師への更新後に必ず再集計する。

| Event | Positive | Negative（10秒観測） | Unknown |
|---|---:|---:|---:|
| tip-over | 5 | 172 | 15 |
| near-fall | 6 | 171 | 15 |
| linear-stall | 19 | 154 | 19 |
| turn-stall | 2 | 170 | 20 |

これらはtrajectory数であり、独立事故数ではない。全データは仕様設計・ラベル確認に使ったため、今から分ける部分も正式な未閲覧testとは呼ばない。

## 3. 原本・派生データ・監査の契約

### 3.1 原本を読み込む条件

adapterは次を全て確認し、一つでも満たさなければdataset生成を停止して理由を出力する。

1. pilot manifest、assignment、各host manifest、decision、verification、内外SHA ledger/seal、completeの整合。
2. terrain、level、seed、env index、candidate、Actor checkpoint/env YAML、config、collector、follower、protocolのSHAと時間設定の整合。
3. candidate間で同一env indexをpairし、pre-Path gate pass。4 envを同じStateGroupに潰さない。
4. traceのstep連続性、20 Hzのstamp差、cycle境界、要求200 step、自然termination/censor理由、reset前stateの存在。
5. initial DEM・state・Pathのshape、frame、有限値、mask、入力時刻。未来の値を初期入力へ混ぜない。

本pilotのActor env SHAは`14fc2441d928a92b1ba45c06ccd8603eb0cf71aebb5d216efc4005f09696545c`、Actor checkpoint SHAは`b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d`。protocolのfrozen SHAは`1df876cc86e749dbff56b9b81ab1fd10e2ad69790f576bbda08db0c82844d9a7`。収集後に追記された仕様書の現在SHAを、原本のprotocol SHAと比較して棄却しない。

### 3.2 派生出力

学習用adapterの出力を新規directoryに保存する。原本・旧v1教師へ上書きしない。

| Artifact | 内容 |
|---|---|
| `dataset-manifest.json` | source root/manifest SHA、全source SHA、input/target schema、定義revision、adapter SHA、作成環境 |
| `samples.npz` | StateGroupごとの初期入力、4 candidate、target/mask。pickleを使わない |
| `sample-index.json` | 各candidateのseed/env、原本trace・label path/SHA、decision時刻、終了理由、追跡ID |
| `targets.jsonl` | event onset/confirmation、cycle根拠、horizonごとの値・観測・mask・unknown理由、recovery診断 |
| `label-diff.json` | v1/v2の変更前後、理由、件数、該当trajectory/cycle。変更なしも明示 |
| `review.csv` | auto labelと数値監査の照合、信号不足、曖昧例、独立した人間による確認状況 |
| `split.json` | base seedを明示した分割と各headの正負/unknown support |
| `sha-ledger.json` | 上記派生成果物のSHA。学習runはこのledger SHAを固定 |

datasetの1 sampleは1 StateGroup、4 candidateを持つ。現pilotは48 sample / 192 candidate outcomeが上限。途中windowは監査用として保存し、初期入力を複製した別sampleにはしない。原本にない候補を陰性で埋めず、candidate maskをfalseにする。

教師v2実装段階の[派生成果物](fall-stall-targets-v2-audit-20261019/artifacts/dataset-manifest.json)は`teachers_and_differential_audit_only`であり、`targets.jsonl`、差分、数値根拠、監査図、manifest/sealを保存した。`samples.npz`・`sample-index.json`・`split.json`による入力dataset化は後段で行う。

その後の[入力dataset manifest](fall-stall-dataset-freeze-20261019/artifacts/dataset-manifest.json)は`initial_paired_learning_dataset`。上記の教師artifactを保持したまま、新規directoryに`samples.npz`、`sample-index.json`、`split.json`と教師コピー、検証記録、SHA/sealを保存した。入力datasetの仕様と配列順は[生成報告](fall-stall-dataset-freeze-20261019/dataset-report.md)に記録した。

## 4. 時刻と教師ラベル v2

### 4.1 共通ルール

decision時刻はpath activation時刻（本pilotではlogical time 1 s）。row iの`root_state`は時刻`i×0.05 s`、post-stateは`(i+1)×0.05 s`として扱う。後処理の将来区間はdecisionから`(0,H]`。最初のpost-action sampleも含める。Hは1/2/5/10秒、主学習は10秒。

自然termination時は`terminal.pre_reset_state`を優先する。reset後のroot/joint/contact/body velocityを同一episodeに接続しない。終端body angular velocityはreset前rootのworld angular velocityを同じquaternionでbody frameへ変換する。snapshotにない必須信号はunknown。parallel env censorはphysical failureとは別コードのまま保持する。

全イベントで`label ∈ {0,1,null}`と`target_mask`を別に保存する。区間内にイベントを確認したら1、区間全体を観測してイベントなしなら0、それ以外はnull/mask=false。別イベントで早期終了しても未観測の陰性を補わない。`observed_until_sec`と各eventの`reason`を保持し、horizon共通の一つのcensoredフラグだけで学習を制御しない。

### 4.2 転倒・near-fall

- quaternionはfiniteかつnormが0でないことを確認し正規化。`tilt = acos(clamp(1−2(x²+y²),−1,1))`。
- `tip_over`: tilt >=85°、またはreset前termination理由`tilt_over`。最初の到達sampleをevent時刻とする。
- `near_fall`: tilt >=60°の連続5 control sample。onset時刻と5番目のsampleのconfirmation時刻を両方保存する。future targetを1にする時刻は**confirmation**とし、horizon外のsampleを使って区間内のイベント成立を先取りしない。
- 5 sampleという離散条件を優先し、sample時刻間の幅は4 interval =0.20秒、5 control intervalの占有は0.25秒と区別する。
- tiltの最大値、60°以上のrun長、tip-over時刻、終了理由も監査用に残す。near-fallとtip-overを排他的なclassにしない。

### 4.3 Linear-stall（並進停滞）

これは「約2秒の指令に対して移動と経路進捗が小さい」という操作的定義。脱出不能な物理拘束の断定ではない。各cycleに揃った40 control intervalについて以下を全て満たす場合を1 windowとする。

1. `D_cmd = Σ abs(follower_command.vx) × 0.05 >= 0.20 m`。
2. `D_actual = Σ norm(post_xy − pre_xy)`、`D_actual / D_cmd <= 0.25`。
3. cycle開始時のPathにpre/post XYを投影し、`best_remaining(t) = min_{u<=t} remaining(u)`を毎sample計算する。`best_remaining(start) − best_remaining(end) <= 0.05 m`。

event時刻は40番目のinterval終了時刻。cycleをまたいで積算しない。Pathは同じcycleの再計画poseとfrozen body Pathから復元し、frame変換を検証する。投影は各線分への最近点、同距離のtieは始点寄りを選ぶ。Path外では始終端へclampする。実移動距離はworld XY、進捗はPath arc座標である。

現followerの`best_remaining`は0.10 m超の改善時しか更新しないため、そのdiagnostic値をこの連続bestの代わりに使わない。旧builderの始終remaining差、1.95秒のdiagnostic積算も新定義の代わりにしない。40 interval未満・信号不足・Path復元失敗はunknownとする。

36 windowという旧集計は`linear_stall_v1`として保存する。v2の変更数と全差分を出し、図と信号で再監査してから教師をfreezeする。今回確認した「diagnostic best条件なら5 window追加」は、このv2の最終件数ではない。

### 4.4 Turn-stall（旋回応答不足）

同じ40 intervalで、全sampleの`abs(follower_command.wz) >= 0.01 rad/s`、かつ対応するpost-stateの`abs(body_omega_z) <= 0.5 × abs(follower_command.wz)`なら1 window。途中で条件が切れるcycle、信号不足、partial cycleは陰性またはunknownを区別する。全40 intervalを観測し全信号があれば条件不成立は0、足りなければnull。

これはyaw-rate応答のheadであり、linear-stallと合算して「移動不能」教師にしない。今回の2例は並進progressが約0.45 m/2 s以上ある。body omega zとworld heading yawの微分を混同しない。

### 4.5 Recoveryとevent-free

`recovered`は当初は**監査用metadataのみ**。near-fallはconfirmation後、stallはwindow終了後を起点に5秒を観測する。同じcandidateの走行を続け、Path方向progressが0.10 m以上増加し、以後5秒以内のtip-overがなければtrue。tip-overしたらfalse。5秒未観測・progress不足を判定できない場合はnull。candidate変更・controller recovery開始・外部操作は別フィールドに残す。

progressは各cycle固定Pathの投影arc増分をcycle境界で接続し、後退は負の増分として残す。境界のreplan自体で0.75 mを加算しない。物理的並進がないreplanだけで回復扱いにならないことを検証する。診断図と数値を照合し、pilotの回復候補を新定義で再判定する。

`event_free_10s`は10秒を全観測し、全event headが既知で全て0の場合のみtrue。既知のeventが一つでもあればfalse、残りはnull。event-freeの10秒完走とglobal goal成功は区別する。

## 5. 入力契約と情報漏洩の防止

入力はdecision時刻以前だけから作る。

| 入力 | Shape（B=StateGroup、K=candidate） | 内容 |
|---|---|---|
| DEM | `[B,4,2,20,20]` | height[m] + valid mask、0.1 m grid。既存prepath.npzの4-frame history |
| Actor state | `[B,22]` | `actor_pre_noise`。列順はActor env/schemaとSHAで固定 |
| Body state | `[B,9]` | body projected gravity 3、body linear velocity 3、body angular velocity 3 |
| Candidate Path | `[B,K,8,2]` | decision時のbody-yaw frame、単位m |
| Direction | `[B,K,1]` | forward +1 / reverse −1 |
| Candidate mask | `[B,K]` | 実在し有効な候補のみtrue |

Body stateはprepath root quaternionから`R_world_from_body`を計算し、`Rᵀ[0,0,−1]`、`Rᵀv_world`、`Rᵀω_world`をこの順に連結する。世界座標x/y、env tile位置、seed、terrain ID/level、candidate IDの任意整数、将来cycle、termination、reward、Critic、将来contact/effortは入力にしない。必要なflipper姿勢等がActor22列にどう含まれるかはActor obs契約で確認し、列を推測で入れ替えない。

DEMは既存evaluatorのheight/mask処理を参考にするが、metadataの`robot_base`という文字列だけでyaw frameと断定しない。ray座標、root yaw、Path変換のfixtureで同じ基準であることを確認する。frameや時刻を確認できないsampleは棄却する。

正規化はtrain seedのみでfitし、checkpointに保存する。mask=0のheightは入力値0へ置換、maskは維持。既存preprocessingと同じく各frameの有効率25%未満、valid部分の非finite、nonbinary maskは入力不正として棄却。全candidate無効は明示エラー。candidateの順を変えると出力も同じ順に変わること。

初期DEMを途中時刻へ再利用しない。初期局所DEMの外へ出た地形は観測されていないので、10秒予測には不可避の情報不足がある。DEM追加による改善がない場合も結果として残し、Actor状態や地形IDによる代替説明と区別する。

## 6. 最初のモデルと学習方法

### 6.1 比較モデル

| モデル | 入力 | 目的 |
|---|---|---|
| Always-event-free | 全event logitを陰性に固定 | 陰性多数でaccuracyだけが高くなる罠を可視化 |
| Train-prior | trainの既知ラベル頻度だけを予測 | 定数確率baseline |
| State + candidate | Actor22 + Body9 + Path + direction | DEMの寄与を切り分ける |
| DEM + state + candidate | 全入力 | 主モデル |

既存`src/visualnav_transformer/eat_nav/evaluator/`の前処理・CNN+MLP・checkpoint契約を再利用候補とする。既存scalar progress checkpointをrisk checkpointと互換扱いしない。新しいdataset adapterとheadは新schemaで作る。

主モデルはDEMを8chへ展開し、Conv(8→16,3,pad1)→ReLU→AvgPool2、Conv(16→32,3,pad1)→ReLU→AvgPool2、Conv(32→32,3,pad1)→ReLU→AdaptiveAvgPool1→Linear(32,64)→ReLU。state31列はLinear(31,32)→ReLU、Path16列+directionはLinear(17,64)→ReLU→Linear(64,32)→ReLU。共有featureをcandidate別に連結し、Linear(128,64)→ReLU→Linear(64,32)→ReLUから各headへ出力する。State-onlyはDEM64列を除く64列をfusionの入力とする。

- 主出力: `[B,K,4]` の10秒event logits、順番はtip-over、near-fall、linear-stall、turn-stall。
- 補助出力: `[B,K,4]` の1/2/5/10秒Path progress[m]。その時刻まで観測できたものだけを教師にする。転倒で早期終了した10秒progressはnull。
- 1/2/5秒event head、severity回帰、recovery headは最初のモデルでは学習しない。教師metadataとして保存する。

### 6.2 Lossと重み

主lossは既知ラベルだけのBCEWithLogits。trainの各headに正負両方がある場合だけ有効化し、`pos_weight = min(n_negative/n_positive, 20)`とする。重みはtrainからのみ計算。正負どちらかが0ならそのheadを`unsupported`にし、予測確率や性能を学習済みとして出さない。

集約順は、headごとの有効candidate平均→StateGroup平均→base seed平均→有効head平均。4 envや多数の陽性windowを持つseedが重くならないようにする。window単位のoversamplingはしない。

`L = L_event + 0.1 L_progress + 0.1 L_pair`。progressはtrainの既知値のmean/std（std floor 0.001 m）で標準化したSmoothL1(beta=1)。同じStateGroup・同じheadで両ラベルが既知かつ異なる候補のみ、危険候補jのlogitが安全候補iより大きくなるよう`softplus(logit_i−logit_j)`をpair lossにする。pairなしのgroupではpair項を0とし、候補ごとの絶対event教師は残す。progressを候補横断のgoal成功順位の教師にはしない。

初回設定はAdam、lr=1e-3、weight_decay=0、batch=全train StateGroup（提案foldでは16）、最大300 epoch、schedulerなし、seed=123/456/789の3回。validationで有効なevent headのunweighted BCE平均が最小のepochを選ぶ。同値なら早いepoch。validationに正負が揃うheadが一つもなければ学習runを中止し、固定epochを後付けで選ばない。分類閾値は診断用0.5固定。重み付きBCEのsigmoid値を校正済み事故確率とは呼ばない。

pair-lossとprogress-lossのmaskが空ならその項を0にし、NaNを作らない。forward/backward、保存/reloadの一致、異schemaの拒否を確認してからrunを受理する。

## 7. 分割と探索的評価

base seedの全4 env・全candidate・全windowを必ず同じroleへ置く。trainで正規化・class weightをfit、validationでepochを選択、exploratory holdoutはその後に一度集計する。

[split-proposal.json](paired-long-horizon-review-20261019/split-proposal.json)は、各terrain×levelのreplicate 0/1/2を結果で並べ替えず3回rotationする。

現在は同じseed集合を[split.json](fall-stall-dataset-freeze-20261019/artifacts/split.json)へ固定済み。各roleのsample ID/index、head別positive/negative/unknownとStateGroup/base seed support、unsupported理由を保存した。原proposalとのseed集合一致とsplit漏洩なしを検証した。

- fold f: holdoutはreplicate f、validationは(f+1)%3、trainは(f+2)%3。
- 各roleは4 base seed、16 StateGroup、64 candidate trajectory。
- これは設計時に全traceを見た後の探索的分割。正式testは今後収集する未使用seedに設ける。
- fold1のtrainは転倒・near-fall正例0、fold0のholdoutは転倒・near-fall正例0。該当head/recall/AUPRCはN/Aとし、seedを移して帳尻を合わせない。
- 現規模で3fold全ての転倒学習を成立させるという合格条件は置かない。unsupportedを含めて結果を残し、追加収集の判断材料にする。

評価結果には必ずbase seed / StateGroup / trajectory単位のsupportを添える。

| 評価 | 報告内容 |
|---|---|
| 転倒・near-fall・各stall | 既知targetだけのTP/FP/TN/FN、見逃し数とrecall、precision、false positive rate、PR-AUC |
| 確率出力の診断 | Brier scoreと信頼度別の件数。希少eventで校正済みと結論しない |
| Candidate ranking | 同じStateGroupの既知かつ異なるeventラベルのpair agreement、pair数、該当seed数 |
| 選択の診断 | tip-over headがsupportされ、4候補の転倒教師が既知のgroupだけで、最小logit候補の観測転倒数。tieはfrozen candidate順 |
| Progress | 各horizonのMAE[m]、mask件数。candidate固有経路の進捗でありgoal到達ではない |
| 情報不足 | unknown/censor率、unsupported head、対象外group、入力不正、初期DEM外走行の限界 |

PR-AUCは正負両方がある場合のみ。macro平均にN/Aを0で混ぜず、有効fold/head数を出す。全windowを独立とした信頼区間は計算しない。pilotではpoint estimateと原数を中心にし、正式評価時にbase seed単位の不確実性評価を別途freezeする。

安全性・progressを一つの恣意的な報酬へ潰さない。最初は各riskとprogressを並べて報告し、「どれだけ進捗を失って危険候補を避けたか」を記述する。オンライン選択ルールや許容危険確率の設計は本pilotから決めない。

## 8. 追加収集の仕様

2026-10-20に[追加収集protocol v0.1](additional-passive-collection-protocol-v0.1.md)と[収集前manifest](fall-stall-additional-collection-freeze-20261020/artifacts/collection-manifest.json)を固定した。24新seed（train12 / validation4 / reserved test8）、120 process / 384候補を事前割当てした。既存pilotとMotion14の48 seedとの非重複、testの非開封条件、source/Actor/教師SHAを記録した。その後、manifest対応runner・blind verifier・role-aware教師/dataset adapterを実装し、全120/120 process・384候補を受動収集した。全試行とbatchのseal・別プロセス再検証はpass。reserved testの意味情報は未開封。2026-10-21にtrain/validationの64 StateGroup・256候補を教師v2・datasetへ変換し、[coverage監査](additional-development-dataset-20261021/dataset-report.md)を完了した。batch01時点では全4 headでtrain/validationの最低coverageは未達であり、追加データでの本学習は未実施。収集の詳細は[追加収集完了報告](additional-passive-collection-20261020/collection-report.md)を参照。

2026-10-21に[batch02 protocol v0.2](additional-passive-collection-protocol-v0.2.md)と[batch02 manifest](fall-stall-additional-batch02-freeze-20261021/artifacts/collection-manifest.json)も固定した。24新seed・120 process・384候補を同じ3/1/2比率で事前割当てし、既知72 base seed・864 sub-seedとの非重複を確認した。batch01＋batch02の同role累積poolとcoverage規則を記録し、testは非開封のまま保持する。続いてbatch02用runner・blind検証・開発専用adapterを実装し、[実containerのbinding固定と別プロセス再検証](batch02-runtime-binding-20261021/runtime-binding-report.md)まで完了した。固定bindingによりbatch02も120/120 process・384候補を受動収集し、全sealと別プロセスblind再検証がpassした。train/validation教師化と同role結合も完了し、累積poolはtrain24 / validation8 seed・128 StateGroup・512候補となった。陽性base seedはtip-over 10/1、near-fall 7/1、linear-stall 8/4、turn-stall 2/1で、linear-stallのみ開発側coverageがpass、他3 headは未達。testはpendingのまま保持する。[累積dataset・coverage報告](additional-development-batch02-cumulative-20261021/dataset-report.md)を参照。[収集完了報告](additional-passive-batch02-collection-20261021/collection-report.md)を参照。詳細は[固定報告](fall-stall-additional-batch02-freeze-20261021/freeze-report.md)を参照。

2026-10-21に[batch03 protocol v0.3](additional-passive-collection-protocol-v0.3.md)と[manifest](fall-stall-additional-batch03-freeze-20261021/artifacts/collection-manifest.json)も事前固定した。24新seed・120 process・384候補を従来のcell別3/1/2へ割当て、既知96 seed・1152 sub-seedとの重複0を確認。3 batch完了後の同role累積poolはtrain36 / validation12 / reserved test24 seed、開発側192 StateGroup・768候補を予定する。batch03の収集・runtime/binding準備は未実施でready_to_collect=false。詳細は[固定報告](fall-stall-additional-batch03-freeze-20261021/freeze-report.md)を参照。

最初の実装・baseline診断で不足を把握した後、通常条件の独立seedを追加する。収集前に新protocolをfreezeし、pilot seedとは重複させない。

最初の追加batch案は現在の4 terrain×level cellごとに6 base seed、計24 seed。4 env×4 candidateで384 trajectory、pre-Pathを含め120 process。pilot実測の比率なら約130分・約562 MBだが、環境やterminationで変わる見積りであり実行実績ではない。受動eventが得られる保証はない。

seedをcell内の事前決定順にtrain 3 / validation 1 / reserved test 2へ割り当て、seed一覧・generator・Actor・candidate順・終了規則・label revisionを収集前manifestへ固定する。reserved testのtrace・event集計を設計調整に使わない。全件seal/protocol verificationは行い、ラベル意味の調整はtrain/validationで行う。

正式モデル比較へ進むための**運用上の最低coverage**は、主headのtrainに陽性を含むbase seedが5以上、validationに3以上、reserved testに3以上あり、各splitに既知陰性もあることとする。これは統計的十分性や安全性の保証ではない。上記1 batchはvalidation seedが合計4なので、自然eventの頻度次第で満たさない。

coverage未達なら正式比較をN/Aのままにし、追加batchを事前割当てで収集する。結果を見て陽性seedを別splitへ移動しない。reserved testを開いた後に閾値・モデル・ラベル定義を変更したら、その集合を以後validation扱いとし、新たな未使用test seedを用意する。希少なturn-stallは独立にcoverageを判断し、転倒headのcoverageで代用しない。

途中時刻→1/2/5秒予測を行う場合は別protocolで、各decision時点の4-frame DEMとstamp、pre-noise Actor/body state、Path、direction、cycle pose、controller状態を同時保存する。同じtrajectoryのwindowは同じbase seed splitへ置く。新decisionからcandidateを切替えるcounterfactualが必要なら、その状態での候補別paired rolloutを別に収集する。

## 9. 実装順と受入条件

この節は仕様の着手順。task 1は実装・差分監査済み、task 2はdataset生成・split固定済みで、task 3は[baseline比較報告](fall-stall-baseline-comparison-20261020/comparison-report.md)のとおり実装・実行済み。task 4の[独立seed追加収集protocol](additional-passive-collection-protocol-v0.1.md)を収集前に固定済み。runner・test非開封の検証処理を実装し、[追加収集](additional-passive-collection-20261020/collection-report.md)も120/120 processで完了した。追加データのtrain/validation教師化・dataset生成・[coverage監査](additional-development-dataset-20261021/dataset-report.md)まで完了した。batch02との[累積dataset・coverage監査](additional-development-batch02-cumulative-20261021/dataset-report.md)も完了し、linear-stallは開発側coverageがpass（正式testはpending）、転倒・near-fall・turn-stallは未達で正式比較N/A。本学習は未実施。

1. **教師v2 adapterと差分監査。** 原本seal検証、t0/interval契約、連続best、距離積算、終端snapshot、missing→unknownを実装。v1との差分を全件列挙し、回復候補と全event trajectoryの図・根拠を確認する。原本は変更しない。
2. **dataset/split生成。** 48 StateGroupを保ち、candidateとseedのリークがないこと、inputに未来信号がないことを確認。全schema/SHAとsupportをfreezeする。
3. **定数baselineと小型model。** 同じsplit/maskで4方式を比較。unsupported headを隠さず、classification・ranking・progressを別々に記録する。
4. **結果を見て追加batchをfreeze。** collector変更や未知地形評価を混ぜず、独立seedと運用範囲内の通常条件を受動観察する。

受入確認には最低限次のケースを含める。

- 60°近傍・5 sample未満/ちょうど5・85°・near-fall confirmationがhorizon境界を跨ぐ例。
- initial `(0,H]`の最初/最後のpost sample、200 interval完走、partialとparallel censor。
- 最終stepのreset後値が大きく変わってもreset前教師を使う例、snapshot不足のunknown。
- 2秒cycleの最終50 msを含む距離積算、replanだけではprogressを稼げない例、remaining始終差と連続bestが異なる例。
- 欠損距離・欠損yaw-rateを0として陰性/陽性にしない例。turn-stallでも並進できる例。
- 同じseedの別env/candidate/windowがsplitを跨がない例、正例0のhead/foldをunsupported/N/Aとする例。
- 全candidate無効、candidate順変更、train-only正規化、mask空loss、checkpoint round-trip/schema拒否。
- 既存`test_turn_stall_requires_continuous_command_and_low_body_yaw_response`をpytestが独立testとして収集すること。

仕様作成時の関連4ファイルは**8 passed**だった。教師v2実装時にturn-stall testのネストを解消し、指令の中断・応答比閾値超過の陰性を検証した。最新の関連テストと全体suiteの環境制約は[検証記録](fall-stall-targets-v2-audit-20261019/verification.txt)に分けて保存した。

## 10. 完了・保留の境界

本仕様、trace監査成果物、教師v2実装と差分監査は完成。受動収集の完了、原本整合、保存v1ラベルの再現、全192本のv2教師を確認した。v2のevent数は36/6/5/2でv1と一致し、near-fall確認時刻6箇所、partial cycleのstall unknown化40箇所が変わった。初期1/2/5/10秒の正負/unknown変更は0件。回復候補はconfirmation後5秒のprogress増分約0.914 mでrecovered=trueとなった。

初期入力datasetとseed splitも生成・固定済み。全48 StateGroup・192候補を保持し、各roleは4 seed / 16 StateGroup / 64候補。原本・教師v2 artifactを保持し、初期pre-noise状態、frame、mask、教師対応、seed familyと固定rotationを検証した。

2026-10-20にtask 3を完了した。固定datasetでtrain-only正規化をfitし、2学習方式×3 fold×3最適化seed=18学習runと6定数baselineを実行。各300 epochからvalidationのみでcheckpointを選び、探索的holdoutでclassification・ranking・progressを別に報告した。詳細・checkpoint・検証は[baseline比較報告](fall-stall-baseline-comparison-20261020/comparison-report.md)を参照。linear-stallのAP記述平均はState 0.278、DEM 0.410、prior 0.111だが、fold 2ではDEMが悪化し、転倒評価可能な陽性は1候補・1 seedだけである。unsupportedはN/Aのまま保持し、正式test性能や実機安全性を主張しない。

追加batch01も2026-10-20に完了した。24独立seed、96 StateGroup、384候補を固定role（train12 / validation4 / reserved test8 seed）のまま収集し、全120試行のstage parity・inner/outer sealとbatch sealを確認した。整合性passはevent-freeや性能向上の判定ではない。testのevent/termination/progressは未開封である。2026-10-21にtrain/validationの64 StateGroup・256候補を教師v2と固定role datasetへ変換した。陽性base seedはtrain/validation順にtip-over 4/1、near-fall 4/1、linear-stall 5/2、turn-stall 1/0で、train≥5・validation≥3の条件は全headで未達。test coverageはpendingのまま保持する。同日に通常条件・独立seedの[batch02 protocol/manifestを収集前に固定した](fall-stall-additional-batch02-freeze-20261021/freeze-report.md)。batch02も120/120 process・384候補を受動収集済みで、protocol failure・自動retryは0件。batch02用contract/coordinator・blind検証・開発専用exportの実装と関連94 tests、実containerのsource/Actor staging・binding固定・別プロセス再検証がpassした。[固定済みbindingによる収集と全件blind再検証](additional-passive-batch02-collection-20261021/collection-report.md)まで完了した。その後、batch02のtrain/validation教師化とbatch01との同role結合・単独/累積coverage監査を完了した。累積128 StateGroup・512候補で陽性base seedはtip-over 10/1、near-fall 7/1、linear-stall 8/4、turn-stall 2/1。linear-stallは開発側coverageがpassし、他3 headは不足を維持する。関連97 tests、教師/tensor全horizon対応、別生成byte一致がpass。追加例のtrain/validation trace reviewは残る。[通常条件のbatch03事前freeze](fall-stall-additional-batch03-freeze-20261021/freeze-report.md)も完了し、次はbatch03対応runner・blind検証/開発専用exportの回帰確認と実行時bindingの準備。batch03収集は未開始。[最新報告](additional-development-batch02-cumulative-20261021/dataset-report.md)を参照。testは意味情報・coverageとも未開封pendingで、本学習は未実施。原本・既存sealed artifactを保持し、関連69 testsと教師/tensor全horizon対応検証がpass。詳細は[開発dataset・coverage報告](additional-development-dataset-20261021/dataset-report.md)を参照。

独立した人間による全件annotation、追加データでの本学習、未使用seedでの正式test、未知地形評価、実機センサparityと実機安全評価は未実施である。

この順でまず確かめるのは「定義と入力を揃えたときに、候補条件付きの危険識別を学習できるか」である。pilotのevent発生、train lossの低下、同じseed内の順位一致だけで未知地形一般化・因果効果・実機安全性を結論しない。
