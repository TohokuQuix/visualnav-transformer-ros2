# 独立seedによる受動追加収集 protocol v0.2（batch02）

- 固定日: 2026-10-21（JST）
- Batch ID: `eatnav-fall-stall-additional-v0.2-batch02`
- 状態: **収集前protocol・seed・split固定。収集未実施。**
- 対応: [学習仕様v0.1](fall-stall-learning-spec-v0.1.md) task 4 / §8、[baseline比較結果](fall-stall-baseline-comparison-20261020/comparison-report.md)。
- 機械可読の正本: [freeze manifest](fall-stall-additional-batch02-freeze-20261021/artifacts/collection-manifest.json)、[seed一覧](fall-stall-additional-batch02-freeze-20261021/artifacts/seed-registry.json)、[全120 processの割当て](fall-stall-additional-batch02-freeze-20261021/artifacts/assignment.json)、[固定split](fall-stall-additional-batch02-freeze-20261021/artifacts/split.json)。この文書のbyte copyも同じsealに含む。

## 1. 目的と選択した範囲

初期入力から10秒先の候補条件付きrisk/progressを学習・評価するため、既存pilot・Motion14・batch01と重複しないseedで受動観察データを増やす。batch01の[train/validation監査](additional-development-dataset-20261021/dataset-report.md)では、陽性base seed数がtip-over 4/1、near-fall 4/1、linear-stall 5/2、turn-stall 1/0であり、全headで開発側coverageが未達だった。batch01のreserved testは未開封で、その結果を割当てに使っていない。baselineではDEM追加のlinear-stall APが平均で改善した一方、fold 2では悪化し、転倒の評価可能な陽性は1候補・1 seedだけだった。観測event数を増やすために転倒・スタックを誘発する操作、初期姿勢、物理、地形分布へ変更を加えない。

候補案は次の3つだった。

| 案 | 利点・影響 | 採否 |
|---|---|---|
| 同じ4 cellに各6新seed | 既存条件との比較を保ち、事前splitを固定できる | 本protocol |
| level 4へ割当てを偏らせる | 陽性が増える保証はなく、観測分布が変わる | 今回採用しない |
| 未知地形・途中時刻入力へ拡張 | 別の一般化や予測問題になりcollector変更が必要 | 別protocol |

この固定はデータ収集の仕様であり、正式なモデル・閾値・評価計画の最終freezeではない。未知地形評価、camera同期、オンライン候補切替え、recovery操作、実機走行は対象外。

## 2. Seedとroleの事前割当て

cell順は `k_rail_step_field/1`, `k_rail_step_field/4`, `pyramid_stairs/1`, `pyramid_stairs/4`。各cellのslot `0..5`に順に `train, train, train, validation, reserved_test, reserved_test` を割り当てる。結果でseedやroleを並べ替えない。pilotの3-fold rotationは新batchへ適用しない。

base seed生成はUTF-8 token `eatnav-fall-stall-additional-v0.2|batch02|{terrain_type}|{level}|{slot}` のSHA256先頭4 byteをbig-endian unsigned整数へ変換する。0、新規同士の衝突、既知seedとの衝突があればfreezeをエラー終了し、結果を見たseed引き直しはしない。今回の24値はseed一覧に固定する。

既知除外集合は既存長時間pilotの12 seed、既存Motion14 manifestの36 seed、[batch01固定registry](fall-stall-additional-collection-freeze-20261020/artifacts/seed-registry.json)の24 seedの和集合（72 base seed・864 sub-seed）。batch01のreserved testもseed IDだけを除外に用い、trace・ラベル等の意味情報を開かない。親freezeのledger SHAとregistry SHAを検証してから生成する。manifestの所在・SHA・列挙値をseed-registryへ保存する。探索済みsmoke等で別の既知seedが判明した場合は実行前に照合し、重複時は本batchを開始しない。この照合は既知集合についての確認であり、過去の全RL学習seedまで網羅した独立性の証明ではない。

各base seedのenv `0..3`、candidate `0..3`、全cycle/windowを同じroleに置く。terrain/command/initial streamは既存`EATSeedPlan`の `SeedSequence([base_seed, stream, env]).generate_state(1)[0]`（stream=0/1/2）を使い、計288 sub-seedも固定一覧へ保存。base seedとは別に、既知集合から計算したsub-seedとの非重複を検証する。同じseedであることだけでcandidate間の状態一致を仮定せず、実測pre-Path gateを使う。

| role | 新base seed | paired StateGroup | candidate-process | candidate trajectory | prepath込みprocess |
|---|---:|---:|---:|---:|---:|
| train | 12 | 48 | 48 | 192 | 60 |
| validation | 4 | 16 | 16 | 64 | 20 |
| reserved_test | 8 | 32 | 32 | 128 | 40 |
| 合計 | 24 | 96 | 96 | 384 | 120 |

batch02単独のsupportと、batch01＋batch02の累積coverageを別々に記録する。累積poolは両batchの同じroleの和集合とし、base seed・全env・全candidateを維持する。収集と全件integrity完了時の予定母数はtrain24 / validation8 / reserved test16 seed、計48 seed・192 StateGroup・768候補。開発側exportはtrain/validationだけとし、予定128 StateGroup・512候補。test側は開封前に陽性数・coverageを集計しない。pilotとMotion14はこのpoolへ混ぜない。実際のpool manifest・教師datasetは収集後に別途生成・sealし、本freezeを学習dataset完成と扱わない。

batch02のbase_groupは `batch02-{terrain_type}-level{level}-slot{slot}`、sample IDはその末尾へ `/env{env}` を付ける。batch01と結合してもsample IDが衝突しない。各データのbatch ID・元root・trace SHAも保持し、同じroleでもbatchの出典を失わない。

## 3. 収集条件

- 地形は既存の2種、levelは1/4。既存generatorのdifficulty bandはlevel 1 `[0.2,0.4)`、level 4 `[0.8,1.0)`。地形生成・sub-seed処理のコードSHAを固定する。
- 4 env/process、flipper `front_folded`、repeat=0。通常のroot reset分布はx/y各±0.1 m、yaw `[-pi,pi]`、roll/pitch・初期twistは0 range。追加の傾斜、押し込み、障害物移動、contact/material/noise変更を加えない。
- RL Actorは共有`rl_actor_model_10000.pt`、SHA `b789f3ab4820a62fffeaaf5a600e5d027772bb3fee8d51a3d`。Actor env YAML SHAは`14fc2441d928a92b1ba45c06ccd8603eb0cf71aebb5d216efc4005f09696545c`。GUIの学習済みMotion Predictor/Evaluatorを走行判断へ入れない。
- Actor観測は既存pre-noise22列。`use_last_action_obs=false`, `use_joint_effort_obs=true`, `use_height_scan_obs=false`を維持。physics/material/noise/settlingは固定された既存設定・実装を使う。
- candidate順は常に `forward_direct(+1,0°)`, `forward_-30(+1,-30°)`, `forward_+30(+1,+30°)`, `reverse_direct(-1,0°)`。候補を結果やイベント発生に応じて変更しない。
- 20 Hz、最大200 control interval=10秒。40 interval=2秒ごとに同じcandidateを現在poseから再計画しfollowerを初期化する。0.75 mの既存Path bankと8点body-yaw Pathを維持する。
- 10秒、既存85°tilt termination、numeric failure、既存resource stopが終了条件。自然転倒は実験のprotocol failureとは別に記録する。他env終了によるparallel censorをphysical failureへ変換しない。
- termination前snapshotを保存し、reset後の値を同じepisodeへ接続しない。転倒・stallの観測を理由に追加操作や再走行を行わない。

## 4. 実行順・保存・異常時

全120行をassignmentのindex順に実行する。まずtrainの12 seed、次にvalidationの4 seed、最後にreserved_testの8 seedを処理する。各role内はcell順→slot順。1 seedの5行はprepath→固定4候補。candidateのreferenceは必ず同seedのprepath出力とし、各env indexを対応させる。hostへ渡すhorizonはprepathも含め200/40で、prepath_only=trueの行はPathを公開せず実測control step数0。candidateのrequested_control_stepsは200だが、実測control_stepsは自然termination/parallel censorの証拠があれば1..200を許す。200未満という理由だけでprotocol failureにしない。

予定のhost保存rootは `/home/ubuntu/Projects/onix_isaac_lab/.eatnav-fall-stall-additional-v0.2-batch02`。新規rootのみ使用し、既存runへ上書きしない。このprotocol固定作業では当該rootを作成しない。role別に`train/`, `validation/`, `reserved_test/`を分け、各試行のrelative pathとreferenceをassignmentに固定する。

各試行は既存host runnerと同じinput manifest、prepath/raw DEM、trace、activation、complete、decision、inner/outer ledger/seal、resource記録を保存する。旧v1 labelsがcollector内部で生成されても監査資料として保存するだけで、学習教師にはv2を用いる。test由来のlabel/terminal/decision詳細を開封前のstdout、集計表、可視化、エラー通知へ転記しない。

既存hostのresource cap（GPU使用量指標6656 MiB、host available memory 2 GiB以上、attempt storage 800,000,000 byte、attempt wall 600秒）を維持し、正確なfield名・値をsource-identitiesに記録する。batch全体はbatch01実績（約128.4分・562.4 MB）から約130分・約562 MBを見積もる。これはbatch02の実績・保証・新しい終了閾値ではない。

protocol/integrity failure、numeric failure、resource stopなら新規processの起動を停止し、失敗attemptを消さずに残す。イベント不足を理由とするearly stopや追加走行を行わない。自動retry・seed代替は禁止。再開は完了sealを検証した行をskipし、未着手行のみ同じ順に進める。失敗行を再実行する必要がある場合は同じseed/role/条件のappend-only recovery planを別途固定し、全attemptの対応と採用規則を残す。成功結果を見てattemptを選ばない。

全24 seedのprepathと4候補を保持して120行のintegrityが確認できた場合のみbatchをcompleteとする。pair不一致・入力不正はprotocol failureとして記録し、該当groupを有効paired datasetへ混ぜない。欠けた候補を安全例で埋めたり、候補単位でsplitを移動しない。不完全batchを完了数384の学習datasetと報告しない。

## 5. 時刻・教師・入力

教師schemaは`eatnav_fall_stall_targets_v2`。decisionはpath activation、将来区間は`(0,H]`、H=1/2/5/10秒。主学習は初期入力→10秒の4 eventと補助progress。教師コード・その依存コード・既存教師sealをsource-identitiesへ固定する。

- tip-over: effective tilt≥85°またはreset前`tilt_over` termination。
- near-fall: tilt≥60°を連続5 control sampleで確認。onsetとconfirmationを別保存し、陽性成立時刻は5番目のsample。4 interval幅0.20秒と5 sample占有0.25秒を区別する。
- linear-stall: cycleに揃った40 intervalのD_cmd≥0.20 m、D_actual/D_cmd≤0.25、Path投影による連続best改善≤0.05 m。followerの0.10 m刻みdiagnostic bestや単純な始終差を代用しない。
- turn-stall: 同じ40 intervalの全sampleでabs(command wz)≥0.01 rad/s、対応するpost-stateのbody abs(omega_z)≤0.5×abs(command wz)。並進の有無とは別head。
- 既知event確認なら1、全horizon観測してなしなら0、残りはunknown/mask=false。partial cycle・missing signal・parallel censorを陰性に補完しない。終端はpre-reset snapshotを優先する。
- progressはcycle固定Pathの投影増分を接続し、replan自体を進捗に加えない。回復は既存v2の監査metadataのみ。学習用recovery headを追加しない。

入力は既存dataset契約のDEM `[4,2,20,20]`、pre-noise Actor22、body state9、候補Path `[4,8,2]`、direction `[4,1]`、candidate mask。初期DEMを途中時刻の入力へ再利用せず、windowでsample数を水増ししない。native rayからのframe/height/mask再構成、timestamp、prepath一致、非finite/nonbinary、各frame valid率25%以上を検証する。正規化とclass weightは固定した累積poolのtrainだけからfitする。batch02固定作業ではfitしない。

## 6. Reserved testの保留と開封

reserved testのseed/role/割当て/ファイルSHA・process完了・protocol gateの成否は確認してよい。trace、イベント陽性数、tilt、termination種類、progress、候補順位、図、教師ラベルやモデル予測を設計者向けに開封・集計しない。protocol failureを通知する場合も外向きにはrow IDとallowlist済みの整合エラーcodeのみとする。

保存先を分けるだけではアクセス制御にならない。既存host decisionやcompleteにはtermination情報が含まれるため、実装するblind verifierは内部で必要なprotocol確認をしても上記の意味情報をログへ出さない。raw/sealは保持する。これは運用上の保留規則であり、現時点でACLや暗号化による強制隔離を実装したとは扱わない。

1. train/validationだけでラベル監査、モデル・入力・閾値・epoch選択を確定する。testは旧pilot用全件監査やdataset CLIへ渡さない。
2. 開封前に別のevaluation freezeへモデルcheckpoint全SHA、normalization、training pool、code/config、比較方式、metric定義・集約・base-seed単位の不確実性評価方法、除外規則、閾値、test seed一覧、評価日時を固定する。該当manifestがない限りtestはreservedのまま。
3. 開封記録をappend-onlyで保存し、固定された評価を一度実行する。その時にtestのcoverageと性能を同時に評価する。test陽性数を先に見てモデルやラベルを変更しない。
4. coverage不足のheadは正式比較N/A。test開封後にラベル・閾値・モデルを変更した場合、そのtest集合は以後開発用として扱い、新しい未使用testを事前割当てする。test結果で都合のよいseedだけを残さない。

## 7. Coverageと追加batchの判断

4 headを個別に判定する。陽性を含むbase seed数がtrain≥5、validation≥3、reserved test≥3、かつ各roleに既知陰性があることを正式比較の運用上の最低coverageとする。turn-stallをlinear-stallや転倒のcoverageで代用しない。これは統計的十分性・校正・安全性の保証ではない。

test未開封時に判断できるのはtrain/validationのcoverageだけであり、testの充足はpending。train/validationで未達なら正式比較へ進まず、次batchを通常条件で事前割当てして固定する。次batchでもcellごとtrain3/validation1/test2を維持し、過去の割当てを移動しない。本batch02は新namespace・新manifest・全既知72 seed/864 sub-seedとの非重複を収集前に固定する。batch03以降も全過去registryへ除外集合を拡張し、新規freezeを作る。本batchの24 seed/120行を結果を見て延長しない。

test開封後にtest coverage不足が判明した場合もN/Aを報告し、当該集合を使った選択を行わない。新testの収集計画を別freezeする。event-window数、4 env、4候補、最適化seed反復を独立事故数として数えない。

## 8. Freeze検証と実行開始の境界

今回確定するものはprotocol、24 base seedと288 sub-seed、role、120行の順序・保存先・reference、候補・通常条件・終了規則、既存Actor/collector/generator/config/教師のSHA、既存seed除外集合、test保留・coverage規則。結果に基づく差し替えはrevision変更として記録する。

機械的な受入条件は、seed生成の再計算一致、既知seed/sub-seed非重複、全env/candidate/windowの同role、各cellの3/1/2比率、120行/96候補process/384trajectory/96group、prepath先行reference、protocol/Actor/teacher/source SHA、ledger/sealの一致。freezeの別directory再生成もbyte一致を確認する。

**実行前に必要な対応**はbatch02の新ledger・batch ID・protocol・root・source inventoryを受け付けるcontract/coordinatorと、blind verifier・role-aware教師dataset adapterの回帰検証である。batch01実装はledger/batch IDをhard pinしており、このfreezeを既存CLIへ渡して実行できる状態とは扱わない。batch01原本のexecution bindingが参照する既存実装を不用意に変更せず、batch02対応の新しい配線と実行SHAを別途固定する。今回のmanifestは `ready_to_collect=false` を保持する。

実行開始時にexecution binding manifestへ追加実装のSHA、protocol ledger SHA、container image ID/Isaac・Onix・依存version、staged全source/Actor/config SHA、資源状態、未開封testの保存先を記録し、固定ソース一致を確認する。現在のOnix source treeもSHAで固定するが、container runtimeの照合は本作業では実施していない。新規実行処理は条件・seed・role・ラベルを変えない配線に限定する。条件を変える修正が必要なら本protocolの新revisionを収集前に作る。

この固定作業ではIsaac/Docker実行、新規地形の生成、追加trajectory収集、モデル学習、reserved test評価を行わない。既存原本・教師v2・dataset・baseline artifactは保持する。

## 9. Revisionの範囲

v0.1/batch01のprotocol・seal・raw・既存datasetは保持する。v0.2/batch02では新しいbatch identity、seed namespace、親registry除外、結合時のID衝突防止、累積pool/coverage規則だけを追加する。走行条件・教師定義・候補・各cellのrole比率・停止規則・test非開封規則を変えない。24新seedの結果を見た後で本batchを延長せず、必要なら別batchとして事前freezeする。自然陽性seedを必要数得られる保証はない。
