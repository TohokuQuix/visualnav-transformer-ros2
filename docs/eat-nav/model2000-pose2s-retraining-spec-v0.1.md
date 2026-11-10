# model_2000による2秒後状態予測：再収集・学習・評価・GitHub保存 変更仕様 v0.1

作成日：2026-11-10。状態：2026-11-10ユーザー承認済み。実行状況は同名成果ディレクトリの記録を参照。

## 1. 目的と完了条件

旧Actorを、提供された `model_2000.pt` と対応設定へ置き換える。それ以外の条件を前回の実行に揃え、新たに走行記録を収集し、2秒後の相対位置・回転予測NNを最初から学習する。精度評価、前回と同じ条件Bのシミュレーション評価、成果物整理を行い、新規GitHubブランチへ通常pushするところまでを承認後の作業範囲とする。

旧dataset・旧予測NNの重みを新しい学習へ混ぜない。RealSense、追加センサ、実機走行、実機plannerの自動変更、RL Actor自体の再学習は対象外。旧成果物を上書きしない。

完了には、新dataset監査、実際の学習完了、独立精度評価、条件B評価、checkpoint取得・推論確認、remoteへのpushとcommit照合を必要とする。精度目標の達成と、実験・公開工程の完了は分ける。未達時は未達の暫定モデルとして公開し、達成済みとは報告しない。学習・評価が実行不能な場合は未完了として原因を報告する。

## 2. 前回の実行を基準とする参照

基準リポジトリ：`/home/ubuntu/eatnav-moving-state-pose2s-20261103`。
参照commit：`285c25696382e4b0f7f253780fe4de48830abe9b`。
参照branch：`exp/eatnav-pose2s-selector-b-20261104`。

同リポジトリ内の以下を参照する。原仕様の古い提案より、実行設定・変更記録を優先する。

- `docs/eat-nav/single-dem-pose2s-spec-v0.1.md`
- `docs/eat-nav/single-dem-pose2s-20261103/actual-fastlio-thirty-amendment.md`
- `docs/eat-nav/single-dem-pose2s-20261103/stage-status.json`
- `docs/eat-nav/single-dem-pose2s-20261103/models/seed123/config.json`
- `docs/eat-nav/single-dem-pose2s-20261103/models/seed123/training-complete.json`
- `src/visualnav_transformer/eat_nav/single_pose2s_contract.py`
- `scripts/eat_nav_single_pose2s_collect_sim.py`
- `docs/eat-nav/single-dem-pose2s-20261103/selector-b-evaluation/` のmanifest・設定・結果

前回は1,330親試行、有効pose区間6,638。train有効1,744、validation有効2,444。最大200epoch設定に対して53epochで早期終了し、best epoch28を採用した。厳密基準は490組中323組pass、167組未達。今回も最大200epochという設定を揃えるが、53epochやbest epoch28に固定しない。

旧予測checkpoint SHA256：`2f2e5ff3b8805e159b23b1e2708211edb0749a4749d46cb4fdc97fbf9619f009`。
旧raw保存先：`/home/ubuntu/eatnav-single-dem-pose2s-data-20261103/actual-fastlio-v2/production`。比較・監査専用として保全する。

## 3. 変更するものと変更しないもの

### 新しいActorと対応設定

提供元：`https://github.com/TohokuQuix/onix_automation.git`。
ローカル確認済み版：`a43de8de76fcfe2113eac8708c9b21a36f2af2bd`。
保存先：`/home/ubuntu/onix_automation-real-policy-20261105`。

|ファイル（提供元repo内）|SHA256|
|---|---|
|`onix_policy_bridge/checkpoints/onix_rough_standard/model_2000.pt`|`56c16ecca41b315ef57aea57beae17a05d94c5c0f54ede88066e9688b7f8a770`|
|`onix_policy_bridge/config/onix_rough_standard/agent.yaml`|`1d8186221ad7db89bc41578f1a4fe4fbd7b699f8169c0bb1965b0202e59d11bb`|
|`onix_policy_bridge/config/onix_rough_standard/env.yaml`|`8e2a592169d3dc4cf193f049f481d35b3c6ff265ae118bb0f8ecf588551d5e2d`|

3ファイルを必ずセットで固定する。mainの将来更新を自動取り込みしない。Actorは22入力→1024→512→256→6出力、ELU、empirical normalizationなし。Actorは駆動・フリッパー制御用で、今回学習する予測NNとは別である。agent.yamlのPPO設定でActorを再学習しない。

変更対象はActorとその必須インターフェース（観測順序・scale、action変換、対応する制御設定）。提供env全体を無条件で置き換え、地形・摩擦・reset・外乱・周期まで変更してはならない。原設定、前回の実効設定、新しい実効設定のキー単位差分を保存する。

Actor入力順はgyro3、gravity3、現在follower指令3、fold mode1、flipper角4、左右track速度2、effort6。出力はRR/RL/FR/FL flipper4、右track、左track。track変換は×14.2、clip±14.2rad/s。flipperは提供設定のoffset・rate limit・方向制約・干渉処理を適用し、固定角度制御へ置換しない。fold modeは前回と同じ0のため、既存のmode依存の方向強制は非作動とする。実機bridge独自の方向判定は追加しない。Actor推論は決定的な推論出力を使い、探索ノイズを足さない。

速度上限を改善目的で新しく調整しない。4.0rad/sを独自に物理上限へ強制適用しない。提供設定と現在runtimeで確認された実効上限12.56rad/s（track）・2.0rad/s（flipper）は事前検査の参照値とし、実際の値を記録する。前回収集runtimeの実効上限・physicsログとの照合も行う。差があれば、対応設定由来かruntime不一致かを分類する。ポリシー対応以外の未説明の物理条件差は本収集前に解消し、解消不能なら報告する。提供環境が学習時に対応することは正常仮定であり、学習当時の版を実証したとは記載しない。

### 維持する条件

- DEM生成・欠損感度条件・inpaint処理・位相・取得時刻は前回の実行manifestから固定する。RealSenseは追加しない。
- 地形、親seed、初期状態生成、候補列、split、FAST-LIO設定、follower、物理周期、予測NN構造・損失・学習設定・評価定義を維持する。
- Actor変更により生じる走行軌跡、センサ値、欠測、終了時刻、教師、正規化統計の変化は結果であり、旧値へ強制しない。

## 4. 入出力と制御の固定

DEMは4Hz更新、4m×4m、0.1mグリッド、1frame。高さ・height-valid・raw-observedの3channel `[3,40,40]`。予測起点以前に取得・受信した最新frameを使い、取得stamp、受信stamp、age、局所化poseを保存する。未知を有効な平面で埋めない。mask・補間・高さ基準・座標配置は前回と同一。前回contractのstate age上限0.20秒、DEM age上限0.50秒、dem_contextのage・availability・取得frameから起点への相対変換を維持する。正規化の列順・方法も維持し、統計値だけを新trainから算出する。

状態27成分：body gravity3、body gyro3、flipper角4と角速度4、駆動回転速度4、FAST-LIO由来body並進速度3と角速度3、fold one-hot3。値・mask・ageを保存。TFは変換・教師・出力合成用で、絶対地図位置をNNに入れない。goal、未来の実測指令・状態、field/level ID、真値速度を入力しない。Actor内部effortと、予測NN入力を混同しない。

FAST-LIOは前回の実本体・設定・因果的pose差分方式を維持する。内部LiDAR 16×512・10Hz、IMU20Hz、初期settling2秒＋LIO初期化2秒は継承する。理想オドメトリや車輪速度へ黙って置換しない。

14候補は前後各7本、角度順0,-30,+30,-60,+60,-90,+90度、長さ0.75m、8点。区間開始時yaw水平座標で生成。経路は2秒固定、followerは20Hz、lookahead0.5m、速度上限0.30m/s、angular gain0.5、上限1.6rad/s、goal tolerance0.2m。前回NominalFollowerの実装を使う。物理dt0.005秒、制御dt0.05秒。2秒40step、10秒200step。

出力は起点の全3D車体座標で相対XYZ[m]と相対回転6D。教師は `R0^T(p2-p0)` と `R0^T R2`。quaternion内部wxyz、ROS変換時xyzw。RPYはZYXで評価し角差をwrapする。goalやイベントheadを追加しない。14候補へ一括推論できる構造を維持する。

## 5. 再収集とdataset監査

対象は `k_rail_step_field`, `pyramid_stairs`, `pyramid_stairs_inv`, `random_rough`, `boxes`, `hf_pyramid_slope`, `k_rail_field` のlevel0〜4。

|split|親試行|予定2秒区間|
|---|---:|---:|
|train|350|1,750|
|validation|490|2,450|
|final_test|490|2,450|
|合計|1,330|6,650|

前回の親manifestとseed・初期状態seed・候補順・DEM位相を再利用し、policy/runtime identityだけ新規にする。14候補のpermutation・trial r/segment jの割当 `π[(r+3j) mod14]` を維持する。親IDは旧との対応表を残し、新run IDで区別する。

各親を10秒連続で実行し、2秒ごとに候補を切り替える。車体・関節・Actor/action履歴・センサ位相は継続し、path固有状態だけ更新する。区間開始時の入力と、その候補を実行した40step後の教師を結ぶ。root/post-rootの1stepずれ、reset跨ぎ、未来DEMの混入を検査する。

欠測・転倒・早期終了による2秒未満区間を正常教師として埋めない。有効数・除外理由をrole/field/level/candidate別に保存する。旧より有効区間が減っても、同じ数に見せるために除外を変えない。新たな追加収集は不足と必要数を示して判断し、無制限に増やさない。

同一親の5区間をsplit間へ分散しない。独立親seedと初期状態hashを検査する。地形重複は集計し、前回と同じ「既知地形での独立走行評価」とする。旧finalはすでに既知なので、前回との比較は対応seedでの再現比較として扱い、完全に新しい未知testとは主張しない。新モデルの調整には新train/validationだけを使用する。

## 6. 小規模検証と資源

本収集前に、3ファイルhash、22→6入力再生、action変換、関節対応、fold、実効上限、実import先、物理・制御・センサ周期を検査する。前回とのruntime差分を確定し、小規模で10秒5区間、FAST-LIO、DEM、教師生成、保存・sealを一通り検証する。

headless・最大30台を基本とし、起動前にGPU/CPU/RAM/空き容量と既存プロセスを確認する。GPU利用は排他し、既存収集やR6を重複起動しない。30台が不安定なら小規模benchmarkで並列数だけ調整し、seed・物理時間は維持する。OOM・異常終了は報告し、失敗出力を正常データに混ぜない。物理dtやセンサ周期を変更して高速化しない。

datasetが完成し監査passした時点でユーザーに通知し、承認済み工程の学習へ続行する。dataset本体をGitHubへ入れない。

## 7. 学習と予測精度評価

既存予測NNと同じ構造・初期化方式でscratch学習する。旧予測checkpointの追加学習は行わない。正規化は新trainのみで再計算する。

前回config：seed123、batch128、AdamW lr0.001、weight decay0.0001、最大200epoch、min epochs30、patience25。損失は標準化XYZ SmoothL1＋SO(3) chordal＋6D退化抑制、重み1:0.5:0.001。validationによる選択指標を維持する。最終testをモデル選択や追加探索へ使わない。新規の大規模探索・学習seed増加は行わない。

候補14×field7×level5の490組について、実行候補の正解だけで評価する。有効区間数・独立親数、前後/左右/高さの絶対誤差、wrap済RPY誤差、SO(3)誤差を保存する。平均・中央値・P95・最大値、許容範囲割合、同時達成割合、符号付き前後bias、前後進・左右旋回・開始segment/動作状態別を報告する。データなしは欠測として表示する。

前回の厳密判定は各組5独立例以上、最大左右10cm・前後30cm・RPY各10度以内。高さに独自の合否基準を足さない。経路に対する前後・左右は前回評価コードの同じ投影基準を使用し、曲線・後退で新たな定義へ変更しない。全体平均だけで490組合格としない。現在状態維持・速度角速度維持・planned arcのbaseline、前処理込み/モデル単体14候補推論時間も再評価する。

旧結果との集計比較に加え、両policyで有効な共通親/segmentの比較と、欠測が異なる件数を分けて示す。改善/悪化をActor重みだけの効果と断定せず、対応設定を含むpolicy交換の結果と表現する。

## 8. 前回と同じ条件Bの走行評価

前回のB manifestの350走行と初期条件・目標列・走行時間を再利用する。新Actorと新予測NNの組で実行する。経路選択はゴールへの予測進捗＋予測重力傾斜60度超の候補除外のみ。goalはNNへ入れない。全候補除外時の処理、切替タイミング等は前回B実装を維持する。DEM段差/支持面/スタック確率/45度STOP等の制約を追加しない。

本仕様の承認は、厳密490組基準が未達の場合も、前回と同じ範囲の探索的B評価を行う承認を含む。未達を解消した正式評価とは区別する。Aは追加せず、B単独から傾斜除外の有効性を断定しない。

有効走行数、転倒観測、route進捗、除外STOP秒、followerゼロ指令秒、直線/旋回スタック陽性window等を前回と同じ定義で保存する。実機精度・実機安全性は未検証として残す。

## 9. 保存先・GitHub公開

実装開始時に基準commitから隔離worktreeと新branch `codex/eatnav-model2000-pose2s-20261110` を作る（既存なら重複しないsuffix）。旧repoの未コミット作業を取り込まず保全する。本仕様書もそのbranchへ収録する。

新raw保存先案：`/home/ubuntu/eatnav-model2000-pose2s-data-20261110/`。
Git内成果先案：`docs/eat-nav/model2000-pose2s-20261110/`。
旧modelとは入力shapeが同じでも対応する制御policyが異なるため、新profile SHAとpolicy identityをcheckpoint/入力例/推論設定へ保存し、取り違えを検出する。

公開対象：承認仕様・差分監査、コード/検証、収集manifest/split/件数/除外理由、学習config/正規化、checkpointとSHA256索引、予測精度CSV/表、B走行結果/代表trace、再現手順、14候補の最小推論入力例。

Actorはライセンス・ファイルサイズを確認し、許可される場合は対応設定とセットで収録する。再配布できない場合は上記固定commitからの取得方法・hashを残す。予測checkpointは前回の通常Git保存方式を第一候補とし、100MiB以上の場合は通常Gitへ強制追加せず、利用可能なLFS等の取得・検証手順を確定する。巨大raw/点群/dataset、一時ファイル、認証情報はcommitしない。

push先は `https://github.com/TohokuQuix/visualnav-transformer-ros2.git`。作業開始時とpush直前に実remoteを確認する。現在の古いbackup worktreeのoriginはローカルパスなので、そのままpushしない。隔離repoの専用remoteで上記URLを確認して通常pushする。force pushしない。

最終報告：branch/commit、GitHubリンク、モデル取得方法・SHA256、別PCのfetch・checkout・14候補推論確認コマンド、実施件数/学習終了epoch、490組合否、B結果、旧との比較、未達/未検証事項、ローカル限定データ保存場所。コマンドは実装後に実在・動作確認したものを掲載する。

## 10. 承認後の実行順序

1. 参照・旧成果の保全、新branch、Actor3ファイル固定と差分監査。
2. 小規模headless収集、入力/教師/FAST-LIO/実効設定検証。
3. 同じmanifestで本収集、dataset生成、全件監査、完成通知。
4. scratch学習、validation選択freeze、最終精度評価、旧結果との比較。
5. 同じ条件Bの350走行、結果監査。
6. 成果物整理、checkpoint SHAと別環境向け推論確認、秘密/巨大ファイル検査。
7. 新branchへpush、remote commit一致確認、最終報告。

通常の修正・再実行は承認範囲内で進める。仕様外のセンサ/制御/物理条件変更、データ破損の危険、必要環境・権限不足は報告する。今回の承認前には本収集・学習・pushへ進まない。
