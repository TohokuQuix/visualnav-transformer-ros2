# 1フレームDEMによる2秒後状態予測・経路選択 仕様案 v0.1

2026-11-03作成。**仕様承認待ち。本仕様の収集・学習・経路選択実験は未実施。**

本仕様は今回のユーザー指示を基準にdataset設計から作り直したもの。以前の`real-conditions-pose2s-spec-v0.1.md`にある約3Hz・複数DEM・状態履歴・18,816区間の計画は継続しない。既存成果・branchは保全する。

新branch：`exp/eatnav-single-dem-pose2s-spec-20261103`。調査元visualnav commit：`2b7bc31b919fd2256f07586010c6b6e20eb0b37e`（製品コードは`43bb7b69fde5d29095b6add17965aacd48ae58fa`と同じ）。参照実ファイルとSHAは[reference-index.json](single-dem-pose2s-20261103/reference-index.json)、実機側確認は[確認票](single-dem-pose2s-20261103/real-pc-checklist.md)。

## 1. 目的、順序、承認事項

指定した14候補のうち1本を、固定したfollowerと既存学習済みサブクローラActorで2秒追従した結果を、現在の観測から予測する。主出力は相対XYZと相対3D回転。goalはNNへ入れない。予測精度を先に検証し、承認した精度条件を満たしてから、ゴール進捗＋予測傾斜除外だけの経路選択を評価する。実機確認は最後の別段階。

実行順は次のとおり。

1. 今回：既存実装・データ・Actorの調査、仕様・参照commit・未確定事項を共有。
2. 実機PCとの仕様共有、ユーザー承認。制御の参照版・入力変換の確認を行う。
3. 承認後：収集／教師／監査／学習／推論／精度評価を実装、小規模検査、dataset作成、学習、独立評価。
4. **承認した予測精度条件を満たしたときのみ**、コード・manifest・モデル・精度結果をGitHubへ第1回アップロード。
5. 同じ予測・追従定義でheadless経路選択評価。結果とコードを第2回アップロード。
6. 実機でshadow予測・走行を確認するための引継ぎ。実機走行は本仕様の自動実行範囲外。

今回承認を求める設計判断：①1330試行の初期収集計画、②490組ごとに5件以上・5独立試行を要求する精度判定案、③最大絶対誤差を合否に使う案、④予測重力傾斜60度超の候補除外案、⑤旧sim followerの追従則を使い自動退避・boostは使わない固定2秒制御案。いずれもまだ承認済みと扱わない。

予測精度未達なら経路選択へ進まない。失敗条件と改善案を報告し、必要な追加収集・学習の根拠を示す。最終testへ再適合させる反復はしない。未達報告の共有は可能だが、精度達成済みモデルとして公開しない。

## 2. 調査結果と方式の選択

### 実ファイルで確認したこと

- `demo_terrain.py:TERRAIN_TYPES`に7フィールドが定義される。levelは0〜4の5行。`terrain_band`は元difficulty_rangeを5分割するため、level番号だけで全生成地形が同一とは限らない。
- `bidirectional_bank.py`は前後各7本、長さ0.75mの直線／円弧、始点を含む8点。
- `model_10000.pt`のActor重みは入力22→1024→512→256→出力6のfeedforward構成。実checkpointをCPUで読み、層shapeを確認した。今回の新予測NNとは別モデルである。
- 既存sim入力DEMは20×20の複数frame。保存済み20×20から観測のない周囲を埋めて40×40の新観測を作ることはできない。
- 既存train5,831区間、全dataset7,765区間・完全pose7,748という前回監査結果が残る。代表rawは400rayとroot/joint traceであり、4m四方の1600セル観測ではない。これらの全件を新仕様適合と認定しない。
- 実機の過去bagは2m四方20×20、約3Hzの観測。**今回の指示は4Hzで確定**し、過去bagへ合わせる変更はしない。実機側で4m四方と4Hzが設定・取得できるか確認する。古いbagを4Hzへstamp改変しない。
- 実機専用cycle followerはユーザー報告ではlookahead0.8m。一方、従来simの`p1_follower_core.cpp`は0.5m。この差を隠して実機同一とはしない。

### 比較と採用提案

| 方針 | 適合性 | 判断 |
|---|---|---|
| 旧20×20・複数frameのテンソルだけで学習 | 4m一枚DEMの空間範囲、level網羅、10秒5区間が不足 | 主datasetにしない |
| 旧生記録のうち本当に4m観測・全必須状態が残るものだけ再加工 | 監査で成立すれば補助利用できる | 元記録は保存。未確認データを水増ししない |
| 既存Actorと14候補を再利用し、指定の連続10秒方式で新規収集 | 今回の入力・切替・490組の評価を直接満たせる | **主方式として推奨** |

新モデルはまず新規trainだけから学習する。旧データの補助利用は検証可能な適合subsetがある場合のvalidation比較に限定し、final-testをtrainへ移さない。

## 3. 正式な7フィールドとlevel

| field ID | コード上の正式名称 |
|---|---|
| 0 | `k_rail_step_field` |
| 1 | `pyramid_stairs` |
| 2 | `pyramid_stairs_inv` |
| 3 | `random_rough` |
| 4 | `boxes` |
| 5 | `hf_pyramid_slope` |
| 6 | `k_rail_field` |

各level0,1,2,3,4、計35cell。flat等の別フィールドを追加しない。`k_rail_field`と`k_rail_step_field`は別物。level0を完全平面と呼ばない。元generator・subterrain設定・difficulty値・mesh hashを記録する。並列数を変えてもcellや地形assignmentを変えない。

## 4. 使用するサブクローラポリシー：ユーザー確認済み

ユーザー回答：**旧R4*/R5のシミュレータ用ポリシーを使用**。

| 対象 | 固定する参照 |
|---|---|
| Actor checkpoint | `checkpoints/eat_nav/rl_actor/model_10000.pt` |
| checkpoint SHA256 | `b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d` |
| 保存env設定 | `checkpoints/eat_nav/rl_actor/env.yaml` |
| env SHA256 | `c2b7e9b7959a231c5391ed714a4de4be958fe5812cf03c10ab60f5636598233b` |
| 既存sim制御の入口 | `scripts/eat_nav_moving_pose2s_collect_sim.py`、`demo_reset_policy.py`、`mi_fi_v1.py` |
| 外部simコード | `/home/ubuntu/Projects/onix_isaac_lab` commit `87fe97d266ae2f64d802a80471727d3429b05ff4` |
| flipper action | `mdp/actions_flipper.py:RateLimitedJointPositionAction` |
| fold mode | このsim profileでは明示的に0を継続。実機未記録modeの代用という意味ではない |

Actor観測は保存envの順序・scaleを再現する：gyro3、gravity3、現在のfollower速度指令3、fold mode1、flipper相対角4、左右track平均回転速度2、effort6で計22。`use_last_action_obs=false,use_joint_effort_obs=true,use_height_scan_obs=false`。Actorの観測と今回予測NNの入力は別schemaとする。criticだけの真値を予測NNへ流さない。

Actor出力はflipper4（RR,RL,FR,FL）＋右track＋左trackの6。保存設定のflipper scale1、default offset使用、clip[-π/2,3π/2]、target-rate limit2.0rad/s、干渉limit2.09439510239rad、rear/front pair(0,2),(1,3)を維持。track scale14.2とclip±14.2rad/s、broadcast mappingも維持する。2.0rad/sはsim targetの変化率設定であり、実機position制御の実速度とは主張しない。

実装順は、既存JointPositionAction変換→directional clip→干渉処理、apply時の既存latency→target rate limit→directional clip→干渉処理→物理joint target。fold mode0では正負fold条件が立たず、方向強制は適用されないが、Actor・rate limit・干渉処理は動く。初期姿勢だけRR,RL,FR,FL=`0,0,π,π`。**走行中に前π・後0を固定する処理を作らない。**

10秒内はActor、action managerのprevious target、latency buffer等をリセットしない。feedforward ActorにRNN hidden stateはない。実行前に依存ファイルhashと実import先を照合する。異なるActorへ黙って置換しない。保存envだけでなく上記wrapperのreset・command上書きを含む実効設定をmanifestへ保存する。

実機bridgeの6/2action、mode有無、ゼロ速度保持、方向clip等は別profile。今回sim用Actorに未確認の実機bridge処理を足さない。実機へ接続する段階で不一致を検証し、未確認のまま同一制御として移植しない。

## 5. 候補形状と2秒の追従条件

候補index0〜6は前進、7〜13は後退。各群の角度順は`0,-30,+30,-60,+60,-90,+90`度。directionは+1/−1。legacy名`forward_direct`等と`forward_+0`等の表記差はindex対応表で固定する。

L=.75m、s_i=iL/7、i=0..7、θ=angle_rad。θ=0ならp(s)=(s,0)、それ以外はp(s)=(L/θ sin(θs/L), L/θ(1−cos(θs/L)))。後退は全XYに−1を掛ける。候補は区間開始時の**yaw水平座標**、x車体前方、y左。曲線角は全接線回転であり終点の方位角ではない。各区間で現在位置・yawからworld pathへ変換し、その後2秒はpathを固定する。

追従profile提案：旧simのpure follower追従則（lookahead.5m、linear upper.30m/s、angular gain.5、angular cap1.6rad/s、20Hz、goal tolerance.2m、方向判定80/100度、heading絶対値.6rad以上は並進0、それ以外v=d*.3*max(0,1−|heading|/π)）。旧envの10〜15秒間隔の外力pushは今回の10秒試行には適用せず、無外力pushの共通profileとする。既存の終点tangent extensionはlookaheadの幾何処理として維持し、path末尾へ新目標をappendしない。2秒途中でgoalへ着いたら同じfollower定義の出力0を維持する。

**自動退避、スタック判定による逆走、angular boost、実測45度STOP、リスクNN gateは今回の追従profileには入れない。** 旧follower全機能を無条件継承せず、nominal追従部分だけを明示profile化する。これらは候補を固定2秒追従する定義に影響するため、収集・精度評価・経路選択の全てで同じ設定を使う。新しい回復制御は作らない。follower .8mを使用する実機とは設定差として残す。

切替時は既存path更新に相当する処理を定義し、path固有のprojection index・goal到達flag・path進捗を新pathへ更新する。方向判定は新path接線で再決定。退避timerは使わない。車体・joint・Actor・action history・sensor更新位相は継続する。これはpathの更新であり、車体の停止・pose/速度resetではない。境界でゼロ指令を挟まず、次候補の最初のcommandを次の制御tickへ直接渡す。切替前後の実指令と実状態を記録する。

予定速度を今回のNNへ別sensor入力として追加しない。同一固定profileの属性をcheckpointへ保存し、異なる速度上限・follower設定で利用する際は互換性なしとする。未来の実速度・実指令は教師・解析専用。

## 6. NN入力schema：単一DEMと現在状態

`eatnav_single_dem_pose2s_input_v1`。Bは観測batch、Kは評価候補数（1または14）。観測encoderは候補間で共有し、候補ごとに同じscorerを使う。

### DEM：4Hz更新、40×40の1frame

DEM高さは`elevation_inpainted`相当。**4Hzは観測更新周期0.25秒を意味し、4frameを意味しない。** 入力shape `[B,3,40,40]`：高さ1枚、height-valid mask、raw-observed support maskの3channel。同じ1frameの付随maskであり複数時刻ではない。raw maskは既存GridMapの元elevation有限性から作り、新sensorを要求しない。元layerがなければraw-supportをunknownとしてmask availabilityを別1数値で示す。

セル面積の範囲は取得時車体yaw水平座標x,y∈[-2,+2)m、中心は−1.95,−1.85,…,+1.95m。行は+y昇順、列は+x昇順、0.1m分解能。**点中心のspan3.9mとセル面積4.0mを区別**する。sim ray1600点を正確にこの配置へ生成し、実機GridMapのcircular buffer/軸/poseを復元して同じ配置に変換する。車体yawに対し前後左右2mで固定。

高さ基準z_refはそのframeの取得時点に利用可能な測定車体基準点高さ。height=z_inpainted−z_ref[m]。絶対地図高さは入力しない。基準poseのsource・stamp・TFを保存する。raw値と変換後値を両方保持。simではvirtual odom/IMU/TFとして公開する測定相当の値で同じ変換を実行し、教師のroot真値を別の隠れたchannelへ追加しない。初版simの観測が理想的な場合は明記し、実機観測誤差を再現できたと主張しない。

inpaint済みの有限heightと元観測supportを区別する。height無効・地図外は正規化後の値0＋height-valid=0とし、0mの有効地形と区別する。raw-support不明は値0＋availability=0。未知セルを真の平面として埋めない。raw観測穴の補間方式・設定を保存し、true mesh全セル観測版と欠損付き版の成績を分ける。元bagの実観測率約53〜58%は参考例で、全実機の分布とはしない。

予測起点t0までに取得・受信できた最新の**1frame**を使用する。更新stamp、受信stamp、clock domain、frame ID、t0、age=t0−取得stampを保存。取得stampをt0へ書き換えない。DEM age[s]とraw-mask availabilityの2数値を入力へ付加する。取得時の局所地図を維持し、取得frameからt0のyaw水平frameへの相対Δx,Δy,Δyawを4数値（m,m,sin,cos）で付加する。これは同じframeの座標情報でありDEM履歴ではない。変換はt0までに利用可能なTFのみ。

4Hz scheduleと制御20Hzの位相を親seedで0/.05/.10/.15/.20秒から均等に変え、起点が常にage0の構成にしない。初版simは遅延なしでもage0〜.20秒を含む。試行開始前に最初のframeが存在するようsensor初回取得を行い、センサの全体更新位相は10秒内で維持する。欠落・受信遅延はrawに記録し、age>.50秒は`input_valid=false`のデータ品質判定案（走行停止ルールではない）。この許容値は仕様承認対象。2秒終点に存在するDEMで区間開始入力を補わない。

### 現在状態：27数値＋mask・age

全てt0までに到着した最新値。状態履歴や追加カメラ等を必須化しない。

| index | 数値・次元 | 単位・座標・変換 |
|---|---|---|
| 0..2 | gravity direction 3 | 無次元unit vector、body x前y左z上、水平時(0,0,−1)。IMU姿勢＋sensor→body外部変換から生成 |
| 3..5 | angular velocity 3 | rad/s、body軸roll(x)/pitch(y)/yaw(z)。IMU gyroを外部変換。軸角速度をEuler角の時間微分と混同しない |
| 6..9 | flipper position 4 | rad、RR,RL,FR,FL、外向水平0・上90度。encoder符号/原点を変換。連続関節の生angleは保持し、物理可動域内を単純wrapしない |
| 10..13 | flipper velocity 4 | rad/s、同じ順序・正符号 |
| 14..17 | sprocket velocity 4 | rad/s、FR,RR,FL,RL。実機joint名ごとの出力軸速度。simでは対応track駆動を固定mappingし、8roller平均等の選択をmanifestで明示 |
| 18..20 | odom linear twist 3 | m/s、body基準点。child_frameから回転・必要な原点移動を適用。未観測z等を真値で補わずmask=0 |
| 21..23 | odom angular twist 3 | rad/s、body軸。IMUと同一値だと推測せずsourceを保存 |
| 24..26 | fold mode one-hot 3 | mode −1,0,+1の順。今回simでは(0,1,0)。未知をmode0へ置換しない |

状態値`[B,27]`、有効mask`[B,27]`、各元観測age`[B,27]`（秒、同一元観測は同一age）。raw JointStateのposition/velocity/effortは全列・名前順を保存する。effortは単位と集約が不明なので**初版予測NNへ入れず**解析用に保存する。既存Actor内部のeffort入力は既存処理のまま残す。角速度は自由ベクトルであり測定位置を変えても同じ剛体のωは同じ。IMUの加速度は今回NN入力に追加しない。

gravity/gyro、4flipper角と速度、4sprocket速度、odom水平速度とyaw角速度、fold modeは必須。planar odomのz/roll/pitch速度等の未観測成分は値0＋mask=0をsim/real両方で一致させられる。必須項目欠測は`input_valid=false`として理由を保存し、教師を正常入力として学習しない。状態age上限.20秒を初期案とし、単なるmissingを物理ゼロへ置換しない。実機driverがIMU covariance−1を返す場合は有効sourceを確認する。新sensorを勝手に要求するものではなく、既存入力の意味の確認事項。

tfはsensor/body、odom/body、map/yaw間の座標変換、観測局所化、教師・出力の合成に使う。map原点依存の絶対XYZ・絶対yaw・field ID・level・terrain seedはNNへ入れない。現在q0は結果合成と評価のmetadataであり、NN姿勢入力はgravityのみ。

### 候補と正規化

`paths[B,K,8,2]`（m）、`direction[B,K,1]`（±1）。goalなし。候補IDは対応付け用metadataとし、geometryを無視したIDだけの学習をしない。

高さ、状態、age、相対translationは**trainのみ**の有効値でmean/stdを決め、mask・one-hot・sin/cos・gravity・directionは物理表現を維持。pathは固定.75mで除算。std floorはheight.001m、state各量.001（各単位）、age.001s。欠損値は標準化後0とmask。位置教師std floor(.05,.05,.02)m。正規化の単位・列順・train ID・SHAを保存する。未来のTF、実速度、指令、接触・終了を入力へ混ぜない。

## 7. 出力と教師の座標

body B0はt0時点の全3D車体座標（yaw水平path frameとは異なる）。world poseを(p0,R0),(p2,R2)として、教師はΔp_B0=R0ᵀ(p2−p0)、ΔR=R0ᵀR2。x前・y左・z上、m/rad/s。NN出力は位置3と回転6D（ΔRの第1列と第2列）を基本案とする。Gram–SchmidtでSO(3)へ変換し、退化時はvalid=false、真のidentityと混同しない。

quaternionは内部wxyz、ROS入出力xyzwを明示変換。q2=q0⊗Δq、p2=p0+R0Δp。relative RPYと合成future RPY、SO(3)全体角誤差、future gravity tiltを確認できる。Eulerは固定ZYX（Rz(yaw)Ry(pitch)Rx(roll)）で表示するが直接Euler回帰を主損失にしない。

初版はpose-only。過去の4イベントhead・スタック確率・進捗headを勝手に付けない。終端やtip/stallは解析ラベルとして別保存できる。入力schema・horizon2.0sをcheckpointとAPIに含め、既存R4*/R5やmoving-state版と誤読しない。

## 8. 10秒連続試行と教師作成

各親試行はseed固定した地形・初期状態から一度だけ開始し、5候補を各2秒、計10秒追従する。境界は0,2,4,6,8,10秒。physics dt=.005s、control dt=.05sを維持。各区間40制御更新・400物理step。DEM更新は.25秒で独立。

開始位置・yawは既存reset実装の範囲（tile基準x,y各±.1m、yaw±π）をseed化し、適用後pose・初速を保存する。旧resetのvelocity_rangeは全6成分(0,0)と実ファイルで確認したため初速0とする。実際のreset後値もmanifestへ保存する。2区間目以後は前区間の結果が開始状態になる。初期resetの安定化・warmupが必要ならその長さ・指令・更新を別prefixとして記録し、10秒へ混入させない。

区間開始では、先に利用可能な入力を確定し、予定候補を対応させ、2秒間追従し、次候補へ切り替える**前**に教師を確定する。root_state(t)は作用前、post_root_stateは作用後。40回目のpostが正確にt0+2秒となることをtimestampで検査する。2秒時点のpostを次区間のstartとし、1stepずらさない。

- 10秒中の車体、joint、ポリシー、action、sensorをリセットしない。
- 転倒／自然終了／非有限／地図境界等で2秒未満ならその区間pose_mask=false。未来を外挿・ゼロ埋めしない。最後のpre-reset state、終了理由、観測済みprefixは保持。
- 参照rough_env_cfgのtilt terminationは85度。新profileもこの既存85度終端を維持し、実効設定を保存する。終端を候補除外制約へ使わない。auto-resetされても新episodeを前episodeの未来へ接続しない。終了後の残り区間は未実行であり0件。
- スタックは途中終了や救済resetの理由にせず、そのまま予定候補を切り替えて10秒まで記録する。
- 予定外candidate変更、append、外部指令介入、profile変更は別理由で除外・集計。followerの自然な減速は正常な実行結果。
- `parent_id,episode_id,window_index,candidate_id,t0,t2,field,level,seed,terrain_hash,source observation IDs,profile hashes`を保存。root/joint/command/action/DEMはrawへ記録し、教師・NN入力のallowlistを分ける。

20Hzの途中軌跡も保存できる構成とするが、今回NNが予測するのは2秒終点のみ。各区間には**実行した候補1本分だけ**の教師を付ける。残り13候補は未観測maskであり、同じ結果を複製しない。同一状態から14分岐したdatasetとも呼ばない。

## 9. 収集数・候補割当・分割

### 計画案の比較

| 案 | train | validation | final評価 | 合計試行・予定区間 | 限界 |
|---|---:|---:|---:|---:|---|
| 最小網羅 | 10/cell | 3/cell | 3/cell | 560試行・2,800区間 | 評価1候補1件程度。最大値・分布の判断が弱い |
| **推奨初期計画** | **10/cell** | **14/cell** | **14/cell** | **1,330試行・6,650区間** | 評価各候補5独立例。まだ大規模な信頼性保証ではない |

trainは35cell×10=350試行・予定1,750区間でユーザー最低条件を満たす。validationは490試行・2,450区間、final評価も490試行・2,450区間。**train350試行とは別に980試行を確保する案**。final評価だけの理論最小は35×ceil(14/5)=105試行（525区間）だが、初期案は各組5独立例を確保するため490試行とする。trainの各候補は各cell3〜4件程度で初期学習には少ない可能性があり、成功を保証しない。増量は偏り・learning curve等の根拠を報告して決める。

### 候補割当

cellごと、roleごとにseed固定した14候補のpermutation πを作り、trial r、区間j=0..4で`candidate=π[(r+3j) mod14]`。trainはr=0..9、validation/finalはr=0..13。14試行を揃えると各候補は正確に5回、かつ5つの独立親・5つの区間位置へ1回ずつ現れる。選択は観測した成績で変えない。roleごとにoffset/permutationを固定しmanifestへ事前保存する。

候補順により前区間の影響が変わるため、candidate×stage×previous candidate×前後切替×field×levelの予定・実行・有効数を集計する。今回全ての候補遷移を網羅する設計ではない。学習候補の偏りも隠さない。

### splitと漏洩

terrain seed・初期状態seed・候補順seed・sensor phase seedをroleとparentの独立streamで生成する。近接5区間は必ず同じrole。親試行・分岐元・prefix・raw観測共有・初期状態hashの跨ぎを禁止。splitは結果を見る前に固定。normalize/epoch/model選択はtrain/validationだけ。

生成mesh・difficulty・生成設定のhashを比較する。同一生成地形を異なるseed名で独立と偽らない。原則terrain hash groupもroleを跨がせない。候補manifest固定前にgeometry重複を検査し、別seedの実生成結果へ置換する。**そのfield/levelが決定的形状で分離不能なら、未知地形評価を達成した扱いにはしない。** 該当cellと理由を共有し、既知地形・独立初期状態評価を許容する仕様変更の承認が必要。field/levelを勝手に増やして穴埋めしない。

### 欠損・早期終了

6,650は予定区間であり有効数ではない。特に後半区間の生存選択・転倒欠測を報告する。final評価の各組は有効区間数n、独立親数n_parentを持つ。n=0は**評価データなし**、1〜4件は**評価件数不足**、5以上で判定可能とする。

不足があれば、candidate×cell×stage・終端理由を提示する。収集前に独立seedの予備manifest（上限14追加試行/cell/role）を作っておく案だが、**不足を理由に自動で無制限追加しない**。必要な追加数・根拠・coverage見込みを報告してから追加batchを固定する。未実行予定を有効件数に数えず、補充例は元欠測を隠さない。final値を見て都合の良いseedを選ばず、補充は有効性/coverageだけで事前順に行う。困難な状態を除くことで精度を良く見せない。

## 10. 学習・効率化

GUI off。10env単一processを第一候補とし、2/5/10envの小規模benchmarkを先に実施する。必要なら2process×5も比較するが、並列起動を増やすほど速いとは仮定しない。prototype/benchmarkは本格datasetの別role、承認後に実施し、finalへ混入しない。benchmarkは各構成2反復×2親/env程度、全体上限100親を案とし、計画件数と別記する。

有効window/壁秒、総sim秒/壁秒、1envあたり速度、VRAM peak、RAM、CPU、I/O、sensor/Actor/保存のstage時間、失敗率を実測する。10envで悪化すれば原因と採用並列数を報告。physics dt、20Hz、2秒は変えず、realtime sleep除去・vectorize・観測共有・非同期writer・shard圧縮で壁時間を短くする。地形・seed assignmentを並列数に依存させない。raw保存は1観測1回＋ID参照、完了shardをatomic sealして再開可能にする。

既存helperの`configure_selected_terrain`は1/4env制限なので、選択cellを変えず10envへ拡張する実装が必要。rayも20×20から40×40へ変更が必要。いずれも**現時点では未実装**。物理・Actor設定は変えない。

モデル案：単一3channel DEMの小型CNN→64次元、現在状態/mask/age・DEM age/相対poseのMLP→64、候補geometry＋directionのMLP→32を融合し位置3・回転6を出す。goal・field/level・絶対poseなし。旧4frame encoderを同一入力としてロードせず、新schemaで初期学習する。旧Actorは推論のみで学習し直さない。

主損失はtrain標準化XYZ SmoothL1＋SO(3) chordal loss＋6D退化抑制、初期重み1:.5:.001。初期optimizer AdamW lr1e−3,weight_decay1e−4、最大200epoch、30epoch後patience25、seed123。batch64/128/256・loader0/2/4をtrainだけの速度測定で選ぶ。validation scoreは前後/.30・左右/.10・RPY/10degの正規化MAEの平均を案とし、成分値も保存する。最大6設定の事前列挙比較、finalはbest設定/epoch/正規化freeze後に開く。

no-motion・現在速度/角速度維持SE(3)・予定arcを上限速度で進むbaselineと比較する。幾何baselineのz/roll/pitchは未予測とし、その項目の勝利を主張しない。旧NNは入力空間・制御が違うため同条件の正式比較と混同せず、別の参考評価とする。停止例だけで良く見える偏り、予測変位の分散・符号・大傾斜の過小予測を確認する。

実行時GPU/CPU/空き容量/既存jobを再確認し、旧作業を無断停止しない。量産時間見積りはbenchmark有効rateから出す。現在の仕様段階で未測定の所要時間・精度を保証しない。

## 11. 予測精度：経路座標と490組

### 経路に対する前後・左右の定義

pred/GTを同じt0 yaw水平frameへ変換する。B0のroll/pitchを含むΔpをそのまま水平XYと呼ばず、R0とyaw0の関係を使って変換する。実行した候補の解析arcへ**GTの終点XYを投影**し、s_GT∈[0,.75]を求める。同距離のtieは小さいsと固定する。予測位置を別の投影基準で評価しない。

GT投影位置の、進行方向に向いたunit tangent T=d*(cos(θs_GT/L),sin(θs_GT/L))、N=(-T_y,T_x)を定義する。δ=predXY−GTXYとして、前後残差e_long=T·δ、左右残差e_lat=N·δ。後退でも**候補の進行方向が正**で、正の前後残差は進み過ぎを意味する。左は進行方向左。curveをbody前方固定軸で評価しない。path端を超えたGTは端点接線を使いclamp flagを付ける。s_GTは評価時だけ使用しNNへ入れない。

高さ残差はworld重力軸のΔz_pred−Δz_GTを報告する。body z誤差も補足できるが混同しない。高さに合否しきい値を追加しない。3D距離・SO(3)全体角も参考値。

### 姿勢誤差

相対回転をq0と合成した2秒後姿勢の同じZYX RPY同士を比較する。各軸は`wrap(pred−GT)=atan2(sin差,cos差)`の絶対値、度で集計し、+179°と−179°は2°とする。相対RPYの値も別列に保存するが、futureと混同しない。

pitch±90度付近はEuler非一意性があるため、SO(3)角とsingularity flagを必ず併記する。公式RPYのcanonical branchはpitch∈[−90,90]、roll/yaw∈[−180,180)に固定。特異点のroll/yawを任意の0誤差で埋めず、unassessableとして件数を出す。必要なRPY誤差が算出不能な例を合格例に数えない。

### 全490組の保存

`candidate_id(14) × field(7) × level(5)`のCartesian product **490行を必ず生成**する。各行で**実際に追従した候補**の有効endpointだけ集計する。未実行候補へ正解を複製しない。

各組の必須列：

- candidate ID/方向/角度、field、level、予定/実行/有効/欠測区間数、独立親試行数、各成分の評価可能件数、status。
- 前後cm・左右cm・height cm・roll deg・pitch deg・yaw deg：絶対誤差の**平均・中央値・P95・最大**。
- 左右≤10cm、前後≤30cm、roll/pitch/yaw各≤10度に入る割合、および5条件を同時に満たす割合。高さは同時合格条件に加えない。
- 前後残差の符号付き平均bias、SO(3)誤差、3D位置誤差、path projection clamp・Euler特異例数。

n=0の統計値はnull、表示は「評価データなし」。0.00やpassへ置換しない。RPY欠測もnullにする。主集計はpredictionが無効だった有効GT例の数を保持し、無効predictionは未達として合格率の分母から捨てない。

`evaluation/candidate_field_level.csv`、`per_window.csv`、`trial_summary.csv`、`missing_reasons.csv`、`coverage.csv`を保存。Markdown/HTMLで候補別（14表×35行）、field別（7表×70行）、level別（5表×98行）を同じCSVから生成し、フィルタやリンクで確認できるようにする。

さらに前進/後退、左旋回/右旋回/直進、field、level、候補、j=0の初期区間、j>=1の途中区間、開始時停止/移動/旋回/スタックで集計する。開始状態資格はt0以前の測定/解析ログから決定し、未来状態で分類しない。スタック中は直近1秒の要求|v|≥.1m/sかつ実測水平進行≤.02m等の**解析定義案**をコード/manifestへ固定する。t0以前1秒がない初期例はunknown。資格判定のraw履歴を保存してもNNへ履歴を追加しない。各表にwindow数と独立親数を併記する。

### 合否判定の提案：最大値を使用（未承認）

平均だけ／中央値だけでの合格は一部条件の大誤差を隠す。P95も5件/組では不安定なので、初期段階では**各組の最大絶対誤差**を正式判定に使う案を推奨する。

各490組で、有効GT・有効入力が5区間以上かつ5独立親以上、必要な全予測が有効で、max|e_lat|≤10cm、max|e_long|≤30cm、max|roll|,max|pitch|,max|yaw|≤10degの全てを満たすとpass。**490組全てpassした場合のみ今回の観測範囲で精度達成**とする。件数不足、評価なし、必要軸判定不能、無効predictionが残る場合は全体達成を宣言しない。有限サンプルでの合格であり未知走行の保証ではない。

平均／中央値／P95／最大の代替案も表へ保存するが、どれを正式採用するかはユーザー承認で固定する。平均が通ったから最大値基準を後から緩めない。高さ誤差が大きい場合は報告するが、指定のない高さ閾値で新たな不合格条件を追加しない。

## 12. 未達時の原因分析と追加の根拠

順番に切り分ける：①境界40step、quaternion/TF、DEM軸・高さ、reset跨ぎ等の実装不具合、②candidate/速度/follower/Actorのprofile不一致、③490coverage・動作状態・candidate遷移の偏り、④摩擦・滑り・遮蔽・現在入力だけでは見えない内部状態、⑤network容量・損失・学習不足。

成分別のworst例、前後bias、frame age、mask率、速度/姿勢/開始状態、Action/target遅れと誤差の対応を保存する。GT no-motionやconstant-twistとの差、train/validation learning curveを示す。追加収集は悪化cell/candidate/開始状態と必要独立親数を根拠として提案する。追加sensorや履歴を勝手に必須化しない。1frameでは識別困難なら限界として報告する。

final開封後の変更は既存testへの反復選択に使わず、testを既知の診断用と明記し、新しい独立seedの最終評価batchを別に固定する。失敗例や終了例を除いて目標達成扱いにしない。精度未達のまま経路選択の実験へ移行しない。

## 13. 精度達成後の最初の経路選択評価

NN入力は引き続きgoalなし。planner外側にだけgoalを与える。2秒周期で14候補を一括予測し、p_future=p_current+R_current*Δpから`score=distance(current,goal)−distance(pred_future,goal)`を計算する。進捗距離はworld水平XYのユークリッド距離[m]を初期案とする。NN出力の重力基準future tilt=acos(−g_future,z)を使い、**60度を超える候補を除外**する案を提案する。これは経験的な初期比較設定で、学習で安全性が証明された閾値ではない。承認前に確定・実行しない。

残候補の最大scoreを選ぶ。同点はcandidate index最小。負のscoreでも最大を選び、進捗下限や退避条件を追加しない。全候補除外時だけ2秒command0で待ち次周期に再評価する（filter結果を実行するための定義）。無効入力/NaN等は試行invalidとして記録・終了し、隠れた別経路評価へ切り替えない。

**持ち込まない条件**：DEM段差、支持面、踏面、スタック確率、転倒イベント確率、実測45度STOP、退避時間/距離、候補dwell、切替margin等。追加する場合は最初の結果を示して、理由と変更案の別承認を得る。follower固有のheading減速・goal toleranceは上記共通の実行条件であり、planner候補除外として使わない。

初回比較はA=同じ予測終点のゴール進捗だけ、B=同じ進捗＋60度予測傾斜除外の2条件。14候補、同じ初期状態/地形/goal列/走行seed/10秒時間上限/20Hz/Actorを揃える。各35cellで10独立seed×2条件=700走行案。予測精度評価とは別の事前固定seedを用い、同じ親のA/Bだけはmatched評価として同一地形を使用する。NNの再学習・閾値探索は行わない。

goalはtile上の初期yaw前方3mを初期案とし、最終到達判定.2mと10秒打切りを両条件共通に固定する。地形境界で成立しなければそのseedの設定不備を結果前に修正・記録する。未達到達は失敗/打切りであり完走扱いにしない。ゴール到達、進捗、転倒、スタック、filterによる停止秒、経路切替回数を記録する。転倒観測85度やsim衝突は結果ラベル・終端用途であり、追加の候補除外条件にはしない。

この最初の関数が実機で最適・安全と断言しない。実機確認前に4m/4Hz入力、joint符号、実Actor/bridge/follower差、計測source、shadow結果を点検する。実機走行は別途ユーザー操作/承認で実施する。

## 14. 成果物・共有・Git運用

| 共有段階 | 保存物 | 次の条件 |
|---|---|---|
| 今回の仕様共有 | 本仕様、参照commit/checkpoint/SHA、未確定事項、実機側確認票 | ユーザー仕様承認。収集・学習なし |
| 実装・少量検査 | schema/変換/teacher/validator、fixture、control profile、benchmark、候補割当manifest | 必須の制御・入力確認とprototype監査。量産と混同しない |
| 精度達成後の第1回upload | 収集/学習/精度評価/推論コード、全manifest/split/欠損集計、checkpoint/SHA/正規化/schema、490表と再現コマンド | 承認した全条件で予測精度達成 |
| 経路選択後の第2回upload | 最初の評価関数/設定、A/B結果、代表・失敗例、実機へ渡す設定/取得/検証手順 | 結果と未解決点を確認・共有 |

新branchを用い旧branch・checkpoint・実機plannerを上書きしない。originは実際には`/tmp/eatnav-r4-github-20261027-01`を向くため共有設定を勝手に変更せず、URL確認済み`github-pose2s`（https://github.com/TohokuQuix/visualnav-transformer-ros2.git）を明示してnormal pushする。force push禁止。今回の文書共有と参照commitの通知はユーザーが指定したS0共有であり、学習完了のuploadではない。

rawは新たな独立directoryへ保存し、path/取得方法/sha manifestを記録。巨大raw/optimizer/tensor全量/認証情報を通常Gitへ入れない。小型モデルは実sizeとrepo制約を確認して通常Git、必要ならLFS等を使い別PCからの取得とSHAを検証する。CPUでも14候補のΔXYZ、quaternion、relative/future RPYを取得できるsampleとCLIを付ける。

再現コマンドは実装後に実在するCLI名・引数・commitで記載する。未実装コマンドを動作確認済みとしない。推論時間はモデル単体と前処理込み、14候補batch、CPU/GPU別の平均/中央値/P95/maxを記録する。

## 15. 未確定事項と承認待ち

- 収集数：推奨1330試行、最大値判定、各組5独立例という提案の承認。
- 最初の経路選択：予測重力傾斜60度除外、XYゴール進捗、同点/全除外時の定義の承認。
- follower：今回のsim固定profile .5mと自動回復なしの承認。実機cycle .8mとは別設定で、将来同一視しない。
- 実機の4m/40×40/4Hz、raw観測mask、時刻、joint/IMU/odom/TF意味。取得不能な任意項目を追加sensor要件へ置き換えない。
- 外部sim依存コードとActorの実import/hash照合。参照先は特定済みだが新collectorの再現検査は未実施。
- 決定的地形hashがsplitを跨がざるを得ないcellの有無。実生成監査で判明した場合の限定主張・仕様変更。

本仕様の承認前には本格収集・学習を開始しない。承認した精度条件が未達なら経路選択へ進まない。今回の仕様共有は完了しても、モデル・dataset・精度・経路選択はまだ完成していない。
