# 実機条件に合わせた2秒位置・方向・姿勢予測 仕様案 v0.1

作成：2026-11-03。**承認待ち・実機整合性確認待ち。現在、本格収集と学習は禁止。**

参照コードcommit：`43bb7b69fde5d29095b6add17965aacd48ae58fa`。新branch：`exp/eatnav-real-conditions-pose2s-20261103`。本commitで行うのは調査・仕様・参照資料の共有のみ。既存モデル・実機planner・既存branchを変更しない。

[調査結果と実件数](real-conditions-pose2s-20261103/investigation.md)／[実機側確認票](real-conditions-pose2s-20261103/real-pc-compatibility-checklist.md)／[参照コード・SHA](real-conditions-pose2s-20261103/reference-index.json)／[gateの初期状態](real-conditions-pose2s-20261103/approval-gates.json)。

## 1. 目的と完了範囲

現在の実機plannerの14候補について、現在までに利用可能な観測と、これから適用する制御条件から、2秒後の相対3D位置・3D回転を予測する。今回は予測精度と正しい測定を目的とする。実機ではshadow推論・記録のみを想定し、走行指令・候補除外・順位付け・停止閾値へ自動接続しない。

完了成果は、仕様、実機と共通の変換・教師・検査コード、因果的dataset、checkpointと正規化、14候補推論例、独立評価、失敗例・未達目標、取得・再現手順。完全な14候補登坂順位の実機比較や安全保証は含めない。実機を走らせる操作は本仕様から自動的に許可されない。

**本格収集・学習の開始条件はAND条件**：

1. ユーザーによる本仕様の明示承認。
2. S1共有物を実機PCで確認し、入力・候補・制御profileの整合性結果を返却。
3. 必須未確定事項が解消し、control/observation contractが固定される。
4. 試作データ監査と小規模benchmarkがpass。

仕様承認だけでは2〜4をpassへ変更しない。確認待ち中に本格収集や試し学習を先行しない。任意項目の欠測を許容する場合も、双方で同じmask・source profileにする合意を記録する。

## 2. 調査で分かったこと

- 旧DEMは0.25秒間隔。4Hzを実機比較で最適化した根拠はなく既存設定の継承だった。実機bagはheader間隔中央値0.300秒、最小0.200160／最大0.399845秒。約300〜333msへ対応する必要がある。
- 実機bagの記録時刻−header中央値はDEM−0.320秒、IMU/joint/odom約−0.517秒。clock domain・stamp生成点が未確定で、この差を通信遅延として学習に使えない。
- elevation_inpaintedは全セル有限だが元elevationの有限セルは約53〜58%。inpainted有効と実観測ありを別に扱う。varianceには有限値がないため既知の高さ信頼度とは扱えない。
- 旧simはfront_foldedで開始してもfold mode0のActorが角度を変える。実機の方向clip等を再現した継続制御と同じではない。
- 実機の対象はユーザー確認の専用 **eat_nav_cycle_follower**。lookahead0.8m。旧simの0.5m followerとは異なる。専用source・完全runtime設定はまだ受領していない。
- 実機wheel odomは全件planar。map→body TFには3D成分があるが精度・生成元が未確定。既存bagで算出した15〜21cm／14〜17度は代理評価であり、独立した実機3D精度ではない。
- 旧train5,831件のDEMは全て250ms。完全な新契約に適合すると実証できた件数は現在0。ただし旧2秒教師や一部joint履歴は再利用候補であり、生記録は捨てない。

## 3. 比較した方針と推奨

| 案 | 利点 | 限界 | 判断 |
|---|---|---|---|
| A：旧生データの再加工だけ | 早い。2秒教師・既存splitを使える | 真の300〜333ms観測、欠けた指令/joint履歴、実機bridge処理を復元できない | 新実機条件の主datasetとしては不採用 |
| B：旧データを補助利用し、実機contract固定後に新規収集 | 生データを生かしながら入力・制御の差を分離できる | 実機側確認と新logger・制御parityが必要 | **推奨案。本仕様の承認対象** |
| C：手動実機bagの候補近似で直ちに再学習 | 実機観測は使える | 実行候補・採用指令・正確な3D教師が未確定。代理値へ適合する危険 | 入力検証・診断に限定 |

速度については、全てを0.30m/sの固定実速度とする案を採らない。**follower/bridge/flipperの制御profileを固定し、予定速度上限と角速度上限を明示入力するhybrid案**を推奨する。profile以外のgainやモードを変えた場合は別contractとし、未学習profileで無条件に推論を続けない。profile hashはアルゴリズム・単位・有効feature・分布設定を識別し、親ごとの乱数seedや摩擦の実現値をNN入力するためのIDではない。

## 4. 候補経路・予測する実行条件

### 4.1 14候補

順序は前進 `0,-30,+30,-60,+60,-90,+90`、後退も同じ7角度。directionは+1／−1。8点は始点を含み、解析arc長0.75mを等arclengthの7区間へ分割する。曲線の点間chord総和は僅かに0.75m未満。角度は接線の総回転で終点方位ではない。

`L=.75, s_i=iL/7, θ=angle_rad` とし、θ=0なら`p_i=(s_i,0)`、それ以外は `p_i=(L/θ sin(θs_i/L), L/θ(1−cos(θs_i/L)))`。後退はXY全体を−1倍する。t0のyaw水平座標、x前・y左、m。前後反転と左右符号をgolden testで確認する。候補ID・点配置・方向・geometry SHAを固定する。

### 4.2 実機follower

ユーザー確認値：linear_speed0.30m/s、lookahead0.8m、angular_gain0.5、max angular1.6rad/s、control20Hz、goal tolerance0.2m。launchは `/eat_nav/path`、`/eat_nav/follower_cmd`、TwistStamped、`chassis/body`、専用unused recovery topic、`publish_subcrawler_commands=false` を上書きする。

ただし完全onix.yaml、launch、source、実行binary、runtime dumpは未受領。減速式、angular_direction_sign、加減速、前後進判定、path終端/延長、recovery内部状態まで旧simと同一とはしない。0.8m lookaheadは0.75m候補より長く、終端接線・goal判定の差は特に検査する。

指令は次の3層を区別する：

`follower要求 → gate入力/出力 → bridge採用出力 → 駆動部実測状態`。

`/cmd_vel_auto`はbridge入力であり実出力ではない。bridge出力Twistも実際の車体移動速度ではない。NNへ入れる指令履歴と教師から計算する移動量を混同しない。

### 4.3 フリッパーは動的制御

物理角度は外向き水平0°、上向き90°、内側約180°。実encoderの符号・原点・offsetを別途確認する。

- 6action policy：policy出力、directional clip、干渉制約、ゼロ指令時保持等の実装順序を再現。
- fold modeを観測するpolicy：明示mode0は方向clipなし。未記録をmode0へ補わない。
- fold modeを観測しないpolicy：速度由来方向判定・履歴・drive-direction処理を再現。
- 2action policy：実測flipper角保持の別分岐を再現。
- front-fold方向の既定clipは前側≥π／後側≤π/2であり、前π・後0固定ではない。reverse時の実効前後はprofile・drive-direction次第。
- position/velocity出力モードを別ABIにする。velocity用kp3.0、limit1.5rad/s、deadband.02rad、およびenv max_velocity2.0はposition制御の実速度ではない。

policy checkpoint SHA、action次元、観測構成、mode取得、frontend/backend/launch/env/runtime設定を固定してから収集する。Actorへ渡す観測の組立・正規化・方向反転も実bridgeと照合し、旧simの観測ベクトルを同じ意味と決めつけない。RNN hidden state・filter・rate limiter・follower recovery timer等があれば継続状態を記録する。観測できない内部状態はlatentな制約として扱い、分岐時にゼロへresetして同じ初期状態と呼ばない。低位Actor checkpointと今回学習する予測NN checkpointは別物として索引へ記録する。

## 5. 時間・座標の共通contract

### 5.1 時間

t0は予測入力を確定したreference clock時刻。未来は`t0+2.000s`。header原値、bag/log時刻、実受信ROS時刻、実受信monotonic時刻、clock domainと変換根拠を保存する。記録時刻を取得時刻へ名前だけ変更しない。

NNが使えるのはt0までに受信・利用できた値だけ。headerとreference clockの対応は校正済み変換と誤差boundを記録する。未解消の負ageを0へclipしたり、中央値offsetを根拠なく差し引かない。clock reset/jump/episode境界を検出し履歴を切る。headerがmap更新時刻ならその意味を保持し、raw sensor exposure時刻とは呼ばない。

DEMは異なる4観測。4Hzへのstamp書換え・同じframeの複製は禁止。header間隔、t0からのage、到着からのage、利用可能なら配信lagを入力する。未知のageは既知の0秒ではなくvalid=false。重複ID・未到着・未来観測は採用しない。

state履歴は直近1秒を含む21参照時点（50ms間隔）へ、既に到着した観測を因果的ZOHで配置し、source age/maskを付ける。同じstateを参照した場合も同一source ID・元stampを保持し、新観測と呼ばない。連続状態の補間は両端がt0以前かつ利用可能な場合だけ許可し、方法をcontractに固定する。未到着の将来点から補間しない。

最大age・gap許容・clock誤差boundはS1の実測と実機確認でprofileへ数値固定する。未確認を旧250msや旧1秒へ自動設定しない。実行可能profileに必須値がなければcollector/trainerを拒否する。prototype parserは未確認理由を出せるが、本格profile成立とは扱わない。

### 5.2 座標・回転

body=B=chassis/body、x前・y左・z上。ROS quaternion xyzwは内部wxyzへ明示変換。物理量はm、s、rad、m/s、rad/s。予定pathはt0 yaw水平座標、位置教師はroll/pitchを含むt0全body座標であり区別する。

`Δp=R0ᵀ(p2−p0)`、`ΔR=R0ᵀR2`。NNは位置3成分と相対SO(3)の6D表現（第1列・第2列の連結）を出す。Euler直接回帰を主損失にせず周期境界を避ける。Gram–Schmidtの退化を検出しvalid=falseとする。

確認用にΔquaternion wxyz、relative roll/pitch/yawを返す。現在姿勢q0が有効なら`q2=q0⊗Δq`。`g0=R0ᵀ(0,0,−1)`、`g2=ΔRᵀg0`、future tilt=`acos(−g2_z)`。world高さ変化は`−g0·Δp`で、body zそのものではない。q0の観測誤差と相対回転予測誤差を別に評価する。

## 6. 提案入力schema v2

名称案 `eatnav_real_conditions_pose2s_input_v2`。S1でfeature manifest・unit・profile・dtype・maskを固定しhash化する。旧v1 loaderへ黙って読み替えない。B=batch、K=14（学習は実行候補K=1でも同じnetwork）、H=4、T=21。

| 入力 | shape / 意味 |
|---|---|
| dem | [B,4,5,20,20]。相対inpainted高さ、raw観測support、height-valid、vertical variance表現、variance-valid |
| dem_times_s / valid | [B,4,4]。取得age・受信age・観測間隔・配信lagと各valid |
| dem_relative_pose / valid | [B,4,5]。取得時局所化からt0への測定Δx/Δy、参照高さ差、sin/cosΔyawとvalid |
| state_history / valid / source_age_s | [B,21,59]。下表の固定順序。欠測をmaskで表現 |
| mode_history / presence | 操作mode、fold mode、drive direction等の版管理されたcategorical列。unknownと明示0を区別 |
| controller_context / valid | [B,21,6]の案。progress age、motion-window age、recovery active、boost残り時間、bridge travel direction、rate-limiter direction。実機でread-only取得できる列だけ有効 |
| paths / directions / candidate_mask | [B,K,8,2] / [B,K,1] / [B,K] |
| execution_condition | [B,K,2]。t0で決まっている予定並進速度上限・角速度上限 |
| control_profile_sha / observation_profile_sha | hashとsupport検証。学習で扱っていないprofileを無条件に受け付けない |
| current quaternion / source quality | 将来姿勢合成用。世界位置・地形IDはNN入力へ入れない |

state59成分の案：

| index | 成分 | 扱い |
|---|---|---|
| 0–2 | body projected gravity | 有効な姿勢sourceが必要 |
| 3–5 | body gyro xyz | source/単位/IMU extrinsicを確認 |
| 6–8 | body線速度xyz | wheel odom等の推定値。未観測zを真の0としない |
| 9–11 | body加速度xyz | 任意。重力を含むか確認後に有効化 |
| 12–15 | sprocket角速度 FR,RR,FL,RL | 軸・gear・単位を確認 |
| 16–19 / 20–23 | flipper角度／角速度 RR,RL,FR,FL | physical0/符号を校正 |
| 24–27 / 28–31 | flipper／sprocket effort | 物理単位が確認された場合のみNN有効 |
| 32–34 | follower要求vx,vy,wz | 要求層として保存 |
| 35–37 | gate出力vx,vy,wz | bridge入力層として保存 |
| 38–40 | bridge実出力vx,vy,wz | 実測車体速度ではない |
| 41–44 | flipper位置目標 RR,RL,FR,FL | positionモードでrad |
| 45–48 | flipper速度目標 RR,RL,FR,FL | velocityモードでrad/s。位置列と混用しない |
| 49 / 50 | 電圧／出力制限fraction | 任意・source/意味が確認された場合のみ |
| 51–58 | flipper4＋sprocket4のモータ電流A | 任意。実取得できなければ全sourceでmask=false |

controller_contextも既存topicにあるという主張ではない。未公開内部状態はsimだけNNへ与えず、両側でmask=falseとしlatent状態の限界を明記する。完全内部stateは分岐再現の解析用に保存できる。

これは全59成分を取得できるという主張ではない。S1で共通profileの有効列を固定し、実機で不明なeffort・電流等はsimでもNN入力を無効にする。raw native値と未知unitは解析用に保存できる。任意欠測は値0＋mask=falseとしてencoderで無効化し、学習平均で実測値を捏造しない。必須観測の欠測は予測不可と返す。

IMU covariance先頭−1は規格上その推定を使用しない指定。実driverの意図・有効性を確認し、数値があるだけで採用しない。代替TF姿勢／因果的角速度推定を使うなら別source profileとしてsim側も同じ観測モデルにする。

### DEM変換

元GridMapのheight・raw mask・inpaint mask・confidence源・grid pose・resolution・circular buffer indexを保存。inpaintedが有限でもraw観測済みとはしない。varianceがない場合は値0＋variance-valid=falseで、確信度が高い意味のゼロ分散ではない。

relative poseの高さ差は各frameの保存z_refと最新の利用可能frameのz_refとの差であり、t0に存在しない新DEMやsim真のbody高さから作らない。XY/yawはt0までに利用可能な測定TFから計算する。

各frameは取得時の測定body yawで局所化し、bilinear kernel、軸向き、annulus 0.3–0.8mの相対高さ基準を両PCで一致させる。高さ基準値とsupport数も保存。raw観測supportは補間に使う観測済みneighborの重み和で、確率的信頼度とは呼ばない。地図外・欠損は明示mask。過去frameを現在frameのコピーで置換しない。TFの補間両端はt0以前に利用可能なものだけ。

新規simはroot真値速度をNNへ直接流さず、実機の推定sourceを模したvirtual wheel odom・IMU等を使う。真のroot pose/速度、摩擦・接触・地形IDは教師／解析領域へ隔離しallowlistでNN入力漏洩を検査する。

## 7. 出力・不確かさ・イベント

`eatnav_real_conditions_pose2s_prediction_v2`、horizon2.0s、candidate ID対応を必須にする。

- Δposition_body_m [B,K,3]。
- Δrotation_6d [B,K,6]、Δquaternion_wxyz、relative_rpy_rad、回転validity。
- q0が有効ならfuture quaternion/RPY、future projected gravity/tilt。Euler特異点flag付き。
- contract/profile/normalization SHA、使用観測時刻・欠測・未確認source、support/OOD注意情報。
- 任意の校正済み誤差半径とcalibration provenance。未校正の分散を「信頼度」と呼ばない。

推奨初版はpose主目的のnetworkとし、4イベントheadは主学習から外す。旧4headは既存モデルで維持。新datasetにはtip85度、near60度5sample、2秒stall等の旧定義の派生ラベルを解析用に保存できるが、姿勢の正解と混同しない。イベントhead追加はvalidation上のablationでposeを損なわない場合の別版とする。

不確かさはまず独立calibration親のresidualによるsplit-conformal型の位置／角度誤差半径を検討する。windowを独立として水増しせず親単位max残差等で校正し、評価側でcoverage・幅・条件差を測る。各95%半径から同時95%や候補条件付き保証を主張しない。少数親／未知profileでは未校正としてnull理由を返す。入力自体が無効な場合は半径で正当化しない。

## 8. 再利用と新規収集

### 8.1 再利用

旧train5,831件は旧時刻・旧制御を維持したlegacy profile。2秒教師、途中GT、利用可能なstate/commandは残す。旧validation/audit/reserved testを新trainへ移さない。legacy trainだけから新しいfeature mask・source・profileを監査して補助事前学習に使うか比較する。実機適合データへ名前だけ変更しない。

rawの250/500ms間隔を実時刻のまま扱うことは可能だが、300–333msの取得を新たに作れない。joint履歴pool916枝も上限で、確定有効数ではない。再加工後にfield/level/profile/欠測別の認定数を出す。

### 8.2 新規取得の対象

flat control plane、random_rough L1、boxes L1、hf_pyramid_slope L4、pyramid_stairs L4、pyramid_stairs_inv L4、k_rail_field L4、k_rail_step_field L4の8cell案。flat controlは新しい平面診断sceneで、rough L1を完全平面と呼ばない。

停止開始だけでなく、途中移動・旋回・後退・登坂前/途中・下り・段差前/乗り上げ・低速・進行阻害を含める。状態は現在以前の速度・gyro・DEM勾配・world vz等の解析真値で資格を付け、field名やpitch符号だけで決めない。解析真値はNNへ入れない。滑り資格はsimの実移動と駆動部から判定でき、実機wheel odomだけで同じ正解が得られるとはしない。

主収集は継続episode中に2秒単位で候補を切替える方式。t0ごとに現在までの履歴を作り直し、2秒区間内部では予定候補・制御profileを変更しない。区間終端で次の候補を現在位置から出し直し、短い初回goal付近で停止し続けるだけのデータにしない。これは制御された予定候補の変更であり、区間内appendではない。

予定量は **8cell×3速度上限(.10/.20/.30m/s)×56親episode×14区間=18,816区間**。1親はwarmup後に28秒、14候補を各1回の事前固定順序で実行する。同一初期状態の14分岐ではない。速度上限の低い2値は設計した感度実験で、実機でその分布を観測済みという意味ではない。全cellの実機対応を主張せず、実機の通常上限.30を主評価subsetにする。

56親/cell-speedはtrain36／selection6／calibration6／final-test8へ開始前固定（予定区間12,096／2,016／2,016／2,688）。実際の有効件数は終了・欠損で減る。転倒やスタックを増やすまで無制限にseedを足さない。再収集が必要ならS2で理由と別batch計画を明示し、既存結果を隠さない。

代表状態の同一初期条件14分岐は任意の小さな追加層（上限20親×14=280枝）。復元できなければmatched-prefixか単独枝にし、反実仮想completeを偽らない。網羅的登坂順位比較は完了条件にしない。

### 8.3 観測・物理の変動

- 0.300秒／1/3秒の取得schedule、実bagで得た間隔列のblock samplingを検討。後者はこの26秒観測の再現に限る。physics tickへ量子化した実取得時刻を保存し、理想stampへ戻さない。
- 配信遅延はclock校正後に測れたものだけを実機由来profileにする。現時点では未確定。欠損・遅延stressの別profileを作る場合も、実機分布と偽らない。
- 元観測maskとinpainted高さを別に生成・記録。同じ上流filterが再現不能ならempirical-mask＋明示した補間という近似profileとし、物理的な実LiDAR再現を主張しない。
- 摩擦旧nominalはstatic/dynamic各1.0。実機同定値ではない。例としてnominal倍率.7/1/1.3の感度実験を提案するが、同定済み分布とはせず、根拠・範囲をS1で承認して固定。乱数値は環境metadataのみ。
- actuator遅れ・負荷・出力制限は実出力とjoint応答がある記録で同定できた範囲を優先。不明値はstress profile。positionモードの実速度へenv値をコピーしない。
- 加速度・電流・effort・電圧等の未知物理対応をNNへ混ぜない。任意channelを外す場合はsim/realで同じmask policyにする。

カメラ・大量点群は保存しない。低解像度DEM、元height/mask、変換pose、state/command、設定・タイムスタンプを保存する。真値root/joint/contactはsim解析・教師専用の別領域。

## 9. 教師・介入・欠測

t0直前の状態と、正確にt0+2秒のpre-reset状態を対応付ける。20Hzなら40制御区間、physics .005sなら400step。root_stateは区間開始、post_root_stateは作用後と明示し、最後のpostを40区間目の終点にする。境界時の次候補変更前に教師を確定する。

- 自然終了・転倒が2秒未満：未来poseを捏造せずpose_mask=false。終端理由・観測済みtrajectory/派生イベントを保持。
- auto reset・別episode・clock jumpを跨がない。最後のpre_resetを記録し、reset後rootを未来へ接続しない。
- 区間内の予定外候補変更、手動介入、mode/profile変更、外部gate停止、設定変更は検出し、主たる「同じ予定制御の2秒結果」から除外／別層。自然なfollower減速や同profileのflipper調整は除外しない。
- 指令欠測で介入有無を判断できない場合は適合不明。候補近似だけで実行済み14候補の教師を作らない。
- teacher補間は同episodeの有効poseで短いgapのみ。simはexact tickを優先。実機proxy補間の許容gap・品質は別schemaで固定し、proxyをsim真値と混ぜない。

20Hzの途中GTは保存するが、初版NNは終点だけを予測する。最大傾斜の将来予測は今回の完了条件ではない。2秒完全観測例に偏る生存選択の限界を明記する。

## 10. 効率化と量産前benchmark

GUI OFF、sensorは必要なもののみ。旧sim physics dt=.005s、decimation10、20Hzの制御を維持する。実機のfollower/bridge実rateが違えば黙って変更せずS1でprofileと実装を整合させる。センサー取得をphysics eventで行い、control .05sへ理想stampを押し込まない。

benchmark案：single-process 2/5/10env、可能なら2process×5envを同じ親・地形assignmentで比較。num_envs変更で地形tileやseedが変わらないよう親manifestへ固定する。multi-processはGPU競合・DDS・保存帯域も測る。実時間sync/sleepを外してwall時間を短縮し、物理dtや制御periodを変えて倍速を作らない。

S1のprototype・benchmark・再現性診断を合わせて新規保存上限320区間。benchmarkは4構成×2回×各32予定windowを上限とし、prototype24枝と少量診断を含めても上限を超えない。有効数を揃えるための無制限な取り直しはしない。

各設定はwarmup後に同じ事前予定window数を2回測り、打切りで減った実際の有効数を含めて次を保存：

- 有効完全2秒window/秒、観測sim秒/wall秒（総env値とper-env値）。
- qualification待ち・初期化・sensor生成・follower/Actor推論・保存別時間。
- GPU peak、CPU/RAM、disk帯域・必要容量、失敗/欠測/終了率。
- I/O queue長・最大待ち。losslessで遅ければbackpressureし、sampleを黙ってdropしない。

観測payloadはevent単位1回保存し、20Hz state rowからID参照。巨大frameを毎row複製しない。数値保存はshard化・圧縮・別writerを測定し、途中終了でも閉じたshardとledgerを再開可能にする。atomic manifest/seal、source/config SHAを付ける。

速度だけで選ばず、制御/観測/教師検査pass、安定反復、VRAM余裕、同じ有効結果を満たす設定のwindow/秒で選ぶ。最初から10台の成功を断定しない。量産所要時間はbenchmarkの有効rateから再計算して共有する。既存R6等を無断破壊しない。

## 11. 学習案

初版は各DEMを共通CNNで符号化し、実time/relative poseを伴う4-token temporal attention、state59＋mask/ageの小型GRU、候補path MLP、予定速度条件MLPを融合する。瞬間値だけでなく指令→駆動回転→車体推定移動・負荷履歴を使う。body速度がwheel由来ならslipを直接測れたとはしない。

旧path encoderの入出力が一致する部分は転移候補。DEMは旧8channel一括入力からframe-wise5channelへ変わるため、旧CNNを同一機能のままコピーできるとはしない。state追加・unit変更にも同じ注意を適用。旧モデルは上書きせず、新schemaのモデルにする。

主損失案：trainだけで標準化したXYZ SmoothL1＋相対SO(3) chordal loss＋6D退化抑制。初期比率は旧実績の1:.5:.001を出発点とし、pose-onlyでまず比較。位置std floor .05/.05/.02m、入力std floor .001、角度6Dは非標準化。forward biasは符号付き残差で監視し、testを見てlossを調整しない。

予定比較：①新規主データのみ、②旧legacy補助事前学習あり、③time featureなし、④指令/状態履歴なしの限定ablation。全て同じ親split・予算・初期seed123で比較し、追加seedは採用候補のばらつき確認に限定。ablationのため再収集しない。hyperparameter試行はselection-validation上の事前列挙・上限6設定まで。最大150epoch、最短30後patience20、best epochをvalidationで固定。

GPU学習はcollectionと競合させず、batch64/128/256を短いgateで比較。loader worker0/2/4、pinned memory、shard cache、GPU利用率とsample/秒を測る。正規化・分位点・標準化範囲はtrainだけ。保存済みtrain IDsと全source/hashを残し、校正/最終testをfitに使わない。

## 12. 分割・評価・原因分離

### 分割

親episode・近接window・prefix・全分岐・観測再利用・初期状態を同じroleへ置く。source ancestry・hashで交差を検査する。地形hashも集計し、未知地形を主張するsubsetはhash非重複とする。K-Railの同一形状がseedだけで再生成される場合は既知地形として明示する。

selection-validationとcalibrationとfinal-testは別親。final-testはモデル/normalization/epoch/評価式/閾値のfreeze後に開く。既存R6/reserved testを開かない。既存26秒実機bagは観測設計に使用済みのdevelopment入力検証用で、未知の正式実機testではない。

### baseline

- 現在位置・姿勢維持。
- t0までの実測/推定速度・角速度によるconstant-body-twist SE(3)積分。wheel由来であるsourceを明記。
- 旧M_move/M_final。厳密250ms CLIは新契約を受理しないので正式に互換があるとはしない。比較用shadow adapterで4枚の実観測を時刻を変えず旧NNへ通す場合、時間不一致のout-of-domain診断baselineとして明記し、実機運用へ置かない。受理率も出す。
- 新dataset上で同じ条件のtime-blind ablation。旧architecture再学習を使う場合も新時刻のまま扱う。
- 予定速度上限×候補arcの幾何baselineは比較可能なXY/yawだけ。減速/滑りを無視する上限走行を正解にしない。

### 指標

全体window加重とfield/state macroを併記し、独立親数・window数・欠測/除外を出す。平均・中央値・P95・最大、XYZ成分MAE/RMSE、3D距離、SO(3)角、wrapped roll/pitch/yaw、将来重力tiltを報告する。成分別分位点は絶対誤差、符号付きbiasは別列。

forward biasはbody dxのpred−GTを前後進別に示し、direction×dx残差の過大走行量も出す。真の大傾斜帯で`max(0,tilt_GT−tilt_pred)`と大きな過小予測件数を示す。Euler特異点ではSO(3)/gravityを主としyawを無理に0誤差にしない。

平坦・登坂・下り・旋回・後退・乗り上げ・低速/停止・進行阻害、速度上限、観測age/mask率、mode、同定済み/感度profileで層別。親bootstrap95%CI。20未満の独立親など小さい層の推定は不確かと明記する。

ユーザー目標案：3D平均≤.10m/P95≤.20m、future roll/pitch各平均≤5°/P95≤10°、yaw平均≤10°/P95≤20°。まずheldout simの目標とする。既存独立simの値と比べ実現性はあるがP95・下り・低速等で未達の可能性がある。formal realの同じ主張には精度が確認された独立3D referenceと指定候補の実行記録が必要。TF代理だけで合格にしない。未達でも隠さず成果と原因を共有する。

原因分離は同じ入力/同じlabelを固定して実施：

1. 両PCの同一fixtureで変換tensor/順序/時刻を比較（float目安atol1e−6、整数ID/時刻完全一致）。
2. 同じ観測・mode・内部状態で実機ソフトとsimのfollower/bridge action処理をreplayし、指令・flipper target parityを検査。
3. sim内で観測proxyと真値の差を計測。NNだけにoracle観測を与えた診断は別表示し、同じ実行結果に対する入力感度として使う。oracleを実機用NN入力へ混ぜない。
4. 整合した候補/制御subsetでモデル残差を評価。幾何近似候補のmanual bagは別のproxy診断。

NN単体と前処理込みAPIを分け、CPU/GPUの14候補batch1、warmup20・200回、median/P95/maxとcold loadを保存。GPUは同期計測。real-PC時間は実機PCで別に計測し、学習PCの数値を流用しない。

## 13. 共有地点と実行順

| 地点 | 共有内容・Git運用 | 次に進む条件 |
|---|---|---|
| S0：今回 | 本仕様、調査、参照code/config commit・SHA、確認票を新branchへ。product実装/新収集/学習なし | 仕様承認。未確定項目を自動承認しない |
| S1：本格収集前 | 承認後に変換/教師/validator、少量prototype、14候補入力例、検査commands、control/observation profileとbenchmarkを同branchの別commitで共有 | **実機側の整合性report＋必須事項解消＋prototype監査pass＋benchmark pass**。返答待ちは停止 |
| S2：dataset完成 | manifest/source/split/件数/除外・欠測/代表sample/検査/再現commandsをcommit。raw全体はlocal別保存 | 監査pass。重大な不一致なら学習を止め報告 |
| S3：学習完了 | checkpoint/SHA/正規化/feature schema/inference例/依存/互換性をcommit | CPU再load・14候補/K1/順序置換・旧schema拒否・有限性検査pass |
| S4：評価完了 | baseline/条件誤差/失敗例/採用理由/未解決点/再現commandsをcommit・push | 独立レビューとremote/model bytes確認。未完了を完了としない |

承認後S1のprototypeは、control profileの暫定実ファイルが揃った範囲で平坦途中・旋回・後退・段差前の4状態×2親×3実行候補程度（上限24枝＋小さい再現性診断）。未実行11候補等に教師を付けず、全14候補の入力例は保存できる。controller情報が届かなければschema/real bag変換検証だけを進め、推測controllerで実機一致のprototypeを作らない。新モデルの学習はS1合格まで行わない。

S2以後は仕様内の通常実装・検証に再承認を求めず進める。ただし必須契約変更・破損危険・権限不足・実機側確認の未解決で依存作業を止める。正式testを見て同じtestへ再適合しない。

## 14. 保存・取得・実機側確認

既存branchを上書きせず、force push禁止。originの旧ローカルpathは共有repoのため勝手に変更せず、正しいGitHub URLを確認したremoteを明示してpushする。最終branch/commit、各共有commit、参照code/config commitとhashを返す。

Gitにはコード、仕様、small prototype、manifest、評価、model index、通常Gitサイズ内のcheckpointを保存。旧モデルは約150KiBなので通常Gitが使えたが、新checkpointは実sizeを確認し、巨大ならLFS等の利用可能な方式を選び取得とSHAを実証する。raw、全tensor、optimizer、大量点群、認証情報、一時runtimeはcommitしない。外部codeは文書参照snapshotと実装依存を区別する。

実機PCでは現在の走行branchをswitch/上書きせず、別worktreeで仕様とsnapshotを確認する。確認票のread-only commandsでruntime parameter・publisher・型・順序・使用checkpointを収集する。**ディスク上の既定yamlや同じsource commitだけでは実行設定の一致を証明しない。**

本仕様の未確定事項は確認票のU01〜U12。特に専用cycle follower、Actor/bridge分岐、時刻domain、状態source/単位、出力topicと配列順序は本格収集前の必須確認。電圧・電流・未校正effort・未知の摩擦分布は取得不能なら任意mask/近似profileとして扱い、実機一致と偽らない。独立3D referenceがなければformal real精度は未評価と報告する。
