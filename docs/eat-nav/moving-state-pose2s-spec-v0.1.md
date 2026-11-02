# EAT-Nav：走行中状態からの2秒位置・回転・イベント予測 仕様 v0.1

作成：2026-11-02 JST。状態：承認待ち。関連：[事前調査TXT](r4star-r5-2s-pose-data-investigation-20261102.txt)、[実行計画](moving-state-pose2s-execution-plan-v0.1.md)、[再監査集計](moving-state-pose2s-pre-spec-audit-20261102.json)。

本書のモデル構成・数値・収集件数は、実ファイルを確認して作成した提案であり、学習・精度確認の完了結果ではない。利用者の承認を受けるまでは、文書以外の変更、収集、学習、pushを行わない。

## 1. 目的、範囲、期限

現在までのDEM・車体状態と候補経路から、その候補を固定の追従制御で実行した2秒後の位置変位・相対回転と、今後2秒以内の4イベント確率を予測する。停止状態だけでなく、実際の後続意思決定時点の入力を使って学習する。

必須は、14候補一括推論、実行可能な学習済みcheckpoint、独立走行による基本精度評価、漏洩監査、CPUで動く取得・推論例である。2026-11-03 23:59 JSTまでに利用可能な初期モデルと推論コードを優先して用意する。追加収集後の最終版も同日中を目指すが、実測資源・障害に依存する見積りであり保証ではない。

今回は実機plannerの除外・順位・STOP条件を変更せず、ROSへの走行指令も送らない。候補選択による登坂性能の最終対照実験、大規模な閾値探索、接触前性能の保証は対象外。物理的な接触識別は学習開始条件にしない。

## 2. 作業開始時の実測と保全

2026-11-02 22:44 JSTの読み取り結果：

| 項目 | 実測・状態 |
|---|---|
| 作業ブランチ | exp/eatnav-r6-2s-pose-thresholds-20261030 |
| HEAD | 5d50730a96ca072a4cd482a8d1c7dab9defbcca6 |
| 作業場所 | /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree |
| origin | /tmp/eatnav-r4-github-20261027-01。現時点で実体なし。GitHub remoteと同一視しない |
| 既存未追跡変更 | r6-2s-pose-20261030内のcollection-plan-before-anchor-schedule.json、collection-plan-pre-exclusion-expansion-draft.json、collection-source-hashes.json |
| GPU | RTX 4090 Laptop GPU、16,376 MiB、使用7,426 MiB、利用率96% |
| CPU／RAM | 32論理CPU、RAM62 GiB、利用可能約53 GiB |
| 保存容量 | /と/tmpは同一ディスク。空き261 GiB |
| 稼働処理 | R6評価pipeline PID91045とIsaac評価PID152579が稼働。PIDは実行時に再確認する |
| 既存成果物 | R4*関連約2.7 GiB、R6関連約15 GiB。削除・上書きしない |

仕様承認後に専用worktreeと新規ブランチ exp/eatnav-moving-state-pose2s-20261103 を作成する。基点は上記HEADを記録し、既存R6の改変されていないコードから必要部分を参照する。既存の作業場所でcheckoutを切り替えない。新しいraw・dataset・モデル出力は専用ディレクトリを使い、既存seal・dataset・checkpointに書き込まない。

利用者は資源調整について「仕様承認後、R6評価をバッチの区切りで保全・一時中断し、今回の追加収集を優先」を選択済み。承認後、まず実行中バッチの完了・出力・完了マーカーを確認して保全し、次バッチの投入を保留する。途中バッチ・Isaac全体・コンテナを強制終了しない。現在のpipelineに安全な保留APIがあるとは未確認であり、専用監視から親スケジューラのみを安全に保留できるか実装前に調べる。できなければ無理な停止は行わず、CPUで初期学習を先行し、GPU資源調整が未完了であることを報告する。R6のキュー、現在バッチ、再開コマンド、完了済みhashをpause-record.jsonに保存する。今回のGPU処理が終わり復帰条件を確認した後、保留したR6を再開する。

## 3. 既存モデルの再確認と採用案

対象は exp/eatnav-r4star-r5-code-checkpoints-20261028 の random_rough / seed123。

checkpoint：docs/eat-nav/r4-star-fourteen-training-six-evaluation-20261026/models/random_rough/seed123/checkpoint.pt

SHA-256：c29bef5a56947db03f91ae2261dd8d259762479ebd9d38553017f3d5bd3c0421

Git内とローカル実ファイルで一致する。既存リスクheadの学習は fall_stall_learning.py の event_targets[:, :, 3, :]、すなわち10秒教師である。教師配列自体には1・2・5・10秒分がある。既存進捗headは4時点のスカラー進捗を予測し、XYZ・回転は出力しない。指定checkpointの実学習はfresh seeded 28D、300epoch、best170であり、既存出力を2秒予測と読み替えない。

| 案 | 利点・制約 | 判断 |
|---|---|---|
| 既存特徴抽出部＋新しい終点・2秒イベントhead | 既存入出力準備を利用できる。小型でCPUでも学習可能。移動中の正規化変更を含め全体を微調整する必要あり | 推奨・本仕様の採用案 |
| 同じ構造を全て初期化して学習 | 転移への依存がないが、少数の希少イベントで不利な可能性。追加比較の時間が必要 | 時間が残れば補助比較 |
| R6の20点軌跡モデルをそのまま使用 | 軌跡を出せるが、yaw基準の位置とEuler派生表現であり、今回の全車体座標・6D回転契約と異なる | そのまま転用しない |

DEM encoder、28D state encoder、candidate encoder、fusionのみを名前・形状で検証して転移する。既存の10秒risk headと4時点progress headは転移しない。2秒イベントhead、XYZ head、6D回転headを新規初期化する。転移元の最適化状態は使わない。転移元の学習に使ったepisodeを新評価側へ入れない。

進捗headとネットワークによる途中軌跡予測は初期版では設けない。XYZと経路から導く幾何学的な進行量を必要なら補助表示できるが、既存progress出力と同一とは呼ばない。途中40点の真値軌跡は保存し、後続拡張を可能にする。

## 4. 入力契約

入力schema：eatnav_moving_pose2s_input_v1。float32、有限値必須。未来観測・実行後指令・未来姿勢・イベント結果は入力へ渡さない。

| 入力 | 形状 | 内容 |
|---|---|---|
| dem | [B,4,2,20,20] | 現在まで4枚の高さ／有効mask。モデル内で8channelに変換 |
| actor_state | [B,22] | 既存pre-noise・observation-scaled値。現在指令のindex6/7/8を除外 |
| body_state | [B,9] | projected gravity、車体座標の線速度、車体座標の角速度 |
| paths | [B,K,8,2] | 現在の車体yaw座標での局所XY候補。K=14で一括推論 |
| directions | [B,K,1] | 前進+1／後進−1 |
| candidate_mask | [B,K] | 入力候補の有効性。教師の観測有無とは別 |
| current_quat_wxyz | [B,4]、任意の付帯情報 | 絶対未来回転を合成する場合に使用。NN特徴には追加しない |

車体状態の28成分は次で固定する。actor部分は既に観測scaleを適用済みで、APIへ二重適用しない。

| 28D内index | 内容 | 物理値からactor値へのscale |
|---|---|---|
| 0–2 | actorの車体角速度xyz | 0.25、元単位rad/s |
| 3–5 | actorのprojected gravity xyz | 1 |
| 6 | fold mode | 0自由、+1前fold、−1後fold |
| 7–10 | フリッパー相対関節位置 RR,RL,FR,FL | 1、rad、default joint positionとの差 |
| 11–12 | 右・左trackの平均関節速度 | 0.05、元単位rad/s。車体速度とは別 |
| 13–16 | フリッパーeffort RR,RL,FR,FL | 0.01、通常N m |
| 17–18 | 右・左trackの合計effort | 0.01 |
| 19–21 | root由来のprojected gravity xyz | 1 |
| 22–24 | root由来の車体線速度xyz | m/s |
| 25–27 | root由来の車体角速度xyz | rad/s |

傾きはgravityに含まれ、roll/pitch角の直接channelはない。角速度・gravityの重複も旧特徴の一部として維持する。フリッパー関節速度は4関節分の独立入力としては入っていない。今回必須の入力追加はしない。

実機側はIMU／TFの取付変換・速度の表現座標・JointStateとdefault相対角度・trackの集約・effort単位を対応付ける。利用者の「水平0度、上垂直90度」は実機角度の基準であり、シミュレータのjoint defaultや各軸の符号と同一と仮定しない。校正差を学習平均で埋めない。入力が取得できない場合は欠落を検出し、補完や実機での性能保証を黙って行わない。実機PCの実測topic検証は別環境への接続がない限り未検証とする。

DEMは既存20×20・4Hzの履歴、各取得時点のyaw座標、既存の相対高さ基準を保つ。過去画像を現在位置へ再投影した履歴とは解釈しない。原則t0直近0.25秒以内の最終DEMと、0.25秒間隔の4frameを要求する。実機側の履歴不足・古すぎる観測はエラーにする。

## 5. 座標系・時刻・回転・出力

t0は候補を再設定した直後、最初の制御を加える直前の車体状態。20Hzで40回の制御区間後をt0+2.00秒とする。入力snapshot・trace pre-state・時刻の一致を確認し、行数だけで合わせない。

rootの位置pはworldのm、qはworldへ写す車体回転のwxyz。車体座標はx前・y左・z上、右手系。候補XYはyaw座標であるが、位置教師はroll/pitchを含むt0の全車体座標で定義する。rootと実機base_linkの原点・固定回転の一致をadapter契約で明示する。

    Δp = R(q0)^T (p2 − p0)
    ΔR = R(q0)^T R(q2)
    Rfuture_world = R(q0) ΔR

回転headはΔRの第1列3成分と第2列3成分をこの順で連結した6Dを出す。Gram–Schmidtで右手系SO(3)へ変換する。Eulerの±π折返しとquaternionの符号二重性を回帰損失へ持ち込まない。identity biasは[1,0,0,0,1,0]。復元epsilon=1e−6。退化出力を検出しrotation_valid=falseを付け、fallback identityを正常な予測として扱わない。

推論schema：eatnav_moving_pose2s_prediction_v1、horizon_sec=2.0を必須にする。

| 出力 | 形状 | 単位・意味 |
|---|---|---|
| delta_position_body_m | [B,14,3] | t0車体座標の前後・左右・上下、m |
| delta_rotation_6d | [B,14,6] | 上記列順、非標準化 |
| delta_quaternion_wxyz | [B,14,4] | 正規化済み相対回転 |
| relative_rpy_rad | [B,14,3] | roll,pitch,yaw、R=Rz(yaw)Ry(pitch)Rx(roll) |
| future_projected_gravity | [B,14,3] | ΔR^T g0 |
| future_tilt_rad | [B,14] | acos(clamp(−g_future_z,−1,1))、正立0 |
| future_quaternion_wxyz／future_rpy_rad | [B,14,4]／[B,14,3] | current_quatがある場合のみ合成。ない場合は欠測＋理由 |
| event_probability | [B,14,4] | tip_over,near_fall,linear_stall,turn_stallの順、2秒以内 |
| validity／support | 候補別・head別 | 入力、回転復元、学習陽性支持、範囲外警告 |

入力quaternionは有限・norm>1e−8を要求して正規化する。出力quaternionはw>=0を標準とし、w=0付近は最初の非zero成分を正にする。Euler診断はgimbal lockを明示し、主たる回転誤差はSO(3)角度で計る。future tiltは現在のgravityだけでも計算でき、world yawを追加学習する必要はない。旧checkpointは新loaderが拒否し、新checkpointを旧risk10s loaderへ渡すことも拒否する。

## 6. 既存元記録の実確認と採用件数

元データの基点をO=/home/ubuntu/visualnav-transformer-ros2とする。

- 新14候補dataset：O/docs/eat-nav/r4-star-fourteen-training-six-evaluation-20261026/datasets/{field}/。
- その旧4候補raw：O/docs/eat-nav/all-fields-learning-20261023/collect-*/sim-output/。
- 追加10候補raw：O/docs/eat-nav/r4-star-fourteen-training-six-evaluation-20261026/collection-runtime/collect-*/sim-output/。
- 共通dataset：O/docs/eat-nav/additional-development-batch02-cumulative-20261021/cumulative/。
- 共通raw：/home/ubuntu/Projects/onix_isaac_lab/.eatnav-fall-stall-additional-v0.1-batch01、および同v0.2-batch02。

sample-index.json、targets.jsonl、plan.json、input-manifest.json、terrain.jsonとraw traceを実読した。指定モデルの初期736件は全て2秒姿勢あり、延長13件を除くと723件で再確認できた。7field全体の初期・後続を再集計した採用前件数は次の通り。

| 区分 | 入力開始点 | 完全2秒姿勢 | 早期終端で姿勢欠測 |
|---|---:|---:|---:|
| 14候補の初期、延長なし | 1,290 | 1,288 | 2 |
| 共通4候補の初期 | 512 | 512 | 0 |
| 14候補rawの後続、延長なし | 5,095 | 5,080 | 15 |
| 合計 | 6,897 | 6,880 | 17 |

これはcontent dedup前の記録件数であり、独立事例数・事故件数ではない。以前の「後続5,835件」は延長ありも含む完全2秒区間数。実際には入力がある後続開始点5,851、完全2秒5,835、早期終端16。完全区間内の延長755件と、延長あり早期終端1件を除いたものが主採用候補である。

後続5,851件全てで現在rootとpre-state、前postとの連続性、body再計算、actor angular/gravity、DEMの因果的時刻、anchor offset、候補・direction・path originを照合した。現在DEMを原rayから再構成する層別spotcheckは52/52一致。履歴4枚全てのraw ray再構成を全件実施したとは主張しない。

代表例：collection-runtime/collect-640/sim-output/input-0040.npzは3→5秒、DEM時刻2.25/2.50/2.75/3.00秒、forward_-60。all-fields-learning-20261023/collect-324/sim-output/input-0240.npzはanchor10＋相対40stepで13→15秒であり、filenameのglobal offsetを無視しない。

後続の完全姿勢5,080件は全てlevel4：

| field | 元train | 元validation | 合計 |
|---|---:|---:|---:|
| boxes | 564 | 191 | 755 |
| hf_pyramid_slope | 476 | 163 | 639 |
| k_rail_field | 461 | 0 | 461 |
| k_rail_step_field | 604 | 211 | 815 |
| pyramid_stairs | 529 | 173 | 702 |
| pyramid_stairs_inv | 639 | 217 | 856 |
| random_rough | 642 | 210 | 852 |

速度ノルム<0.05m/sが897、0.05–0.2が2,943、0.2以上が1,240。車体|yaw角速度|>=0.2rad/sが949。pitch<−10度が654、±10度が3,939、>10度が487。これは瞬間状態であり、登坂・接触等の地形局面の確定件数ではない。

後続の独立terrain seedは元train41／元validation12。全候補・anchor0/10・後続区間は共通の親seedに紐付ける。後続区間の未実行13候補に正解を複製しない。1つの観測候補結果として学習し、14候補の反実仮想比較データとは呼ばない。

## 7. 教師生成とイベント

snapshot t0ごとに入力を読み直す。初期入力の使い回し、未来値からの補間・逆算、future命令の特徴入力は禁止。40個の0.05秒区間、時刻差、root13の有限性と単位quaternion、episode・reset・境界、実行候補を検証する。自然終端ではterminal.pre_reset_stateを優先し、リセット後rootを採用しない。

主学習は「t0で作った候補＋固定follower/Actor、次の定期再計画直前まで」の2秒に限定する。t0自身のreplanは許可し、区間内部の候補切替・replan・append・外部STOP・recovery命令介入を検出する。append付き既存区間は主datasetから除外し別集計する。追加収集では2秒内のappend・再選択・保護STOPを無効にし、通常のpath終端停止はfollowerの結果として扱う。転倒等の自然終端を強制的に回復させて教師を延ばさない。実行指令は監査用に保存する。

2秒未満の自然終端はpose_mask=false、観測済みのイベントは残す。完全イベント陰性は必要な全観測が揃う場合だけ確定し、打切り陰性を作らない。trace切断・manual stop・境界越えは自然転倒と区別する。全ての欠測にreasonを保存する。

イベント条件は既存fall_stall_targets.pyを基準とし、区間をt0からの時刻・stepにrebaseする。

| イベント | 固定する意味 |
|---|---|
| tip_over | post-actionの重力基準傾き>=85度、またはtilt_over自然終端 |
| near_fall | >=60度のpost-action状態が5sample連続し、5点目で成立。20Hzの5区間=0.25秒、最初と最後の時刻差=0.20秒 |
| linear_stall | 完全2秒で積分絶対前後指令距離>=0.20m、実XY移動距離／指令距離<=0.25、path上最大進捗<=0.05m |
| turn_stall | 完全2秒の全sampleで|指令wz|>=0.01rad/s、|実body wz|<=0.5|指令wz| |

near_fallはt0より前の連続カウントを持ち越さず、未来区間内の5点で確認する。t0時点で既に傾いている／スタックしている可能性は状態別に記録し、予兆例と分けて集計する。今回の2秒イベントは「新たな初回事故だけ」ではなく、未来区間で条件が成立した確率である。

主採用候補6,897記録の再監査値（重複除去・追加監査で更新し、確定版はmanifestで固定）：

| head | 陽性 | 陰性 | 欠測 |
|---|---:|---:|---:|
| tip_over | 18 | 6,879 | 0 |
| near_fall | 25 | 6,868 | 4 |
| linear_stall | 377 | 6,503 | 17 |
| turn_stall | 98 | 6,782 | 17 |

完全2秒の後続5,080件だけにするとtip陽性は0になるため、早期終端prefixを捨てない。poseと各eventのmaskは独立させる。位置・回転headの教師は2秒状態を観測できた区間に限られ、生存・打切りによる選択バイアスが残る。高い2秒イベント確率の候補でも数値のpose出力は返すが、転倒・終了後まで正しい姿勢を学習できたとは保証しない。prediction metadataにpose_supervision=complete_2s_onlyを記載する。

途中のGTは40点のΔp・相対quaternion、観測mask、time offsetを保存する。末尾終端の後をゼロ埋めの正常値にしない。これはGT保存であり、初期ネットワークの途中予測やADE評価を実装したことにはしない。

## 8. 分割・重複・漏洩

source trace SHA、t0 stamp、実候補ID、input SHA、元seed／episode／anchor／親prefix IDを索引に保存する。完全duplicateを除き、同一初期状態署名がsplitを跨ぐ場合は関連する親グループを結合する。元trainを評価へ移動させない。転移元の学習親と新評価親の交差も検査する。

既存trainはtrainのまま使用する。既存validationの親seedをfieldごとに固定文字列moving-pose2s-v1とseedのSHA-256順で並べ、偶数番目をselection-validation、奇数番目をaudit-evalに分ける。共通データも同じ親seed単位で分ける。1親しかないfieldはselection-validationに置き、独立評価不足と記載する。既存reserved testとR6の評価testは開封しない。

追加収集のtrain／validation／audit-eval親を収集前に固定し、候補・状態・後続を跨がせない。audit-evalの入力・教師は学習・正規化・early stopping・閾値調整に使わない。最初のaudit評価前にM_stop／M_moveの重みとM_finalの学習recipe・収集選択規則を凍結する。M_stop／M_moveは既存audit-evalで先に評価して初期版を提供でき、M_finalの学習完了を待たない。後でM_finalを同じ既存audit-evalと追加audit-evalで評価するが、先に見た結果でM_finalの構造・loss・hyperparameter・対象親を調整しない。追加audit-evalはM_final凍結後まで開かない。既存validationから分けたaudit-evalは以前の研究で既に集計された開発由来holdoutであり、完全未観測の正式testと呼ばない。追加audit-evalは今回の新規未学習親として別報告する。

既存k_rail_step_fieldの選択地形hashはtrain／validationで重複あり。主評価は既知地形上の独立走行への予測と位置付ける。地形hash交差表と、重複地形を除いた評価subsetも出す。共通rawの地形hashが復元不能ならunknownと記録し、そのsubsetを未知地形評価に含めない。初期停止データと移動データで同じ親のsplitを共有する。

## 9. 代表的走行中状態からの追加収集

GUI OFF・sensor ON。既存元記録を使うM_moveの学習開始を追加収集の完了待ちにしない。既存actor checkpoint・環境設定を固定し、通常の目標・速度指令でprefixを走らせる。転倒を人工的な外力や不連続な姿勢操作で作らない。

基本案：7状態×20独立親×14候補=1,960枝、140anchor。状態ごとの20親は原則別seedとし、12train／4validation／4audit-evalに事前固定する。候補結果を見て親を選ばない。異なる状態で親を共用した場合は親roleを共有し、独立親数を別掲する。

| 状態 | 主なfield・level | anchor資格の提案 |
|---|---|---|
| 平坦部を移動中 | random_rough L1の低勾配patch | 速度>=0.10m/s、傾斜<10度、低勾配DEM |
| 旋回中 | boxes L1、平坦な空き領域 | |body wz|>=0.20rad/s、通常の曲線prefix |
| 登坂開始直前 | hf_pyramid_slope L4 | 前方地形上昇の境界まで0.2–0.5m、速度>=0.05m/s |
| 登坂途中 | hf_pyramid_slope L4 | 地形上昇区間、速度>=0.05、0.25秒平均world vz>=0.02m/s |
| 段差進入直前 | pyramid_stairs L4 | 既知の段差前縁まで0.2–0.5m、速度>=0.05m/s |
| 降坂中 | pyramid_stairs_inv L4 | 地形下降区間、速度>=0.05、0.25秒平均world vz<=−0.02m/s |
| 停止・低速 | k_rail_field L4 | 速度<0.05、停止と非zero指令下の低速を区別して記録 |

傾斜はquaternion由来の重力基準tilt。資格はprefixの現在以前の情報と既知地形位置から判定し、接触前と保証しない。速度は物理値で記録する。登坂・降坂はpitch符号だけで判定しない。定性的な地形境界を実装で勝手に変えず、地形座標・DEM勾配の具体判定を収集manifestへ固定し、状態別に代表prefixを目視でなく数値traceで検証してから量産する。

各親のprefix探索上限20秒、資格成立は原則0.2秒継続、DEM履歴1秒を確保する。到達不能はunavailableとして保存し、同じseedで失敗結果を消して取り直さない。予備親を使う場合も収集前のリスト順と上限を固定する。

並列数は10を候補とし、2／5／10envの短いheadless gateでメモリ・整合性・host実時間を実測して選ぶ。固定親の地形tile幅と初期化IDを並列数変更で変えない。OOM・timeout時は失敗記録を残して小さい並列数へ変更し、未完了枝から再開する。collect上限3時間を設け、期限優先の縮小案は7状態×10親×14=980枝、状態ごと6／2／2親とする。縮小は開始前の時間予算で選び、途中で評価親をtrainへ振り替えない。基本案を始めた後の時間切れは完了分・未試行・失敗をそのまま報告し、980枝を完了したと読み替えない。

## 10. 分岐状態の復元と同一性の限界

再利用候補のR6 collectorにはscene state、root、joint pos/vel、joint targets、action/command tensor、latency ring、観測history、DEM history、Torch/CUDA/NumPy/Python RNGの復元がある。ただしgateは現在root/joint/DEM中心であり、そのままで完全同一とは言えない。prefixも現状は固定直進で、7状態recipeは承認後に追加実装が必要。

actorが実際に非再帰型であることをcheckpointとロード済みpolicyで確認する。再帰型ならhidden stateも保存し、未対応のまま走行中分岐しない。command samplerの個別RNG、actuator/filter状態、manager timer、sensor timestamp、前回action／指令遅延、全フリッパー状態も列挙し、保存・復元・照合の可否をmanifestへ出す。t0で高位followerを新候補用に初期化することは意図した介入であり、低位Actorの内部状態をリセットしてよいという意味ではない。

PhysX接触warm-start cache等は通常のscene snapshotで完全保存できない。表現は「観測可能な状態を一致させたsnapshot分岐」であり、厳密な物理状態同一とは主張しない。settlingを追加して動いているt0を停止状態へ変えない。

量産前、7状態からtrain／validation各1anchor以上で同じ候補の重複continuationを調べる。最初と全候補実行後で同一候補を再実行し、候補順序を反転した場合も照合する。開始root／joint／controller buffer／DEM／actor入力の差を保存し、許容する開始差は位置1e−5m・関節1e−5rad・速度1e−5・正規化済みquaternion角1e−4rad、その他数値bufferはatol1e−5・rtol1e−5とする。重複2秒結果は位置差<=0.02m、回転差<=2度、イベント一致を工学的再現性gateとする。接触力・最初のaction差も診断する。これは性能精度の根拠ではなく、復元不良を見つけるgateである。

gate不成立は原因を調べ、未保存の可視bufferを修正する。接触cache等で回復できなければ各候補を新規resetから同じprefixで再走行し、anchor状態・入力を再照合する。これをmatched-prefix replayと呼び、厳密snapshotとは区別する。再現不能なanchorは除外し、同一状態14候補を取得できたと主張しない。単独枝として使う場合もその開始入力と結果が正しいことを別途確認し、counterfactual-complete=falseとする。

## 11. dataset形式、正規化、学習

新dataset schema：eatnav_moving_pose2s_dataset_v1。観測済みの候補だけを教師にする。原則1行は1つの実行候補・t0で、学習時K=1、推論時K=14を同じcandidate encoderで処理する。14候補の元入力も保存し、未実行候補の教師は作らない。入力candidate_maskとpose/event/trajectoryの教師maskを分ける。

manifest、sample-index、split、SHA ledger、source binding、teacher reasons、coverage、trajectory付きsamplesを作る。GT・rawはローカル保存で、通常Gitに無条件追加しない。

DEM高さ・state・pathの平均／標準偏差はtrainだけでfitし、旧入力契約を保ちながら新統計で特徴抽出部全体をfine-tuneする。旧normalizationと新normalizationを混用しない。既存のstd floor0.001を入力に適用し、channelごとの統計、分位、正規化後の範囲を出す。分布外を隠す入力clipは追加しない。位置教師はtrain平均と成分stdで標準化し、std floorを[0.05,0.05,0.02]mとする。回転6Dは標準化しない。

| 損失 | 定義・重み |
|---|---|
| Lpos | pose有効例の標準化XYZ SmoothL1、beta=1、3軸平均 |
| Lrot | 0.25 × ||Rpred−Rgt||F²、pose有効例、無次元 |
| Levent | head別mask付きBCEWithLogits、pos_weight=1。陽性oversampling・陽性重みによるposterior変形を初期版では行わない |
| Lreg | 6Dの1列目と直交化2列目の長さを1へ寄せる二乗誤差 |
| 合計train | Lpos + 0.5 Lrot + 0.5 Levent + 0.001 Lreg |

損失の平均はfield→親seed→時点→候補の階層で重みを正規化し、長いスタック走行・14枝の初期点だけが支配しないようにする。有効maskごとに分母を再計算する。validation選択はtrain統計によるLpos＋0.5Lrot＋0.5のunweighted BCEで行う。event_probabilityはsigmoid(logit)で得る未校正の確率推定値であり、正確な実発生率を保証しない。固定0.5は評価用閾値であって実機STOP閾値ではない。Brier/BCEと10binの信頼度集計・件数を出し、少数headの校正不確実性を明記する。別のクラス重みや校正器を後付けする場合は変更を記録してvalidationだけで選び、audit結果から選ばない。head間平均では支持のないheadを除外し、陽性・陰性・独立親件数を保存する。陽性0のheadは学習済みとして主張せず、train事前確率の定数出力とsupport=falseを出し、4出力を欠落させない。

seed123を固定。最初5epochは新headのみAdam lr0.001、その後head0.0003／encoder0.0001、weight_decay1e−5、grad clip5、最大150epoch、最短30epoch後patience20。batch64、OOMなら32→16。epoch数と所要実時間を必ず保存し、best checkpointをvalidationで選ぶ。CPUは8threadを上限として先行可能、GPU学習・収集はheadlessで排他的に実行する。120分/fitを予算上限とし、上限時は完了epochのbestをearly-stoppedとして保存する。発散、非有限値、loader不一致は成功扱いにしない。

比較モデルは同じ構造・seed・split・評価集合で固定する。

- M_stop：既存初期区間だけ。初期点は必ずしも完全停止ではないため「初期区間限定baseline」と明記する。
- M_move：既存初期＋走行中区間。期限前にまずこのcheckpointと推論例を確保する。
- M_final：M_moveと同じ転移元から、既存＋追加trainで学習。追加前後で評価集合を揃えた比較も出す。

M_finalはM_moveの学習続きを暗黙に増やすのではなく、同じ初期化・予算から学習し、差の解釈を容易にする。train統計の差も記録する。scratch比較・複数学習seed・軌跡headは期限内初期成果の必須条件にしない。

## 12. 精度評価と推論確認

モデル設定・best epoch・分類閾値0.5をvalidationで固定し、学習に使わないaudit-evalを§8の段階順で評価する。位置・回転のmean／median／P95、位置はMAE・RMSEも示す。

- 位置：dx/dy/dzの誤差、XY距離、3D距離誤差、予測／GTの移動量分布。
- 回転：SO(3) geodesic角、wrapped yaw/pitch/roll、future gravity tilt。Euler特異姿勢は別掲する。
- イベント：head別陽性／陰性／欠測、独立陽性親数、AP・AUROC・Brier・BCE、閾値0.5のprecision/recall/FPR/F1。片方のclassがない指標はN/Aと理由を出す。
- 層別：field、level、候補方向、速度<0.05／0.05–0.2／>=0.2m/s、|wz|>=0.2、姿勢帯、資格付き登坂・降坂・停止、既にイベント条件に近い状態、完全／早期終端。
- 親単位のbootstrapで誤差・モデル間差の95%CIを出す。多数windowを独立走行と数えない。

必須baseline：位置変化なし＋回転identity、現在速度・角速度のconstant-body-twist SE(3)積分、候補arcを既存follower上限速度で2秒進めるXY/yawの幾何予測。速度・角速度baselineはt0の実測値だけを用い、v(t)=R0 exp([ω]×t)v_bodyを0から2秒まで積分し、ΔR=exp([ω]×2)とする定義を固定する。候補幾何baselineは地形由来z/roll/pitchを予測しないので、比較可能なyaw水平座標XY/yawに限って対比し、全6DoF予測と同等とは扱わない。この比較ではNNとGTの全車体座標変位もt0のyaw水平座標へ写し直し、異なるXY座標の誤差を混ぜない。

M_stop対M_moveを同じ移動中holdoutで比較して、走行中データの効果を出す。M_move対M_finalは同じ既存holdoutと新規audit-evalの双方で示す。評価改善がなくてもその結果を保存する。「ほぼ不動」への崩壊は、非停止subsetでの予測移動量・回転量、no-motion baselineとの比較、成分別誤差で検出する。

新loaderでcheckpoint再読込、CPU推論、K=1と14の候補ごとの一致、候補順序置換、schema/time幅の混用拒否、有限出力を検証する。14候補計測はbatch1・warmup20回・計測200回、CPUのNN部分と入力変換込みの双方のmedian/P95、GPUは使用できる場合に同期して測る。実機PC自体の速度とは混同しない。目標はCPUで14候補P95<100msだが実測前の保証・合格実績ではない。

## 13. 時間予算、縮小案、失敗時

22:44時点で期限まで約25時間15分。承認後の所要目安は実装・監査4–6時間、初期学習と比較1–3時間、復元gate・追加収集2–4時間、追加後学習0.5–2時間、評価・配布1–2時間、障害対応余裕3時間。合計8.5–17時間＋余裕。承認が遅れた場合は残時間で再計算し、期限超過が見込まれた時点で報告する。

既存R6 benchmarkは10env約31.24記録row/秒、5env17.46、2env8.24。ただし今回の多様prefix・復元gateを含む実測ではない。1,960枝×40rowの本体だけなら約42分、10row/秒なら約2.18時間で、起動・prefix・重複gateは別。学習時間は今回モデルで未計測なので、承認後10epochで残時間を更新する。

優先順位は、因果的dataset＋M_move＋CPU推論→漏洩・baseline精度確認→代表状態14分岐＋M_final→成果整理・push。全工程を進めるが、追加収集の遅れで既存データによる利用可能な初期モデルの保存を遅らせない。

縮小してよいもの：追加20親/状態→10親/状態、epochのearly stopping、scratch／追加seed／軌跡headの延期。縮小してはいけないもの：2秒XYZ・回転・4イベント、走行中snapshotの因果性、早期終端mask、漏洩検査、未学習データ評価、14候補CPU推論、checkpoint取得性。状態資格を満たせない、復元不能、支持がないhead、精度がbaseline以下などは不足として明記する。

構造・教師・座標系の重大変更、既存処理の安全な保留不能、データ破損リスク、権限不足は報告し、依存する危険な操作を進めない。通常の修正・batch縮小・テストのたびに再承認を求めない。失敗時はログ・再現条件を保全し、入力→教師→損失→実行環境の順で原因を切り分ける。初期モデルの精度がbaselineを改善しない場合はauditを開く前のvalidation上で限定的な修正・再学習を行う。audit結果を見て同じ集合へ再適合しない。auditで実装不具合が判明した場合は再現テストで修正し、再評価した既存auditは再利用開発評価と明記する。追加audit-evalは最終凍結まで保留する。

## 14. 成果物、配布、完了条件

新規コードはmoving_pose2s専用のschemaとファイル名にし、既存R4*/R5・R6 loaderを上書きしない。保存するもの：承認仕様・実行計画、teacher生成・監査、追加収集設定と実行記録、学習・評価・推論、checkpoint、normalization、SHA・index、入力契約、splitと件数・欠測理由、metric・時間、最小14候補例、互換性・未検証事項。

既存checkpointは約480KBの通常Git blobで、.gitattributesはbinaryを-text指定。現環境にはGit LFSがない。新小型モデルも通常Gitを第一候補とし、1file10MiB超ならoptimizer/historyを別のローカル再開用へ分離する。重み・統計・schema・推論に必要なメタデータだけの推論checkpointは10MiB以下を目標とし、100MiB以上を通常Gitへ追加しない。大きい場合はGitHub Release assets等を利用できるか確認し、取得URLとSHAをindexへ記載する。raw・GT datasetは新規20GiBを上限目安とし、ディスク空き50GiB未満で新規収集を保留する。

GitHubへ含めるのはコード・文書・小さな推論例・必要checkpoint・SHA/index・集計であり、生記録や大量datasetはローカル一覧とhashだけを共有する。認証情報、環境の秘密値、不要なログは含めない。

承認後に専用worktreeでgithub remoteをhttps://github.com/TohokuQuix/visualnav-transformer-ros2.gitと明示し、owner/repoと既存HEADを読み取り照合してから新規ブランチへpushする。現在の壊れたローカルoriginへpushしない。force pushを使わない。remoteの最終commitとcheckpoint bytes/SHAを検証する。

最小推論CLIの予定契約：scripts/eat_nav_moving_pose2s_predict.py --checkpoint PATH --input example-input.npz --output predictions.json --device cpu。将来の具体コマンドは実装・再実行で確認してからREADMEへ記載し、未実装CLIを現在動くと案内しない。

初期モデルの利用可能条件は、teacher監査・分割検査・厳密再読込・有限SO(3)出力・14候補CPU例・baseline付き未学習データ評価が揃うこと。性能がbaseline以下・希少head未検証なら研究用初期モデルとして制約を明記し、改善済み・実機使用推奨とは報告しない。

最終完了は、承認された範囲の実装・試行・学習・評価と成果物公開を実記録で確認して判定する。未完了の追加収集・学習がある状態を最終完了としない。最終報告はブランチ・commit・GitHubの仕様/評価/codeリンク、モデル取得方法とSHA、別PCのfetch/CPU推論、実施件数、誤差と走行中データ効果、未完了/未検証を含む。

## 15. 承認前レビュー記録

独立した読み取りレビューを実施。指摘された「M_final待ちによる初期提供の遅れ」と「陽性重み付きBCEの値を確率として返す問題」は、§8の段階的audit凍結と§11のpos_weight=1・未校正確率の明示に修正した。再レビューで実装前に解決すべき重要な矛盾の追加指摘はなかった。これは仕様の実装可能性レビューであり、未実施の学習・性能・物理復元の合格を意味しない。
