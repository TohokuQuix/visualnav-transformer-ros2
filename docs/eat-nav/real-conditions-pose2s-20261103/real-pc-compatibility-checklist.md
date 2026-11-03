# 実機PCとの整合性確認票 S0/S1

状態：仕様承認前。**未確認はpassではない。仕様承認と本票の必須確認の両方が完了するまで、本格収集・学習へ進まない。**

本票は入力・制御の互換性を確認するもので、実機を動かす許可、予測精度の合格、走行安全の保証ではない。コマンドは原則read-only。走行指令のpublish、param set、controller切替、実機起動はここでは行わない。

## 1. 共有commitと参照版

| 対象 | 確定している識別子 | 実機で確認すること |
|---|---|---|
| 学習repoの旧コード | 43bb7b69fde5d29095b6add17965aacd48ae58fa | 旧moving-state-pose2sモデル/変換/評価の参照版 |
| ローカルbridge参照 | 39bd4b2431dfb7b049204133e1a52548d03dcbe0 | 実機版と同一かは未確認。ファイルSHAを比較 |
| 10/30handoff | 元commit不明、reference-indexに各SHA | 元commitを推測しない。凍結コピーはこの仕様commitで共有 |
| 現在の実機cycle follower | source/runtime hash未受領 | 専用packageのsource、install、起動上書き、実行parameterとbinary |
| 今回の仕様commit | このファイルを含むcommitを最終回答で通知 | fetch後のgit rev-parse HEADを返却 |

`references/`は文書添付の凍結コピーであり、実行用にinstallしない。実機の既存branch・install環境を上書きしない。

## 2. 実機PCで仕様だけを取得する

既存cloneのrootでremoteが正しいことを確認する。異なる場合は止め、正しいremote名を選ぶ。学習PCのoriginは古いローカルpathだったが、その設定を実機PCへコピーしない。

```bash
git remote get-url origin
# expected: https://github.com/TohokuQuix/visualnav-transformer-ros2.git
# SSH形式で同じrepositoryを指す場合も可。
git fetch origin exp/eatnav-real-conditions-pose2s-20261103
git worktree add --detach ../eatnav-real-pose2s-review origin/exp/eatnav-real-conditions-pose2s-20261103
cd ../eatnav-real-pose2s-review
git rev-parse HEAD
cd docs/eat-nav/real-conditions-pose2s-20261103
sha256sum -c SHA256SUMS
```

既に同じreview用directoryがある場合は別名にする。現在の走行branchをswitch/resetしない。上記はROS環境やcontrollerを変更しない。

## 3. 確認済みのユーザー報告（2026-11-03）

- 使用予定は専用 `eat_nav_cycle_follower`、node `/eat_nav_follower`。
- config installed：`/home/yasuda-haruki/Documents/Codex/2026-10-27/git-clone-branch-exp-eatnav-r4/fusion_simple_planner/install/eat_nav_cycle_follower/share/eat_nav_cycle_follower/config/onix.yaml`。
- 編集元：同workspaceの`src/eat_nav_cycle_follower/config/onix.yaml`。
- nominal：linear_speed.3m/s、lookahead.8m、angular_gain.5、max angular1.6rad/s、control20Hz、goal tolerance.2m。
- `real_robot_cycle.launch.py`がpath/cmd/frame/recovery topicsと`publish_subcrawler_commands=false`等を上書き。
- follower→gate→bridge→diff_driveの流れ。`/cmd_vel_auto`は採用出力ではない。
- bridge既定の出力候補：`/diff_drive_controller/cmd_vel`（TwistStamped）、`/flipper_joint_position_controller/commands`（Float64MultiArray）、`/chassis_mode_state`（String）。**runtime確認前に確定topicとはしない。**
- flipper既定配列FL,FR,RL,RR。NNのRR,RL,FR,FLと区別。
- 前側を畳み後ろを広げる運用。ただし固定180°/0°ではなく、policy出力とdirectional/interference等の動的処理。
- 6action／2action、fold-observation有無、drive-direction、ゼロ速度保持等で分岐。mode未記録を0で埋めない。
- position/velocity出力modeは変更可能。velocity用kp3/limit1.5/deadband.02やenv2.0をpositionの実速度とみなさない。

これらはローカルコード・既定値・運用の報告であり、記録時点/今後の実行設定を全て確定したものではない。

## 4. 未確定事項とgate

| ID | 確認事項 | 必要な証拠 | 本格収集への影響 |
|---|---|---|---|
| U01 | 専用cycle followerの正確な実装・全設定 | repo commit/dirty状態、source/config/binary SHA、runtime param dump、launch上書き | **必須blocker**。旧0.5m followerで代用不可 |
| U02 | 低位Actor checkpointと観測/action構成 | 実ファイルSHA、action次元2/6、fold入力有無、RNN/stateful有無、obs/action順序 | **必須blocker**。予測NN checkpointとは別 |
| U03 | bridgeの実行分岐と状態 | frontend/backend/launch/env SHA、mode/drive-direction/source、zero保持・clip・干渉・rate limitの順序 | **必須blocker**。固定角／mode0補完不可 |
| U04 | 実採用速度・flipper出力のtopic/型/順序 | runtime parameters、publisher、frame、position/velocity、配列順 | **必須blocker**。未知ならread-only loggerを追加して確認 |
| U05 | clockと取得/受信時刻の意味 | use_sim_time、header生成点、各node clock、同期根拠、offset/jump/誤差bound、受信monotonicログ | **必須blocker**。負ageを0へclipしない |
| U06 | DEM layer・局所化・補間 | elevation/inpainted/raw mask、confidence source、grid/frame/TF、kernel、参照高さ | **必須blocker**は入力意味・変換。未知filter物理再現は近似profileと明記して承認可 |
| U07 | 現在姿勢・gyro・速度の有効source | IMU covariance−1のdriver意図、extrinsic、単位、TF生成元、odom frame | **必須blocker**は有効な必須入力。未観測z/加速度等は合意したmask可 |
| U08 | jointの原点・符号・gear・対応 | 実joint名、水平0/上90への変換、出力軸rad/s、drive aggregation、位置/速度追従 | **必須blocker**は必須joint観測とcontrol mapping |
| U09 | effort/電流/電圧/制限 | unit・取得元・gear・モータか軸か・欠測定義 | **任意**。未知ならNN無効、raw native＋unknown unitで保存可 |
| U10 | actuator応答・摩擦・負荷・遅延分布 | adopted output→実測joint、独立速度の記録、物理同定条件 | 同定不能部分は明示したsensitivity profile。実機同定済みとは呼ばない。基準応答が不明ならprofile承認で扱いを決める |
| U11 | 正式な実機3D正解 | 独立3D localization/reference、精度/時刻/source、実行候補記録 | **実機精度の正式主張のblocker**。sim学習のblockerではない。なければproxy評価のみ |
| U12 | 実機で新schemaを作れるか | S1 fixture/入力NPZ/report、負例テスト、候補hash/順序/時刻/parity、source可用性 | **必須blocker**。ユーザーの結果返却と承認が必要 |

新たに確認不能な任意項目が出ても勝手に必須条件を増やさない。必須に変える合理的理由がある場合は仕様変更として説明する。任意欠測は同じmask policyをsimと実機へ適用し、その制約込みでモデルを学習する。

## 5. 実行中システムを変更しない確認コマンド例

ROS環境は実機側の既存設定を使う。node名・topic名は`list/info`とruntime parameterで確認してから選ぶ。設定ファイルがあるだけでload済みとは判断しない。

```bash
ros2 node list
ros2 node info /eat_nav_follower
ros2 param dump /eat_nav_follower > follower.runtime.yaml
ros2 param get /eat_nav_follower use_sim_time
ros2 pkg prefix eat_nav_cycle_follower
# bridgeの実node名を確認後：
ros2 node info <actual_bridge_node>
ros2 param dump <actual_bridge_node> > bridge.runtime.yaml
ros2 param dump /diff_drive_controller > diff_drive.runtime.yaml
ros2 topic list -t
ros2 topic info /eat_nav/follower_cmd -v
ros2 topic info /cmd_vel_auto -v
ros2 topic info <resolved_bridge_velocity_output> -v
ros2 topic info <resolved_bridge_flipper_output> -v
ros2 topic info <resolved_mode_state> -v
```

`ros2 param dump /diff_drive_controller`がcontroller manager構成により使えない場合は、実際のcontroller params sourceを示す。wheel radius/separation/filter/limit/実行周期を推測しない。

実機source・設定の所在と使用checkpointを確認後、実ファイルへ`sha256sum`、対応repoで`git rev-parse HEAD`と`git status --short`。installとsrcのyamlの両方を確認する。実行中processがloadしたpath/起動argvも確認する。共有には必要なcontrol情報を選び、認証情報を含むdump全体を無条件commitしない。

flipper arrayがruntimeでもFL,FR,RL,RRならNN順へのpermutationは3,2,1,0。ただし名前・順序のruntime確認後に使う。各jointの符号/offset変換も別に検査する。位置列へ速度値を入れない。

## 6. S1で承認後に用意する共有物と検査

**以下の新しいtoolとprototypeは、この仕様commit時点では未実装。** 実装済みのように実行結果を報告しない。

- `real_pose2s`版のschema validator、ROS/bag→current-only入力変換、同一parser、教師生成、control-profile replay検査。
- real bagの代表4frameとsource/mask/clock qualityを保持した入力例。clock未解消なら不適合理由が出る例として保存。
- 実profileが揃った範囲で最大24枝程度の小さいsim prototypeと、正解pose/trajectory/mask/設定。
- 全14候補入力、各候補の予定速度とprofile ID、teacherの有無を区別したsample。
- 正常例と、未来入力・重複frame・stamp改変・mask/shape誤り・joint permutation・position/velocity混同・別episode等の負例。
- benchmark report（2/5/10env・multi-process、有効window/秒、sim/wall、メモリ、失敗理由）。prototype・診断と合わせて新規保存上限320区間。
- `feature_manifest.json`、`control_profile.json`、`observation_profile.json`、各SHA、検査コマンドとexpected report。

予定検査は、①schema/units/availability、②観測の到着≤t0、③4実観測IDの相異、④clock domain/age、⑤DEMのraw/inpaint mask分離、⑥TF/回転/軸順序、⑦bridge順序/実action処理、⑧2秒教師のpre/post/reset対応、⑨両PC同一fixtureのtensor差を確認する。

S1コード・sampleのcommitを別途共有するので、そのcommitを別worktreeでfetchして検査する。本仕様commitだけで「同じ形式の入力を既に生成できた」とは扱わない。実機PCでは駆動publisherを持たないoffline converter/shadow inspectorだけを使う。

## 7. 実機側から返すreport

```text
spec_commit:
s1_code_sample_commit:
real_repository_commits_and_dirty_files:
follower_runtime_source_config_binary_hashes:
actor_checkpoint_sha256_action_dim_obs_schema:
bridge_runtime_params_and_output_publishers:
flipper_mode_source_order_units_mapping:
clock_domain_alignment_and_header_semantics:
DEM_layers_observation_mask_interpolation_profile:
state_sources_units_covariance_handling:
optional_features_disabled_in_both_sim_and_real:
fixture_and_prototype_validation_report_sha:
unresolved_items:
formal_real_3d_reference_available: yes/no
compatibility_accepted_for_full_collection: yes/no
spec_approved: yes/no
```

必須項目が未記入、または最後の2項目がyesでない場合、本格収集・学習は待つ。「コードを見た」「既定値はこうだった」はruntime/profileの確定とは分ける。S1結果の合意後は通常の学習・検証のたびに再承認を求めず、仕様に沿ってS2〜S4を進める。
