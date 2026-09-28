# EAT-Nav Core v1 設計

## 目的と合意事項

Onix の RL Actor が多様な速度指令で実行した成功・失敗軌道を収集し、Critic の時系列値から得る品質を条件として、DEM と相対 goal から局所軌道を生成する。NoMaD の構造、diffusion 実装、軌道表現を再利用し、EAT-Nav の全重みは scratch から学習する。実機では RViz goal から局所軌道を生成し、既存 follower に `nav_msgs/Path` を渡す。

合意済みの境界は次の通り。

- EAT-Nav と `onix_global_planner_adapter` は同時起動しない。follower は今回変更しない。
- 経路の参照座標は既定 `map` とし、ROS config で変更できる。車体座標は既存設定に合わせて `chassis/body` を既定とする。
- DEM は相対高さと有効セル mask の2チャンネル。実機とシミュレーションで同じ前処理 API を使う。
- 収集用 RL checkpoint の既定値は `/home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt`。CLI または config で上書きでき、ファイルの SHA-256 と対応する `params/env.yaml` を manifest に記録する。

## 既存実装の事実

| 系統 | 参照 | 確認結果 |
| --- | --- | --- |
| Isaac 環境 | `/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/config/onix/rough_env_cfg.py` | `OnixRoughEnvCfg`、`OnixObservationsCfg`。Actor/Critic の観測は別 group。Critic は `CommonCfg` の `velocity_commands` を継承し、接触・高さ scan 等を追加する。既存 scanner は `ray_alignment="yaw"`、1.6 × 1.0 m、0.1 m 間隔。現在の terrain generator は10行×7列の tile 設定で、100環境の独立 terrain seed 要件にはそのままでは足りない。制御周期は 0.005 × 10 = 0.05 s。 |
| RL モデル | `config/onix/agents/rsl_rl_ppo_cfg.py` と選択した run の `params/agent.yaml` | `ActorCritic`、Actor/Critic MLP はそれぞれ `[1024, 512, 256]`、ELU。保存済み config は actor/critic 観測 group を指定する。 |
| RL 指令 | `rough_env_cfg.py`、`mdp/commands.py`、IsaacLab `velocity_command.py` | `v_x∈[-1,1]`、`v_y=0`、`ω∈[-0.5,0.5]`。学習時は heading command が有効で、実際の yaw 指令は heading error から計算して範囲内に clip される。 |
| RL 終了 | 選択 run の `params/env.yaml` | `time_out`、`non_finite_state`、`tilt_over`。他の終了理由を創作しない。IsaacLab `step()` は reward と終了判定の直後に該当環境を reset する。 |
| NoMaD | `src/visualnav_transformer/train/config/nomad.yaml`、`models/nomad/nomad_vint.py`、`train.py` | context_size=3 なので4観測 token、埋め込み256、Transformer 4層/4 head、軌道8点、DDPM 10 step。diffusion は `ConditionalUnet1D(input_dim=2, global_cond_dim=256)`。 |
| NoMaD 軌道 | `data/vint_dataset.py`、`training/train_utils.py`、`deployment/src/robot_interface.py` | 現在姿勢基準の局所 XY 点列を学習し、diffusion は正規化された XY 差分を予測する。推論後に差分を逆正規化・累積し、ROS 境界で距離係数と TF を適用する。既存 `action_stats` は EAT データに流用しない。 |
| 実機 DEM | `/home/kojima/onix/dep/onix_automation/onix_automation/config/elevation_mapping.onix2.yaml` | `/elevation_map`、`grid_map_msgs/msg/GridMap`、`elevation` layer、`map` frame。2.0 × 2.0 m、0.1 m/cell なので設定上20 × 20セル。追跡点は `chassis/body` 原点。地図は車体を追って移動するが軸は `map` に固定。未知値は NaN。 |
| ROS 経路 | `onix_global_planner_adapter`、`simple_trajectory_follower` | RViz goal は `/goal_pose` の `geometry_msgs/PoseStamped`。follower 入力は `/teaching_playback/path` の `nav_msgs/Path`、車体 frame は `chassis/body`。planner adapter も同じ path topic を使うため排他起動する。 |

保存 checkpoint の tensor shape と実際の GridMap メッセージの data layout は、このホストの通常 Python 環境に PyTorch がなく実行確認できていない。実装時は安全な `weights_only=True` のロードと、実メッセージまたは記録 bag による検証を必須 gate とする。設定値から実測値を推測して固定しない。

## NoMaD 論文と現行実装の差分、EAT-Nav の採用値

[NoMaD 論文 Figure 2 と Training Details](https://arxiv.org/pdf/2310.07896) は過去5時刻の観測、8 future actions、30 epochs を示す。一方、現在使用している `src/visualnav_transformer/train/config/nomad.yaml` は `context_size: 3`、`len_traj_pred: 8`、`epochs: 100` である。現行 `vint_dataset.py` は `range(curr_time - context_size * waypoint_spacing, curr_time + 1, waypoint_spacing)` で履歴を切り出し、`NoMaD_ViNT` は `context_size + 1` 個の観測 token へ reshape する。したがって `context_size=3` は **過去3枚＋現在1枚＝4枚の DEM frame**、`context_size=5` は **過去5枚＋現在1枚＝6枚**を意味する。EAT-Nav v1 は現行実装に合わせて3を既定とし、3/5を config から変更できる。履歴 buffer、dataset window、DEM encoder token 数、位置埋め込み長、checkpoint metadata を同じ設定から構築する。Quality 有効時の Transformer 入力 token 数はそれぞれ6/8（DEM 4/6＋goal＋quality）、無効時は5/7とする。異なる履歴長の checkpoint を黙ってロードしない。

現行 NoMaD の training loop は `latest` と epoch 番号付き checkpoint を保存するが、validation metric による best 選択と early stopping は備えない。`epochs` はモデル構造を決めない。EAT 学習 config は `max_epochs: 100` を初期上限とし、`validation_metric: diffusion_loss`、`early_stopping_patience: 10`、`early_stopping_min_delta: 0.0`、`validation_interval_epochs: 1`、`best_checkpoint_metric_mode: min` を変更可能にする。validation goal/window は評価ごとに再サンプルせず固定し、train と同じ正規化統計を使う。各評価で validation metric を記録し、改善時に `best` checkpoint を保存する。patience に達したら100 epochs 前でも停止する。最新 checkpoint も resume 用に保存する。raw model と EMA のどちらを best 判定・保存対象にするかも config で明示し、既定は EMA とする。

## 1. DEM 座標と前処理

EAT 専用の Isaac RayCaster を追加する。既存 RL 用 scanner と観測 group は変更しない。EAT scanner は `ray_alignment="world"` とし、中心は車体位置を追い、格子軸は world に固定する。IsaacLab の `GridPatternCfg` は端点を含むため、`size=(1.9,1.9), resolution=0.1` の20 × 20 ray をセル中心として使う。実機 GridMap の位置と位相はメッセージ metadata を優先し、scanner 側にもセル中心の phase を適用してテストする。取得した world 固定格子を実機 GridMap と同じ resolution、grid size、source axis、未知値表現（NaN/無効 mask）へ変換してから、共通 adapter で学習用入力に変換する。

出力格子は車体 yaw に固定された20 × 20セル、各セル0.1 m。有効セルの高さを robot/reference height で差し引く。参照高さは車体中心から0.3–0.8 mの環状領域にある有効セルの median とする。有効セル率が既定の25%を下回れば観測を棄却し、この閾値は config で変更できる。高さが無効なセルは高さチャンネルを0、有効 mask を0にする。補間は全ての寄与元セルが有効なときだけ行い、地図外や NaN を高さ値へ外挿しない。

各車体格子セル `(x_b,y_b)` を観測時刻の車体 pose で source map frame に写し、GridMap の円形バッファ開始 index、position、resolution に従って source height を取得する。`map` に固定された2 m四方を車体へ回すと斜め姿勢で四隅が地図外になるため、その mask を残す。シミュレーションにも同じ地図外 mask を適用する。実機固有の LiDAR 欠測を完全再現できない部分は、実地図の mask 分布を測った後に収集時の欠損付与で補う。

検証は非対称な高さ配置で yaw 0°/45°/90°、地図中心のセル単位移動、円形バッファ折り返し、NaN、中心欠損、標高オフセット不変性を含める。単純な左右対称地形だけでは軸反転を検出できない。

## 2. 収集と raw schema

収集スクリプトは checkpoint path、環境数、収集秒数、記録周期、command mixture、hold 時間、出力先、seed を CLI/config で受ける。既定は100並列環境を各3600秒の simulation time だけ動かす設定であり、約100 robot-hours を一回の収集で得る。各環境に異なる terrain seed、command seed、初期位置、初期 yaw を割り当て、100環境での重複を manifest と validation run で検査する。checkpoint に付属する `params/env.yaml` から Actor/Critic 観測 toggle を復元し、起動時に両 group と checkpoint の tensor shape を照合する。Actor は推論モードで実行し、Critic を同じ時刻の critic 観測で forward する。Critic 値の実測範囲と時間変動を短時間収集で確認する。

収集用 command は env ごとに独立 seed を持つ piecewise constant `(v_x, ω)`。heading command による上書きを無効にし、`v_y=0`、上記の学習済み範囲を守る。保持時間の初期範囲は0.5–3.0 s、カテゴリは前進・後退・弱/強旋回・その場旋回・ほぼ停止を含め、実頻度を診断する。コマンド分布の変更と checkpoint の評価値への影響を短時間収集で調べる。

1 control tick の論理 record は `timestamp, env_id, episode_id, terrain_id/seed, pose, orientation, body/world velocity, command, actor action, critic_value, reward, termination flags/reasons` を持つ。これを20 Hzで保存し、DEM source/height/mask は5 Hzと終了 tickで保存する。`V_t` は `t` の観測と command に対応し、`reward_t` と終了は `t→t+1` の実行に対応する。終了 tick の状態は IsaacLab の pre-reset recorder hook で取得し、その環境だけ新 episode を開始する。終了理由は `TerminationManager.get_term(name)` から取得する。reward component は manager で利用可能なら別系列として保存する。Critic sequence は各 control tick の raw `critic_value` として保存し、収集時に P10 その他の集約を行わない。failure 軌道も消さず、`termination_reason` と episode metadata を残す。

保存形式は少数の圧縮 chunk と manifest。chunk は env/episode 境界を索引で保持し、原子的 rename で確定して resume 時に上書きしない。DEM の source/world 固定観測と変換後 height/mask の両方、生成地形の config/seed と再生成に必要な terrain ID を保持する。収集開始前に、既存 terrain generator の tile 数と seed 処理を検査する。100環境ごとに独立 seed から terrain tile を生成し、terrain 種別・難易度も分散させる。単一の global seed で生成した少数 tile を100環境に再利用するだけでは要件を満たさない。manifest は checkpoint path/hash、RL/env config、terrain config/seed、command config、DEM config、git commit、開始時刻、chunk checksum を記録する。

初期 DEM 記録周期は5 Hz とし、20 Hz の RL 制御および scalar 記録と区別する。20 × 20 × 2チャンネルをそれぞれ float32/uint8 で保存すると、DEM だけで約2 kB/frame、5 Hz ×100環境×1時間で約3.6 GB（圧縮前）になる。これに20 Hzの scalar 記録と終了時 DEM が加わる。実測書き込み帯域と容量から本収集前に chunk 長と圧縮方式を確定する。

## 3. Dataset と品質

split は terrain seed と episode ID を単位とし、test に未使用 terrain seed を確保する。manifest に割り当てを固定し、隣接 window を別 split へ流さない。各 sample は `context_size + 1` 枚（既定4枚）の DEM 履歴、現在 pose から将来位置へ変換した相対 goal、5 Hzで切り出した8点の実行 XY 軌道、その時間区間に含まれる20 Hzの全 Critic/reward/command 列、終了情報と有効長 mask を持つ。既定では未終了の1.6秒区間に32個の Critic 値が対応し、P10 はこの raw 列から計算する。history と将来点の間隔は記録周期と同じ config で管理し、実機地図の実測更新周期に合わない場合は本収集前に config を変更する。

将来 goal は同じ episode の位置から選ぶ。局所 goal の最大半径は初期値0.8 mとして方向を保って clip し、有効セル率が不足する方向の goal は sample から除外する。半径は config 化し、短時間収集で得る有効 DEM 領域から本収集前に検証する。終了を含む window は削除せず、終了後の軌道を最終位置で pad し、loss mask で未実行点を除く。次 episode の点を混入させない。学習軌道は車体座標のメートル単位で定義し、diffusion 用 XY 差分の正規化統計は train split のみから計算する。

有効な Critic 列の P10 を raw quality `c` とする。train split の `c` だけで経験 CDF を作り、train/validation/test に同じ写像を適用して `q∈[0,1]` にする。低品質 sample は削除しない。品質と reward/終了率の関係、terrain/command 別の分布、品質極値と mask 率を学習前に診断する。

## 4. EAT-Nav model と training

EAT-Nav は RGB 画像を入力しない。既存 NoMaD は各 RGB 観測を EfficientNet-B0 で token 化するが、EAT-Nav はその画像 encoder を2チャンネル DEM 専用の小さな **共有 CNN encoder** に置き換える。各時刻の `DEM[2,20,20]` は convolution 層、空間 pooling、256次元への線形 projection を経て **1 DEM token `[256]`** になる。生の DEM 格子を flatten して MLP に渡す方式は採用しない。全時刻の DEM は同じ CNN の重みを使う。相対 goal `[x,y]` と品質 scalar `q` にだけ、それぞれ独立した MLP を使って256次元 token を作る。既定 `context_size=3` かつ Quality 有効時の Transformer 入力は `DEM token ×4 + goal token ×1 + quality token ×1 = [B,6,256]` である。

NoMaD と同じ位置埋め込み・4層4 head Transformer 設定を使い、token を平均 pool して既存 `ConditionalUnet1D` の global condition へ渡す。diffusion は固定8点の XY 差分、DDPM 10 step、epsilon prediction。全重みを scratch から初期化し、NoMaD pretrained checkpoint をロードしない。品質なし baseline は config で品質 token を除いて別モデルとして学習する。

train/validation の diffusion noise prediction loss と trajectory error を記録し、品質 bin 別の trajectory/failure 傾向も評価する。checkpoint は model state、optimizer/scheduler state、DEM/時間/軌道 config、train 由来の action stats と ECDF、split manifest hash をまとめる。ロード時は全 shape と config の整合を厳密に検証する。

## 5. ROS 推論

ROS node は `/elevation_map`、`/goal_pose` を購読し、品質 target の既定値を0.95とする。参照座標の既定は `map`、設定で変更可能。DEM history は学習と同じ間隔で `context_size + 1` 件（既定4件）保持し、欠落・時刻逆行・TF 失敗では history を無効化する。各 GridMap はその header frame から `chassis/body` への時刻付き TF と共通 DEM adapter で処理する。RViz goal は goal 自身の frame から車体 frame へ TF 変換し、遠い goal は向きを保って局所到達範囲へ clip する。goal provider は後に Nav2 global path を採用できるよう独立させる。

EAT 出力は車体 XY の8点。観測時刻の TF で設定された参照座標（既定 `map`）へ変換し、`nav_msgs/Path` を follower の `/teaching_playback/path` へ publish する。goal は逆向きの TF で参照座標から車体へ変換する。軌道の座標変換は network の外、ROS 境界だけで行う。EAT 用 launch は planner adapter を起動しない。dry-run では follower へは publish せず debug Path、入力 DEM、requested/local goal、品質 target を RViz で確認する。follower のコード・設定は変更しない。

既存 follower は空 Path を無視し、Path 自体に有効期限がない。このため EAT 専用 launch では follower の `cmd_vel_topic` を `/eat_nav/follower_cmd_vel` に設定し、別プロセスの EAT command guard だけが `/cmd_vel_auto` へ `TwistStamped` を publish する。guard は EAT の `std_msgs/Bool` ready heartbeat、最新 Path の時刻、follower command の時刻を steady clock で監視する。既定 TTL は heartbeat と command が0.3 s、Path が0.8 sで、いずれかの失効または EAT の明示的 invalid で20 Hzのゼロ速度を publish する。無効化後は invalid 以降に生成した新しい有効 Path と heartbeat を受けるまで再開しない。DEM/goal/TF 欠落、推論失敗、EAT node 停止、時刻逆行をこの条件へ接続する。follower のソースと設定ファイルは変更せず、EAT launch のパラメータ上書きだけで経路を差し替える。guard が落ちた場合は下流 chassis controller の `cmd_vel_timeout`（現行3.0 s）に依存するため、実機接続前にその停止動作と、`/cmd_vel_auto` の publisher が guard だけであることを確認する。条件を満たせなければライブ launch は有効化しない。

## 6. 段階的検証と完了条件

1. 既存3 workspace の shape/topic/座標調査を記録し、指定 checkpoint の安全な tensor shape 検証を Isaac 実行環境で行う。
2. DEM adapter の軸・欠損・相対高さテストを通す。実メッセージで20 × 20、resolution、frame、start index、NaN、更新頻度を確認する。
3. 100環境の短時間収集で command/Critic/reward/終了が変化し、env ごとの episode reset と多様な terrain が観測されることを確認する。MB/robot-hour、GB/100 robot-hours、書き込み帯域を報告する。
4. 小規模実データから split、window、P10、train ECDF、モデル1 batch の学習、checkpoint の保存/再読込、推論8点を実行する。
5. ROS dry-run で DEM、goal、TF、quality、車体/参照座標 Path の整合を確認する。guard の DEM/TF/goal/Path/heartbeat 欠落と回復を模擬し、`/cmd_vel_auto` が設定 TTL 内にゼロになり、新規 Path 前には再開しないことをテストする。下流 controller timeout を確認してからライブ接続を有効化する。

Isaac GPU、実機 GridMap または bag が使えない環境では、対応する integration test を「未実行」として記録する。合成 fixture の unit test を実機検証の代わりとしない。
