# 走行中状態からの2秒位置・姿勢・イベント予測

## 現在の状態

**既存データで学習した初期モデルM_moveとCPU推論例を利用可能。追加収集・追加後のM_finalは進行中。最終版の完了報告ではない。**

[承認仕様](../moving-state-pose2s-spec-v0.1.md)、[実行計画](../moving-state-pose2s-execution-plan-v0.1.md)、[初期評価結果](evaluations/M_move-existing-audit/report.md)、[モデル索引](model-index.json)。実機走行、plannerの選択条件変更、接触前性能の保証は実施していない。

旧random_rough/seed123の特徴抽出部を初期値に、7fieldの既存初期・走行中データで新しいモデルを学習した。旧checkpointを上書きしていない。旧10秒risk/progress headは転移していない。

## 別PCで取得・CPU推論

TohokuQuix/visualnav-transformer-ros2のclone内で実行する。現在の未コミット作業があれば、Gitが切替を拒否したときに強制上書きしない。

    git fetch origin exp/eatnav-moving-state-pose2s-20261103
    git switch exp/eatnav-moving-state-pose2s-20261103
    git pull --ff-only origin exp/eatnav-moving-state-pose2s-20261103

既存のNumPy/PyTorch環境を使える。新規CPU専用環境の例（Python3.10以上）：

    python3 -m venv .venv-moving-pose2s
    source .venv-moving-pose2s/bin/activate
    python -m pip install numpy==1.26.4
    python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu

CPU wheelの指定は[PyTorch公式の過去バージョン手順](https://pytorch.org/get-started/previous-versions/#v270)による。推論にtorchvision、Isaac Sim、ROS、学習datasetは不要。作成環境はPython3.10.13、NumPy1.26.4、PyTorch2.7.0+cu128で、推論はdevice=cpuを検証済み。別PC自体の速度・実機topic接続は未検証。

checksumは成果物ディレクトリから確認する：

    cd docs/eat-nav/moving-state-pose2s-20261103
    sha256sum -c SHA256SUMS
    cd ../../..

リポジトリrootから14候補を予測する：

    PYTHONPATH=src:scripts python scripts/eat_nav_moving_pose2s_predict.py --checkpoint docs/eat-nav/moving-state-pose2s-20261103/models/M_move/checkpoint.pt --input examples/eat_nav_moving_pose2s/example-input.npz --output /tmp/eatnav-pose2s-predictions.json --device cpu
    less /tmp/eatnav-pose2s-predictions.json

checkpointは通常Gitの小型binaryであり、Git LFSは不要。M_stopは初期区間限定の比較用モデルで、初期値はM_moveを使用する。

## 入出力

input_schema=eatnav_moving_pose2s_input_v1、prediction_schema=eatnav_moving_pose2s_prediction_v1、horizon_sec=2.0。

- DEM [B,4,2,20,20]、actor_state [B,22]、body_state [B,9]、paths [B,14,8,2]、directions [B,14,1]、candidate_mask [B,14]。
- CLIにはcurrent_time_ns [B]とdem_timestamps_ns [B,4]が必須。未来・0.25秒以上古い最終DEM・不正な履歴間隔を拒否する。NNの特徴ではなく入力時刻検証に使う。
- actorの現在指令index6/7/8を除外し、19＋body9=28成分をtrain統計で標準化する。速度・角速度・gravity・flipperは仕様§4のscale/順序を厳守する。
- delta_position_body_m [B,14,3]は、roll/pitchを含めた現在の全車体座標のx前・y左・z上、単位m。候補XY自体のyaw座標とは区別する。
- delta_rotation_6dは相対回転の第1列・第2列を連結。delta_quaternion_wxyz、relative_rpy_rad（roll,pitch,yaw）も返す。
- future_projected_gravity、future_tilt_radは2秒後の重力基準の傾き。**2秒間の最大傾斜の予測ではない。**
- current_quat_wxyzを渡せばfuture_quaternion_wxyz／future_rpy_radを合成する。渡さない場合は欠測理由を返す。渡したquaternionとbody projected gravityの不整合を拒否する。
- event_probability [B,14,4]はtip_over、near_fall、linear_stall、turn_stallの2秒以内の未校正確率推定値。0.5は評価用で、実機STOP閾値ではない。
- validityには回転復元、Euler特異点、候補有効性と、大きな標準化入力（絶対値10超）の警告がある。値を隠すclip・学習平均補完はしない。これは厳密なOOD検知器ではない。

古いR4*/R5の10秒risk/progress loaderとの混用をschemaで拒否する。進捗headとNNによる途中軌跡は今回の初期版にはない。40点のGT軌跡はローカルdatasetに保存している。早期終端後の2秒姿勢を補完していないため、pose性能は2秒状態を観測できた区間に限られる。

## 初期モデルの評価

既存807区間・8独立親のaudit-eval。以前のvalidationから親単位で分けた開発由来holdoutであり、完全に未観測な正式testとは呼ばない。モデル選択・正規化には使用していない。地形hashがtrainと共通のsubsetがあり、未知地形汎化の主張ではない。

| 予測 | 平均3D誤差m | P95 3D誤差m | 平均SO(3)誤差rad |
|---|---:|---:|---:|
| M_move | 0.07614 | 0.20450 | 0.12292 |
| 不動 | 0.23329 | 0.46736 | 0.17556 |
| 現在body速度・角速度一定 | 0.26629 | 0.55772 | 0.90076 |

同じ後続590区間・6独立親では、初期区間限定M_stopの平均3D誤差0.09017mから、M_moveの0.07545mへ低下。SO(3)は0.14206radから0.12478radへ低下した。親bootstrap95%CIと双方の状態別指標は評価JSON/CSVにある。構造・seed・上限設定は同じだが、追加データ・正規化・early stoppingの結果が異なる比較であり、あらゆる走行で改善する保証ではない。

このholdoutのtip/near-fall陽性は0であり、その検出性能は未評価。直線スタック19陽性（3親）、旋回スタック11陽性（4親）で、少数例である。イベントhead全てを十分に検証できたとはしない。

14候補CPU計測はNN・入力変換・全APIを区別して[実測JSON](full-api-benchmark.json)に保存した。実機PC上の速度とは異なる。

## データの所在と再現

ローカル専用：/home/ubuntu/eatnav-moving-state-pose2s-data-20261103/

- reused-dataset：6,897区間、完全pose6,880、早期終端17。train5,299／validation791／audit_eval807。
- 元記録は[承認仕様§6](../moving-state-pose2s-spec-v0.1.md)の絶対パスを参照。GitHubへ生記録や大量テンソルを追加していない。
- [dataset監査](reused-dataset-audit.json)、[固定split](reused-dataset-split.json)、学習config/historyとsource/hash bindingを保存した。
- 親走行・初期状態・転移元学習親の評価側への混入は0。完全重複0、経路延長936区間を除外。既存reserved testとR6 testは開封していない。
- models/*/training-resume-local.ptはサーバー上のみ。Gitには推論checkpointと再現メタデータを入れている。

再学習にはGitHubのコード・checkpointだけでなく、ローカルdatasetまたは元記録の転送が必要。最小推論例はtrain由来の現在入力のみで、未来GTを含まない。

## 残作業

7状態の追加headless収集は、復元gateで検出したlazy cacheと状態整合性の修正・再検証中。ベンチマーク成功を14分岐の復元成功と読み替えない。採用済み追加データ、追加後M_final、追加audit-eval、収集後の全成果まとめはまだ完了していない。進行中状態は[モデル索引](model-index.json)で確認する。
