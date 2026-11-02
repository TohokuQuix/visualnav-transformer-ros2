# EAT-Nav moving-state pose2s 実行計画 v0.1

状態：仕様と同時に承認待ち。承認前に本計画を実行しない。

Goal：走行中の因果的入力から2秒XYZ・6D相対回転・4イベントを予測する学習済みモデルと14候補CPU推論を2026-11-03 JST中に優先確保し、追加収集・最終学習・検証・新規ブランチ公開まで進める。

Spec：[moving-state-pose2s-spec-v0.1.md](moving-state-pose2s-spec-v0.1.md)。構造・数値・mask・split・公開条件は仕様を正とする。実装は小さなtaskごとに検証し、独立した教師/学習/収集の作業を必要に応じて並行する。通常の判断で再承認を要求しない。

## 全体制約とレビュー重点

- 既存branch・checkpoint・sealed datasetを上書きせず、専用worktreeで作業する。
- 学習/収集はGUI OFF。実機を動かさず、planner順位・STOP条件を変えない。
- 20Hzのpre→40回後post、自然終端のpre-reset、anchor offsetを明示的に検証する。
- 未実行候補の教師、未来入力、打切り陰性、同一親のsplit跨ぎを作らない。
- pose欠測でも早期転倒のevent陽性を残す。head別maskと入力candidate_maskを混同しない。
- 既存10秒head、R6のyaw座標軌跡head、新しい全車体座標6D headを混用しない。
- snapshot復元はphysics cacheの完全同一を保証しない。duplicate gateとmatched-prefix fallbackを実装する。
- 期限短縮でも漏洩監査・未学習データ評価・14候補推論・checkpoint取得性は省略しない。

## Task 0：分離、資源の保全、再開可能な台帳（目安0.5–1時間）

成果物：新branch/worktree、source-binding.json、resource-plan.json、R6-pause-record.json、progress.json。

- [ ] 現在branch/HEAD/status、元checkpoint SHA、外部actor/env/config、ソフトウェア版、GPU/CPU/diskを再確認する。
- [ ] 新branch exp/eatnav-moving-state-pose2s-20261103 と専用worktreeを作り、未追跡の既存3ファイルを移動・追加しない。
- [ ] 実データは絶対パスとhashのmanifestで参照し、新しい出力は既存seal配下に書かない。
- [ ] R6の現在バッチ・親子PID・キュー・完了markerを確認する。出力を保全してバッチ境界で新規投入を保留し、再開手順を記録する。安全な保留ができない場合はGPUを奪わずCPU先行とし、不足を報告する。
- [ ] runtimeの新規出力は一意なrun IDを使い、完了・失敗・途中状態を別扱いにする。再開時はrunningを完了と読み替えず、出力照合して同一実験を重複起動しない。

完了確認：元checkpoint SHA不変、既存未追跡変更不変、実際に動いているジョブとresource台帳が一致。

## Task 1：時刻・回転・イベント教師の純粋関数（目安1–1.5時間）

Create：src/visualnav_transformer/eat_nav/moving_pose2s_geometry.py、moving_pose2s_targets.py。
Test：tests/eat_nav/test_moving_pose2s_geometry.py、test_moving_pose2s_targets.py。

Interface：relative_endpoint(root0, root2)→delta_position_body_m/rotation6d/quaternion；build_window(rows, start, snapshot, contract)→inputs、pose/event/trajectory targets/masks/reasons。

- [ ] 既知のroll/pitch/yaw付きrootで全車体座標変換・wxyz・第1/2列6D・identity・逆回転・future gravityを検証するテストを先に作る。
- [ ] 40intervalと41行取り違え、t0 replan許可/内部replan拒否、anchor10/global240、1.95秒/2.00秒終端、reset poseの混入、境界・append・manual stopを検証するテストを作る。
- [ ] 5sample near-fall、直線/旋回stall閾値、early tip positive＋pose欠測＋stall未知を既存教師と比較する。
- [ ] 必要な実装だけを追加し、上記と既存教師関連テストを実行する。

完了確認：解析解と実trace代表例の2秒終点一致、自然終端後の捏造0、イベントheadの時間幅が2秒。

## Task 2：走行中dataset生成、監査、split固定（目安1–2時間）

Create：scripts/eat_nav_moving_pose2s_dataset.py、eat_nav_moving_pose2s_audit.py。
Create：config/eat_nav/moving_pose2s/dataset.json。
Test：tests/eat_nav/test_moving_pose2s_dataset.py。

Interface：build_dataset(source_manifest, output)→dataset manifest；audit_dataset(dataset)→counts/motion/event/split/hash reports。outputが既存なら同じhashの検証だけを行い、上書き生成しない。

- [ ] 既存1,290＋512＋5,095開始点を再走査し、現在snapshot、root、DEM時刻、実候補IDの因果性を照合する。
- [ ] 完全重複を検出し、親prefix・seed・初期署名のsplit関係を固定する。既存validationを仕様のhash順でselection-validation/audit-evalへ分ける。
- [ ] pre-dedup 6,897record／pose6,880／partial17との差分を理由付きで出す。共通rawの地形hashは取得できる範囲を確認し、unknownを未知地形扱いしない。
- [ ] 28D、正しいDEM履歴、1実行候補、pose/event/40点GTと別maskを保存する。未実行13候補を教師にしない。
- [ ] field/level/速度/姿勢/方向/初期・後続/イベント陽性陰性未知/独立陽性親を集計する。
- [ ] テストで未来snapshot、t0不一致、旧reverse slot3→canonical7、重複、train/eval親衝突、GTのみ欠測を検証する。

完了確認：manifestとNPZ/索引hash一致、未来入力0、親・転移元学習親とaudit-evalの交差0、全除外理由に件数あり。

## Task 3：モデル・train・schema厳密推論（目安1.5–2時間）

Create：src/visualnav_transformer/eat_nav/moving_pose2s_model.py。
Create：scripts/eat_nav_moving_pose2s_train.py、eat_nav_moving_pose2s_predict.py。
Create：config/eat_nav/moving_pose2s/train.json。
Test：tests/eat_nav/test_moving_pose2s_model.py。

Interface：predict(checkpoint, inputs, device='cpu')→仕様prediction_v1。train(dataset, split, variant, output, config)→checkpoint/normalization/config/history/support。

- [ ] 旧encoder/fusionのみのtransfer key許可リスト、新headのshape/identity bias、旧risk/progressの転移拒否を検証する。
- [ ] 位置SmoothL1、回転chordal/正則化、pos_weight=1のhead別masked BCE、train限定統計、階層重み、支持なしheadの定数/supportを実装する。
- [ ] 28Dの状態scaleとnormalizationは学習/推論で同一にし、command perturbation不変をテストする。
- [ ] 6D退化、yaw折返し、Euler特異点、非有限入力、wrong schema/horizon、異なる候補順とK=1/14同値を検証する。
- [ ] checkpointは推論用と再開用を分け、CPU strict reload・同値・finiteを確認する。

完了確認：実入力の14候補出力とcurrent gravity合成、既存loader混用拒否、raw future情報を特徴へ渡す経路なし。

## Task 4：既存データによる初期学習（目安1–3時間、Task5と資源を分けて並行可）

成果物：M_stop、M_moveのcheckpoint・学習履歴・validation結果・凍結後の既存audit評価。初期モデルの利用可能条件は仕様§14。

- [ ] Task2で固定したtrain/validationを用い、M_stopとM_moveを同じseed・構造・予算でfitする。この学習中にはaudit-evalを開かない。
- [ ] 10epoch実測で所要時間を更新し、必要ならCPU8threadで続ける。GPUと収集を競合させない。
- [ ] 非有限・loss停滞・ほぼ不動予測への偏りは、まず単位/mask/正規化・GT・サンプル重みを調べて修正する。
- [ ] M_stop/M_moveの重みとM_finalのrecipe・収集選択規則を凍結し、既存audit-evalをbaseline付きで評価する。この結果でM_finalをtuningしない。
- [ ] 学習終了・early stop・CPU reload・14候補例・未学習データ評価まで確認してから「初期モデルあり」と台帳に記録する。

期限対応：このM_moveを失わないよう成果一式を保存し、追加収集が遅れても利用可能な初期版として識別する。

## Task 5：代表状態のheadless分岐と追加dataset（目安2–4時間）

Create：scripts/eat_nav_moving_pose2s_collect.py、eat_nav_moving_pose2s_collect_sim.py。
Create：config/eat_nav/moving_pose2s/collection.json。
Test：tests/eat_nav/test_moving_pose2s_collection.py。

Interface：prepare_collection(protocol)→frozen parent manifest；collect(manifest, runtime, output)→anchor/branch/duplicate/gate records；pack_fresh→Task2同schema。

- [ ] 保存可能なscene/controller/RNG/historyと保存不能なphysics cacheを列挙し、実Actorが非再帰型であることを確認する。
- [ ] 7状態prefix recipeと資格判定を実装し、代表traceの時刻/DEM/速度/地形位置を監査する。前方直進だけのprefixで代用しない。
- [ ] 10／5／2envのgateでhost実時間とVRAM、親tile不変を調べ、残時間から基本20親/状態か縮小10親/状態を開始前に選ぶ。
- [ ] 7状態のtrain/validation anchorで開始入力と重複2秒結果、候補順序の影響を検証する。未保存bufferを修正し、必要ならmatched-prefix replayへ切り替える。
- [ ] 主実験の2秒内append/replan/guardを禁止し、14候補をそれぞれ実行する。自然終端をマスクし、失敗・unavailable・partialを保全する。
- [ ] 3時間collect予算・disk/timeout gateを守り、試行前固定manifestと実完了数の差を報告する。
- [ ] 新しい親と既存train/evalの衝突、geometryhash、counterfactual-completeを監査してpackする。

完了確認：原理上保存不能な状態を明記、合格した復元方法と代表状態件数が実記録で裏付けられる。予定枝数を実績と読み替えない。

## Task 6：追加後の最終学習と共通holdout評価（目安1.5–3時間）

Create：scripts/eat_nav_moving_pose2s_evaluate.py。
Create：src/visualnav_transformer/eat_nav/moving_pose2s_metrics.py。
Test：tests/eat_nav/test_moving_pose2s_metrics.py。

Interface：evaluate(checkpoint, dataset, audit_eval, baselines)→metrics.json、state-stratified.csv、report.md。

- [ ] 既存＋fresh trainからM_finalを同じ元checkpoint・同じ学習予算でfitし、validationでbestを固定する。
- [ ] 事前凍結recipeによるM_finalの重みをvalidationで選んで固定し、既存audit-evalとまだ開いていない新規audit-evalを評価する。M_stop/M_moveも同じ新規audit-evalで評価し、既存auditの保存予測と合わせて比較する。
- [ ] no-motion、constant-twist、candidate arcの適用可能成分のbaselineを作り、解析例で正しさを検証する。
- [ ] 各軸・3D・SO(3)・wrapped RPY・gravity tilt、イベント指標と分母、mean/median/P95、状態/field別、親bootstrap CIを出す。
- [ ] M_stop→M_moveとM_move→M_finalの差を同じholdoutで出す。陰性しかないheadのAUROC等はN/Aにする。
- [ ] CPU14候補のNNのみ/変換込み時間、save/load、candidate permutation、run-time schemaを実測する。

完了確認：全値に分母とscopeがあり、未評価head・欠測pose・既知/未知地形を区別。audit-evalに合わせた追加tuningをしない。

## Task 7：成果整理・実機PC向け最小例・新branch push（目安1–2時間）

Create：docs/eat-nav/moving-state-pose2s-20261103/README.md、evaluation-report.md、model-index.json、SHA256SUMS、models/M_move/checkpoint.pt、models/M_final/checkpoint.pt。
Create：examples/eat_nav_moving_pose2s/example-input.npz（小さい実入力1件）、README.md。

- [ ] 推論用weights/schema/統計だけを小型checkpointに保存し、SHAとモデル入力契約・event horizon・supportをindexへ記録する。
- [ ] raw/datasetのローカル場所・hash・利用数をREADMEに書き、GitHubには巨大データを追加しない。
- [ ] fresh CPUプロセスで14候補CLIを実行し、別PC向けfetch・取得・SHA確認・推論コマンドをその結果から記載する。
- [ ] ステージ対象を確認し、未追跡の既存作業やsecret・rawが入っていないことを確認する。必要な関連テストとdiff checkを実行する。
- [ ] github remoteのURL/owner/repoを読み取り照合し、新規branchへ通常pushする。remote commitとcheckpoint byte SHAを検証する。
- [ ] 保留したR6評価の状態・資源を確認し、安全な再開を記録する。
- [ ] branch/commit/リンク、取得方法、誤差、走行中データの効果、実施/未完了/未検証を最終報告する。

## 期限時の報告基準

初期モデルと推論コードの確保を最優先するが、未完了の追加収集・最終学習を完了としない。11/3の期限内に全工程が終わらない見込みなら、利用可能なM_move、実際に完了した評価、残作業・原因を先に整理する。必要に応じ初期版を明示した新branchへpushし、その後も承認済み範囲を継続する。最終版と初期版はindexとcommitで区別する。
