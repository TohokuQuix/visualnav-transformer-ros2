# 全7フィールド継続走行・学習・Evaluator比較 v0.1

2026-10-23 JST。利用者の実行指示により承認された K-Rail 仕様の全field展開。
基準: [K-Rail仕様](krail-field-learning-and-evaluator-spec-v0.1.md)。以下以外は同仕様を継承する。

- 対象は `demo_terrain.TERRAIN_TYPES` の7種類。順序は `k_rail_field`, `k_rail_step_field`, `pyramid_stairs`, `pyramid_stairs_inv`, `random_rough`, `boxes`, `hf_pyramid_slope`。
- 各fieldはlevel4・16m tile・1env・front_folded・20Hz。同じActor、候補preview、2秒更新、残り0.30mでのpath継足し、順序付きwaypointを適用。
- 転倒教師は現行v2の85度以上/自然tilt_overのみ。接地判定を追加しない。near-fallと各stallも既存v2。踏ん張りが必ずstallになるとは仮定せず指令・実移動・進捗から算出。
- 各fieldに独立した8収集seed(train6/validation2)、anchor0/10秒、4候補、64 branch。全体448 branch。prefix失敗、anchor不一致、boundary、制御不成立を含め全件保存。
- fieldごとに新規モデル1本を学習する。新fieldのtrainと固定既存trainの2:1 source損失、trainだけで正規化、seed123、300epoch。固定旧モデルと比較する。全field合同モデルへの変更は行わない。
- 各field3独立development seed×R0/R1/R2/R3各60秒、12比較。全体84走行。新旧モデルで同じ初期状態・同じfieldを使う。
- k_rail_fieldの配置範囲はpallet実配置、stepfieldはcell実配置、その他は生成tile範囲とし、tileとの共通範囲を2m控える。元generator寸法/配置/source SHA/実meshを記録。地形固有のholeや階段形状は変更しない。
- 新seedは全既知registryのbase/派生seedおよびstepfield比較seedから除外。全field/role間も非重複。新reserved testは作らず既存test非開封。
- 収集時にrisk/45度STOPは使わない。評価時のみ共通45度保護と既存risk guardを使う。物理とActorは変更せず受動観察する。
- GUIは実走行を表示し、host/Docker実起動コマンドを保存・表示。終了まで新しいgoalを順番に設定する。学習中は保存済みdatasetを使うため新走行ではない。
- 固定件数を終えたら利用可能データで学習・評価し、陽性不足による追加収集は自動で行わない。全失敗/N/Aと境界censorを報告する。

実装・収集・学習・比較・監査の状態は新しい実験rootへ記録する。旧sealed資料とbyte-bound sourceを改変しない。

## 承認済み更新: K-Railの学習乱数6通り

利用者が同一architectureの学習seed6通りを選択。`k_rail_field`のみ optimization seed `[123,456,789,101112,131415,161718]` で6モデル。他の6fieldはseed123の1モデル。
K-Railの各モデルで同じdevelopment3 seed×R1/R2/R3を行い、旧モデルR0は共通3走行。K-Rail57走行、他field72走行、合計129評価予定。モデルを結果に応じて除外せず6つすべて報告する。
収集seed/64branchと教師条件は変更しない。従来84評価のmanifestは初期計画として保持し、実行前の追補manifestにモデルseedを含む129割当を固定する。
