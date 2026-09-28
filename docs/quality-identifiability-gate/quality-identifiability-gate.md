# EAT-Nav Quality conditioning 診断結果

## 判定

| 項目 | 判定 | 根拠 |
| --- | --- | --- |
| データ監査 | **warning、window整合性は通過** | 一部terrain×q層が疎い。破損・split leakage・ペア不足はない |
| Gate A：実qの予測上の利益 | **fail** | held-outでProbe 0に対する改善が負、事前設定の実用効果・permutation条件も未達 |
| Gate B：同一入力のlow/high qを学習可能か | **inconclusive** | qによる出力差はあるが、scratch overfitの収束条件に届かない |
| 総合 | **audit warning / Gate A fail / Gate B inconclusive** | Quality conditioning の有効性は立証できない |
| Full collection | **未実施・許可しない** | Gate A/Bの通過条件を満たさない |

実行順はデータ監査 → 閾値・matched pair固定 → Gate A → Gate B → 層別・内部感度解析 → 総合判定。Gate Aがfailでも、指示どおりGate Bと層別解析まで実施した。モデル構造、loss、Quality定義、ECDF、split、pilot成果物は変更していない。

## データ監査とwindow waterfall

同じ360,000件のextended pilot記録から次を再現した。[全段階とsplit別内訳](window-waterfall.json)、[監査詳細](data-audit.json)、[全window ID](window-identities.jsonl.gz)を保存した。

| 段階 | 件数 | 定義 |
| --- | ---: | --- |
| DEM sample | 72,871 | 20 Hz記録中の4 Hz周期sampleとterminal sample。残る287,129行は意図的にDEMなし |
| 同一episode内で4 frame履歴完成 | 69,259 | currentを含む4枚、0.25 s間隔 |
| 現行EAT dataset window | **69,180** | 非terminal episodeの記録末尾でfuture不足の79 anchorを除外 |
| 8点教師軌跡が完全 | 61,977 | 全点有効 |
| terminalで1–7点のみ有効 | 6,302 | 後続点はゼロ埋めしmaskをfalseにする |
| terminal anchorで有効点ゼロ | 901 | action lossへの寄与なし |
| 旧「complete」定義 | **59,151** | episodeローカルindexの5 step strideかつ8点完全。現行completeの真部分集合 |
| 現行completeのみ | 2,826 | reset後のepisode開始global step位相が1–4 mod 5。旧ローカルstrideがglobal DEM anchorを見落とす |
| held-outの現行window | **6,928** | 変更なしのseed 123 split |
| held-outの8点完全eligible | **6,208** | 6,928 − partial 630 − zero-valid 90 |

旧59,151は69,180から順に追加フィルタした結果ではなく、別の厳密なwindow定義である。旧定義だけに属するwindowは0件。実記録で異なるepisodeからのsource参照は0件、goal-source不一致も0件。[境界テスト](/tmp/eatnav-vnt/tests/eat_nav/test_window_waterfall.py)は大きなreset位置ジャンプを含む2 episodeを構成し、全context frame・goal・有効な教師8点がsource episodeに留まり、padding点はzero/false maskであることを検証した。splitとデータは再調整していない。

train/validation/held-outはterrain seed 80/10/10個で分離し、trainだけのCritic P10 ECDFでqを変換する。split hash `ae584ee5813be6a09f16baa4fae8271ed20c457ef40d9dd44005636a71e9234f` は既存checkpointと一致する。終端理由はtimeout 884件、tilt_over 17件、success 0件。timeoutは必ずしも走行失敗を示さず、qは成功ラベルではなくCritic由来の連続量であるため、success 0件のみでは停止しない。

### terrain×q support と適用範囲

validationの `k_rail_step_field` はq bin 3が6/679件、`k_rail_field` は25/693件。各q bin最低30件という事前基準に従い、この2 terrainのvalidation層別性能は **NA / insufficient support** とし、層別pass/failを出さない。Gate Aのcheckpoint選択は全validationを集約したlossだけを使い、terrain別結果は副次評価とした。held-outはcheckpoint選択に使っていない。

train/validation/held-outのすべてに存在するterrainは `k_rail_step_field`, `pyramid_stairs`, `pyramid_stairs_inv`, `random_rough` の4種。そのうち全splitで各q bin≥30を満たすのは後ろの3種。validationに `boxes`、held-outに `hf_pyramid_slope` と `k_rail_field` は存在しない。評価の適用範囲は実際のheld-out 5 terrainに限り、全7 terrainへの一般化は主張しない。[split別support表](probe-results.json)の `support` を参照。

## Gate A：固定qと実qの対照実験

Probe 0にもQuality MLPとtoken経路を残し、すべての入力を **q=0.5** に固定。Probe Qは同じ構造・初期seed・data order・optimizerで実qだけを使用した。train 54,638、validation 6,803の有効点を持つwindowでseed 11/29/47を対で学習。全validation aggregate lossでbest checkpointを選択してから、held-out 6,838有効windowを一度評価した。train由来の単純平均軌跡のvalidation MSEは1.0471、Probe 0/Qのbest validation lossは約0.44で、双方が単純平均を十分上回った。上限100 epoch時点で改善中のrunはなかったため、underfitによる `inconclusive` 条件には該当しない。

| held-out主評価 | 実測 |
| --- | ---: |
| Probe 0 − Probe Q、episode clusterごとのpaired window MSE差 | **−0.000921** |
| 同差の95% bootstrap CI | **[−0.002326, 0.000483]** |
| 物理座標RMS誤差の相対改善 | **0.10%** |
| ΔR² | **−0.00100** |
| terrainを保ったq permutation 10回 | observedが全shuffleを上回らず、p=0.818 |

事前基準は相対RMS改善5%、ΔR² 0.02、効果のCI下限正、permutation優越およびterrain頑健性。実測は境界付近でなく、10 shuffleから100へ増やす事前条件にも該当しなかった。held-out terrain別paired MSE差は `boxes` +0.000892、`k_rail_step_field` −0.000580、`pyramid_stairs` −0.001745、`pyramid_stairs_inv` −0.001764、`random_rough` −0.001074。方向一致の一部があっても、予測利益の主基準は満たさない。[全run・CI・support](probe-results.json)、[学習ログとcheckpoint SHA](probe-training-summary.json)、[学習曲線](plots/probe-learning-curves.png)。

補助的なA2近傍比較では、held-outのcross-q 1,000ペアの教師軌跡差平均0.1439 m、same-q 1,000ペアは0.0723 m。差0.0716 mのepisode-cluster 95% CIは[0.0228, 0.1058] m。endpoint、前進量、経路長、旋回量の各差のCIには0が入る。またcross/sameでterrain構成が異なるため、このA2結果は関連性の観測でありqの因果効果を証明しない。[ペアID](heldout-pair-controls.json)。

## Gate B：train-only scratch overfit

trainのlow/high候補580,477ペアから、開始前に候補分布の75パーセンタイルで軌跡RMS閾値 **0.211888 m**、方向指標中央値で閾値 **0.341000** を[config](overfit-config.yaml)に固定。重複windowを使わず、1 episode最大3回の64ペアを83 episode・7 terrainから選んだ。同一ペアのDEM historyとgoalは完全一致し、qと教師軌跡が異なる。[選定ID・検査値](matched-pairs.json)。選択ペアの教師軌跡差中央値は0.7844 mで候補全体に比べ大きく、この診断の難度に影響しうる。閾値・ペアは再選択していない。

seed 11/29/47で既存checkpointを使わずscratchから各300 epoch学習した。固定noise bank lossは初期0.974/0.936/0.966から最終 **0.2100/0.2026/0.2080** へ低下したが、事前設定の十分なoverfit基準0.05に届かず、末尾でも緩やかに低下中だった。学習を十分と判定できないため、Gate Bは **inconclusive** とする。これは「構造的にqを表現できない」という判定ではない。[training summary・checkpoint SHA](overfit-training-summary.json)、[loss/gradient曲線](plots/overfit-learning-curves.png)。

同じDEM history・goalと初期/各stepのdiffusion noiseを固定し、q=0.1, 0.3, 0.5, 0.7, 0.9, 0.95をsweepした。3学習seed×3 diffusion seedの中央値で、qによる生成軌跡差は **0.1443 m**、教師low/high差の **18.4%**。同一qでnoiseだけを変えた差の **2.81倍** で、qへの非ゼロ応答はある。しかしlow/high教師の正しい両側選択は **3.65%**、方向一致の最小率は **45.8%**。低qの出力も高q教師に近い傾向だった。観測された応答は意図した条件付けをまだ実証しない。[詳細](overfit-results.json)、[q sweep](plots/q-sweep.png)、[効果比較](plots/target-vs-generated-effect.png)、[noise比](plots/effect-sampling-ratio.png)。

## 層別・内部感度解析

以前のQuality pilot checkpointをそのまま用い、held-outの同じ50入力でqだけを0.1/0.5/0.95に変更した。Quality MLPの相対活性差0.798、Quality token入力0.287、Transformer layer 3 Quality token 0.217、最終context 0.123、Diffusion U-Net初段0.041、最終予測noise 0.049。全10 timestepの予測noise差は非ゼロだが、最終生成軌跡のq効果/サンプリング変動は **0.100**。q情報はDiffusionまで届き、出力で小さくなる。attention weight単独からの解釈は行っていない。[全層・step別値](layer-sensitivity.json)、[図](plots/layer-q-sensitivity.png)。

Gate Aの不合格とGate Bの収束不足に整合する原因候補は、近傍内の軌跡分散に対するq効果が小さいこと、選択したtrain pairの教師差が大きくoverfitしにくいこと、入力・教師の多峰性、Diffusion下流での条件信号の減衰である。現段階で因果要因は確定していない。結果を受けたloss・モデル構造・データsplitの変更は行っていない。

## 再現情報と成果物

- 元manifest SHA-256: `55d9e2d89631c2afa19dabfb02b2753616bc2b71907843f5570bd3a8e41e1475`。window ID gzip SHA-256: `a54b868a7e9961eb68ea1937e5cdba397893fd1a30aab51b9df7013c32b96213`。既存Quality checkpoint SHA-256: `fcaf3b9397f3344a0017a52e324ad11ce879c29d9737e1b924d3800293e00987`、baseline checkpoint: `c8187d5b57baf6ddfb6ab36029de7b644b8dcfa699553bb6fbe260942c7465fb`。
- 実行基盤: Docker `isaac-lab-onix`、Python 3.11.13、PyTorch 2.7.0+cu128、CUDA 12.8、NumPy 1.26.0、SciPy 1.15.3、diffusers 0.35.1、RTX 5070 Ti 16.3 GB。Probe全16 runの学習部分は19.6分、PyTorch peak allocation約790 MiB。Bの3 run学習部分は42.7秒、peak約165 MiB。短時間smoke test後、想定外の時間・VRAM増大がなかったため実行した。
- 主要entrypoint: `scripts/eat_nav_quality_gate_audit.py`、`eat_nav_window_waterfall.py`、`eat_nav_quality_gate_pairs.py`、`eat_nav_quality_gate_probe.py`、`eat_nav_quality_gate_evaluate.py`、`eat_nav_quality_gate_overfit.py`、`eat_nav_quality_gate_overfit_evaluate.py`、`eat_nav_quality_gate_layers.py`。入力configは[Probe](probe-config.yaml)と[Overfit](overfit-config.yaml)。対応するDocker上のPythonは `/isaac-sim/python.sh`、`PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src`。
- 全checkpointと生ログは `/tmp/eatnav-quality-identifiability-gate/probe-full/` および `/tmp/eatnav-quality-identifiability-gate/overfit-full/` に保存した。Gitには小さい機械可読結果と図を保存し、checkpoint本体は含めない。[判定JSON](quality-identifiability-gate.json)。
