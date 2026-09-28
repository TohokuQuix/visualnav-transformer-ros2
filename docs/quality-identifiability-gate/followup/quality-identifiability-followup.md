# EAT-Nav Quality identifiability follow-up

## 総合判定

| 診断 | 判定 | 主要根拠 |
| --- | --- | --- |
| Complete-only Gate A（主評価） | **fail** | 実qのheld-out予測利益はごく小さく、95% CIに0を含む。物理RMS誤差は悪化 |
| 既存partial-inclusive Gate A（副次結果） | **fail** | 既存の事前登録済み結果を保持 |
| Gate B1 / B8 / B64 | **すべてinconclusive** | 3 seedとも事前のfixed-noise loss 0.05へ届かず、plateau条件で停止 |
| Full collection | **未実施・許可しない** | Quality conditioningの有効性を立証できていない |

既存の[Gate A/B報告](../quality-identifiability-gate.md)とcheckpointは上書きしていない。pilotのsuccess終端は0件だが、timeoutは必ずしも走行失敗ではなく、qは成功ラベルでなくCritic由来の連続量である。モデル構造、loss、ECDF、split、actor latent、embodiment token、Critic蒸留も変更していない。

## 学習前監査と固定条件

同一manifestとsplit hashを照合し、8点すべてのaction maskがtrue、全教師点・DEM・goalがfiniteのwindowに限定した。train **49,598**、validation **6,171**、held-out **6,208**件で、[既存waterfall](../window-waterfall.json)と一致する。61,977 complete windowsのcontext・goal・教師点を元episodeの時刻・poseへ戻して照合し、episode跨ぎ0件、target routing不一致0件、split間重複0件だった。[件数](complete-window-counts.json)、[全window ID](complete-window-identities.jsonl.gz)。partial/zero-valid windowは今回の主Gate Aに含めていない。

Gate Bは[凍結済み64ペア](../matched-pairs.json)のみ使用し、新規候補探索や結果を見た再選択をしていない。B1は教師軌跡RMS差が64ペアの中央値に最も近い `21162:31409`。B8はB1を含む8ペアで、7 terrainの多様性を確保した。B64は元の64ペア全体。window重複はなく、1 episodeの使用上限3を守る。各ペアのlow/highはcanonical化したDEM history・goalが完全一致し、教師軌跡とqをそれぞれ元windowへ照合した。[subsetの全ID](overfit-ladder-pairs.json)、[固定config](followup-config.yaml)。

## Gate A：complete-onlyと従来結果

Probe 0はQuality MLP/token経路を保持して全サンプルへq=0.5を入力し、Probe Qは実qを入力した。両者は同一architecture、seed 11/29/47、data order、optimizer、epoch上限を使用し、scratchから学習した。best checkpointは**全validationのaggregate lossだけ**で選択し、held-outは選択に使用していない。学習前の平均軌跡baselineもcomplete-only trainから再計算した。

| 指標 | Primary：complete-only | Secondary：既存partial-inclusive |
| --- | ---: | ---: |
| held-out有効window | 6,208 | 6,838 |
| Probe 0−Q paired MSE差（正なら実qが改善） | **+0.000185** | −0.000921 |
| episode-cluster bootstrap 95% CI | [−0.001032, +0.001519] | [−0.002326, +0.000483] |
| 物理RMS誤差の相対改善 | **−0.135%** | +0.104% |
| ΔR² | +0.000201 | −0.001000 |
| terrain内q shuffle | 10回、p=0.273 | 10回、p=0.818 |
| Gate | **fail** | **fail** |

Primaryの平均軌跡validation MSEは1.0493、Probe 0/Q best validation loss平均は0.4455/0.4442で、双方が平均軌跡を十分上回る。上限epoch時に改善中のrunはなく、underfitによる`inconclusive`条件には該当しない。observed効果+0.000185は10 shuffleの最大+0.000320を上回らず、実用改善5%、ΔR² 0.02、CI下限正、terrain頑健性も満たさない。境界付近ではないため、事前規則に従い100 shuffleへの拡張は不要だった。[主結果・seed別・terrain別・CI](gate-a-complete-results.json)、[比較表](gate-a-comparison.md)、[学習summary](gate-a-complete-training-summary.json)、[曲線](plots/complete-probe-learning-curves.png)。

Complete-onlyでもvalidationの`k_rail_step_field`高q binは6件、`k_rail_field`は25件。30件未満のterrain×q層は **NA / insufficient support** とし、その層の性能値やpass/failを出さない。全splitにあるterrainは `k_rail_step_field`, `pyramid_stairs`, `pyramid_stairs_inv`, `random_rough`。全splitで各q bin≥30のcommon supportは後ろの3種だけ。validationに`boxes`、held-outに`hf_pyramid_slope`と`k_rail_field`はなく、全7 terrainへの一般化は主張しない。[complete-only support](complete-support.json)。正式なterrain別表示は[common-support表](gate-a-common-support.json)で、`boxes`と`k_rail_step_field`はNA、`pyramid_stairs`、`pyramid_stairs_inv`、`random_rough`のみ副次性能値を示す。生のheld-out地形値は[主結果JSON](gate-a-complete-results.json)へ監査用に残すが、support外の頑健性主張には使わない。

## Gate B：scratch overfit ladder

B1→B8→B64をseed 11/29/47ごとに**毎回scratch**から学習した。既存checkpointはwarm startに使っていない。architecture・diffusion noise loss・optimizerは既存と同じ。各stageで最大1,000 epoch、fixed-noise-bank loss≤0.05、または直近50 epochの相対改善<1%を停止条件とした。評価ではDEM history・goal・初期および各stepのdiffusion noiseを固定し、q=0.1/0.3/0.5/0.7/0.9/0.95をsweepした。3つのdiffusion seedを各学習seedへ適用した。

| Stage | 停止epoch範囲 | 最終fixed loss中央値 | 生成差 / 教師差 | 両教師の正しい選択率 | 最小方向一致率 | q効果 / sampling変動 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B1（1ペア） | 96–129 | **0.9904** | 0.0017 | 0% | 33.3% | 0.005 | **inconclusive** |
| B8（8ペア） | 148–197 | **0.3578** | 0.0577 | 0% | 45.8% | 0.539 | **inconclusive** |
| B64（64ペア） | 176–250 | **0.2548** | 0.1700 | 3.13% | 46.4% | 1.952 | **inconclusive** |

B1/B8/B64の生成q差中央値は0.0013/0.0453/0.1334 m、教師low/high差中央値は0.7868/0.7844/0.7844 m。B64ではq変更の効果が同一qのsampling変動を上回ったが、低q出力も高q教師へ寄る傾向で、両教師の正しい選択は少ない。seed間でB64の生成差/教師差は0.092〜0.318と変動した。各runの終端lossは0.05を大きく上回り、前提の「十分overfit」を満たさないため、現構造の表現能力に対するpass/failには使わない。[ladder結果](overfit-ladder-results.json)、[seed別loss・gradient・停止理由](overfit-ladder-training-summary.json)、[loss曲線](plots/overfit-ladder-loss.png)、[q sweep](plots/overfit-ladder-q-sweep.png)、[効果とsampling比較](plots/overfit-ladder-effects.png)。endpoint・前進量・経路長・旋回方向のseed別値は各stageの[B1](gate-b1-results.json)、[B8](gate-b8-results.json)、[B64](gate-b64-results.json)に保存した。

**解釈上の制約：** plateau判定は50 epoch前のfixed-bank lossと現在値の相対差で行った。lossは揺れ、B1 seed 11/29/47では最良値0.842/0.745/0.725に対して停止時0.990/0.948/1.140だった。停止規則の下では「1ペアすら構造的に学べない」とは判定できない。十分にoverfitできない原因候補は、選択ペアの教師差の大きさ、確率的diffusion学習の変動、最適化時間・条件信号の利用不足である。結果を見て停止規則、ペア、loss、モデルを変更していない。

## 実行・再開・検証

実行は未接続の`tmux` session `eatnav-quality-followup`で行い、SSH/VS Code接続を保持しなかった。[完了status](status.json)、[全stdout/stderr log](runner.log)、[smoke実測](smoke-results.json)を保存した。実行中の確認コマンドは`tmux ls`、`tmux capture-pane -pt eatnav-quality-followup`、`tmux attach -t eatnav-quality-followup`、`cat /tmp/eatnav-quality-identifiability-gate/followup/status.json`。完了後はsessionが終了し、statusとlogが残る。

各stageの`*.done.json`には入力・config・出力SHAとseedを保存し、不一致ならそのstageから再実行する。Probeは毎epoch、Bは25 epoch以内の間隔と終了時にmodel・optimizer・乱数状態を保存する。全7 stage markerのSHA照合を確認し、選択済み25 checkpointはコンテナ内とhost archiveでSHA-256一致を確認した。[SHA台帳](artifact-hashes.json)。元manifest SHA-256 `55d9e2d89631c2afa19dabfb02b2753616bc2b71907843f5570bd3a8e41e1475`、split hash `ae584ee5813be6a09f16baa4fae8271ed20c457ef40d9dd44005636a71e9234f`、follow-up config SHA-256 `05f8021b20ac9ee61b26ef647f0e3b444a7ebc1252394207b9f915b9ea`。実行環境はDocker `isaac-lab-onix`、Python 3.11.13、PyTorch 2.7.0+cu128、CUDA 12.8、NVIDIA GeForce RTX 5070 Ti。実行コードの事前コミットは`d164621`。実測全体時間は29分34秒で、smoke後の45〜90分見積もりより短かった。
