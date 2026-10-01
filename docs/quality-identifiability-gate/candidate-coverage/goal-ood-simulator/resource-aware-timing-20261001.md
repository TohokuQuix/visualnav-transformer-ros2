# Goal-conditioned local subgoal: resource-aware timing probe

## Decision

`runtime_gate_fail`。固定済み `train-003` の fresh-process / full-prefix anchor replay は 1621 control step、4回の正常な episode reset を経て完走した。anchor pose、DEM history、command は既存記録と一致した。失敗の理由は資源不足ではなく、同じ方式で4 stateのanchor 4 runとlive 44 runを実行した場合の4時間予算超過である。実測に基づく予測はpoint **17.63時間**、P95 step timeによるconservative **18.24時間**、その20% margin込み **21.89時間**、worst observed **21.80時間**。live run、candidate outcome、GS1–GS4は開始していない。

## 固定条件と前回停止の解釈

出力rootは `/tmp/eatnav-candidate-coverage/goal-ood-resource-timing-20261001/`。旧rootと失敗証拠は保持した。R0 provenanceはpass（SHA256 `7838307568620d1007e765280f27a2a09af78f0aed74120e45edaa199d61e638`）。開始前にsimulator、bridge、follower、monitorの残留processはなかった。Docker `memory.max=max`、`memory.events` のOOM系countは0、host `MemTotal=100,954,218,496` bytes、`MemAvailable=96,073,982,976` bytes。host `/usr/bin/python3` と NumPy 1.26.4を固定した。4 state、16 goal、32 Path、seed、44 runの順序、NoMaD/RL checkpoint、manifest/splitのSHAはR0に保存した。

前回のstop条件は cgroup `memory.current > 8 GiB`。この値には回収可能なfile cacheが含まれ、前回の停止後には約5.88 GBのfile cacheがあった。今回は `memory.current` を警告値として記録し、hard capを `anon + shmem` の6 GiBに固定した。known-good anon peak `2,493,247,488` bytesの2倍より6 GiBが大きい。host `MemAvailable` は10% `MemTotal`（`10,095,421,850` bytes）未満が連続3 heartbeatで停止、GPUは6656 MiB、step stallは120秒、heartbeat欠損は90秒、保存量上限は0.8 GB。固定config SHA256 `fa312f532881044d4675222c209ab4c251865928825a66f364122271b115b2e7`。cache削除や閾値の事後変更はしていない。

monitor、停止判定、原子的marker、cleanup、NumPy後処理、予測reportの10 testは通過した。今回のsimulator source SHA256 `e177be4772718dce6f951e4998dd7511d979b64f8cbc6212b421bab47b908f87` は前回parity sourceと異なるため、現sourceで120-step parityを再実施した。30 field一致、raw joint effort対Actor pre-noise effortの最大差 `3.33786e-6`、parity結果pass。NoMaD、Evaluator、RL checkpoint、follower制御則、exact-stamp protocol、bridge、reset/dynamicsは変更していない。

## Anchor実測

`train-003` episode key `[0,8,4]`、target global step 1620、episode-local step 20。resetはglobal step 399、799、1199、1599で発生し、予期しないresetはない。full-prefix 1621 stepのhost wallは **1370.18秒（22.84分）**、prefix区間は1351.31秒。通常step（reset前後2 stepを除く1601件）はmean 0.83540秒、median 0.83908秒、P90 0.86107秒、P95 0.86718秒、P99 0.87655秒、max 1.05317秒。reset step 4件のmeanは0.92630秒、maxは1.12172秒。Docker exec→Isaac開始0.129秒、scene構築12.695秒、initial resetとpolicy準備0.259秒、prefixの非step部分0.823秒、anchor照合0.015秒、cleanupとDocker return5.691秒。simulator process CPUはprefix開始時40.59秒、anchor到達時1620.58秒で、この間約1579.99 CPU秒。

Pose位置差0 m、姿勢差0.00122 rad、速度最大差0、DEM RMS 0 m、DEM history SHA一致、mask IoU 1.0、prefix command差0。logged actionは今回の `action_manager_action` と6成分すべて一致した。今回の別時点の `actor_action` とlogged actionの最大差は0.5993で、同一の観測量として扱わない。Critic value差は0.01587。joint position/velocity、Actor observation/latent、Critic observation、previous action、raw effortは今回のanchorで保存・finite確認したが、旧logged anchorには同じ詳細fieldがなく、過去値との照合はNA。これを完全な全field再現と主張しない。

10秒周期のheartbeat 130件で最大間隔10.65秒。cgroup `memory.current` は6.23→10.11 GB、内訳のpeakはanon 2.502 GB、shmem 1.415 GB、file 7.295 GB、slab 0.291 GB。`anon+shmem` peak **3.917 GB**で6 GiB cap未満。simulator RSS peak 5.704 GB、PSS peak 5.682 GB。host `MemAvailable`最低91.36 GBで下限より十分高い。GPU peak5550 MiB。`memory.events`のlow/high/max/oom/oom_killの開始前後差分はいずれも0。step stall、heartbeat欠損、resource stopは0。simulator exit 0、cleanup後の残留process 0。

## 4時間予測と適用範囲

固定prefixは `train-003=1621`、`train-004=421`、`train-019=436`、`train-021=3241` step。各stateでanchor 1 runとfresh-process live 11 run（nominal duplicate 3＋candidate 8）を実行するため、full-prefixは合計 **68,628 step**、live suffixは44×120 step。予測は各stateについて `anchor_s + 11×(launch_s + full_prefix_s + 120-step suffix_s + cleanup_s)` の和である。suffixは現sourceの120-step exact-stamp parityから、follower/bridge setup1.085秒、stepとcommand105.441秒、trace保存1.691秒、host copy/cleanup9.355秒を使用した。probeとparityの失敗・stall時間は通常step分布に含めていない。

| 予測 | 時間 |
| --- | ---: |
| point（通常step mean） | 17.63 h |
| conservative（通常step P95） | 18.24 h |
| conservative +20% | **21.89 h** |
| worst observed | 21.80 h |

保存量予測は約239.9 MBで0.8 GB以内、今回rootの実測は6,819,136 bytes。GPU peakは上限未満。4時間超過が明確であり、GPUやメモリによる判定変更はない。これは `train-003` のstep/reset分布を他の3 stateへ外挿した予測で、他stateの実測時間ではない。前回の6.3時間外挿は履歴として保持し、今回の判定には使用していない。

## 証拠と停止状態

- R5 anchor: `resource-anchor-result.json` SHA256 `a535d39b48f637ef867e28cf709805bb2c15808ff341659e7b53a3c338ed8aa7`。
- Step durations: `runs/train-003-resource-anchor/timing-step-durations.json` SHA256 `35a25a0fb7faab97fe7096ebc1b43e0db30d33998eab80c58e1a4895ba0dd88f`。
- Resource timeline: `runs/train-003-resource-anchor/resource-timeseries.jsonl` SHA256 `7ca8bbe55a66b658839e9d4827f5d31fa2649e2f51b18398317e1194bfb15fc0`。
- Forecast: `runtime-forecast.json` SHA256 `7159bbf7d8769ae6625f82e808ed0d78a6087dede265ee5d36b453f23fdcf3e3`。
- Gate result: `resource-gate-result.json` SHA256 `d7272d45f712f15ddbcd6cac99e433edbd2721ff3bd879b684a9d0b9e8421979`。
- Source/input SHAは `resource-probe-provenance.json` と `resource-probe-source.json`、parity結果は `parity/parity-result.json` に保存した。

一度だけの承認済みtiming probeで、retry 0、anchor 1、nominal duplicate 0、candidate rollout 0。完了後に専用tmux routerも停止し、tmux sessionとsimulator/bridge/follower/monitorの残留は0。固定state/goal/Path/seedは変更していない。44 live run、GS1–GS4、モデル再学習、実機制御は未実施。
