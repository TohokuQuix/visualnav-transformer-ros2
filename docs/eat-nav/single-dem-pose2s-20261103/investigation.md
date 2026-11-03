# 今回の再調査記録

新収集・学習・製品コード変更なし。参照版はreference-index.jsonで固定する。

- `demo_terrain.py`の正式7名称とlevel0〜4を確認。8番目のフィールドを追加しない。
- `bidirectional_bank.py`で14候補、円弧0.75m、8点、前後反転、角度順を確認。
- `checkpoints/eat_nav/rl_actor/model_10000.pt`をCPUで`weights_only=True`で読んだ。Actor weight shapesは[1024,22]、[512,1024]、[256,512]、[6,256]。既存学習済みActorを使い、今回はActor自体を再学習しない。
- Actor envと現在onix_isaac_labの`actions_flipper.py`を実際に確認。fold mode0で方向強制なし、rate limit・干渉制約を継続する。Actor/現在予測NN/実機bridgeを区別する。
- `rough_env_cfg.py`実ファイルで初期xy±.1m、yaw±π、6成分の初速0、既存tilt終端85度を確認。
- `p1_follower_core.cpp`でlookahead.5m、v上限.30、gain.5、ω上限1.6、heading減速、goal.2、退避/boostが存在することを確認。新仕様はnominal追従のみを共通profile化する提案で、自動回復を無断継承しない。
- 既存`make_collection_cfg`は20×20のrayとupdate .25秒を作り、既存field helperは1/4env対応。40×40・10envは新しい実装が必要。

## 保存済み代表サンプルを開いた結果

`/home/ubuntu/eatnav-moving-state-pose2s-data-20261103/reused-dataset/samples.npz`：

| array | 実shape |
|---|---|
| dem | [6897,4,2,20,20] |
| actor_state | [6897,22] |
| body_state | [6897,9] |
| position_targets | [6897,1,3] |
| pose_mask | [6897,1] |

raw代表：`/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/all-fields-learning-20261023/collect-320/sim-output/raw-dem-0080.npz`：

- ray_hits_w、ray_starts_w：[1,400,3]
- root_state_w：[1,13]
- env_origins_w：[1,3]
- ray_valid：[1,400]

したがって、この代表データは4m/1600セルの観測ではない。旧データを新schemaへ単純reshapeして使わない。上記6897はreused pool単体で、前回freshを含む合計7765と混同しない。過去監査の件数を新仕様で使える件数と認定しない。

旧実機bagは20×20・約3Hzだった。今回ユーザー指示で4Hzを維持するため、3Hzへ合わせる案を撤回し、新仕様は1frameの実ageを扱う。過去bagの取得stampを変えて4Hzに見せない。

## 仕様段階の算術監査

14試行でcandidate=(r+3j) mod14を割り当てると、各候補は5つの独立親・各stageに1回ずつ現れる。35cell×14試行=490final試行、予定2450window、490評価組×5window。train350試行だけでfinal490組を満たすとは扱わない。これは計画の算術確認であり、有効データ収集完了や精度達成を意味しない。
