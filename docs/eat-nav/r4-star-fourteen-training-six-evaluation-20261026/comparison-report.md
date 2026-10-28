# R4 / R4*: 全14円弧候補を使う比較

予定36、処理済み36。既存14円弧候補で学習し、評価も同じ14候補を使用。

| field | 条件 | 有効case数 | 走行seed数 | 転倒観測走行 | 平均進捗m | 平均STOP秒 | 平均観測秒 | 正常waypoint | linear陽性window | 評価可能走行秒 | 未知走行秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k_rail_field | R4 | 11 | 2 | 6 | 2.067 | 0.36 | 41.86 | 6 | 47 | 448.00 | 8.45 |
| k_rail_field | R4* | 11 | 2 | 1 | 2.240 | 0.00 | 56.60 | 6 | 50 | 622.00 | 0.65 |
| k_rail_step_field | R4 | 3 | 3 | 0 | 1.772 | 0.00 | 60.00 | 1 | 46 | 180.00 | 0.00 |
| k_rail_step_field | R4* | 3 | 3 | 0 | 2.473 | 0.00 | 60.00 | 0 | 3 | 180.00 | 0.00 |
| pyramid_stairs | R4 | 3 | 3 | 0 | 1.969 | 19.33 | 60.00 | 1 | 0 | 122.00 | 0.00 |
| pyramid_stairs | R4* | 3 | 3 | 1 | 2.099 | 4.00 | 58.05 | 2 | 1 | 162.00 | 0.15 |
| pyramid_stairs_inv | R4 | 3 | 3 | 1 | 1.527 | 0.00 | 54.87 | 1 | 12 | 164.00 | 0.60 |
| pyramid_stairs_inv | R4* | 3 | 3 | 0 | 1.877 | 0.00 | 60.00 | 1 | 32 | 180.00 | 0.00 |
| random_rough | R4 | 3 | 3 | 0 | 2.725 | 0.00 | 60.00 | 1 | 0 | 180.00 | 0.00 |
| random_rough | R4* | 3 | 3 | 0 | 1.368 | 0.00 | 60.00 | 0 | 0 | 180.00 | 0.00 |
| boxes | R4 | 3 | 3 | 0 | 1.863 | 0.00 | 60.00 | 1 | 6 | 180.00 | 0.00 |
| boxes | R4* | 3 | 3 | 0 | 2.487 | 0.00 | 60.00 | 1 | 7 | 180.00 | 0.00 |
| hf_pyramid_slope | R4 | 3 | 3 | 0 | 2.279 | 0.00 | 60.00 | 1 | 0 | 180.00 | 0.00 |
| hf_pyramid_slope | R4* | 3 | 3 | 0 | 2.479 | 0.00 | 60.00 | 2 | 1 | 180.00 | 0.00 |

## R4*候補別の実行時間

| field | 候補 | 指令採用step | 採用時間秒 | decision選択回数 |
|---|---|---:|---:|---:|
| k_rail_field | forward_direct | 480 | 24.00 | 12 |
| k_rail_field | forward_-30 | 0 | 0.00 | 0 |
| k_rail_field | forward_+30 | 240 | 12.00 | 6 |
| k_rail_field | forward_-60 | 0 | 0.00 | 0 |
| k_rail_field | forward_+60 | 520 | 26.00 | 13 |
| k_rail_field | forward_-90 | 0 | 0.00 | 0 |
| k_rail_field | forward_+90 | 3440 | 172.00 | 86 |
| k_rail_field | reverse_direct | 320 | 16.00 | 8 |
| k_rail_field | reverse_-30 | 320 | 16.00 | 8 |
| k_rail_field | reverse_+30 | 560 | 28.00 | 14 |
| k_rail_field | reverse_-60 | 1000 | 50.00 | 25 |
| k_rail_field | reverse_+60 | 360 | 18.00 | 9 |
| k_rail_field | reverse_-90 | 1853 | 92.65 | 47 |
| k_rail_field | reverse_+90 | 3360 | 168.00 | 84 |
| k_rail_field | STOP | 0 | 0.00 | 0 |
| k_rail_step_field | forward_direct | 0 | 0.00 | 0 |
| k_rail_step_field | forward_-30 | 0 | 0.00 | 0 |
| k_rail_step_field | forward_+30 | 0 | 0.00 | 0 |
| k_rail_step_field | forward_-60 | 0 | 0.00 | 0 |
| k_rail_step_field | forward_+60 | 360 | 18.00 | 9 |
| k_rail_step_field | forward_-90 | 0 | 0.00 | 0 |
| k_rail_step_field | forward_+90 | 560 | 28.00 | 14 |
| k_rail_step_field | reverse_direct | 80 | 4.00 | 2 |
| k_rail_step_field | reverse_-30 | 0 | 0.00 | 0 |
| k_rail_step_field | reverse_+30 | 480 | 24.00 | 12 |
| k_rail_step_field | reverse_-60 | 0 | 0.00 | 0 |
| k_rail_step_field | reverse_+60 | 720 | 36.00 | 18 |
| k_rail_step_field | reverse_-90 | 80 | 4.00 | 2 |
| k_rail_step_field | reverse_+90 | 1320 | 66.00 | 33 |
| k_rail_step_field | STOP | 0 | 0.00 | 0 |
| pyramid_stairs | forward_direct | 0 | 0.00 | 0 |
| pyramid_stairs | forward_-30 | 0 | 0.00 | 0 |
| pyramid_stairs | forward_+30 | 120 | 6.00 | 3 |
| pyramid_stairs | forward_-60 | 0 | 0.00 | 0 |
| pyramid_stairs | forward_+60 | 160 | 8.00 | 4 |
| pyramid_stairs | forward_-90 | 200 | 10.00 | 5 |
| pyramid_stairs | forward_+90 | 1683 | 84.15 | 43 |
| pyramid_stairs | reverse_direct | 0 | 0.00 | 0 |
| pyramid_stairs | reverse_-30 | 80 | 4.00 | 2 |
| pyramid_stairs | reverse_+30 | 0 | 0.00 | 0 |
| pyramid_stairs | reverse_-60 | 0 | 0.00 | 0 |
| pyramid_stairs | reverse_+60 | 280 | 14.00 | 7 |
| pyramid_stairs | reverse_-90 | 240 | 12.00 | 6 |
| pyramid_stairs | reverse_+90 | 480 | 24.00 | 12 |
| pyramid_stairs | STOP | 240 | 12.00 | 6 |
| pyramid_stairs_inv | forward_direct | 80 | 4.00 | 2 |
| pyramid_stairs_inv | forward_-30 | 160 | 8.00 | 4 |
| pyramid_stairs_inv | forward_+30 | 80 | 4.00 | 2 |
| pyramid_stairs_inv | forward_-60 | 120 | 6.00 | 3 |
| pyramid_stairs_inv | forward_+60 | 0 | 0.00 | 0 |
| pyramid_stairs_inv | forward_-90 | 240 | 12.00 | 6 |
| pyramid_stairs_inv | forward_+90 | 1200 | 60.00 | 30 |
| pyramid_stairs_inv | reverse_direct | 360 | 18.00 | 9 |
| pyramid_stairs_inv | reverse_-30 | 280 | 14.00 | 7 |
| pyramid_stairs_inv | reverse_+30 | 640 | 32.00 | 16 |
| pyramid_stairs_inv | reverse_-60 | 360 | 18.00 | 9 |
| pyramid_stairs_inv | reverse_+60 | 0 | 0.00 | 0 |
| pyramid_stairs_inv | reverse_-90 | 80 | 4.00 | 2 |
| pyramid_stairs_inv | reverse_+90 | 0 | 0.00 | 0 |
| pyramid_stairs_inv | STOP | 0 | 0.00 | 0 |
| random_rough | forward_direct | 0 | 0.00 | 0 |
| random_rough | forward_-30 | 0 | 0.00 | 0 |
| random_rough | forward_+30 | 0 | 0.00 | 0 |
| random_rough | forward_-60 | 80 | 4.00 | 2 |
| random_rough | forward_+60 | 360 | 18.00 | 9 |
| random_rough | forward_-90 | 80 | 4.00 | 2 |
| random_rough | forward_+90 | 2640 | 132.00 | 66 |
| random_rough | reverse_direct | 0 | 0.00 | 0 |
| random_rough | reverse_-30 | 0 | 0.00 | 0 |
| random_rough | reverse_+30 | 0 | 0.00 | 0 |
| random_rough | reverse_-60 | 0 | 0.00 | 0 |
| random_rough | reverse_+60 | 200 | 10.00 | 5 |
| random_rough | reverse_-90 | 120 | 6.00 | 3 |
| random_rough | reverse_+90 | 120 | 6.00 | 3 |
| random_rough | STOP | 0 | 0.00 | 0 |
| boxes | forward_direct | 80 | 4.00 | 2 |
| boxes | forward_-30 | 80 | 4.00 | 2 |
| boxes | forward_+30 | 0 | 0.00 | 0 |
| boxes | forward_-60 | 200 | 10.00 | 5 |
| boxes | forward_+60 | 240 | 12.00 | 6 |
| boxes | forward_-90 | 720 | 36.00 | 18 |
| boxes | forward_+90 | 1800 | 90.00 | 45 |
| boxes | reverse_direct | 0 | 0.00 | 0 |
| boxes | reverse_-30 | 200 | 10.00 | 5 |
| boxes | reverse_+30 | 0 | 0.00 | 0 |
| boxes | reverse_-60 | 0 | 0.00 | 0 |
| boxes | reverse_+60 | 0 | 0.00 | 0 |
| boxes | reverse_-90 | 0 | 0.00 | 0 |
| boxes | reverse_+90 | 280 | 14.00 | 7 |
| boxes | STOP | 0 | 0.00 | 0 |
| hf_pyramid_slope | forward_direct | 0 | 0.00 | 0 |
| hf_pyramid_slope | forward_-30 | 0 | 0.00 | 0 |
| hf_pyramid_slope | forward_+30 | 0 | 0.00 | 0 |
| hf_pyramid_slope | forward_-60 | 0 | 0.00 | 0 |
| hf_pyramid_slope | forward_+60 | 120 | 6.00 | 3 |
| hf_pyramid_slope | forward_-90 | 520 | 26.00 | 13 |
| hf_pyramid_slope | forward_+90 | 800 | 40.00 | 20 |
| hf_pyramid_slope | reverse_direct | 80 | 4.00 | 2 |
| hf_pyramid_slope | reverse_-30 | 80 | 4.00 | 2 |
| hf_pyramid_slope | reverse_+30 | 0 | 0.00 | 0 |
| hf_pyramid_slope | reverse_-60 | 200 | 10.00 | 5 |
| hf_pyramid_slope | reverse_+60 | 280 | 14.00 | 7 |
| hf_pyramid_slope | reverse_-90 | 520 | 26.00 | 13 |
| hf_pyramid_slope | reverse_+90 | 1000 | 50.00 | 25 |
| hf_pyramid_slope | STOP | 0 | 0.00 | 0 |

## 全予定の処理結果

| field | 学習seed | 走行seed | 状態 | 対応比較 | 理由 |
|---|---:|---:|---|---|---|
| k_rail_field | 123 | 1791182661 | completed | True | horizon_complete |
| k_rail_field | 456 | 1791182661 | completed | True | horizon_complete |
| k_rail_field | 789 | 1791182661 | completed | True | horizon_complete |
| k_rail_field | 101112 | 1791182661 | completed | True | horizon_complete |
| k_rail_field | 131415 | 1791182661 | completed | True | natural_termination |
| k_rail_field | 161718 | 1791182661 | completed | False | horizon_complete |
| k_rail_field | 123 | 1831975652 | completed | True | horizon_complete |
| k_rail_field | 456 | 1831975652 | completed | True | horizon_complete |
| k_rail_field | 789 | 1831975652 | completed | True | horizon_complete |
| k_rail_field | 101112 | 1831975652 | completed | True | horizon_complete |
| k_rail_field | 131415 | 1831975652 | completed | True | horizon_complete |
| k_rail_field | 161718 | 1831975652 | completed | True | horizon_complete |
| k_rail_field | 123 | 423440295 | failed | False | termination during settling |
| k_rail_field | 456 | 423440295 | failed | False | termination during settling |
| k_rail_field | 789 | 423440295 | failed | False | termination during settling |
| k_rail_field | 101112 | 423440295 | failed | False | termination during settling |
| k_rail_field | 131415 | 423440295 | failed | False | termination during settling |
| k_rail_field | 161718 | 423440295 | failed | False | termination during settling |
| k_rail_step_field | 123 | 1752506436 | completed | True | horizon_complete |
| k_rail_step_field | 123 | 779651991 | completed | True | horizon_complete |
| k_rail_step_field | 123 | 2177554285 | completed | True | horizon_complete |
| pyramid_stairs | 123 | 2153479260 | completed | True | natural_termination |
| pyramid_stairs | 123 | 2465645667 | completed | True | horizon_complete |
| pyramid_stairs | 123 | 1005734882 | completed | True | horizon_complete |
| pyramid_stairs_inv | 123 | 565464307 | completed | True | horizon_complete |
| pyramid_stairs_inv | 123 | 4270400402 | completed | True | horizon_complete |
| pyramid_stairs_inv | 123 | 2340268683 | completed | True | horizon_complete |
| random_rough | 123 | 1796165707 | completed | True | horizon_complete |
| random_rough | 123 | 1870190470 | completed | True | horizon_complete |
| random_rough | 123 | 1137732034 | completed | True | horizon_complete |
| boxes | 123 | 2113840787 | completed | True | horizon_complete |
| boxes | 123 | 2399893319 | completed | True | horizon_complete |
| boxes | 123 | 3298388622 | completed | True | horizon_complete |
| hf_pyramid_slope | 123 | 2353927512 | completed | True | horizon_complete |
| hf_pyramid_slope | 123 | 2685046136 | completed | True | horizon_complete |
| hf_pyramid_slope | 123 | 2292600284 | completed | True | horizon_complete |

候補別採用stepはSTOPを除いた実際の候補ID、decision数は2秒再評価時の最終選択。候補継続と新規選択を混同しない。
対応するR4/R4*の初期状態・共通入力一致と再現監査が成立した集合だけで比較する。集計途中は結論を出さない。
学習成功モデルには全14候補の照合済み教師を使用。教師不足・学習失敗はN/A。R4との差は学習データと評価候補の拡張の組合せ。
転倒・STOP・進捗・観測時間を併記する。早期転倒でSTOP秒が短くなる可能性がある。linear陽性windowは独立事故件数ではない。
走行seed反復、既存K-Rail validation不成立、初期PhysX警告とsettling失敗の制約を引き継ぐ。reserved test未開封。
