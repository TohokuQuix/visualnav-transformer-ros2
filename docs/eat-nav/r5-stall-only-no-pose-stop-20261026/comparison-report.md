# R4* / R5: 直線を維持して旋回スタック予測を追加

予定36、処理済み36。再学習なし、R4*と同じ14候補学習済みweights・同じ14候補。

両条件で直線risk有効、転倒/near-fall/実測45度STOP/35度復帰は無効。R5は旋回risk重み0.5と既存R2の除外条件を追加。
STOP秒は保護による速度指令ゼロの時間であり、スタックしている時間とは別。

| field | 条件 | 有効走行数 | 走行seed数 | 転倒観測走行数 | 平均route進捗m | 平均STOP秒 | 平均観測秒 | 正常waypoint |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| k_rail_field | R4* | 12 | 2 | 1 | 2.211 | 0.00 | 56.89 | 6 |
| k_rail_field | R5 | 12 | 2 | 0 | 2.387 | 0.00 | 60.00 | 6 |
| k_rail_step_field | R4* | 3 | 3 | 0 | 2.473 | 0.00 | 60.00 | 0 |
| k_rail_step_field | R5 | 3 | 3 | 0 | 2.483 | 0.00 | 60.00 | 0 |
| pyramid_stairs | R4* | 3 | 3 | 1 | 2.099 | 4.00 | 58.05 | 2 |
| pyramid_stairs | R5 | 3 | 3 | 0 | 2.312 | 5.33 | 60.00 | 2 |
| pyramid_stairs_inv | R4* | 3 | 3 | 0 | 1.877 | 0.00 | 60.00 | 1 |
| pyramid_stairs_inv | R5 | 3 | 3 | 0 | 1.614 | 0.00 | 60.00 | 1 |
| random_rough | R4* | 3 | 3 | 0 | 1.368 | 0.00 | 60.00 | 0 |
| random_rough | R5 | 3 | 3 | 0 | 1.830 | 0.00 | 60.00 | 1 |
| boxes | R4* | 3 | 3 | 0 | 2.487 | 0.00 | 60.00 | 1 |
| boxes | R5 | 3 | 3 | 0 | 2.347 | 0.00 | 60.00 | 1 |
| hf_pyramid_slope | R4* | 3 | 3 | 0 | 2.479 | 0.00 | 60.00 | 2 |
| hf_pyramid_slope | R5 | 3 | 3 | 0 | 2.626 | 0.00 | 60.00 | 2 |

## 動作中のスタック観測

| field | 条件 | 直線陽性window | 旋回陽性window | 直線評価可能秒 | 旋回評価可能秒 | 直線未知秒 | 旋回未知秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| k_rail_field | R4* | 59 | 6 | 682.00 | 682.00 | 0.65 | 0.65 |
| k_rail_field | R5 | 76 | 3 | 720.00 | 720.00 | 0.00 | 0.00 |
| k_rail_step_field | R4* | 3 | 0 | 180.00 | 180.00 | 0.00 | 0.00 |
| k_rail_step_field | R5 | 8 | 0 | 180.00 | 180.00 | 0.00 | 0.00 |
| pyramid_stairs | R4* | 1 | 1 | 162.00 | 162.00 | 0.15 | 0.15 |
| pyramid_stairs | R5 | 0 | 1 | 164.00 | 164.00 | 0.00 | 0.00 |
| pyramid_stairs_inv | R4* | 32 | 1 | 180.00 | 180.00 | 0.00 | 0.00 |
| pyramid_stairs_inv | R5 | 29 | 1 | 180.00 | 180.00 | 0.00 | 0.00 |
| random_rough | R4* | 0 | 5 | 180.00 | 180.00 | 0.00 | 0.00 |
| random_rough | R5 | 1 | 7 | 180.00 | 180.00 | 0.00 | 0.00 |
| boxes | R4* | 7 | 1 | 180.00 | 180.00 | 0.00 | 0.00 |
| boxes | R5 | 0 | 4 | 180.00 | 180.00 | 0.00 | 0.00 |
| hf_pyramid_slope | R4* | 1 | 1 | 180.00 | 180.00 | 0.00 | 0.00 |
| hf_pyramid_slope | R5 | 0 | 4 | 180.00 | 180.00 | 0.00 | 0.00 |

## 全予定の記録

| field | 学習seed | 走行seed | 状態 | 比較成立 | 理由 |
|---|---:|---:|---|---|---|
| k_rail_field | 123 | 1791182661 | completed | True | horizon_complete |
| k_rail_field | 456 | 1791182661 | completed | True | horizon_complete |
| k_rail_field | 789 | 1791182661 | completed | True | horizon_complete |
| k_rail_field | 101112 | 1791182661 | completed | True | horizon_complete |
| k_rail_field | 131415 | 1791182661 | completed | True | horizon_complete |
| k_rail_field | 161718 | 1791182661 | completed | True | horizon_complete |
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
| pyramid_stairs | 123 | 2153479260 | completed | True | horizon_complete |
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

同じ学習済みweights・seed・入力が一致したR4*/R5だけで対応比較する。学習seed反復は独立走行seedではない。
旧R2はモデル・候補数が異なるため、この比較から45度STOP単独の効果を断定しない。STOP減少だけで改善とせず、転倒/進捗/観測時間/動作中stallを併記する。
windowは独立事故数ではない。早期終了・未知区間・起動失敗・unsupported headを陰性にしない。reserved test未開封。
各走行auditのcandidate_steps/candidate_decisionsに候補別利用を保存する。
