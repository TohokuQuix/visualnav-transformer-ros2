# Candidate coverage train pilot

Completed terrain states: 7/7; completed candidate outcomes: 91/91.
ΔV oracle Recall@K among complete states: {'1': 0.14285714285714285, '2': 0.14285714285714285, '4': 0.2857142857142857, '8': 0.42857142857142855}.
NoMaD K8 beats nominal in 5/7 complete states; beats the fixed bank in 3/7.

Only complete-only train states were used. Formal validation, retraining, and real robot control were not started.
The K32 set was assessed offline only; its oracle outcome is unavailable in this pilot.
The reported recall applies only to the listed train terrain states and the evaluated 13-candidate pool.

## State table

| Terrain | Status | ΔV K1 | ΔV K8 | bank best | hybrid best | K8 command clusters | duplicate XY RMS |
|---|---|---:|---:|---:|---:|---:|---:|
| boxes | complete | 0.751 | 0.947 | 0.813 | 0.947 | 3 | 0.000 |
| hf_pyramid_slope | complete | 0.949 | 0.999 | 1.018 | 1.018 | 5 | 0.000 |
| k_rail_field | complete | 1.353 | 1.580 | 1.472 | 1.580 | 3 | 0.000 |
| k_rail_step_field | complete | 0.238 | 0.267 | 0.282 | 0.282 | 1 | 0.000 |
| pyramid_stairs | complete | -0.096 | 0.050 | 0.144 | 0.144 | 1 | 0.000 |
| pyramid_stairs_inv | complete | 1.008 | 1.008 | 0.975 | 1.008 | 2 | 0.000 |
| random_rough | complete | -0.012 | -0.012 | 0.028 | 0.028 | 1 | 0.000 |
