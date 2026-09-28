# EAT-Nav Quality identifiability gate: stopped at data audit

| Decision | Result |
| --- | --- |
| Gate A | **Inconclusive — not run** |
| Gate B | **Inconclusive — not run** |
| Overall | **Inconclusive: stop at A-1 audit** |
| Full collection | **Do not start** |

The specified audit stop condition was found before pair selection or training: validation `k_rail_step_field` has **6 high-q (bin 3) windows among 679 (0.88%)**. Validation `k_rail_field` has 25/693 (3.61%) in bin 3. The validation set has no `boxes`; held-out test has no `hf_pyramid_slope` or `k_rail_field`. The split is valid, but this uneven terrain/Quality coverage could make terrain-specific probe comparison or robustness claims misleading. No thresholds were relaxed, no data were rebalanced, and the held-out split was not used for training or model selection. The existing Quality loss, token, Transformer, diffusion U-Net, scheduler, ECDF and normalization were not changed. The existing pilot checkpoints/reports were not overwritten.

## Audit evidence

The existing extended 100-env pilot has 360,000 raw records, including 72,871 valid transformed DEM samples. At 20 Hz, non-DEM steps omit the `dem` key; 287,129 such rows are expected at the 4 Hz DEM schedule. No sampled DEM had a missing transform or conversion error. Train-only raw Critic P10 generated the ECDF; all three reconstructed splits used identical train ECDF values. The terrain-seed split hash `ae584ee5813be6a09f16baa4fae8271ed20c457ef40d9dd44005636a71e9234f` matches both existing checkpoints. No terrain seed or episode crossed splits.

| Split | Terrain seeds | Windows | Episodes with windows | q bins 0/1/2/3 | Timeout episodes | Tilt failures | Success episodes |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Train | 80 | 55,358 | 726 | 13,839 / 13,839 / 13,840 / 13,840 | 708 | 12 | 0 |
| Validation | 10 | 6,894 | 94 | 1,887 / 1,582 / 1,436 / 1,989 | 87 | 4 | 0 |
| Held-out test | 10 | 6,928 | 90 | 1,060 / 1,701 / 1,960 / 2,207 | 89 | 1 | 0 |

The q bins are reasonably populated at whole-split level, while the validation terrain stratum above is strongly imbalanced. Every terminal `tilt_over` window is in q bin 0 (train 101 windows, validation 27, test 8); these are windows, not independent failure events. Terminal episode counts above come from raw records and total 884 timeouts plus 17 tilts. The pilot's recorded success reason count is zero.

Spearman correlation of q with goal distance is −0.280 train, −0.198 validation and −0.333 test; q with path length is −0.320, −0.247 and −0.377. q with DEM height standard deviation is −0.258, −0.324 and −0.265. This suggests partial goal-distance/roughness proxy behavior but does not establish whether q explains residual trajectory information. See [complete audit](data-audit.json) for terrain and termination strata, goal/trajectory statistics, DEM mask/height associations and per-point XY-delta correlations.

## Why computation stopped

The user's instruction says to report extreme bin imbalance at A-1 without silently correcting it. The 6/679 high-q validation `k_rail_step_field` count meets that condition. Its likely immediate source is the existing split of only 10 validation terrain seeds combined with strongly varying q distributions by terrain seed; this is a candidate explanation, not a confirmed causal result. The split and its checkpoint hash were preserved. Matched-pair thresholds, Probe 0/Q training, 10/100 q permutations, Gate B scratch overfit and layer sensitivity were **not executed**. Their result files and plots were therefore not fabricated.

The new [probe config](probe-config.yaml) and [overfit config](overfit-config.yaml) fix q=0.5 for Probe 0, underfit/validation-cap inconclusive logic, ten-to-hundred permutation escalation, selection quantiles, and episode reuse cap. Tests cover these decision/data invariants. Scratch overfit and its scratch-start test remain pending because the A-1 stop occurred first.

## Provenance and commands

- Source commit for the audited code: `0aa4a83`; split seed 123; manifest SHA-256 `55d9e2d89631c2afa19dabfb02b2753616bc2b71907843f5570bd3a8e41e1475`.
- Quality checkpoint SHA-256: `fcaf3b9397f3344a0017a52e324ad11ce879c29d9737e1b924d3800293e00987`; baseline: `c8187d5b57baf6ddfb6ab36029de7b644b8dcfa699553bb6fbe260942c7465fb`.
- Runtime: `isaac-lab-onix`, Python 3.11.13, PyTorch 2.7.0+cu128, CUDA 12.8, NumPy 1.26.0, SciPy 1.15.3, PyYAML 6.0.2, diffusers 0.35.1; RTX 5070 Ti 16.3 GB. No new GPU training was started; training VRAM was therefore not measured.
- Audit command in the container: `PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_quality_gate_audit.py --manifest /tmp/eatnav-extended/output/manifest.json --config /tmp/eatnav-train/quality.yaml --output /tmp/eatnav-train/quality-gate-output/data-audit.json`.
- Verification: `PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src /isaac-sim/python.sh -m pytest /tmp/eatnav-train/tests/eat_nav -q` → 57 passed. The raw audit exited 0 and produced the complete JSON.
