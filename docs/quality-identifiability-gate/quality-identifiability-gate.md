# EAT-Nav Quality identifiability diagnostic

| Decision | Current status |
| --- | --- |
| Audit | **Warning; window integrity passes** |
| Gate A | Pending |
| Gate B | Pending |
| Overall | **audit warning / Gate A and B pending** |
| Full collection | Not started; remains disallowed until both gates pass |

## Window-count waterfall

The same 360,000-record extended pilot archive produced the following counts. The old 59,151-window duplication audit is a stricter *parallel definition*, not a filter applied after the current 69,180 windows. The exact [machine-readable waterfall](window-waterfall.json) includes split-specific counts.

| Stage | Count | Explanation |
| --- | ---: | --- |
| Raw DEM samples | 72,871 | 4 Hz periodic samples plus terminal samples; 287,129 non-DEM 20 Hz rows intentionally omit the key |
| Four-frame history complete | 69,259 | Four in-episode DEM frames at 0.25 s spacing |
| Current EAT dataset windows | **69,180** | 79 history-complete anchors lack sufficient future in nonterminal episodes at recording end |
| Current 8-point complete | 61,977 | All eight 0.25 s teacher points are valid |
| Current terminal-padded with 1–7 valid points | 6,302 | Future points after terminal are zero/masked; source rows never cross reset |
| Current terminal-padded with 0 valid points | 901 | Terminal anchor itself; zero loss contribution because all action-mask entries are false |
| Older strict complete definition | **59,151** | Uses local episode index stride 5 and requires eight future points; every one is in the current complete set |
| Current complete outside old strict definition | 2,826 | Every case begins in an episode whose first global step has phase 1–4 modulo 5; the older local stride misses those global DEM samples |
| Held-out EAT windows | **6,928** | Ten held-out terrain seeds under the unchanged seed 123 split |
| Held-out eight-point eligible windows | **6,208** | 6,928 minus 630 partially padded minus 90 zero-valid |

The [episode-boundary integration test](/tmp/eatnav-vnt/tests/eat_nav/test_window_waterfall.py) constructs two episodes with a large pose discontinuity at reset and verifies that DEM history, relative goal and all valid teacher points remain in the source episode; padded points are zero with a false mask. The real archive's current and legacy window-ID sets have zero cross-episode source lookups, zero goal-source mismatches, and no legacy-only windows. These counts explain the earlier report's distinct definitions without changing the split or source data.

## Data-audit warning and scope

Validation `k_rail_step_field` has 6 high-q bin-3 windows out of 679 (0.88%); `k_rail_field` has 25/693 (3.61%). This limits terrain-stratified estimates. Gate A will select checkpoints by **aggregate validation loss**, mark under-supported terrain×q strata `insufficient support`/NA with no performance or pass/fail estimate, and separately show common support among train/validation/held-out. Terrain-level metrics are secondary. Held-out data will be used once for final evaluation, never for checkpoint selection or thresholds.

Validation lacks `boxes`; held-out lacks `hf_pyramid_slope` and `k_rail_field`. The eventual generalization claim will therefore be limited to held-out terrain types, not all seven. The pilot has 884 timeout, 17 tilt and zero success terminal reasons. A timeout need not imply a driving failure; q is a train-ECDF-transformed continuous Critic P10, not a success label. There are no terrain-seed or episode split crossings. Whole-split q bins are train 13,839/13,839/13,840/13,840, validation 1,887/1,582/1,436/1,989, test 1,060/1,701/1,960/2,207. The [complete audit](data-audit.json) includes terrain and termination strata, goal/trajectory associations, DEM mask/height statistics and per-point XY deltas.

## Provenance and current limits

- Source archive manifest SHA-256: `55d9e2d89631c2afa19dabfb02b2753616bc2b71907843f5570bd3a8e41e1475`. Split seed 123; split hash `ae584ee5813be6a09f16baa4fae8271ed20c457ef40d9dd44005636a71e9234f`, matching both pilot checkpoints.
- Quality checkpoint SHA-256 `fcaf3b9397f3344a0017a52e324ad11ce879c29d9737e1b924d3800293e00987`; baseline `c8187d5b57baf6ddfb6ab36029de7b644b8dcfa699553bb6fbe260942c7465fb`.
- Runtime: `isaac-lab-onix`, Python 3.11.13, PyTorch 2.7.0+cu128, CUDA 12.8, NumPy 1.26.0, SciPy 1.15.3, PyYAML 6.0.2, diffusers 0.35.1; RTX 5070 Ti 16.3 GB.
- Audit command: `PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_quality_gate_audit.py --manifest /tmp/eatnav-extended/output/manifest.json --config /tmp/eatnav-train/quality.yaml --output /tmp/eatnav-train/quality-gate-output/data-audit.json`.
- Waterfall command: `PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src /isaac-sim/python.sh /tmp/eatnav-train/eat_nav_window_waterfall.py --manifest /tmp/eatnav-extended/output/manifest.json --config /tmp/eatnav-train/quality.yaml --output /tmp/eatnav-train/quality-gate-output/window-waterfall.json`.
- Existing quality loss, Quality token, Transformer, Diffusion U-Net, scheduler, Quality definition, ECDF, split and pilot artifacts remain unchanged. Gate A/B training and layer analysis have not yet run.
