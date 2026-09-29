# Candidate coverage Stage 0 audit

Status: input audit passed; 4 h total wall-clock budget and 40-step full-prefix cap approved and frozen before state/candidate outcomes. No candidate generation or rollout has started.

- Base commit: `8e8f4fd980e7af78b4c43c79762723a97287392e` on `exp/eatnav-candidate-coverage`.
- Reference: Quality-disabled best EMA, epoch 120, SHA-256 `191d6a03266bca2ce8d43afac44b04828ae0d2571632be4f47a5f0dc3564b4e0`.
- Complete-only train: 49,598 windows; validation: 6,171. Held-out was not used.
- Every audited train window has a complete 8-point future and natural 4-frame DEM history from a single episode.
- Goal distance ≥0.30 m leaves 27,396 train windows.
- Full-prefix cost is **global anchor step + 1**. Across all complete train windows, median 1,781 steps and P90 3,246 steps. Among goal-eligible windows, median 1,776 and P90 3,236.
- Empirical pilot cost model uses state 42 (61 steps, 96.72 s) and boxes #7 (1,251 steps, 1,084.02 s). These two observations support only a rough estimate; actual runtime will be tracked per run.
- At the approved 40-step cap, all seven terrain types have an eligible state. The SHA-ranked 119-run pilot is estimated at 2.20 h, or 2.64 h with 20% reserve. State IDs are **not frozen** yet. The approved hard wall-clock limit is 4 h, including simulator/follower startup, replay, duplicate, candidate rollout, and allowed retry.

The selector verifies the manifest, reference config, and checkpoint SHA before reading the complete-only train view. It rejects missing budget/cap, partial windows, non-train windows, and attempts to overwrite a different frozen selection. It uses only terrain, episode/window identity, goal, DEM input hash, and global step; reward, Critic, termination, candidate results, and Evaluator values are excluded.

The 119 runs are **7 terrain × (1 anchor-only + 3 nominal duplicate + 8 NoMaD + 5 fixed bank)**. The additional seven anchor-only runs freeze each terrain state and its map-frame Path before a candidate is executed; they do not contribute an oracle candidate outcome. Terrain order, candidate seeds, bank names, and per-terrain run order are frozen in `pilot-config.yaml` before candidate outcomes. At 3 h elapsed, the runner must refresh its remaining-time estimate. It must stop starting new candidates whenever completion is forecast past 4 h; completed markers and outputs remain intact. Unexpected runtime (>2× pre-run estimate), duplicate gate failure, repeated follower timeout, or unexpected GPU memory also stop subsequent runs. Formal validation, retraining, and real robot control remain excluded.

See `source-audit.json`, `full-prefix-cost.json`, `budget-options.json`, and `pilot-config.yaml` for machine-readable details.

## Frozen pilot inputs (before candidate outcomes)

- Selection SHA-256: `1bb7865e81d2f0f52fff550037bb2e07b560b73f5de553b6183203bdaa969ddc`.
- Run schedule SHA-256: `8982e628bbb71e50eeea11534ca345bebc15009b33ef27bd16d70cc54182e95d`.
- Sorted terrain/state IDs and full-prefix steps: boxes `coverage-000` / 26; hf_pyramid_slope `coverage-001` / 21; k_rail_field `coverage-002` / 36; k_rail_step_field `coverage-003` / 16; pyramid_stairs `coverage-004` / 26; pyramid_stairs_inv `coverage-005` / 16; random_rough `coverage-006` / 31.
- All selected states are in episode 0, use distinct environment/episode keys, and have a goal distance ≥0.30 m. No reward, Critic, termination, candidate outcome, Evaluator output, or held-out data was used for selection.
- Predicted 119-run runtime: 7,912.67 s (2.20 h); with 20% reserve: 9,495.20 s (2.64 h), below the approved 14,400 s hard limit. The estimate is based on two prior runs and is updated from actual run times during the pilot.
