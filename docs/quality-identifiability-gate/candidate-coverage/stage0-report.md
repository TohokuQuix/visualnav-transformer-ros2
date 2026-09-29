# Candidate coverage Stage 0 audit

Status: input audit passed; outcome-blind pilot state selection pending the approved wall-clock budget. No candidate generation or rollout has started.

- Base commit: `8e8f4fd980e7af78b4c43c79762723a97287392e` on `exp/eatnav-candidate-coverage`.
- Reference: Quality-disabled best EMA, epoch 120, SHA-256 `191d6a03266bca2ce8d43afac44b04828ae0d2571632be4f47a5f0dc3564b4e0`.
- Complete-only train: 49,598 windows; validation: 6,171. Held-out was not used.
- Every audited train window has a complete 8-point future and natural 4-frame DEM history from a single episode.
- Goal distance ≥0.30 m leaves 27,396 train windows.
- Full-prefix cost is **global anchor step + 1**. Across all complete train windows, median 1,781 steps and P90 3,246 steps. Among goal-eligible windows, median 1,776 and P90 3,236.
- Empirical pilot cost model uses state 42 (61 steps, 96.72 s) and boxes #7 (1,251 steps, 1,084.02 s). These two observations support only a rough estimate; actual runtime will be tracked per run.
- At a 40-step cap, all seven terrain types have an eligible state. The SHA-ranked 119-run pilot is estimated at 2.20 h, or 2.64 h with 20% reserve. State IDs are **not frozen** yet.

The selector verifies the manifest, reference config, and checkpoint SHA before reading the complete-only train view. It rejects missing budget/cap, partial windows, non-train windows, and attempts to overwrite a different frozen selection. It uses only terrain, episode/window identity, goal, DEM input hash, and global step; reward, Critic, termination, candidate results, and Evaluator values are excluded.

See `source-audit.json`, `full-prefix-cost.json`, and `budget-options.json` for machine-readable details. The next required input is the approved pilot wall-clock limit, which determines the full-prefix step cap before any state or candidate result is examined.
