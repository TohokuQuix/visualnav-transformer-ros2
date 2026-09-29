# Feasible transfer: sample audit and stratified addendum

This addendum extends `analysis-report.md` without changing any original result. No candidate, checkpoint, rollout or gate decision changed.

## Sample-level input check
- Seven pilot states, 13 candidates each; DEM tensor shape 4×2×20×20; selected goal maximum absolute difference 0 m.
- All P/C/R candidate trajectories have shape 13×8×2; increment↔cumulative round-trip maximum absolute error 7.45e-09 m.
- The one Phase B held-out example has 8×2 target increments and round-trip error 3.73e-09 m. Held-out was used only to verify the frozen implementation, never for model or threshold selection.
- Logged teacher endpoint can differ from the clipped relative goal in three pilot states. This is a recorded data property, not a conversion error; all selected goal values match exactly.
- Candidate conversion is numerically consistent across P/C/R. Their semantic sources differ, and the fixed-goal/candidate-endpoint coupling remains a measured distribution shift.

## Within-state stratification
The figures below average each supported state equally. Pair counts are descriptive and are not independent sample counts. Subsets with one candidate have no pairwise statistic.

| Candidate subset | Supported states | GT(P) pairwise | GT(C) pairwise | GT(R) pairwise |
|---|---:|---:|---:|---:|
| all | 7 | 0.467 | 0.669 | 0.534 |
| nomad | 7 | 0.496 | 0.522 | 0.433 |
| fixed_bank | 7 | 0.562 | 0.784 | 0.422 |
| constraint_valid | 7 | 0.460 | 0.640 | 0.517 |
| planned_in_distribution | 7 | 0.453 | 0.636 | 0.517 |
| planned_ood | 3 | 0.667 | 1.000 | 0.000 |

GT(P) remains below chance for planned trajectories screened in-distribution (0.453 over seven states), so the eight p95-OOD planned candidates do not by themselves explain the transfer failure. NoMaD-only GT(P) is 0.496 and fixed-bank-only is 0.562; neither establishes reliable within-state selection. The OOD subset has only three supported pairs across three states and is descriptive only.

Command-cluster strata: multi-cluster states (4) GT(P) pairwise 0.452; single-cluster states (3) 0.488. Both remain near or below chance. Full per-state candidate IDs, generator, seeds, score spreads, calibration and ensemble disagreement are in `evaluator-strata-detail.json`.

## Source integrity

| Added artifact | SHA-256 |
|---|---|
| evaluator-input-audit-detail.json | `716d603d92666814db3d801b614a64b754040c74822a06bd1b2e5bdad68c9baf` |
| evaluator-strata-detail.json | `fa061213b7e36d1032b2b5f6f27e8579fcdccd7dcf141304e5ad68e1c9fbf3e7` |

The original Stage 3 decision remains Gate N `inconclusive` and Gate E `fail`. The proposed long-horizon experiment has not started and requires separate approval.
