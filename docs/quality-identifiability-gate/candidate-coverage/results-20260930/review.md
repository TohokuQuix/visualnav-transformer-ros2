# Candidate coverage train pilot review (2026-09-30)

## Execution and scope

The frozen 7-terrain, 119-run schedule completed at 05:00 JST. The wall clock from the pre-offline budget start was 9,022.9 s (2 h 30 m 23 s), under the approved 14,400 s ceiling. The 3 h forecast checkpoint was not reached. All seven anchor checks and 21 nominal duplicates completed; all seven duplicate gates passed with 40-step XY RMS of 0.0000 m. The remaining 91 candidate runs yielded 91 oracle outcomes. Peak observed GPU memory was 5,549 MiB. Pilot files occupied 213.3 MB, including 132.5 MB of logs and 45.4 MB of compressed traces.

One Isaac Kit startup segmentation fault occurred before simulator initialization on `pilot-013`. The original failure and explicit review are preserved under `/tmp/eatnav-candidate-coverage/pilot/`. The same candidate passed on its sole permitted infrastructure retry. State, candidate, seed, schedule, deadline, and model were unchanged.

Only complete-only train states were used. No formal validation, retraining, Phase D/E, full collection, or real robot control was started. NoMaD K32 was assessed offline only; only its first eight fixed seeds were rolled out. The seven terrain names indicate train pilot coverage, not generalization.

## Primary proposal coverage

With the pre-frozen ΔV epsilon of 0.01, NoMaD Oracle Recall@K across the 13-candidate pool was 1/7 at K1, 1/7 at K2, 2/7 at K4, and 3/7 at K8. K8 best exceeded nominal in 5/7 states and fixed-bank best in 3/7 states (boxes, k_rail_field, pyramid_stairs_inv). Those three wins used different seeds (1006, 1003, 1001) and their winning proposals passed the recorded OOD/constraint screen. NoMaD K8 commands formed more than one rollout cluster in 4/7 states; the other three had one cluster. The K8 path geometry itself formed one cluster under the frozen path threshold in every terrain, so command/outcome variation should not be described as broad geometric maneuver diversity.

All 91 candidate rollouts were noncatastrophic for the 40-step horizon; no stack, excessive tilt, or termination reason was recorded. Consequently this pilot does not establish coverage of failure avoidance or long-horizon safety. Ten of 91 proposals failed the combined OOD/constraint screen. The best NoMaD outcomes above the fixed bank did not depend on those invalid proposals.

**Interpretation:** the train pilot gives partial evidence for proposal coverage: some terrain-dependent improvement and K gains are present, but Recall@8 is 3/7 and several terrains show command collapse. It is insufficient to claim generalization or to start formal validation without a separate approval. The frozen bank remains the better source of oracle-best ΔV in four terrains. No candidate or threshold was changed after seeing outcomes.

## Secondary Evaluator diagnostic

The Phase B GT Evaluator was run only after all oracle outcomes existed and did not select rollout candidates. For ΔV, its oracle-best top-1 recall was 0/7, top-3 recall 2/7, and pairwise accuracy 0.466 over 487 supported pairs. This is a secondary ranking limitation; it does not alter the proposal coverage measurement.

The full raw outcome table, traces, checkpoints, and logs remain in `/tmp/eatnav-candidate-coverage/`. This directory contains small summary JSON, reports, and six diagnostic figures. Source hashes are in `postprocess-complete.json`; frozen inputs and run markers are under the same `/tmp` root.
