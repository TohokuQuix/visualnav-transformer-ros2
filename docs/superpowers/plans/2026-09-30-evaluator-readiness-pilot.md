# Counterfactual Evaluator readiness pilot

Approved scope: train-only seven-state, 112 fresh-process six-second rollouts with the existing exact-stamp follower diagnostic. No retraining, production follower edit, formal validation, or full collection.

1. Audit the clean worktree and exact-stamp provenance; rerun focused Python tests, ROS build, and syntax checks in their correct environments.
2. Select one complete train window per terrain from different episodes using only pre-outcome inputs, frozen SHA-256 seed 20260930, goal distance at least 0.30 m, and episode-local prefix at most 80 steps. Exclude earlier coverage states. Stop if any terrain lacks support.
3. Freeze the seven state IDs, source hashes, the same reference NoMaD checkpoint and eight seeds, five existing bank paths, 112-run order, and online-input versus teacher-only schema before looking at outcomes.
4. Add tests for selection and freeze invariants, candidate IDs, runner resume and stop gates. Extend the exact-stamp diagnostic runner only where new states require it.
5. Run a short smoke, measure runtime and memory, then launch the immutable schedule in detached tmux if the forecast fits seven hours and one gigabyte. Gate each state's three nominal repeats before candidate runs.
6. Analyze constraint and OOD validity, three-space diversity, feasible-union Recall@K, relative-label dispersion, safety labels, input deployability, and frozen Evaluator scores. Preserve raw traces under `/tmp` and commit only small artifacts.

Stop and report on any specified integrity, anchor, duplicate, timing, storage, or memory failure. Preserve completed evidence and do not change thresholds or selected states after freezing.
