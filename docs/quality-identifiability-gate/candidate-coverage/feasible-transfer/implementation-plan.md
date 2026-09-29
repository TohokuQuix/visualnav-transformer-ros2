# Feasible Coverage and Evaluator Transfer Implementation Plan

**Goal:** Diagnose train-pilot feasible mode coverage and frozen Phase B Evaluator transfer from existing artifacts, without any new rollout or training.

**Architecture:** Immutable pilot inputs are verified by a Stage 0 provenance/identity gate. Pure, tested geometry/ranking helpers produce Stage 1 coverage results. A separate Docker GPU process first reproduces original Phase B held-out metrics, then evaluates the same 91 frozen candidates as planned, command-derived, and realized trajectories. A final offline renderer writes traceable JSON, figures, cause classifications, and decisions.

**Tech Stack:** Python 3.12, NumPy/SciPy, PyTorch only for frozen Evaluator inference, Matplotlib, Docker Isaac image for original checkpoints/data.

**Spec:** User attachment `5f0a5247-af3d-43d6-b6d5-44d0416a51c5/Pasted text.txt`.

## Global constraints

- Only `exp/eatnav-candidate-coverage` dedicated worktree; preserve original pilot files.
- No simulator rollout, retraining, formal validation, Phase D/E, full collection, or real robot control.
- Frozen seven states, 91 candidate runs, seeds, paths, checkpoint weights, ΔV, epsilon 0.01, and existing cluster thresholds.
- Train pilot uses state as statistical unit. Held-out split is read only to reproduce published Phase B results and is never used to select thresholds or models.
- Each stage writes to a new `/tmp/eatnav-candidate-coverage/feasible-transfer/` root with input SHA verification and atomic markers.

## Review focus

- Candidate ID/seed/Path/trace mismatch must fail Stage 0, rather than silently align by array order.
- Map-frame Path and base-frame action increments must be converted explicitly before Evaluator inference.
- Tie pairs at |ΔV| ≤ 0.01 must not count as ranking errors or successes.
- Multi-candidate observations from one state must not become 91 independent confidence intervals.
- A Phase B held-out metric reproduction failure must stop all Evaluator transfer interpretation.

### Task 1: Frozen config, tests, and Stage 0 provenance

Files: `feasible-transfer/analysis-config.yaml`, `src/visualnav_transformer/eat_nav/feasible_transfer.py`, `scripts/eat_nav_feasible_transfer.py`, `tests/eat_nav/test_feasible_transfer.py`.

- [ ] Test candidate identity and all required SHA links; watch mismatch test fail.
- [ ] Implement immutable source audit, K32 seed correspondence, worktree snapshot, and atomic status.
- [ ] Run Stage 0; halt on missing or inconsistent artifacts.
- [ ] Commit code/config/small provenance.

### Task 2: Feasibility, modes, regret, teacher support, and K32

- [ ] Test fixed feasibility conditions, three-space cluster labeling, feasible-mode Recall@K and oracle regret; watch failures.
- [ ] Implement per-candidate/state/generator results and bank-winner nearest-NoMaD comparisons.
- [ ] Match teacher windows with pre-existing DEM/goal thresholds and complete-only train; record episode independence and trajectory/direction support.
- [ ] Compute offline K32 geometry/command coverage without assigning feasibility to unrolled candidates.
- [ ] Run Stage 1 and commit small derived summaries.

### Task 3: Phase B reproduction and frozen Evaluator transfer

- [ ] Test increment/cumulative conversion, coordinate frame, head order, score direction, ties, and state aggregation; watch failures.
- [ ] Recompute published Phase B GT held-out metrics from original checkpoint/data. Stop if tolerance fails.
- [ ] Audit all listed input/label transformations and rescore GT(P), GT(C), GT(R) for the frozen 91 runs.
- [ ] Evaluate goal-endpoint, generator/mode strata, fixed score baselines and state-wise calibration/uncertainty.
- [ ] Commit only scripts/tests/small JSON; checkpoints remain outside Git.

### Task 4: Cause classification, figures, and report

- [ ] Test gate classification with supporting/counter evidence and no premature formal-generalization pass.
- [ ] Render ten requested figures and all specified JSON files with traceable IDs.
- [ ] Verify source hashes, atomic markers, runtime/memory, tests, and worktree status.
- [ ] Commit small figures/report. Stop before any additional rollout/training and present conditional next-experiment design for approval.
