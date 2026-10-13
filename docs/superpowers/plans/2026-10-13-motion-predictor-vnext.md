# EAT-Nav vNext Motion Predictor Implementation Plan

> **For agentic workers:** Execute sequentially in this worktree. Review each gate before starting its successor.

**Goal:** Learn goal-independent, two-second body-frame XY motion for fixed body-frame Arc Paths, then select a Path by geometric progress to the current map Goal.

**Architecture:** Preserve the ROS-free PureFollower and RL Actor, along with the baseline common-goal Evaluator. Add a versioned Arc Bank, a paired StateGroup dataset, a two-output CNN/MLP predictor, and a separate demo mode. Runtime collection and training proceed only when the preceding evidence gate passes.

**Tech Stack:** Python, NumPy, PyTorch, Isaac Lab, C++ PureFollower, ROS2 GUI transport.

**Spec:** User-provided `EAT-Nav vNext：Goal-independent Motion Predictorへの移行、Arc Bank構築、データ収集・再学習・GUI統合` (2026-10-13).

## Global Constraints

- Base commit `11b5ef9`; do not edit the sealed G3 trials, original demo branch, baseline checkpoint, RL checkpoint, DEM generator, or PureFollower control law.
- Input `[B,4,2,20,20]`, `[B,22]`, `[B,K,8,2]`, `[B,K]`; output `[B,K,2]`; Goal is excluded from model input.
- Label is `R(-yaw_at_start) * (root_xy_after_40_steps - root_xy_at_start)`; missing/censored labels are not zeros.
- Train/validation/test split by independent StateGroup; normalization from train only; test is final evaluation only.
- Preserve old Goal-progress Evaluator as an explicit baseline; never load its checkpoint as the motion model.
- Stages A→F are gated. A failed gate stops the next stage and requires an evidence report.

## Review Focus

- A zero-length or repeated-waypoint Path must fail geometry validation.
- ±180° Paths with the same endpoint remain separate Path identities.
- A terminal before step 40 produces an unavailable label with a recorded reason.
- Changing Goal alone changes the geometric score and cannot change predicted XY.
- A reused StateGroup or terrain seed cannot leak between train, validation and test.

---

### Task 1: Phase A contracts and preserved baseline

**Files:** `docs/eat-nav/motion-predictor-vnext-contract.md`; existing evaluator, demo, follower and G3 trace code are read-only references.

- [ ] Record current source/checkpoint hashes and map the data, frame and timestamp paths.
- [ ] Freeze the two-second label, masking, split and score contracts; identify all independent StateGroups presently available.
- [ ] Verify the contract against sealed trace examples without altering original artifacts.
- [ ] Mark Phase A Pass/Fail/Inconclusive and commit the contract.

### Task 2: Phase B Arc Bank and offline follower audit

**Files:** `src/visualnav_transformer/eat_nav/arc_bank.py`; `tests/test_eat_nav_arc_bank.py`; `scripts/eat_nav_arc_bank_audit.py`; versioned bank JSON/report under `docs/eat-nav/motion-predictor-vnext/`.

- [ ] RED tests for the constant-curvature body-frame formula, 8-point arclength sampling, symmetry, endpoint/tangent bearing distinction and ±180° identity.
- [ ] Implement Stage 1 angles `0,±30,±60,±90` and Stage 2 angles `±120,±150,±180` at frozen arclength, with fail-closed geometry checks.
- [ ] Audit projection/carrot, heading cap and initial command using the actual PureFollower core; separately label offline geometry and runtime feasibility.
- [ ] Freeze valid candidates and limits before any physics; if infeasible, report before changing the bank.

### Task 3: Phase C paired StateGroup pilot

**Files:** New collector/manifest under `scripts/`, dataset adapter under `src/visualnav_transformer/eat_nav/`, relevant tests and persisted evidence root.

- [ ] Freeze independent terrain seeds, StateGroup IDs, candidate order, tolerances, resource caps and immutable manifest.
- [ ] Verify identical pre-Path physical/DEM/Actor/noise/latency state; run each Arc for exactly 40 control steps using the unchanged PureFollower and Actor.
- [ ] Seal traces, record terminal reason and source hashes, compute body-frame XY only for valid complete windows.
- [ ] Validate StateGroup split independence and candidate motion diversity; stop if pair reproduction or data validity fails.

### Task 4: Phase D pilot training and held-out comparison

**Files:** New motion model, trainer, checkpoint schema, evaluator and tests.

- [ ] Reuse the existing CNN/MLP encoders with a two-output fusion head and explicit candidate masks.
- [ ] Fit output normalization on train StateGroups only; run small overfit, validation model selection and final test.
- [ ] Report XY RMSE/MAE, direction error for nontrivial motion, ranking, and Direct/endpoint heuristic/model/oracle on identical inputs.
- [ ] Gate held-out finite reproducible predictions and candidate ranking; heuristic superiority is diagnostic, not an automatic failure.

### Task 5: Phase E full collection and retraining

- [ ] Use Phase C measured runtime/VRAM/RAM/storage and failure rate to freeze scale and independent terrain/seed coverage.
- [ ] Collect and seal versioned dataset with censored rows and lineage; audit distributions and split independence.
- [ ] Retrain by validation only, evaluate held-out test once, record checkpoint/schema SHA.

### Task 6: Phase F separate GUI demo mode and final evaluation

- [ ] Add explicit bank/model selection without changing the original demo mode.
- [ ] Score frozen body-frame arcs from predicted XY and current Goal, with map transform only at the Path/ROS boundary.
- [ ] Show all arcs, selected arc, predicted XY and geometric score; keep old Evaluator scores separately named.
- [ ] Test Goal-only change, closed-loop stepping, latency and reset; compare Direct, heuristic, baseline and motion predictor with stated distribution limits.
- [ ] Update student setup, pipeline and roadmap and commit reproducible run commands.
