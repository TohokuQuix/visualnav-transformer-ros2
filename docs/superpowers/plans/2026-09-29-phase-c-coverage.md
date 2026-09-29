# Phase C Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Diagnose conditional teacher modes and candidate collapse; attempt frozen sampling and train PCA proposals; stop at the authorized coverage decision.

**Architecture:** An immutable Stage 0 manifest identifies inputs and states. Docker workers write atomic stage artifacts; host ROS bridge supplies actual follower commands. Conditional stages are skipped explicitly when their gate fails.

**Tech Stack:** Python, NumPy, PyTorch, diffusers, ROS2, Docker, tmux.

**Spec:** User attachment `dc728b95-c4e5-4bac-86e6-1d97414a9391/Pasted text.txt`.

## Global Constraints

- Frozen checkpoint `191d6a03…b4e0`, complete train 49,598 and val 6,171; held-out forbidden.
- No retraining, scheduler math changes, formal Phase C, full collection, Phase D/E or hardware.
- Follower command cluster thresholds: linear 0.03 m/s, angular 0.05 rad/s.
- Sampling gate 75% states two command clusters, 50% three Path clusters, ≤5% train P95 OOD violations.

## Review Focus

- Prevent validation/held-out from entering PCA or OOD fitting; test source IDs.
- Detect partial stage artifacts on resume; marker SHA and output SHA test.
- Keep K8 the exact prefix of K32; test generated noise path equivalence.
- Ensure follower startup and state reset; test bridge with duplicate Path.
- Prevent invalid structured candidates from silently being replaced; test counted failure.

---

### Task 1: Frozen inputs and tests

**Files:** `coverage-config.yaml`, `scripts/eat_nav_phase_c_coverage.py`, `tests/test_phase_c_coverage.py`.

- [ ] Write failing tests for forbidden split, state identity, K8 subset, marker validity.
- [ ] Run tests and confirm expected failure.
- [ ] Implement freeze command and provenance hashes.
- [ ] Run tests and freeze before inspecting new outcomes.

### Task 2: Conditional teacher modes

**Files:** `scripts/eat_nav_phase_c_coverage.py`, `tests/test_phase_c_coverage.py`.

- [ ] Write failing tests for geometric neighborhood, cross-episode, cluster metrics.
- [ ] Implement train-only audit and run in detached session.
- [ ] Verify atomic output and summary.

### Task 3: Diffusion trace and sampling grid

**Files:** `scripts/eat_nav_phase_c_coverage.py`, `tests/test_phase_c_coverage.py`.

- [ ] Write failing test that instrumented DDPM exactly reproduces model.sample.
- [ ] Implement per-step trace and K8/K32/grid evaluation.
- [ ] Use actual follower and fixed train OOD; apply frozen coverage gate.

### Task 4: Conditional structured proposal

**Files:** `scripts/eat_nav_phase_c_coverage.py`, `tests/test_phase_c_coverage.py`.

- [ ] Write failing tests for train-only PCA, zero residual endpoint and invalid candidate handling.
- [ ] Fit train basis, generate frozen K8, evaluate same gate if sampling fails.

### Task 5: Evaluator, simulator, decision

**Files:** `scripts/eat_nav_phase_c_coverage.py`, `docs/quality-identifiability-gate/phase-c-coverage/phase-c-coverage-report.md`.

- [ ] Run Evaluator only for coverage-passing proposals.
- [ ] Run 8 train-state serial simulator smoke only if offline coverage passes.
- [ ] Plot and report all measurements; mark skipped artifacts `not_run`.
- [ ] Verify artifact SHA and stop before formal rollout.
