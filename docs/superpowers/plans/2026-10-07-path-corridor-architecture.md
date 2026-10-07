# Path-corridor Architecture Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare A1–A4 on frozen 28-StateGroup offline data without changing Outcomes, splits or candidate definitions.

**Architecture:** One read-only source auditor creates a provenance manifest. A geometry module aligns each saved source DEM to the anchor pose and extracts full-grid and Path-corridor tensors. A training/evaluation runner uses the frozen 40 folds, six-head loss and four model definitions. A report module produces StateGroup-level paired statistics, decisions and figures.

**Tech Stack:** Python, NumPy, PyTorch, YAML, Matplotlib, pytest.

**Spec:** `docs/quality-identifiability-gate/path-corridor-architecture/spec.md`

## Global Constraints

- Source root and all 252 traces are read-only. No simulator, NoMaD, follower or robot invocation.
- Freeze `corridor-config.yaml` and its SHA before reading Outcomes for support classes.
- Use all 40 existing folds and seeds 123/456/789; never split a StateGroup or select epochs using test data.
- Stop on source SHA, coordinate, mask, leakage, or finite failure.

## Review Focus

- A past DEM frame expressed in its own robot pose must map to the anchor pose before Path sampling; test a translated/rotated synthetic grid.
- Bilinear interpolation must reject an invalid contributing cell and out-of-bounds sample; test both.
- A 3-repeat direct group must never cross a fold boundary; test all saved folds.
- Support classes must be identical after labels are perturbed; test Outcome independence.
- A constant prediction must not be called a Pareto improvement merely for recall 1; test precision/set size/all-tied reporting.

### Task 1: Source and coordinate audit

**Files:** `scripts/eat_nav_corridor_audit.py`, `tests/test_eat_nav_corridor_audit.py`.

**Interfaces:** Produce `source-provenance.json` with 252 record IDs, 28 group IDs, 21 source anchors, source SHA ledger, all 40 folds and pose/DEM gate results.

- [x] Write failing tests for sealed SHA mismatch, group/split leakage and old/new anchor source mapping.
- [x] Run targeted tests and observe expected failures.
- [x] Implement the read-only auditor and atomic output.
- [x] Run targeted tests and real C0 audit; stop if any gate fails.

### Task 2: Geometry and support

**Files:** `src/visualnav_transformer/eat_nav/path_corridor.py`, `scripts/eat_nav_corridor_prepare.py`, `tests/eat_nav/test_path_corridor.py`.

**Interfaces:** Consume frozen records and source-frame poses; produce aligned full DEM `[252,4,2,20,20]`, corridor `[252,4,2,16,7]`, profile, per-candidate metrics and 28 Outcome-blind classes.

- [x] Write failing coordinate, bilinear-mask, 16×7 shape and class Outcome-independence tests.
- [x] Run targeted tests and observe expected failures.
- [x] Implement alignment, interpolation, profile, support metrics and classification.
- [x] Run tests, freeze config SHA and execute C1 before opening labels.

### Task 3: Four models and fixed loss

**Files:** `src/visualnav_transformer/eat_nav/path_corridor_models.py`, `tests/eat_nav/test_path_corridor_models.py`.

**Interfaces:** A1–A4 accept the same common vector and specified full/corridor tensors and return `[batch,6]`; return parameter counts.

- [x] Write failing shape, no-BatchNorm, shared-head and finite-gradient tests.
- [x] Run targeted tests and observe expected failures.
- [x] Implement model branches and common fusion; save architecture/parameter freeze before training.
- [x] Run targeted tests.

### Task 4: Frozen training and evaluation

**Files:** `scripts/eat_nav_corridor_train.py`, `tests/test_eat_nav_corridor_train.py`.

**Interfaces:** Consume C0/C1 outputs plus frozen labels/splits; produce 480 per-fold/seed/architecture prediction and metric records with train-only scalers and atomic progress.

- [x] Write failing train-only scaler, fold partition, loss parity and model-shape tests.
- [x] Run targeted tests and observe expected failures.
- [x] Implement fixed 40×4×3 runner, a 1-fold timing probe and resumed detached execution.
- [x] Verify finite loss/predictions, 480 results and no test-fold checkpoint selection.

### Task 5: Statistics, figures and decision

**Files:** `scripts/eat_nav_corridor_report.py`, `tests/test_eat_nav_corridor_report.py`.

**Interfaces:** Produce paired StateGroup bootstrap intervals, all requested metric/stratum tables, final decision JSON, report, seven figure families and artifact SHA ledger.

- [x] Write failing paired-group, Pareto-tie and insufficient-support tests.
- [x] Run targeted tests and observe expected failures.
- [x] Implement analysis and figures without changing frozen inputs.
- [x] Verify outputs, tests and SHA ledger; stop before any downstream phase.
