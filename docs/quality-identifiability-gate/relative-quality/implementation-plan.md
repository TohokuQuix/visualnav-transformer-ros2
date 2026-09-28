# Relative Trajectory Quality Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Diagnose whether 2-second relative Critic/reward labels explain held-out EAT trajectories better than the current absolute Critic P10, without modifying the deployed model or collecting new data.

**Architecture:** A pure timing/label module aligns 20 Hz pre-action values with post-action reward and pose. A detached host runner invokes Docker stages for audit, value extraction, train-only normalization, matching, deterministic probes, evaluation, figures, and hashes. All train/validation/held-out identities remain the frozen terrain-seed split.

**Tech Stack:** Python 3.11 in `isaac-lab-onix`, NumPy/SciPy, PyTorch, Matplotlib, stdlib host runner, tmux.

**Spec:** User attachment `35743525-fe62-4035-b8f3-9d1de2f41c69/Pasted text.txt`.

## Global Constraints

- Do not modify existing Quality, ECDF, split, model, checkpoint, Gate A/B/casebook artifacts, or start simulation collection.
- Checkpoint `model_10000.pt`; gamma 0.99 from its saved `params/agent.yaml`; physics dt 0.005 s × decimation 10 = control step 0.05 s.
- A raw row contains V of its pre-action state; reward and pose describe that row's transition endpoint. Anchor pose on row i therefore uses V on row i+1; rewards use rows i+1 through endpoint j, and endpoint V uses row j+1.
- Twenty-nine complete windows end at a terminal row (train 22, validation 7, held-out 0), so their endpoint V is absent. Preserve all 61,977 identities and mark these labels unavailable; primary fair Probe comparison uses the other 61,948 windows. Terminal return handling is auxiliary and uses the PPO timeout correction.
- Future reward/Critic labels are post-hoc diagnostics, not available inference inputs.

## Review Focus

- One-step value/pose offset and 40 control-step discount exponent: synthetic transition test.
- Reset and terminal endpoint: no cross-episode value lookup, explicit null labels and auxiliary return.
- Train-only normalization and pair thresholds: leakage test.
- Probe parity: scalar graph and identical seed/order/optimizer; multi token adds only 192 input weights.
- Runner interruption: status and SHA-checked resume test.

### Task 1: Timing and value primitives

**Files:** Create `src/visualnav_transformer/eat_nav/relative_quality.py`; test `tests/eat_nav/test_relative_quality.py`.

- [ ] Write synthetic tests for V alignment, 40-step gamma, drawdown/recovery, terminal/timeout correction, reset boundary, original q reproduction.
- [ ] Run tests red in Docker.
- [ ] Implement timestamp-aligned window extraction and label formulas; run tests green.
- [ ] Save `rl-timing-semantics.json/.md` with source paths/hashes and terminal endpoint audit.

### Task 2: Frozen config and value extraction

**Files:** Create `relative-quality-config.yaml`, `scripts/eat_nav_relative_quality_worker.py`.

- [ ] Freeze seeds, thresholds, split/manifest/checkpoint SHA, support and underfit rules before extraction.
- [ ] Smoke 100 windows; verify all timestamps, finite values, 0.05 s cadence, resource use.
- [ ] Extract 61,977 complete identities, 61,948 aligned primary labels, original q and train-only scalers/ECDF; save gzip and NPZ intermediates.
- [ ] Validate split integrity and original q against EATDataset.

### Task 3: Correlations and matched neighborhoods

**Files:** Extend worker; test matching and bootstrap functions.

- [ ] Fix geometric and label thresholds on train-only candidates before held-out evaluation.
- [ ] Select cross-episode held-out cross/same relative-label pairs without repeated windows and compute trajectory differences with episode-cluster CI.
- [ ] Report split, terrain, q bin, terminal reason, confounding, support, and reward-decomposition limitation.

### Task 4: Deterministic Probe comparison

**Files:** Create `src/visualnav_transformer/eat_nav/relative_quality_probe.py`; extend worker; test architecture/input parity.

- [ ] Use the Gate A DEM/goal/Transformer/head graph for Q-0, abs, advantage, drawdown, and multi; freeze 11/29/47 seeds, batch order, optimizer and aggregate validation selection.
- [ ] Save per-epoch and best checkpoints in `/tmp`, with SHA and parameter counts.
- [ ] Evaluate held-out once after validation selection; compute window MSE, physical RMS, R², paired episode bootstrap, terrain support, and underfit.
- [ ] Run terrain-stratified input permutation 10 times per label, 100 only under the preregistered boundary/otherwise-pass rule.

### Task 5: Detached execution, figures, decision

**Files:** Create `scripts/eat_nav_relative_quality.py`; `docs/quality-identifiability-gate/relative-quality/*`.

- [ ] Test host stage markers and resume SHA checks.
- [ ] Run short smoke, then detached `tmux` session `eatnav-relative-quality` if resource use is normal.
- [ ] Generate requested plots, explicit pass/fail/inconclusive decision and report; never start simulator rollout.
- [ ] Verify artifact hashes, tests, status, and clean git diff; commit only compact outputs and plots.
