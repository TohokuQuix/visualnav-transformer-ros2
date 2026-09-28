# Quality Identifiability Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose whether held-out trajectory residuals contain useful Quality information and whether the unchanged EAT policy can express it.

**Architecture:** One diagnostic-only package reads the existing extended pilot and split, freezes gate settings, audits data, matches trajectories, trains deterministic nested probes, performs a scratch same-condition diffusion overfit, traces layer responses, and writes immutable provenance and reports. Each stage is separately resumable and stops at the user's named audit conditions.

**Tech Stack:** Python, NumPy, SciPy, PyTorch, pytest, existing EAT dataset/model, RTX 5070 Ti Docker runtime.

**Spec:** User attachment `/home/kojima/.codex/attachments/716caba4-696f-49a6-956a-36fa2cbc688b/Pasted text.txt` and subsequent four-point approval.

## Global Constraints

- Do not start full collection or alter the Quality definition, ECDF, existing architecture, scheduler, trajectory representation or loss.
- Preserve split seed 123 and train-only ECDF from raw Critic P10; never train/select/tune on held-out seeds.
- Preserve existing artifacts and checkpoint SHA-256. Write new outputs only under `docs/quality-identifiability-gate/` and `/tmp/eatnav-quality-identifiability-gate/`.
- Probe 0 retains Quality MLP/token and supplies q=0.5 for every sample. Probe Q changes only that scalar input.
- If both probes underfit a train-only mean trajectory or validation is still improving at cap, Gate A is inconclusive.
- Each Gate B run starts from scratch, has identical paired DEM/goal, unique windows, capped episode reuse and train-distribution thresholds saved before training.
- Start with 10 terrain-stratified q shuffles; if all other Gate A conditions pass or results are borderline, run 100 before final decision. With 10, pass requires observed effect above every shuffle.

## Review Focus

- Split or episode leakage: abort audit before model training.
- Reused window or episode concentration in Gate B: reject selection.
- Validation underfit or improving at cap: inconclusive, never infer failure.
- Permutation count and decision logic: tests cover ten and hundred paths.
- Paired diffusion noise and no checkpoint warm start: test exact invariants.

### Task 1: Frozen config and invariant tests

**Files:** Create `src/visualnav_transformer/eat_nav/quality_gate_config.py`, `tests/eat_nav/test_quality_gate_config.py`, `docs/quality-identifiability-gate/probe-config.yaml`, `docs/quality-identifiability-gate/overfit-config.yaml`.

**Interfaces:** `QualityGateConfig` loads thresholds, seeds, fixed q and gate rules; `decide_gate_a`, `decide_gate_b` expose deterministic status.

- [ ] Write tests for fixed q 0.5, underfit inconclusive, cap improvement inconclusive, permutation escalation, ten-shuffle strictness, Gate B criteria.
- [ ] Run focused tests and watch them fail.
- [ ] Implement config and decisions; rerun tests and EAT suite; commit.

### Task 2: Audit and train-only matching

**Files:** Create `src/visualnav_transformer/eat_nav/quality_gate_data.py`, `tests/eat_nav/test_quality_gate_data.py`, `scripts/eat_nav_quality_gate.py`.

**Interfaces:** audit three EATDataset splits; match cross-q and same-q controls; choose Gate B pairs with fixed train-derived thresholds and episode cap; serialize all identities.

- [ ] Write tests for split leakage, episode clustering, exact paired-condition identity, duplicate window and episode cap rejection.
- [ ] Watch failures, implement, rerun suite; commit.
- [ ] Run real audit; stop and report if the specified integrity failures occur. Freeze pair thresholds and identities before training.

### Task 3: Gate A probes and permutation control

**Files:** Create `src/visualnav_transformer/eat_nav/quality_gate_probe.py`, `tests/eat_nav/test_quality_gate_probe.py`.

**Interfaces:** paired Probe 0/Q and shuffled Q runs; validation selection; one-time held-out scoring; episode-cluster bootstrap; 10-to-100 shuffle escalation.

- [ ] Test Probe 0 fixed q path, equal architecture, matched seed/data order, no held-out selection, shuffle strata, bootstrap, pass/inconclusive logic; watch fail.
- [ ] Implement and pass suite; commit.
- [ ] Smoke test GPU time/VRAM. If acceptable, execute full Gate A and save curves/results.

### Task 4: Gate B scratch overfit

**Files:** Create `src/visualnav_transformer/eat_nav/quality_gate_overfit.py`, `tests/eat_nav/test_quality_gate_overfit.py`.

**Interfaces:** scratch EATNav runs on fixed-condition pairs for three seeds; common diffusion noise; q sweep; target selection, directions and effect ratios.

- [ ] Test no checkpoint load, exact condition identity, target routing, shared noise, and decision; watch fail.
- [ ] Implement and pass suite; commit.
- [ ] Smoke test GPU time/VRAM. If acceptable, run overfit and save curves/results.

### Task 5: Layer sensitivity and final report

**Files:** Create `src/visualnav_transformer/eat_nav/quality_gate_layers.py`, `tests/eat_nav/test_quality_gate_layers.py`, `docs/quality-identifiability-gate/quality-identifiability-gate.md` and JSON/figures.

**Interfaces:** hook Quality MLP, tokens, Transformer, diffusion blocks; compare activations under paired q/noise; render requested plots; combine gate statuses.

- [ ] Test hook coverage and equal noisy action/timestep; watch fail.
- [ ] Implement and pass suite; commit.
- [ ] Run layer analysis, write reproducible commands, commits, versions, SHA-256 and final ruling; verify artifacts and full tests.
