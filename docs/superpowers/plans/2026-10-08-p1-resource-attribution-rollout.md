# P1 Resource Attribution and Conditional Rollout Implementation Plan

**Execution status (2026-10-08):** Stopped at Task 2 with `p1_resource_attribution_fail`. The single allowed remediation cycle was consumed; Tasks 3–5 were not started. See `docs/eat-nav/p1-resource-attribution-stop-20261008/report.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove live per-PID GPU attribution in one candidate-free radial shard, then run the frozen P1 sequence only if every preflight gate passes.

**Architecture:** Extend the accepted bracketed NVML accounting with exact host PID/start-tick process-tree classification. Reuse the sealed radial fixture as the functional input gate and run one new source-selected shard only for resource measurement. A fresh attempt root owns later X1/X2 and, conditionally, each sequential candidate process.

**Tech Stack:** Python, pytest, NVML, IsaacLab Docker, ROS2 Jazzy/Zenoh, Git worktree.

**Spec:** `/home/kojima/.codex/attachments/72b654a1-9d7d-4670-9378-e45aa5c2a787/Pasted text.txt`

## Global Constraints

- Base commit `3fa65484873492fef4457922d644033da99f29fc`; preserve r3–r5 and prior radial fixture roots.
- Frozen P1 resource v2: baseline 1,024 MiB; process 7,680 MiB; unresolved GPU memory 0; bracket gap at most 256 MiB; 600 s process timeout; 90 s heartbeat; 1 GB storage cap.
- Four anchors, five Paths, 28 logical rows, seed 123, no retry, common goal `[0.75, 0.0]` stay unchanged.
- No candidate Path or control before attribution smoke and all R3–R5 gates pass.
- External router is read-only and operator-owned.

## Review Focus

- PID reuse must not map a GPU process to an old simulator identity; test exact start tick.
- Non-owned graphics memory must remain within device total but not count as unresolved; test classification.
- Empty/missing process tables during nontrivial GPU use must fail; test both.
- Bracket or device cap overflow must stop before a candidate; test exact thresholds.
- A candidate termination is a measured outcome while missing primary trace is an infrastructure failure; test the row auditor.

### Task 1: Attribution collector and tests

**Files:** `scripts/eat_nav_p1_gpu_attribution.py`, `tests/test_p1_gpu_attribution.py`.

- [ ] Write failing tests for owned PID/descendant, PID reuse, graphics process, unresolved process, missing/empty table, bracket gap, UUID, disappearing simulator, cap, and accounting-disabled success.
- [ ] Run tests and capture RED result.
- [ ] Implement pure classification and live NVML bracket capture using the existing `eat_nav_g0e_gpu_probe` collector.
- [ ] Run tests and capture GREEN result.

### Task 2: One candidate-free smoke

**Files:** `scripts/eat_nav_p1_attribution_smoke.py`, `tests/test_p1_attribution_smoke.py`, small report JSON.

- [ ] Freeze source-based shard choice, prior parity SHA, source SHA, output root, and exact 200-step invocation.
- [ ] Add lifecycle tests proving no Path/follower/bridge and requiring native exit/marker/SHA/cleanup.
- [ ] Run one representative original shard with live ownership-aware samples; stop on attribution or resource failure.
- [ ] Seal raw resource series and small smoke result.

### Task 3: Conditional fresh preflight and X1/X2

**Files:** `scripts/eat_nav_p1_attempt.py`, `tests/test_p1_attempt_preflight.py`, small report JSON.

- [ ] Test output-only 28-row overlay and fresh SHA/baseline gating.
- [ ] Capture a fresh pre-X1 NVML baseline, resource forecast, and router X1 snapshot.
- [ ] Run existing candidate-free X2 with actual bridge/eight production followers and fresh post-X2 baseline.
- [ ] Stop before `pilot-started.marker.json` on any failed gate.

### Task 4: Sequential 28-process radial P1

**Files:** dedicated radial candidate simulator/runner, row auditor, tests, raw and small artifacts.

- [ ] Test radial settling contract, Path boundary, frozen row identity, primary 40-step trace, failure semantics, and zero retry.
- [ ] Implement the candidate path on top of the proven same-process radial settling state; retain E1 episode-prefix runner unchanged.
- [ ] Run rows sequentially with per-row GPU attribution, source SHA, markers, cleanup and SHA seal; stop at first invalid row.

### Task 5: Offline signal analysis and final decision

**Files:** `scripts/eat_nav_p1_analyze.py`, tests, report/figures/ledger.

- [ ] If 28/28 protocol-valid, compute duplicate reproducibility and P/C/A/R/Y matrices with anchor/terrain statistical units.
- [ ] Recompute common-goal 0–2 s Outcome from trajectories, preserving locomotion reward separately.
- [ ] Save the prescribed final decision and final router snapshot; commit only code and small artifacts.
