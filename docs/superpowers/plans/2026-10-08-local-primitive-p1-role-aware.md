# Role Aware Local Primitive P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze a versioned terrain-role gate on the eight measured anchors and, only if it passes, execute and analyze one 35-process local primitive P1.

**Architecture:** Keep the prior 601debf offline review immutable. A pure role-audit module consumes its SHA-sealed anchors, Paths, DEM corridors and existing classifier thresholds. A separate P1 runner can consume only a sealed five-anchor/35-row manifest after the gate, router and X2 pass; all process artifacts are fresh and individually sealed.

**Tech Stack:** Python, NumPy, pytest, SHA-256, ROS2/Zenoh and the frozen E1 simulator/follower only if offline gates pass.

**Spec:** `/home/kojima/.codex/attachments/c5d325c5-3e71-4b8a-a0ef-cb55431094d6/Pasted text.txt`

## Global Constraints

- Base commit `601debf88d43530dfb12ef7bcf1047ce47a80994`; new branch/worktree only.
- The prior `local_motion_primitive_terrain_support_fail` remains true under its old gate.
- Five exact prior Path SHAs, eight physically supplied anchors, DEM classifier constants and 0–0.60 m primary interval remain unchanged.
- Use no candidate Outcome in taxonomy, gate or selection. No threshold tuning after observing anchor results.
- Rollout is conditional on every N0–N8 gate; a failure stops before Path publication.
- If rollout starts, no automatic candidate retry and no physics/follower/checkpoint changes.

## Review Focus

- The common transition/graded roles must be based on known 0–0.60 m per-Path DEM support, never merely the anchor's frozen class.
- One anchor cannot fill two of the five selected slots, even when it has secondary role flags.
- Goal metadata must remain separate from Path endpoint in the 35-row physical manifest and in two-second labels.
- X2 must verify the operator router by PID/start tick/ZID/log SHA and must not take ownership of it.
- A measured candidate failure must stop immediately; pre-control remediation may not touch a published Path or control step.

### Task 1: Freeze provenance and taxonomy

**Files:** `docs/eat-nav/local-primitive-p1-role-aware-20261008/role-aware-terrain-taxonomy.json`, `scripts/eat_nav_role_aware_gate.py`, `tests/test_role_aware_gate.py`.

- [x] Write RED tests for role classification and source-identity selection; verify taxonomy SHA and prior provenance integrity in the runner.
- [x] Implement source/anchor/Path SHA checks and refuse a reused output root.
- [x] Freeze taxonomy and its SHA before any anchor-role evaluation.

### Task 2: Eight-anchor role audit and gate

**Files:** pure role module and tests above.

- [x] Write RED tests for distinct primary roles, known support and precedence.
- [x] Evaluate all eight anchors, retain secondary flags and margins, and test five distinct selected anchors and four source-family identities.
- [x] Save role gate and selection ledger. Gate insufficient: stop with `role_aware_terrain_support_insufficient` before X1/X2.

### Task 3: Conditional 35-process manifest and runtime preflight

**Files:** `scripts/eat_nav_local_primitive_p1.py`, tests, frozen manifest.

- [ ] Write RED tests for 5×7 ordering, three 0° direct repeats, separate global goal and exact Path SHA.
- [ ] Forecast 35-process wall/storage/GPU/anon+shmem from sealed measurements with 20% margin.
- [ ] Freeze row order/seeds/config/SHA and reject any resource envelope failure before X1.

### Task 4: Router/X2 and bounded execution, only if Task 2/3 pass

- [ ] Verify router ownership and saved stdout/stderr; run candidate-free X1/X2 with three snapshots.
- [ ] If X2 passes, run 35 fresh processes in order with atomic run markers, SHA seal and zero residual checks.
- [ ] Stop on first protocol, duplicate, resource or identity failure. No candidate retry.

### Task 5: Post-run analysis, only if 35/35 valid

- [ ] Validate 15/15 direct repeats and 20/20 non-direct results against frozen limits.
- [ ] Compute P→C→A→R→Y and 0–39 step reward/progress labels in the common map frame; keep locomotion and navigation separate.
- [ ] Produce role-specific analysis, correlated five-goal relabels, 12 figures, SHA ledger and one N14 decision. Stop before later phases.

## Completion Boundary

Either an offline stop decision before X1, or exactly one post-P1 decision after 35 sealed processes. No scaled collection, training, P2 or robot operation.

**2026-10-08 result:** Task 2 failed because no primary graded-common or flat-control anchor exists under the pre-frozen v1 Path-level taxonomy. Tasks 3–5 remain conditional and were not started. See `docs/eat-nav/local-primitive-p1-role-aware-20261008/review-report.md`.
