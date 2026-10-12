# P1 G3 Period-2 Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute only the frozen first-differential / batch 0 / period 2 and analyze the three-period pilot while preserving periods 0/1.

**Architecture:** A new continuation root holds a fresh period-2 batch and read-only links to the two sealed prior batches. A separate continuation launcher verifies old seal/source/manifest/snapshot SHA before the existing host runs. The corrected host keeps the same physics, Path, follower, and post-Path gates; its source diff from the valid periods is limited to assignment validation.

**Tech Stack:** Python 3.12, pytest, Isaac Lab, Docker, ROS 2, SHA-256, host seals.

**Spec:** User-approved P1 G3 period-2 continuation, 2026-10-12.

## Global Constraints

- Execute at most one new process / four trials; do not repeat periods 0/1.
- Preserve all earlier roots and seals; no candidate/Outcome/threshold changes.
- Stop before Path on any old-reference, strict state, material, or resource mismatch.
- Do not run the remaining 15 processes, Evaluator training, or full collection.

## Review Focus

- Reject a changed old seal, source, manifest, or snapshot SHA before launch.
- Reject a wrong symlink target or changed frozen period-2 Path role.
- Keep external launcher logs outside the sealed batch.
- Distinguish old and new host SHA in aggregate provenance.
- Do not infer generalization from one StateGroup and seed bundle.

---

### Task 1: Freeze cross-attempt provenance and staging

**Files:** New `scripts/eat_nav_p1_g3_crossover_v2_continuation.py`, new tests, new continuation manifest.

- [ ] Write failing synthetic tests for old-reference SHA mismatch and fresh-root-only stage.
- [ ] Verify prior period 0/1 seals, source, manifest, snapshot, assignment and Path roles.
- [ ] Stage only period 2 and link prior sealed roots read-only; write SHA ledger and continuation manifest.
- [ ] Run tests and verify root-level input SHA.

### Task 2: Prove validator scope and cross-attempt gate

**Files:** New tests; use existing corrected `eat_nav_p1_g3_crossover_v2_host.py` unchanged.

- [ ] Compare old and corrected host source and assert diff is assignment validation only.
- [ ] Test period 0/1/2 role acceptance and wrong-assignment rejection.
- [ ] Test old sealed snapshot equality and altered strict field rejection through the existing host gate.
- [ ] Verify the host can resolve and seal-check old roots through continuation aliases.

### Task 3: Execute one period and analyze

**Files:** New continuation outer launcher and aggregate report.

- [ ] Read-only verify external router, GPU/container, old references, and new staged source.
- [ ] Run period 2 once in detached tmux; preserve external logs.
- [ ] Verify material, strict pre-Path, exact-stamp, same-episode, resources, cleanup, SHA, and seal.
- [ ] If protocol-valid, run 12-trial paired analysis with per-period source provenance.
- [ ] Commit code and small reports, then stop.
