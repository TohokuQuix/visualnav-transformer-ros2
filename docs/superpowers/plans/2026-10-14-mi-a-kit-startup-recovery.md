# MI-A Kit Startup Recovery Implementation Plan

> **For agentic workers:** Execute this plan inline with test-first changes and preserve every sealed attempt.

**Goal:** Identify whether Kit startup is stable, verify process 43 under frozen scientific inputs, and resume the v3.1 formal schedule only when its runtime gate passes.

**Architecture:** Keep the frozen simulator, host, manifest, Gate P and analyzer unchanged. Add a small coordinator for process-43 repeatability and a fresh-root continuation which references verified prior seals read-only. Stop on unexplained startup or protocol failure.

**Tech Stack:** Python, pytest, Isaac Sim 5.1, Docker, tmux, SHA-256.

**Spec:** Current user request and `docs/eat-nav/motion-predictor-vnext/mi_a_v31_protocol.md`.

## Global Constraints

- Preserve original 43 valid processes and both failed process-43 attempts.
- Do not change Path, seed, terrain, posture, Actor, follower, DEM, physics, Gate P, 0.03 m threshold or bootstrap.
- Use a new persistent root and record source/manifest/input hashes for each new attempt.
- Stop if startup crashes recur or scientific conditions would change.
- Never start MI-B, Motion Predictor training, MI-C or Phase C collection.

## Review Focus

- A startup-only crash must not become a valid candidate result: test `verify_attempt` rejects a failed seal.
- A repeated test of process 43 must not silently enter the 192-row analysis: keep one designated valid attempt.
- The continuation must not modify or copy old attempts: use read-only links and verify every linked seal.
- A crash during continuation must stop before the next row: test status and stop behavior.
- A 192-row analysis must refuse incomplete schedules: rely on frozen analyzer audit of every assignment row.

### Task 1: Runtime diagnosis and process-43 stability gate

**Files:** Reuse the frozen `scripts/eat_nav_mi_fi_v1_host.py` CLI and its sealed output; new persistent diagnostic root.

- [ ] Record X11, DISPLAY, Xauthority, Kit and GPU evidence; run candidate-free minimal Kit starts sequentially.
- [x] Verify exact row 43 and its sealed pre-Path reference against the manifest.
- [x] Reuse the frozen host CLI to execute two fresh process-43 attempts with the original DISPLAY environment; each attempt records source SHA.
- [x] Compare Gate P, trace/label identity and numeric output. A new startup crash would have stopped the task.

Ruling: Reusing the frozen CLI avoids a new process-43 launcher and keeps the tested scientific execution path unchanged. The cost if this is wrong is a missing coordinator-level lifecycle signal, mitigated by the host's native-exit, resource and seal checks.

### Task 2: Fresh-root formal continuation

**Files:** `scripts/eat_nav_mi_a_v31_formal_resume_v2.py`, `tests/test_mi_a_v31_formal_resume_v2.py`, new persistent formal root.

- [ ] Write failing tests for source/manifest/seal checks, first 43 read-only links, one designated valid process-43 link and continuation at row 44.
- [ ] Implement and test the coordinator. Record the two excluded failures and diagnostic duplicate separately.
- [ ] Commit source and tests; launch remaining rows in frozen assignment order using detached tmux.
- [ ] Verify each attempt before the next. Stop on Gate P, protocol, resource or unexplained runtime failure.

### Task 3: Frozen analysis and report

**Files:** New report under `docs/eat-nav/motion-predictor-vnext/`; new formal root analysis JSON and SHA ledger.

- [ ] After 192/192 valid processes, run frozen `audit(root, 'formal')` and `save` once.
- [ ] Report seed-clustered bootstrap, 0.03 m Primary XY gate, terrain/posture/candidate effect and uncertainty, and seed/terrain/level confounding.
- [ ] Verify all seals, resource and cleanup evidence, source SHA and Git status; commit report and stop before MI-B.
