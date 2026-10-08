# P1 Terminal-Aware GPU Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct normal post-completion PID disappearance without weakening pre-completion ownership or resource checks.

**Architecture:** Keep the startup handshake unchanged. A lifecycle state machine validates an identity-bound completion envelope backed by the frozen simulator's pre-close marker, then classifies the exit-crossing and terminal NVML brackets. Real runs use one fresh preflight and one fresh candidate-free shard-B smoke before any router-dependent stage.

**Tech Stack:** Python, pytest, `/proc`, NVML, Docker, Isaac Sim.

**Spec:** `/home/kojima/.codex/attachments/a217a195-1190-4966-a2f6-73289e8a5f6c/Pasted text.txt`

## Global Constraints

- Base commit `305bdc27eec19296a3316fed33b70fbc3d5068ee` and a fresh output root.
- Preserve prior H4/H5 and all frozen experiment inputs and resource-contract-v2 numeric limits.
- No candidate Path, bridge, follower or Outcome in T3/T4.
- One T3 and one T4 authorized real run; no postauthorization retry.
- Never start, stop or write to the operator's Zenoh router.

## Review Focus

- Marker identity and timestamp must be checked independently of PID absence.
- PID reuse after marker remains a failure.
- A verified terminal absence must not erase unrelated unexpected compute or resource failures.
- Completion cannot be inferred solely from wrapper/client exit 0.
- Native exit remains unknown when App close does not return.

---

### Task 1: T0 provenance and freeze

**Files:** Create `scripts/eat_nav_p1_terminal_prepare.py`, `tests/test_p1_terminal_prepare.py`.

- [ ] Write failing tests for prior artifact SHA verification and unchanged 28-row logical manifest.
- [ ] Verify RED, implement, verify GREEN.
- [ ] Freeze source/test SHA, output overlay and previous evidence in a new root before real execution.

### Task 2: Lifecycle state machine and 18 cases

**Files:** Create `scripts/eat_nav_p1_terminal_lifecycle.py`, `tests/test_p1_terminal_lifecycle.py`.

- [ ] Write all 18 specified failing cases using fake brackets, markers and process tables.
- [ ] Verify RED, implement explicit PREAUTH/AUTHORIZED/RUNNING/FINALIZING/COMPLETE/FAILED transitions, verify GREEN.
- [ ] Preserve original GPU classifier for all nonterminal brackets and unrelated processes.

### Task 3: Host integration

**Files:** Modify `scripts/eat_nav_p1_owner_host.py`, `scripts/eat_nav_p1_gpu_attribution.py`; add host integration tests.

- [ ] Write failing tests for marker ordering and final bracket integration.
- [ ] Verify RED, implement marker read/attestation and terminal bracket sampling, verify GREEN.
- [ ] Verify complete process tables, exact cleanup and SHA for every transition.

### Task 4: T3/T4 and conditional stages

- [ ] Run relevant suites and freeze their SHA before real execution.
- [ ] Run one zero-physics preflight. Stop if invalid.
- [ ] Run one candidate-free shard-B smoke. Stop without retry if invalid.
- [ ] If passing, seal evidence, take fresh baseline and audit only a current operator-owned router.
- [ ] Run X2/28 rows only if all preceding gates and router pass; otherwise report the mandated stop decision.
