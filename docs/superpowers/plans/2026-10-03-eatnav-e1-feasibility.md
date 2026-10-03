# EAT-Nav E1 Feasibility Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run only the 18 authorized fresh-process Bezier candidate continuations, then report P→C→R→Y feasibility using the frozen two-group protocol.

**Architecture:** Preserve G0-E's 20-row manifest and two completed anchor artifacts. Create a distinct 18-row continuation manifest and a guarded sequential runner that verifies each anchor, executes the existing production follower and exact-stamp bridge, enforces per-PID resource gates, and stops on the first invalid run. Analyze accepted traces by StateGroup and produce provenance, figures and a final decision offline.

**Tech Stack:** Python 3.13, pytest, Isaac Lab/Sim in `isaac-lab-onix`, ROS 2 production follower, tmux, NVML.

**Spec:** `/home/kojima/.codex/attachments/8d9db663-e4c6-4561-bc5e-140da0fde5c4/Pasted text.txt`

## Global Constraints

- Exactly two frozen StateGroups, seven unique Paths/group and 18 new processes; no outcome-guided changes or retry.
- Freeze input/source SHA, order, seed 123, 120 steps at 20 Hz, 6,656 MiB device cap, 800 MB storage cap and 3,600 s continuation cap before launch.
- Run direct repeats 0–2 and validate duplicates before any lateral candidate for that StateGroup.
- Keep exact-stamp protocol and production follower control law, RL checkpoint, dynamics, reset and observation pipeline unchanged.
- Never run learning, inference, scaled collection or real hardware.

## Review Focus

- An input Path whose map SHA differs from G0-E must stop before simulator launch.
- A fresh candidate with an anchor mismatch must stop before Path publication.
- A missing or corrupted complete marker must never be treated as a successful resume.
- A duplicate gate failure must prevent lateral candidates and the second StateGroup.
- External GPU occupancy over the device cap must stop the run without signaling unrelated processes.

---

### Task 1: Frozen continuation and provenance

**Files:** Create `scripts/eat_nav_e1_prepare.py`, `tests/eat_nav/test_e1_prepare.py`, and `/tmp/eatnav-e1-feasibility-20261003/continuation-manifest.json`.

**Interfaces:** Consume G0-E decision, 20-row manifest, map Path table and SHA ledger. Produce immutable 18-row manifest and per-run Path/input files.

- [ ] Write tests for exact 2/6/12 role arithmetic, state/Path/seed/order equality, three direct repeats sharing one SHA, and rejected SHA drift.
- [ ] Run tests and confirm RED.
- [ ] Implement immutable manifest/provenance generation and run tests GREEN.
- [ ] Record branch/HEAD, source/binary/exact-stamp SHAs and the prelaunch resource forecast.

### Task 2: Fresh candidate simulator and anchor gate

**Files:** Create `scripts/eat_nav_e1_sim.py`, `scripts/eat_nav_e1_anchor_gate.py`, and `tests/eat_nav/test_e1_anchor_gate.py`.

**Interfaces:** Consume one immutable manifest row and return anchor-check result plus P/C/R/Y raw trace with source/output SHA.

- [ ] Write tests for frozen identity, four-frame DEM gate, exact Path SHA, logical timestamps and no Path publication after failed anchor.
- [ ] Run tests and confirm RED.
- [ ] Adapt the validated goal-subgoal 120-step simulator to G0-E inputs, retaining production follower/bridge calls and observation/dynamics behavior.
- [ ] Run tests GREEN and compare formal anchor capture semantics with the G0-E source.

### Task 3: Sequential monitored runner

**Files:** Create `scripts/eat_nav_e1_run.py`, `tests/eat_nav/test_e1_run.py` and frozen `e1-continuation-config.yaml`.

**Interfaces:** Consume 18-row manifest. Produce one atomic run marker per process, resource/cleanup logs, and a status whose next row is derived from verified markers.

- [ ] Write tests for fixed order, one-run-at-a-time behavior, duplicate gate stop, 3,600 s cap and no automatic retry.
- [ ] Run tests and confirm RED.
- [ ] Implement detached tmux execution with 0.4–1.0 s per-PID monitor, cleanup of owned PIDs only and strict resume checks.
- [ ] Run tests GREEN and inspect the launch command/source SHA before execution.

### Task 4: Run and analyze E1

**Files:** Create `scripts/eat_nav_e1_analyze.py`, `tests/eat_nav/test_e1_analyze.py`, small JSON/figure/report outputs and raw logs under `/tmp/eatnav-e1-feasibility-20261003`.

- [ ] Write tests for direct-repeat aggregation as one oracle candidate, StateGroup as statistical unit, invalid infrastructure exclusion and frozen gate application.
- [ ] Run tests RED, implement analysis, then run tests GREEN.
- [ ] Execute the approved 18-run schedule in tmux, checking the duplicate gate before each lateral sequence and stopping at the first failure.
- [ ] Verify SHA ledger, residual processes, final status and figures; report supported/fail/inconclusive/measurement-invalid without expanding scope.
