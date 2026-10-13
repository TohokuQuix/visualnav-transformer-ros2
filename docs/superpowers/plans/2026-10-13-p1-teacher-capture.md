# P1 Teacher Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Capture reward components and raw mechanical loads from one ROS-free G3 rollout, then produce versioned offline teacher labels without changing control behavior.

**Architecture:** Keep the sealed crossover runner untouched. Add pure validation/logging modules and a new diagnostic runner/host protocol derived from the proven ROS-free Direct path. Reward terms are read after the existing RewardManager compute; contact peaks use a direct PhysX view read at each actual 5 ms physics substep, gated by runtime capability and timestamp checks. Compact chunks and metadata are written under a new persistent root and sealed only after all writers exit.

**Tech Stack:** Python 3, NumPy, PyTorch, Isaac Lab/PhysX, pytest; no new third-party dependency.

**Spec:** User attachment `15bf16bb-6ad6-420f-bc46-93dc6b8f78ab/Pasted text.txt` and `docs/eat-nav/p1-g3-motion-residual-and-teacher-label-review-20261013.md`.

## Global Constraints

- Preserve all sealed 12-trial roots and existing ROS-free crossover source/assignment.
- No follower, Actor/Critic checkpoint, Path geometry, physics, noise, latency, or material change.
- One short Direct-only runtime diagnostic at most; no second-graded crossover, training, or full collection.
- Unknown mechanical allowables remain missing; do not fabricate Pain.
- Reward contribution sum tolerance fixed before runtime at `atol=1e-5, rtol=1e-5` per env/step.
- Contact substep capability must be proved from live 5 ms samples or reported unavailable; never relabel 50 ms samples as substep peaks.

## Review Focus

- RewardManager `_step_reward` is divided by dt; tests must catch a missing dt factor.
- Zero-weight terms have no raw value; tests must enforce explicit missing rather than zero raw.
- Contact force sampling can be stale; tests must require ten distinct physics-step callbacks and shape/finite checks.
- Interrupted chunk must leave prior chunks verifiable and must not create a complete marker.
- Logging must not mutate observation/command or consume RNG; synthetic tests compare off/on control objects.

### Task 1: Reward and load capture contracts

**Files:** Create `src/visualnav_transformer/eat_nav/p1_teacher_capture.py`; test `tests/test_p1_teacher_capture.py`.

**Interfaces:** `RewardTermCapture.from_manager(manager, dt)` returns immutable resolved term metadata; `capture_step(manager, returned_reward, step, stamp_ns)` returns raw/weighted vectors after one existing compute. `SubstepContactAccumulator.accept(sample, substep, physics_stamp_ns)` records peak, index, impulse and contact count. `compute_pain(loads, allowables)` returns missing or separate normalized joint/body components.

- [ ] Write synthetic RewardManager tests for dt, zero weight, sum tolerance, nonfinite, no recomputation; confirm RED.
- [ ] Implement capture and run tests GREEN.
- [ ] Write joint/body mapping, substep aggregation, missing/Pain tests; confirm RED.
- [ ] Implement and run tests GREEN.

### Task 2: Compact durable chunk logger

**Files:** Create `src/visualnav_transformer/eat_nav/p1_teacher_chunks.py`; test `tests/test_p1_teacher_chunks.py`.

**Interfaces:** `ChunkWriter(root, schema, fixed_shapes, chunk_rows)` writes typed fixed-shape arrays into compressed NPZ chunks and ragged JSONL separately; `verify_chunks(root)` checks schema, row counts and SHA. `seal_chunks(root)` marks complete only after writer close and verification.

- [ ] Write round-trip, interrupted-write, corrupt-SHA, legacy-schema and buffer-bound tests; confirm RED.
- [ ] Implement minimal writer/reader and verify GREEN.

### Task 3: New ROS-free diagnostic runtime adapter

**Files:** New `scripts/eat_nav_p1_teacher_sim.py`, `scripts/eat_nav_p1_teacher_host.py`, `scripts/eat_nav_p1_teacher_prepare.py`; tests `tests/test_p1_teacher_runtime_contract.py`.

**Interfaces:** One authorized Direct-only process, four envs, 20-step settling and short frozen control horizon. New schema and source/input SHA in manifest; stage all dependencies under fresh root. Simulator captures reward immediately after `env.step`, joint/body maps at startup, substep contact only if direct PhysX view updates each 5 ms, and compact rows without changing Actor/command path. Host verifies native exit, caps, SHA and seal.

- [ ] Write static/synthetic tests for authorization, step alignment, unchanged command objects, stage SHA, source isolation; confirm RED.
- [ ] Implement/stage and run tests GREEN.
- [ ] Run broader related tests, freeze manifest/source SHA and preflight resource caps.
- [ ] Execute at most one small Isaac diagnostic if all offline gates pass; stop on first invalid/inconclusive result.

### Task 4: Offline labels and review

**Files:** Create `scripts/eat_nav_p1_teacher_labels.py`; test `tests/test_p1_teacher_labels.py`; report `docs/eat-nav/p1-teacher-capture-runtime-review-20261013.md`.

**Interfaces:** Compute fixed 40-step motion, common-goal and `γ=1` reward from the same valid chunk; preserve censor/arrival masks. Pain emits `missing_allowables` unless the independent versioned allowables schema is complete.

- [ ] Write fixed-window, arrival, censor, gamma, missing Pain tests; confirm RED.
- [ ] Implement and run tests GREEN.
- [ ] Report runtime results, logging off/on limits, measured resource usage and 100-env extrapolation. Stop.
