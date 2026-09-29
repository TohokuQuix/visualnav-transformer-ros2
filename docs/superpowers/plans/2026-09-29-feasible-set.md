# Counterfactual Feasible Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Measure whether multiple command-distinct, physically feasible paths exist from the same train-selected simulator state.

**Architecture:** Freeze state IDs and train-derived thresholds before simulator outcomes. Reproduce selected collection anchors by replaying logged actions in the original 100-env scene and verify pose/DEM agreement before serial restoration. Generate five endpoint-matched Path proposals, precheck through the actual follower, then run duplicate and maneuver replay with state-level markers. Compare feasible modes with train teacher neighborhoods; validation is confirmatory only.

**Tech Stack:** Isaac Sim/IsaacLab, Python/NumPy/PyTorch, ROS2 follower bridge, tmux.

**Spec:** `/home/kojima/.codex/attachments/f64899fb-619c-4ba3-93cd-7d53b38519f3/Pasted text.txt`.

## Global Constraints

- Train 56 states, 8 per terrain; validation at most 28, at most 4 per terrain; held-out forbidden.
- Checkpoint, Actor/Critic, follower, 40-step horizon and Phase B outcomes frozen.
- No new model training, formal Phase C, Phase D/E, full collection, or real robot run.
- No changing maneuver bank, scales, state IDs, or thresholds after simulator outcomes.
- Exact anchor reconstruction is a measurement gate: if replay cannot match the logged state, do not call the result a feasible-set measurement.

## Review Focus

- Logged records omit joints/delay state; deterministic prefix replay must match anchor pose/DEM before a state is valid.
- Dataset goal derives from logged future motion; selection may use relative goal as the spec permits, but no future reward/Critic/termination.
- A Path may be command-distinct but fail due to controller startup or state restoration; duplicate controls must separate these.
- Terminal/reset changes must be captured before reset and aligned with reward/value outcomes.
- Invalid proposal or OOD support must yield proposal-inconclusive, never single-solution.

---

### Task 1: Freeze state IDs and thresholds

**Files:** `scripts/eat_nav_feasible_set_freeze.py`, `tests/test_feasible_set_freeze.py`, `docs/quality-identifiability-gate/feasible-set/feasible-set-config.yaml`.

- [ ] Test future-label-free state ranking, episode uniqueness, terrain/stratum quotas, held-out exclusion.
- [ ] Implement complete-only train/validation selection and train-only threshold fitting.
- [ ] Save source/checkpoint/config/dataset hashes, state IDs and maneuver scales before simulator run.

### Task 2: Endpoint-matched maneuver bank and real follower precheck

**Files:** `src/visualnav_transformer/eat_nav/feasible_set.py`, `scripts/eat_nav_feasible_set_precheck.py`, tests.

- [ ] Test shared origin/goal endpoint, left/right symmetry, finite bounds, fixed scales.
- [ ] Generate bank and record OOD and constraint status.
- [ ] Use actual follower bridge to measure command clusters; retain nondistinct states.

### Task 3: Anchor reconstruction and duplicate control

**Files:** `scripts/eat_nav_feasible_set_sim.py`, tests.

- [ ] Replay logged actions from step 0 in the original 100-env runtime; compare logged anchor pose/DEM.
- [ ] Capture and restore root/joint, manager, delay, controller and RNG state; quantify unreplicated variation.
- [ ] Run nominal duplicate at least twice per state, stage marker per state, resume without rerun.

### Task 4: Counterfactual rollouts and outcome classification

**Files:** `scripts/eat_nav_feasible_set_runner.py`, `src/visualnav_transformer/eat_nav/feasible_set.py`, tests.

- [ ] Fix candidate order before outcomes, 40-step replay for each bank Path.
- [ ] Record Actor/Critic, command, reward, contact, DEM, terminal and kinematics.
- [ ] Classify Level 1–3, distinguish no-solution from proposal-inconclusive.

### Task 5: Coverage gap, uncertainty and report

**Files:** `scripts/eat_nav_feasible_set_analysis.py`, report/figures, tests.

- [ ] Assign feasible modes to train teacher neighborhoods, bootstrap by state/episode, terrain/stratum analysis.
- [ ] Apply frozen conditions to validation only after train results, report H1–H4 decision.
- [ ] Verify artifact hashes, stop before collection/retraining/formal Phase C.
