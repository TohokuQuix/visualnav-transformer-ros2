# EAT-Nav ROS2 Adapter Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-testable ROS2 navigation pipeline that reconstructs the frozen 14-candidate model input and safely connects only to a mock follower interface.

**Architecture:** ROS Jazzy Python 3.12 handles typed transport, TF and supervision. A separate CPU-only Python 3.10 worker handles optional PyTorch inference; deterministic mock mode needs no checkpoint. Immutable calculation sources are imported from this isolated worktree. No Stage B/C source, process, artifact or shared ROS workspace is modified.

**Tech Stack:** ROS2 Jazzy, Python 3.12 rclpy, NumPy, Python 3.10 PyTorch worker, unittest/pytest, local colcon build.

**Spec:** `docs/eat-nav/motion-predictor-vnext/ros2-node-separation-phase1-design.md` including user-approved Sections 8.1–8.3.

## Global Constraints

- No Isaac/GPU runtime or robot velocity topic connection.
- Stage B/C code and evidence are read-only; all new source is under `src/visualnav_transformer/eat_nav/ros2_phase2/`, new ROS package under `ros2/eat_nav_interfaces/`, tests under `tests/`.
- The frozen 14-bank, DEM conversion, 22D semantics, Follower core and Stage C checkpoint gate are unchanged.
- Three explicit modes: `mock14_transport`, `legacy6_demo`, `learned14_adopted`; only mock is default for Phase 2 wiring.
- All build/install/log output stays in `/tmp/eatnav-ros2-adapter-build/` or this isolated worktree.

## Review Focus

- A transient-local old Path after Cancel must not result in nonzero mock command.
- A repeated DEM stamp must not renew the Path lease or conceal sensor outage.
- A 14-slot request must never enter the six-slot legacy checkpoint.
- Negative DEM heights and invalid masks must remain distinct after ROS serialization.
- Mismatched goal generation or TF stamp must not publish a Path.

---

### Task 1: Lock the design amendment and source provenance

**Files:** Update Phase 1 design; create this plan.

- [ ] Confirm Sections 8.1–8.3 state exact jitter/outage and stop behavior.
- [ ] Record Python 3.12/3.10 split and source baseline.
- [ ] Run `git diff --check` and commit the two documents only.

### Task 2: Typed ROS interface package and pure wire codecs

**Files:** Create `ros2/eat_nav_interfaces/{package.xml,CMakeLists.txt,msg/*.msg}`; create `src/visualnav_transformer/eat_nav/ros2_phase2/{__init__,wire}.py`; test `tests/test_eatnav_ros2_phase2_wire.py`.

**Interfaces:** `encode_dem/decode_dem`, `encode_actor/decode_actor`, `encode_candidate_bank/decode_candidate_bank` produce/consume generated ROS messages and reject incorrect shape/frame/stamp/contract. Fixed CANDIDATE_IDS and bank SHA come from existing bank.

- [ ] Write tests for positive/negative height, invalid mask, float32 byte round-trip, candidate order/direction/SHA and rejected malformed messages; observe RED.
- [ ] Implement message definitions and codecs; isolated `colcon build` and rerun tests GREEN.
- [ ] Commit task files.

### Task 3: DEM history, GridMap source and Actor22 adapter

**Files:** Create `ros2_phase2/{history,gridmap,actor_state}.py`; tests `tests/test_eatnav_ros2_phase2_inputs.py`.

**Interfaces:** `DemHistory.push(frame)` and `assemble(actor, now_ns)` return model-ready raw `[4,2,20,20]` and state `[22]`; `source_grid_from_gridmap(msg, layout)` delegates to unchanged `robot_dem`; `build_actor22(...)` reconstructs term order and scales from explicit stamped inputs.

- [ ] Write RED tests for history order, episode reset, 300/500 ms boundaries, circular GridMap with negative/invalid cells, Actor term order/fold/effort and missing values.
- [ ] Implement minimal deterministic adapters; run GREEN tests and commit.

### Task 4: Predictor modes, checkpoint loader and global-goal evaluator

**Files:** Create `ros2_phase2/{model_worker,predictor_client,selection}.py`; tests `tests/test_eatnav_ros2_phase2_predictor.py`.

**Interfaces:** `predict(mode, raw_tensors, checkpoint_info)` yields `[K,2]` body metres. Mock14 returns deterministic finite 14 outputs; legacy invokes pinned six-slot old loader only with seven forward bank; learned14 requires Stage C adoption Pass and exact metadata. `score_global_goal` uses unchanged formula and ordered mask argmax.

- [ ] Write RED tests for mode isolation, old/new checkpoint rejection, normalization, target inverse transform, TF/body scoring and tie break.
- [ ] Implement CPU worker with bounded local protocol; run tests with system Python and tinyvla CPU; commit.

### Task 5: Supervisor and mock-only follower safety gateway

**Files:** Create `ros2_phase2/{supervisor,gateway}.py`; tests `tests/test_eatnav_ros2_phase2_safety.py`.

**Interfaces:** `Supervisor.tick(now_ns, data_state)` yields defer/replan/stop and generation; `MockFollowerGateway` records selected/invalidated Path and final command, enforcing 500 ms lease, command age and zero at 20 Hz.

- [ ] Write RED tests for normal jitter, outage, Cancel, Timeout, stale evaluation, lost heartbeat, stale follower command and old transient Path.
- [ ] Implement state machine and mock, run GREEN tests, commit.

### Task 6: ROS nodes, replay and CPU integration

**Files:** Create `ros2_phase2/{source_nodes,inference_node,supervisor_node,display_node,replay}.py`, launch/config and `tests/test_eatnav_ros2_phase2_integration.py`.

**Interfaces:** Source nodes publish typed DEM/Actor from replay or measured GridMap/ROS state; inference node assembles inputs and calls worker; supervisor node owns Goal/Cancel and mock gateway; display node subscribes only. A GUI-less launch path is mandatory.

- [ ] Write RED tests for node topic/QoS contract, Goal/Cancel, duplicate/out-of-order messages, TF failure and mock final zero.
- [ ] Implement nodes and isolated launch; run CPU-only colcon/test commands and exact sealed-sample ROS round-trip.
- [ ] Record SHA, test evidence and remaining real-source Unknowns; commit without changing Stage B/C.

## Completion gate

`git status` clean on this worktree; all relevant CPU tests and isolated ROS round-trip pass; no Isaac process, GPU call, real follower topic or shared ROS workspace rebuild; Stage B/C source HEAD and manifests remain unchanged. If a gate cannot be verified, report it as incomplete instead of claiming Phase 2 Pass.
