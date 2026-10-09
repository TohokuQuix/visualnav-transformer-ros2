# P1 Golden Batch Parity Implementation Plan

> **For agentic workers:** Execute tasks in order with test-first checks and stop at the first frozen gate failure. No subagent delegation is authorized for this run.

**Goal:** Measure whether one 112-env batch with 28 target envs reproduces the SHA-frozen P1 sequential commands, Actor/physics signals, and Outcomes.

**Architecture:** A read-only golden loader validates the 28-run backup. A ROS-independent follower state machine consumes each target's Path, local pose and logical time. A custom per-slot terrain generator and batch runner reproduce four P1 anchors on 28 separated tiles. A host runner enforces pre-Path, parity, resource and artifact gates, then writes a read-only decision report.

**Tech stack:** Python 3.12, PyTorch/IsaacLab inside existing `isaac-lab-onix`, frozen RL checkpoint, production C++ follower source as semantic reference. The production ROS node and follower binary are read-only.

## Task 1: Golden source and tolerance freeze

- [ ] Add `src/visualnav_transformer/eat_nav/p1_batch_golden.py`, `config/eat_nav_p1_batch_parity.json`, and `tests/test_p1_batch_golden.py`.
- [ ] Test missing/altered seal, manifest, ledger, trace, follower SHA, and old-tenth exclusion (RED); implement verification (GREEN).
- [ ] Record source and contract SHA, coordinate mapping, all numeric limits, resource caps, and a fixed 28-slot manifest before examining batch results.

## Task 2: ROS-independent follower core

- [ ] Add a follower core with explicit Path/pose/logical-time/state input and command/state output. Keep production node untouched.
- [ ] Test Path projection, carrot, goal line, yaw, speed clamp, goal stop and state isolation against independent small fixtures (RED→GREEN).
- [ ] Replay all 3,360 golden commands offline. Stop on a mismatch; capture first run, step and internal signal. No model/Path/threshold change driven by results.

## Task 3: Mixed terrain and pre-Path batch gate

- [ ] Add a 112-slot per-tile terrain plan from the frozen four anchors and seven candidate/repeat rows per group. Test seed/type/slot uniqueness and local-to-world translation.
- [ ] In one simulator process, run RL reset and 200-step zero-action settling. Capture all target roots/joints, four DEM frames and Actor pre-/post-noise/pre-/post-delay inputs.
- [ ] Compare every target to golden under the frozen coordinate contract. Seal the pre-Path audit and stop before Path on any failure.

## Task 4: Batch control and measurement

- [ ] After pre-Path pass, activate 28 separate follower states and publish 28 frozen Paths exactly once.
- [ ] At 20 Hz for 120 control steps, use live observation for Actor and live physics. Record P, C, A, R, reward, Critic, DEM/contact, termination and reset per target. Keep the original P1 observation noise/latency contract explicit and test that divergence in pre-noise observation is not masked.
- [ ] Compare every step to golden. Compute 40-step discounted locomotion return and common-goal progress as separate primary labels; 120-step progress/termination as auxiliary.

## Task 5: Lifecycle and decision

- [ ] Test direct clones, per-env terminal isolation, cap attribution, lossless compression roundtrip, atomic seal and verified resume (RED→GREEN).
- [ ] Run one bounded batch under the frozen caps; stop without retry on any gate failure.
- [ ] If mixed-candidate parity passes, write an 8–12 independent-StateGroup pilot design and measured time/storage estimate. Otherwise report the first failing signal and causes. Do not start further collection or training.

## Review focus

- The two P1 terrain batch seeds and reused original env index 48 cannot be represented by the original single-seed generator without a per-slot scene plan.
- A world-coordinate comparison can be invalid solely because target tiles move; the local-frame mapping must be frozen before Path.
- Golden Actor input includes noise and observation latency; forcing golden post-delay input would mask live dynamics and is forbidden.
- The follower's goal-line and progress/stuck state can change a command even when current Path and pose are unchanged.
- A post-terminal reset in one target must not contaminate any other target or that target's 2 s label.
