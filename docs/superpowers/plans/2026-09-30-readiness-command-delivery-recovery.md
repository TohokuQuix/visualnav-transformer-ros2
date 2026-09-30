# Readiness Command Delivery Recovery Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans and test-driven-development task by task.

**Goal:** Locate the first failed command-delivery boundary, repair its confirmed cause, and resume the frozen 119-process pilot only after every recovery gate passes.

**Architecture:** Keep the original 100 completed runs and failed run immutable. Use a diagnostic bridge/follower/simulator copy with event logs, then make one minimal infrastructure correction only when the evidence identifies a cause. A new resume runner must validate every identity and SHA before launching any remaining process.

**Tech Stack:** Python, ROS 2 Jazzy/rclpy, C++ follower diagnostic build, Isaac Sim Docker, pytest, tmux.

**Spec:** User attachment `dcaf1590-968b-46c6-b389-5a897a79d44b/Pasted text.txt`.

## Global constraints

- Freeze `readiness-100` attempt-0 and all 100 completed markers and SHA values.
- Keep candidate, state, seed, Path, order, controller, model, checkpoint, and exact-stamp protocol unchanged.
- At most one diagnostic replay and one new measured run for `readiness-100`.
- Do not start postprocessing before all 119 processes finish.
- Stop on any missing command, stamp mismatch, resource excess, or unresolved root cause.
- Additional caps: two hours for diagnosis and recovery, two hours for remaining runs, one hour for postprocessing; one GB total pilot storage.

## Review focus

- A callback logged before publish does not prove publish returned: assert both boundaries.
- A stopped Docker CLI may leave a simulator process: assert process cleanup within 60 seconds.
- A partial failed directory may look resumable: assert its attempt is immutable and never considered complete.
- Reused markers may conceal changed inputs: verify schedule, candidate Path, source and output SHA.
- Diagnostics may change timing: compare exact stamp, command values, and production fixture results.

### Task 1: Freeze and explain prior evidence

**Files:** `/tmp/eatnav-candidate-coverage/evaluator-readiness-recovery/evidence-freeze.json`, `offline-timeline.json`, `hypothesis-matrix.json`.

- [x] Hash source, binary, config, inputs, failed logs, 100 done markers, and output markers.
- [x] Check all 100 identity and output SHA values.
- [x] Build eight-branch six-stamp timeline, using null for missing events.
- [x] Compare USD warnings with a successful run.

### Task 2: Instrument diagnostic copies and test event delivery

**Files:** diagnostic-only bridge and follower copies under `/tmp/eatnav-candidate-coverage/evaluator-readiness-recovery`; new targeted tests under `tests/eat_nav/`.

- [ ] Write RED tests for callback versus waiter thread, lock release, sequence propagation, executor-stop error, and timeout cleanup.
- [ ] Add flush-on-event diagnostics for bridge executor, subscriber, queue, waiter, TCP, and process resources.
- [ ] Add follower callback, TF, publish entry/return, subscriber count, and shutdown diagnostics in the diagnostic-only binary.
- [ ] Add simulator request, exception, trace flush, and close-stage diagnostics.
- [ ] Run targeted tests GREEN and verify command/Path/stamp behavior is unchanged.

### Task 3: One frozen-input diagnostic replay

- [ ] Launch a detached tmux session with atomic status, log, resource monitor, and two-hour deadline.
- [ ] Replay exactly `readiness-100` once into `diagnostic-attempt-1`; retain attempt-0.
- [ ] Locate the first missing boundary and compare fresh versus old bridge lifecycle.
- [ ] If root cause remains unproven, record `inconclusive` and stop.

### Task 4: Minimal fix and recovery gate

- [ ] Write a failing test for the confirmed cause, then apply one minimal infrastructure fix.
- [ ] Pass production fixtures, one-branch and eight-branch smokes, exact-stamp checks, cleanup under 60 seconds, and three lifecycle runs if changed.
- [ ] Verify diagnostic replay completed 120 steps with identical input SHA and expected resources.

### Task 5: Measured resume and analysis

- [ ] Run one new measured `readiness-100` attempt and stop permanently if it fails.
- [ ] Revalidate 100 completed markers, then run remaining 18 in frozen order under the two-hour cap.
- [ ] Only after all 119 complete, run approved coverage/readiness analysis and publish the SHA ledger and final report.
