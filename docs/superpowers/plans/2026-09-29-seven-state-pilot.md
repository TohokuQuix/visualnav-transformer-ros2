# EAT-Nav State 42 Diagnosis and Seven-Terrain Pilot Implementation Plan

> **For agentic workers:** Execute inline using superpowers:executing-plans. This plan follows the user-provided state 42 / seven-terrain pilot specification.

**Goal:** Classify the state 42 first-command timeout, then run an outcome-blind seven-terrain train pilot only if its infrastructure gate permits it.

**Architecture:** Preserve the frozen 56-state selection and five-Path maneuver bank. A diagnostic ROS bridge records each follower boundary and performs only source-supported readiness checks. A fresh-process runner reconstructs each anchor from the recorded 100-env episode prefix and runs at most three nominal duplicates plus five fixed Paths per selected state. Pure analysis produces per-state gates and stops before any validation or formal rollout.

**Tech Stack:** ROS 2 Jazzy, real `simple_trajectory_follower`, Isaac Lab Docker, Python/pytest, detached tmux, JSON/NPZ/gzip artifacts.

**Spec:** `/home/kojima/.codex/attachments/c50c8a68-2d87-431a-9e04-94f511115266/Pasted text.txt`

## Global Constraints

- Do not alter maneuver bank, scales, Path construction, follower control parameters, simulator dynamics/noise/latency/PhysX, model checkpoints, or frozen 56-state identities.
- State 42 must be classified before any seven-state rollout; infrastructure-invalid or unexplained first-command loss stops the pilot.
- All attempts, including failures, remain in results; no post-result state replacement or unlimited retry.
- Use independent simulator processes and logged episode prefixes; no snapshot restore.
- Use 40 control steps at 0.05 s, duplicate XY RMS limit 0.0105 m, and the existing Level 1/2/3 feasibility thresholds.
- Keep validation 20, formal 56 train, Phase C/D/E, full collection and robot tests stopped.

## Review Focus

- A zero-valued fresh TwistStamped is classified differently from no received message.
- A near-goal normal stop is classified differently from an infrastructure timeout.
- Subscriber discovery alone does not prove Path callback, TF readiness, timer activity or command publication.
- A simulator exception followed by Isaac cleanup exit code zero cannot produce a completion marker.
- A failed third duplicate remains in the denominator and prevents a false pass.

## Task 1: Freeze goal and failure-pipeline evidence

**Files:** New `scripts/eat_nav_seven_state_diagnose.py`, `tests/eat_nav/test_seven_state_diagnose.py`, output `state42-diagnosis.json` and copied ROS logs.

- [ ] Write and run failing tests for goal-distance and event classification.
- [ ] Parse frozen state 42 Path, follower params/source and prior bridge/follower logs; save timestamps, SHA and evidence levels.
- [ ] Establish whether the failure is near-goal, Path missing, TF startup, timer, publish, or bridge reception.
- [ ] Verify all 8 follower logs and the 0.800 m goal distance; record unresolved boundaries explicitly.

## Task 2: Minimal readiness instrumentation and fix, only if Task 1 confirms a startup/TF/clock/Path timing cause

**Files:** `scripts/eat_nav_phase_c_ros_bridge.py`, `tests/eat_nav/test_phase_c_readiness.py`, output `readiness-handshake.json`.

- [ ] Write and run failing tests for bounded readiness, zero command freshness, one Path resend, and final invalid classification.
- [ ] Add bridge-only counters/timestamps for Path subscriber, status/timer proxy, TF lookup, command arrival, and follower process/log capture; preserve follower control code.
- [ ] Establish the active ROS clock mode. If `/clock` is absent because `use_sim_time=false`, record the deviation and use only a user-approved or technically justified clock criterion.
- [ ] Run one bounded diagnostic reset with fixed state 42 Path; then implement the minimal confirmed readiness fix and verify it.

## Task 3: State 42 fresh-process retest

**Files:** `scripts/eat_nav_seven_state_run.py`, `state42-retest.json`, individual atomic markers/traces.

- [ ] Run three fixed-Path nominal repeats from the exact logged episode prefix, preserving each failed attempt.
- [ ] Compare anchor, Path SHA, readiness, 40-step command/Actor/XY/termination and duplicate RMS.
- [ ] Classify `valid`, `no_action_required`, or `infrastructure_invalid` and stop on invalid.

## Task 4: Freeze seven-terrain train selection before pilot outcomes

**Files:** `scripts/eat_nav_seven_state_select.py`, `tests/eat_nav/test_seven_state_select.py`, `pilot-state-selection.json`, `pilot-config.yaml`.

- [ ] Write and run failing tests for deterministic minimum-rank selection, one terrain each, episode cap, goal-distance eligibility, and no refill from outcomes.
- [ ] Select only from the frozen 56 train states using SHA256 rank of `(seed, terrain, episode_key, view_index)` and goal distance >=0.30 m; save all ranked eligible candidates and selected IDs.
- [ ] If state 42 is classified normal near-goal and selected, retain NA without replacement; if infrastructure-invalid, stop before execution.

## Task 5: Seven-state fresh-process pilot

**Files:** `scripts/eat_nav_seven_state_stage.sh`, `scripts/eat_nav_seven_state_analyze.py`, `duplicate-controls.json`, `maneuver-rollouts.jsonl.gz`, `pilot-feasibility.json`.

- [ ] Execute in detached `eatnav-seven-state-pilot` tmux with per-run markers, source/input/output SHA, status and persistent log.
- [ ] For each selected state, require at least two nominal duplicates (three if available) before fixed five-Path rollouts; preserve candidate order seed from the existing config.
- [ ] Store complete ROS/Actor/Critic/action/root/joint/contact/DEM/reward/termination traces, first command divergence and all requested trajectory/feasibility metrics.
- [ ] Stop at two measurement-invalid states or other frozen stop conditions; do not replace failures.

## Task 6: Decision and report

**Files:** `runtime-storage.json`, `pilot-decision.json`, `artifact-hashes.json`, figures, `seven-state-pilot-report.md`.

- [ ] Compute validity, candidate-effect, rho, command-cluster and Level 1/2/3 summaries by terrain without population generalization.
- [ ] Measure runtime, RAM/VRAM and bytes per run; compare to the frozen estimate and record failures.
- [ ] Render the requested seven figures, hash every output, and verify markers and no formal process is running.
- [ ] Report the gate and stop. Do not start 56 train or 20 validation.
