# NoMaD Robot Interface Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task-by-task.

**Goal:** Connect compressed front-camera input to map-frame Path output without changing model architecture.
**Architecture:** Pure interface helpers own RGB/history/planar transforms. A single inference worker runs existing NoMaD inference; the ROS main executor owns state generations, exact-time TF, publishing and a steady-clock watchdog.
**Tech Stack:** Python, ROS 2 Jazzy, NumPy/OpenCV/Pillow, PyTorch, pytest.
**Spec:** docs/superpowers/specs/2026-09-15-nomad-robot-interface-design.md

## Global constraints
- No changes to models or weights, no classifier/state-source additions.
- Required model/checkpoint/topomap, context_interval_sec and action_scale_m.
- Keep existing goal selection, DDPM and first-sample selection.
- Path at observation time, metric map XY, source position plus predicted points.
- Stop on stale/invalid data, exact-time TF failure and rewind; old worker results cannot revive motion.

## Task 1: Interface helpers
Files: deployment/src/robot_interface.py, tests/test_robot_interface.py.
- [x] Write RGB JPEG, malformed image, interval/duplicate/gap/rewind and map-transform tests; verify missing implementation fails.
- [x] Implement decode_compressed(data)->PIL RGB, ImageHistory.add(image, stamp_ns), metric_path(points, scale, transform, frame, stamp).
- [x] Check independent known transforms at yaw 0/±90/180 degrees, preserve order and add origin once; reject invalid/degenerate paths.

## Task 2: Existing-model inference adapter
Files: deployment/src/robot_inference.py, utils.py, tests/test_robot_inference.py.
- [x] Add strict-load option to existing loader defaulting to existing behavior; robot adapter opts in.
- [x] Implement topomap validation and a NoMaD predictor using unchanged encoder, distance-head, scheduler and get_action code path.
- [x] Test nearest/subgoal selection and numerical equivalence with seeded legacy reference on identical model/inputs; incompatible state dict must fail.

## Task 3: ROS node lifecycle
Files: deployment/src/navigate_robot.py, tests/test_navigate_robot.py.
- [x] Use one worker future at a time, no work queue; callbacks own generations and publish only current results.
- [x] Implement readonly startup parameters, compressed-image subscriber, transient-local Path publisher, nonblocking observation-time TF and steady-clock watchdog.
- [x] Test actual ROS subscriptions/TF and output frames/stamps, timeout during blocked inference, clock pause/rewind, invalid worker output, shutdown and final goal.
- [x] Ensure a generation invalidation clears history and pending work, while new images can refill for recovery.

## Task 4: Delivery
Files: config/robot_interface.yaml, README.md, dependency metadata, tests.
- [x] Document required model-dependent settings, baseline checkpoint compatibility and follower rolling launch.
- [x] Run pure/ROS tests; run model tests if dependencies can be installed locally. Report any inference/real-hardware verification limits explicitly.
- [x] Obtain code review, address findings, copy verified changes to feat/nomad-robot-interface, check diff and commit.


## Verification record
- 28 tests passed: RGB/history/coordinate helpers, actual ROS transport/lifecycle tests, and real NoMaD inference comparison.
- The seeded CPU inference result equals the original navigate.py NoMaD branch element-for-element; a synthetic compatible checkpoint was used. Model class and training files are unchanged.
- Conditioned checkpoint keys are rejected by the strict robot loader. Legacy loader callers retain their prior default behavior.
- Review identified independent published-Path expiry; reproduced the failure and added a regression before fixing it. Also prevented same-observation reinference after invalidation.
- Black 24.4.2, isort 5.13.2 and focused flake8 error checks passed. uv lock --check --offline passed.
- Tests ran in an isolated /tmp virtual environment with ROS 2 Jazzy and CPU PyTorch; no robot velocity topics were used.
- Actual checkpoint selection, training interval/scale/frame, topomap selection, bag validation and real-robot testing remain deployment prerequisites; no navigation accuracy claim is made.
