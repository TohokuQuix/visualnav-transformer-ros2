# Minimal Four-Anchor P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze a four-anchor, 28-process P1 input and, only after source, X1/X2 and resource gates pass, execute it once and report the two-second signal.

**Architecture:** Build on committed role-aware and primitive reviews without rewriting their decisions. An offline manifest builder reads SHA-verified pre-run anchors and Path definitions, then emits immutable selection and process manifests. A separate runtime driver consumes those manifests and existing E1 simulator, bridge and follower code under the frozen lifecycle/resource contract.

**Tech Stack:** Python, pytest, NumPy, SHA-256, existing ROS2/Zenoh/Isaac E1 runners.

**Spec:** `/home/kojima/.codex/attachments/bbc8e39b-750e-4ba8-9419-0094a2d52ac5/Pasted text.txt`

## Review Focus

- The prior five-role failure remains unchanged; the new gate requires only two differential and two transition-primary anchors.
- Selection cannot read candidate Outcome; all inputs come from committed audits and durable radial backups.
- The common 0°/0.75 m navigation goal is stored separately from each primitive endpoint.
- Router and X2 are external and candidate-free, with no implicit router ownership.
- No candidate retry or post-publication remediation; the first protocol-invalid row stops the schedule.

## Task 1: New gate and provenance

- [x] Write RED tests for role/seed slots, missing support and SHA tampering.
- [x] Implement `minimal_p1_gate_v1` over committed role grid and durable anchor SHA evidence.
- [x] Freeze new output root, provenance, gate and four-anchor selection before any runtime preparation.

## Task 2: Frozen 28-row manifest

- [x] Write RED tests for four anchors × seven fresh processes, 0° three repeats, unique IDs and frozen Path SHA.
- [x] Implement deterministic row order and explicit separate global goal/endpoint/label identity.
- [x] Store source/config/checkpoint SHA, seed, output root and retry=0; seal input manifest.

## Task 3: Static and resource preflight

- [ ] Verify exact runtime anchor restoration and E1 source interfaces; stopped before simulator by resource gate. Prior offline map/base geometry was verified.
- [x] Forecast 28 processes from preserved measured distributions with 20% margin against one frozen E1 resource contract; GPU margin fails.
- [x] Compile scripts, run focused suite and inspect source diff.

## Task 4: Router/X1/X2 and candidate execution

- [ ] Verify external operator-owned router PID/start tick/ZID/endpoint/log SHA, host/container connectivity.
- [ ] Run candidate-free X2 with actual bridge, eight production followers and three readiness snapshots.
- [ ] If gates pass, run 28 rows in order, sealing each run and checking resource/process cleanup before the next.
- [ ] Stop at first failure without candidate retry.

## Task 5: Post-run analysis if 28/28 valid

- [ ] Apply previously frozen duplicate gates to the twelve direct-repeat runs.
- [ ] Analyze P→C→A→R→Y and 0–39-step locomotion/global-goal Outcome by role and seed.
- [ ] Produce correlated goal relabels, figures, SHA ledger and exactly one final decision. Stop before later phases.

## Stop Rule

Any source, anchor, router, X2, resource or protocol gate failure stops its dependent work and produces a sealed report. Graded and flat roles remain unsupported in this minimal P1.

**Result:** The four-anchor and manifest gates passed. The prospective GPU device peak is 7,138.8 MiB with 20% margin versus the frozen 6,656 MiB cap. Stop before X1/X2 and candidate physics. Tasks 4–5 remain conditional and unstarted; see `docs/eat-nav/local-primitive-p1-minimal-20261008/review-report.md`.
