# EAT-Nav Cancel Residual and Student Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Explain residual motion after Cancel with preserved evidence and document the reproducible research demo.

**Architecture:** Analyze the sealed terrain-v2 trace read only. Add a versioned diagnostic trace for signals absent from v2, then run paired boxes-level-0 conditions in fresh persistent roots. Keep the follower, Actor, physics, terrain, and existing experiment evidence unchanged. Use source-linked student documents to distinguish implemented steps from future research work.

**Tech Stack:** Python, Isaac Lab/Isaac Sim, ROS2 Jazzy, PyTorch, Matplotlib, Git.

**Spec:** `/home/kojima/.codex/attachments/d5380c39-cdb2-44fa-8d33-0181d111c5e7/Pasted text.txt`

## Global Constraints

- Preserve all existing experiment roots and seals.
- Keep checkpoint, follower, Path, Actor observation, noise, latency, terrain, and physics settings fixed.
- Save any new runtime in fresh persistent roots with SHA ledger and seal.
- Do not train or change control behavior.

## Review Focus

- The command entering the Actor may lag the follower command; compare both per step.
- Root displacement is distinct from integrated path length; report both.
- Effort/tensor fields must be labeled by Isaac's actual data meaning.
- A no-goal run must still connect the frontend and advance physics identically.
- A student's copy-paste command must use existing files and Python environments.

## Tasks

### 1. Preserve and analyze v2 evidence

- [ ] Verify the original SHA ledger and host seal read only.
- [ ] Extract Cancel-aligned command, Actor action, root XY/yaw, finite-difference velocities, and missing-field inventory.
- [ ] Generate plots and an analysis JSON with cumulative/net displacement separated.

### 2. Add minimal diagnostic capture

- [ ] Add tests for temporal alignment, finite signal capture, and labels of measured versus commanded effort.
- [ ] Watch the new tests fail, then implement trace-only capture and a new protocol/root identity.
- [ ] Run static/synthetic tests and compare unchanged control path to v2.

### 3. Run bounded paired diagnostics

- [ ] Freeze A (goal, no Cancel), B (goal, Cancel), C (no goal) conditions and source/input SHA before launch.
- [ ] Run sequentially in fresh roots; verify each native exit, trace, resource, SHA, and seal.
- [ ] Analyze the command chain, Actor action, joint motion, and root response; document attribution limits.

### 4. Student handoff and reproduction

- [ ] Write environment setup, experiment pipeline, and research roadmap docs with source references.
- [ ] Follow the documented launch steps for a one-env GUI reproduction and save a fresh sealed root.
- [ ] Run focused tests, verify docs/links, preserve evidence, and commit only task files.
