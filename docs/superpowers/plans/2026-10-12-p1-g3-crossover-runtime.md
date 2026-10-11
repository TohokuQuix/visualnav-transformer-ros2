# P1 G3 Crossover Runtime Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement these tasks in order. No subagent delegation is requested.

**Goal:** Freeze and statically verify a 18-process Direct/Left/Right crossover runtime without executing Isaac.

**Architecture:** Derive a versioned simulator and host from the sealed direct-repeat runtime while leaving its source untouched. A host/simulator pre-Path handshake verifies current inputs against the first period before any Path is published. A separate offline analyzer computes paired candidate effects with seed-level clustering.

**Tech Stack:** Python, Isaac Lab/PhysX/USD APIs at runtime, ROS exact-stamp follower, pytest for static/synthetic tests.

**Spec:** User-approved P1 G3 Path crossover instructions, 2026-10-12; `docs/eat-nav/p1-g3-path-crossover-v2-review-20261012.md`.

## Global Constraints

- No Isaac runtime, curved Path rollout, Evaluator training, or full collection in this implementation turn.
- Preserve the sealed direct-repeat v1 code, manifest, traces, aggregate, and seals.
- Keep 18 process/72 trial assignment, process order, existing physical/shared DEM/Actor and direct duplicate numeric gates, follower/checkpoint/physics/noise/resource contracts.
- Candidate Path publication occurs only after composed terrain/robot material and cross-period pre-Path gate pass.

## Review Focus

- Path-role/seed/env assignment drift: fail before launch.
- USD binding versus PhysX tensor effective robot material: save both; require usable PhysX values.
- Contact/Critic native state may differ: record separately, never silently call it equal.
- Cross-period gate runs after simulator settling but before Path publication, including period 0 self-check.
- Host seal follows all writers and includes new material/prepath artifacts.

---

### Task 1: Contracts and tests

- [x] Add synthetic tests for balance, immutable source SHA, cross-period strict/diagnostic fields, material mismatch rejection.
- [x] Implement pure gate and Path geometry/outcome helpers; run tests.

### Task 2: Runtime integration

- [x] Create a new versioned simulator and host, with a pre-Path approval handshake and live material readout.
- [x] Add host validation of all new artifacts and direct duplicate pairs; run static tests.

### Task 3: Frozen manifest and analyzer

- [x] Stage 18 planned process inputs in a new non-runtime root and produce SHA-frozen manifest.
- [x] Implement paired candidate−direct analysis, preserving censoring and clustering.
- [x] Run static/synthetic tests, source SHA audit, and report pilot procedure/resources.
