# P1 Statistical Path Comparison v1 Implementation Plan

**Execution status (2026-10-09):** stopped as `protocol_invalid`. The v1 pilot
config requested 80 total control steps while this plan requires 120. Five
sealed 80-step processes remain excluded evidence. A versioned v2 config and
execution guards are prepared; no v2 Path rollout has started. See
`docs/eat-nav/p1-statistical-path-comparison-stop-20261009.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze and validate a fair randomized P1 Path experiment, then run only a two-batch variance pilot if every preflight passes.

**Architecture:** Pure Python contract, assignment and analysis modules validate all non-simulator decisions. A versioned simulator adapter reuses terrain geometry, clones pre-Path control state, runs the existing follower and Actor protocol, and emits sealed per-rollout records. A host runner enforces process/resource gates.

**Tech Stack:** Python 3.11, pytest, NumPy, IsaacLab, rsl_rl, ROS 2 follower bridge, SHA-256.

**Spec:** `docs/superpowers/specs/2026-10-09-p1-statistical-path-comparison-design.md`

## Global Constraints

- Preserve all historical evidence and exact-replay thresholds. New protocol version only.
- Use `policy.act_inference()`; Gaussian policy action seed is not applicable.
- Four original StateGroups and five frozen Paths, plus two extra direct repetitions per batch.
- Two independent pilot batch seeds; 20 settling and 120 candidate control steps; only first 40 enter primary Outcomes.
- GPU ≤7680 MiB, RAM available ≥10 GiB, active storage ≤1 GiB, heartbeat ≤90 s, process ≤600 s.
- Production follower, Path geometry, RL checkpoint and Outcome definitions unchanged.

## Review Focus

- `boxes` tile equality must be checked on collision mesh after stage composition, not only cached source arrays; test duplicate source geometry reuse and fail a changed mesh.
- A seed label is not a valid common-random-number pair with `act_inference`; test that paired analysis is rejected unless stream alignment evidence exists.
- A candidate assigned permanently to one slot is confounded; test two-batch slot rotation.
- A terminal event in the 0–2 s interval must not admit post-reset reward or pose; test masked extraction.
- A narrow bootstrap interval from two batches must not be called confirmatory; test the descriptive label and independent-batch count.

---

### Task 1: Pure assignment and preflight contract

**Files:** Create `src/visualnav_transformer/eat_nav/p1_statistical_protocol.py`, `tests/test_p1_statistical_protocol.py`, `config/eat_nav_p1_statistical_v1.json`.

**Interfaces:** `build_assignments(source_contract, batch_seeds, random_seed) -> dict`; `validate_assignment(manifest) -> None`; `initialization_checks(records, tolerances) -> dict`; `terrain_identity_checks(records) -> dict`.

- [ ] Write tests for two-batch permutation, all five frozen Paths, three direct slots, slot rotation, geometry mismatch and initial control-history mismatch.
- [ ] Run tests RED.
- [ ] Implement the four interfaces and frozen v1 config.
- [ ] Run tests GREEN and commit these explicit paths.

### Task 2: Outcome and uncertainty analysis

**Files:** Create `src/visualnav_transformer/eat_nav/p1_statistical_analysis.py`, `scripts/eat_nav_p1_statistical_analyze.py`, `tests/test_p1_statistical_analysis.py`.

**Interfaces:** `extract_outcome(trace, goal, gamma=0.99) -> dict`; `analyze(rollouts, manifest, config) -> dict`; CLI takes sealed run root and new output root.

- [ ] Write tests for 40-step reward, terminal masking, unpaired batch contrast, same-batch dependence, non-significance and sample-size estimate.
- [ ] Run tests RED.
- [ ] Implement extractor, pilot descriptive cluster bootstrap and provenance validation.
- [ ] Run tests GREEN and commit these explicit paths.

### Task 3: Simulator terrain/control adapter

**Files:** Create a new `scripts/eat_nav_p1_statistical_sim.py`, `src/visualnav_transformer/eat_nav/p1_statistical_runtime.py`, `tests/test_p1_statistical_runtime.py`.

**Interfaces:** Geometry generated once per StateGroup and reused for all seven slots; `clone_and_check_state(env, source, targets, limits) -> dict`; record source/target mesh and material fingerprints, observations and explicit unavailable hidden state.

- [ ] Write tests for cached tile reuse, complete control-state fields and fail-before-Path behavior.
- [ ] Run tests RED.
- [ ] Implement runtime adapter using the frozen source fixture and existing follower bridge; do not modify production ROS node.
- [ ] Run tests GREEN, compile simulator code and commit explicit paths.

### Task 4: Candidate-free preflight and limited pilot

**Files:** Create `scripts/eat_nav_p1_statistical_host.py`, `docs/eat-nav/p1-statistical-pilot-20261009.md`; output roots under `/tmp/eatnav-p1-statistical-20261009/`.

- [ ] Verify old seals, freeze code/config/assignment SHA, router identity, resource baseline, terrain and initialization before Path publication.
- [ ] Run candidate-free initialization validation; stop and report on first failed gate.
- [ ] If valid, run two-batch pilot in frozen order with owned process, heartbeat/resource monitor, atomic marker, resume and SHA seal.
- [ ] Analyze read-only sealed Outcomes, report assignment/variance/effect/interval/replication need and decision. Stop before expansion or training.
