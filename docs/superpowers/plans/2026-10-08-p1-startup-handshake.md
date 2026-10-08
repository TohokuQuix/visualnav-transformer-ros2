# P1 Startup Ownership Handshake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish simulator ownership before GPU initialization, validate one zero-physics preflight and one candidate-free 112-env resource smoke, then conditionally continue the frozen P1 protocol.

**Architecture:** A standard-library-only wrapper publishes a versioned owner-ready record and waits for an exact authorization record. The host resolves a single PID/start tick from token, namespace PID and cgroup, starts bracketed NVML monitoring, and authorizes an in-place `exec` into the frozen simulator source. Coordination and physics output use separate directories.

**Tech Stack:** Python, pytest, Docker exec/cp, `/proc`, NVML, Isaac Sim.

**Spec:** `/home/kojima/.codex/attachments/5276c15e-f16a-4dbc-ab14-75c9f4e42ade/Pasted text.txt`

## Global Constraints

- Base commit `a2915537c124ea59061061c8625273a4310c3b30`; no prior attempt is promoted.
- Keep anchor, Path, manifest order, seed 123, simulator/reset/Actor and resource v2 thresholds frozen.
- One H4 real preflight and one H5 real smoke. After authorization, no automatic retry.
- Never start or stop the external Zenoh router.

## Review Focus

- A process with the same output path but a different token must not be owned.
- A reused PID with a different start tick must not be signaled or authorized.
- A stale authorization marker must not advance the wrapper.
- A second run cannot start while an earlier owned PID is live.
- Completion must be written before potentially nonreturning App close.

---

### Task 1: Freeze provenance and output overlay

**Files:** Create `scripts/eat_nav_p1_handshake_prepare.py`, `tests/test_p1_handshake_prepare.py`.

**Interfaces:** `prepare(root: Path) -> dict` verifies required SHA ledgers and writes the 28-row overlay and field diff before launch.

- [ ] Write failing tests for 28 unchanged logical rows and bad SHA refusal.
- [ ] Verify RED.
- [ ] Implement explicit source audit and output overlay freeze.
- [ ] Verify GREEN and save H0 artifacts.

### Task 2: Wrapper protocol

**Files:** Create `scripts/eat_nav_p1_owner_wrapper.py`, `tests/test_p1_owner_wrapper.py`.

**Interfaces:** `wait_for_authorization(coord: Path, identity: dict, timeout: float) -> None`; `main()` emits ready, waits, then executes either zero-physics preflight or the frozen simulator.

- [ ] Write failing tests for exact marker matching, timeout/rejection, stale marker and no preauthorization GPU import.
- [ ] Verify RED.
- [ ] Implement atomic markers and PID-stable `exec` after authorization.
- [ ] Verify GREEN.

### Task 3: Host identity and resource monitor

**Files:** Create `scripts/eat_nav_p1_owner_host.py`, `tests/test_p1_owner_host.py`; modify `scripts/eat_nav_p1_gpu_attribution.py` only to include full process tables.

**Interfaces:** `resolve_owner(table, ready, token, container_id) -> dict`; `authorize(...)`; `run_one(...)` writes complete bracket tables and exact cleanup evidence.

- [ ] Write failing fake-process tests for missing/duplicate NSpid, wrong cgroup/token/start tick, lineage and preauthorization GPU.
- [ ] Verify RED.
- [ ] Implement exact resolution and bracket-before-authorization state machine.
- [ ] Verify GREEN.

### Task 4: Real gates and conditional continuation

**Files:** Create H4/H5 run evidence and final decision under new `/tmp` output root; small JSON/Markdown report in `docs/eat-nav`.

- [ ] Freeze source/test/manifest SHA and run baseline tests.
- [ ] Run one zero-physics preflight. Stop on postauthorization failure.
- [ ] If H4 passes, run one candidate-free shard-B smoke; verify 200 steps, full attribution, SHA and cleanup.
- [ ] If H5 passes, audit router and continue X1/X2/28 rows only when every gate passes.
- [ ] Commit code, tests and small reports after fresh verification.
