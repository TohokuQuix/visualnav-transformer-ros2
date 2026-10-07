# Radial 112-Pose Sharded Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the frozen 112 radial poses in independently sealed shards of at most five selected contact views, then classify eligible terrain-interaction anchors offline.

**Architecture:** Reuse the proven 112-env simulator and lean logger. A deterministic inventory manifest assigns each pose once; a strict host coordinator runs one shard at a time, validates its sealed outputs, and stops on the first invalid shard. An offline analyzer uses only anchor-time DEM and measurement traces.

**Tech Stack:** Python, Isaac Sim/IsaacLab in `isaac-lab-onix`, NumPy, PyTest, Matplotlib, Git worktrees, detached tmux.

**Spec:** `/home/kojima/.codex/attachments/86fbdc54-88b4-4129-aa00-6c2ceb693f9f/Pasted text.txt`

## Global Constraints

- Scene 112 envs; 7 terrains × 16 directions at 22.5° increments; radius 0.30 m; batch seed 20261007; difficulty 0.5.
- RL reset, origin z +0.35 m, XY/yaw-only overwrite, zero 6D action, dt 0.005 s, 200 steps, final40 settled window.
- Five or fewer selected contact views per shard; capacity 4096; active force floor 1e-6 N; four DEM frames; Actor shape [22].
- GPU 12288 MiB; anon+shmem 10 GiB; host available floor 8 GiB; per-shard timeout 900 s; heartbeat 90 s; total storage 500 MB.
- No candidate Path, follower, bridge, Outcome, second seed, model training, threshold change, or physical setup change.

## Review Focus

- A pose duplicated or absent from the interleaved manifest must fail preflight.
- A shard of six views or a changed radius, seed, terrain, or angle must fail preflight.
- A sealed shard with incomplete trace/DEM/Actor, SHA drift, residual PID, or native exit failure must halt the coordinator.
- A numeric exceedance remains measurement-valid but anchor-ineligible; it must not invalidate an otherwise complete shard.
- An interaction inferred only from unknown DEM cells must not enter the candidate-bank shortlist.

---

### Task 1: Deterministic manifest and source freeze

**Files:** Create `src/visualnav_transformer/eat_nav/radial_shard_inventory.py`, `tests/test_radial_shard_inventory.py`; create frozen JSON under the new output root.

**Interfaces:** `build_shards(rows: list[dict], size: int = 5) -> list[list[dict]]`; `validate_inventory(manifest: dict) -> None`.

- [ ] Write failing tests for 112 unique poses, 23 interleaved shards, 1–5 selected views, frozen row/column/seed/radius/yaw, and immutable manifest SHA.
- [ ] Run the targeted test and verify RED.
- [ ] Implement deterministic shard assignment and validation.
- [ ] Run targeted tests and verify GREEN.

### Task 2: Generalize the frozen simulator and one-shard lifecycle

**Files:** Modify `scripts/eat_nav_radial_five_pose_sim.py`, `scripts/eat_nav_radial_five_pose_run.py`, `src/visualnav_transformer/eat_nav/contact_host_gate.py`; extend `tests/test_radial_shard_inventory.py`.

**Interfaces:** Simulator consumes a 1–5-pose shard manifest, emits `sim-finished.json` with exact count 200; host emits `sha-ledger.json`, `status.json`, and `seal.json` only after native exit and all SHA/ownership checks.

- [ ] Write failing tests for shard size, dynamic frozen sensor-path identity, complete marker count, exact PID/start-tick cleanup, and seal rejection on incomplete files.
- [ ] Run tests and verify RED.
- [ ] Change only selection, source metadata, and host lifecycle; keep all frozen physics paths.
- [ ] Run targeted and existing lean-smoke tests and verify GREEN.

### Task 3: Sequential coordinator and resource gate

**Files:** Create `scripts/eat_nav_radial_112_run.py`; extend `tests/test_radial_shard_inventory.py`.

**Interfaces:** Coordinator reads a frozen inventory manifest and starts shards in order; status is atomic and resumable only across already sealed shards.

- [ ] Write failing tests for first-shard calibration, total storage cap, no continuation after invalid seal/resource/physical signal, and immutable shard roots.
- [ ] Run tests and verify RED.
- [ ] Implement the coordinator and host/container SHA preflight.
- [ ] Run tests and verify GREEN; run one calibration shard, then continue only if its frozen gates pass.

### Task 4: Outcome-blind anchor and terrain-interaction analysis

**Files:** Create `src/visualnav_transformer/eat_nav/radial_interaction.py`, `scripts/eat_nav_radial_112_analyze.py`, `tests/test_radial_interaction.py`.

**Interfaces:** Analyzer reads sealed traces and four DEM frames, returns per-pose measurement validity, sustained settled eligibility, profile class/start distance, shortlist, missing classes, aggregate tables and eleven figures.

- [ ] Write failing synthetic tests for flat, transition, graded, differential, and unknown DEM profiles; numeric exceedance classification; shortlist exclusion from unknown-only evidence.
- [ ] Run tests and verify RED.
- [ ] Implement analyzer with prospective, documented geometry thresholds; use no reward/Critic/Outcome.
- [ ] Run tests and verify GREEN; analyze all sealed shards after 112/112 complete.

### Task 5: Final verification and report

**Files:** Output root `final-decision.json`, `report.md`, figure and SHA ledgers; commit only source/tests/config/report, leaving large raw artifacts outside Git.

- [ ] Verify every shard seal, 112 unique trace sets, old-root SHA invariance, zero residual, final figure count, and output SHA ledger.
- [ ] Run relevant tests and record exact results.
- [ ] Report the permitted machine-readable decision and stop before candidate rollout or second seed.
