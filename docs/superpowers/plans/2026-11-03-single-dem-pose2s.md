# Single DEM Pose2s Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Train and independently assess a 40×40 single-frame, 4Hz DEM model of executed-candidate 2s relative pose, gating path selection on all 490 groups.

**Architecture:** Separate immutable collection manifests and raw continuous trials from causal input conversion and endpoint teachers. Share one schema/geometry implementation between training and inference; train only executed-candidate labels, report the full Cartesian evaluation grid including missing groups. Reuse the exact existing low-level Actor and nominal follower semantics with no recovery gates.

**Tech Stack:** Python3.10, NumPy, PyTorch, Isaac Sim/Isaac Lab in existing container, pytest, CSV/Markdown.

**Spec:** docs/eat-nav/single-dem-pose2s-spec-v0.1.md (user approved 2026-11-03).

## Global Constraints

- DEM 4Hz, one frame, 4m×4m, .1m, 40×40; no goal, camera, future observations or simulation-only extra features.
- 7 existing fields × levels0..4; each parent follows 5 candidates for 2s each without body/controller/sensor reset.
- Actor SHA b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d; fold mode0, dynamic control.
- train350/validation490/final490 planned parents, immutable role seeds, no role ancestry/terrain-hash leakage.
- 490 groups; maximum errors lateral10cm, longitudinal30cm, RPY10deg; >=5 valid endpoints from >=5 parents each; height is reported only.
- No path selection unless accuracy passes. First selector only predicted XY goal progress and predicted gravity tilt>60deg removal.
- Headless; dt .005s, 20Hz controls, no force push or overwrite legacy artifacts. Existing R6 job is preserved.

## Review Focus

- Stale or missing DEM and circular/frame transforms must never use a future acquisition or present padded terrain as observed.
- Terminations/auto reset at boundaries must not become a valid 2s endpoint from the next episode.
- Reverse/curved path errors must share the GT-projected tangent rather than separately projected predictions.
- Invalid predictions and missing groups must not vanish from error/coverage/pass rates.
- Parallel environment counts must not silently change parent terrain identities or role assignments.

---

### Task 1: Contract, seeds, geometry and nominal follower

**Files:** Create `src/visualnav_transformer/eat_nav/single_pose2s_contract.py`, `single_pose2s_geometry.py`, `single_pose2s_follower.py`; test `tests/test_single_pose2s_contract.py`.
**Interfaces:** `make_manifest(seed:int)->dict`, `validate_input(sample:dict)->None`, `endpoint_target(root0,root2)->dict`, `path_errors(pred,truth,q0,candidate_id)->dict`, `NominalFollower(path).step(x,y,yaw)->tuple`.
- [ ] Write/run failing tests for 1330 role-unique parents, candidate stage balance, 40x40 shapes, future/stale inputs, wrapped angles and reverse tangent signs, nominal follower without recovery.
- [ ] Implement the fixed schema, train/val/final role streams, exact 14 bank mapping and analytic arc projection, causal input checks and Python nominal follower port.
- [ ] Run `PYTHONPATH=src /home/ubuntu/venvs/isaaclab_py310/bin/python -m pytest tests/test_single_pose2s_contract.py -q`; expected all pass. Commit task.

### Task 2: Continuous collector and prototype benchmark

**Files:** Create `scripts/eat_nav_single_pose2s_collect_sim.py`, `scripts/eat_nav_single_pose2s_pipeline.py`; test `tests/test_single_pose2s_collection.py`.
**Interfaces:** consumes manifest parents and contract; produces sealed trial NPZ plus JSON metadata, raw root/state/command/DEM stamps, 5 endpoint masks, per-parent terrain identity and benchmark timings.
- [ ] Failing test pins 5 windows40steps, no next-episode endpoint, per-parent phase and seeds independent of batch size.
- [ ] Implement headless vectorized simulator using existing Actor/observation reconciliation, continuous action state, exact single DEM rays/masks, phase schedule and raw writes. Preserve old jobs and files.
- [ ] Run CPU tests, then 2/5/10env finite benchmark with fresh diagnostic parents. Expected seals, no boundary/reset mislabels, Actor22→6 parity and finite physical properties. Record measured fastest valid concurrency.
- [ ] Audit duplicate terrain hashes before crossing roles; if deterministic-cell disjoint split impossible, report blocker rather than pretending unseen terrain. Commit task.

### Task 3: Dataset generation and audit

**Files:** Create `src/visualnav_transformer/eat_nav/single_pose2s_dataset.py`, `scripts/eat_nav_single_pose2s_dataset.py`; test `tests/test_single_pose2s_dataset.py`.
**Interfaces:** `build_dataset(raw_paths,manifest,output)->audit`, fixed current-input arrays and body-relative pose targets, causal source IDs, split arrays.
- [ ] Write/run failing exact40steps, future input, early termination/reset, unknown-mask and cross-role duplicate tests.
- [ ] Implement timestamp-driven current input and teacher, per-reason masks, role/parent/terrain overlap audit, finite shapewise arrays and manifests. Preserve raw unaltered.
- [ ] Run tests, prototype dataset audit and roundtrip fixture. Run planned collection once benchmark/identity audits pass. Expected 6650 planned slots distinguished from actual valid labels. Commit task.

### Task 4: Model, train-only normalization, train and inference

**Files:** Create `src/visualnav_transformer/eat_nav/single_pose2s_model.py`, `scripts/eat_nav_single_pose2s_train.py`, `scripts/eat_nav_single_pose2s_predict.py`; test `tests/test_single_pose2s_model.py`.
**Interfaces:** `fit_normalization(arrays,train_ids)`, `SinglePose2s`, `predict(payload,inputs)->dict` with K1/K14 XYZ/6D/quaternion/RPY, schema/profile hashes.
- [ ] Failing tests for train-only stats, masked channels, candidate permutation, legacy schema rejection and relative/future pose composition.
- [ ] Implement CNN/state/path fusion, SmoothL1+SO3 losses, AdamW schedule/validation-only model choice and CPU reloadable checkpoint. CLI exposes raw validated fixtures and 14 candidate outputs.
- [ ] Run tests and finite small overfit diagnostic on train only; train actual dataset with frozen final test. Expected checkpoints/logs/normalization/validation metrics; no accuracy assertion yet. Commit task.

### Task 5: Independent 490-group evaluation and error diagnosis

**Files:** Create `src/visualnav_transformer/eat_nav/single_pose2s_metrics.py`, `scripts/eat_nav_single_pose2s_evaluate.py`; test `tests/test_single_pose2s_metrics.py`.
**Interfaces:** `evaluate(records)->report` + 490-group/per-window/trial/missing/coverage CSV + candidate/field/level tables, pass flag.
- [ ] Failing tests for exact490 rows, null missing values, invalid-prediction denominators, wrapped RP Y, max pass rather than mean, executed candidate only.
- [ ] Implement all mean/median/P95/max, proportions/joint rate, signed bias, height, motion/stage strata, independent parents, baselines and CPU/GPU timings.
- [ ] Freeze model/metrics before final opening. Evaluate final once. Expected honest report; if any group fails, diagnose conditions and propose data/model improvements, leaving selector gated. Commit task.

### Task 6: Accuracy-gated selector and simulator comparison

**Files:** Create `src/visualnav_transformer/eat_nav/single_pose2s_selector.py`, selector simulator CLI and tests only after Task5 passes.
**Interfaces:** input accuracy attestation + prediction outputs + goal; output chosen candidate or all-filtered command0.
- [ ] Write/run failing no-accuracy-attestation, 60deg filter, negative-score selection and deterministic tie tests.
- [ ] Implement only XY goal progress and gravity tilt filter. Run fixed independent matched A/B 700 runs with common follower/Actor and no other gates.
- [ ] Run tests and archive headless results/failures; expected new results without extrapolating real-robot safety. Commit task.

### Task 7: Delivery and approved uploads

**Files:** `docs/eat-nav/single-dem-pose2s-20261103/` reports/model index/commands/small inference fixtures.
**Interfaces:** versioned checkpoint bytes, SHA256, spec/source commits and reproducible fetch/inference/evaluation commands.
- [ ] Accuracy pass required for first release upload; preserve failed results as local honest reports if not achieved. No premature trained-ready assertion.
- [ ] Fresh whole-branch review, meaningful tests, model CPU reload/K14/permutation and checksum validation.
- [ ] Push new branch normally only at approved milestones. Verify remote HEAD and checkpoint availability. Second upload contains selector results only if actually completed.

## User-approved transition amendment (2026-11-04)

- User explicitly elects to proceed with the provisional model because overall means meet targets. Original 490-group maximum gate remains failed (323/490); do not rewrite the strict accuracy attestation.
- User then explicitly requests Task6 before GitHub, and defers all pushes until simulator comparison and available rosbag checks finish.
- Task6 authorization is a separate file bound to checkpoint SHA; scope exploratory_selector_and_offline_bag, not an accuracy pass or real-robot command authorization.
- First verify pure selection and causal input tests. Implement host/worker file handoff: simulator pauses at current observation, host runs actual FAST-LIO only over recorded sensor prefix then predicts K14 and returns chosen candidate. Do not substitute root velocity for LIO.
- Small A/B prototype before full 700 runs. Both conditions restart fresh with the same parent seeds, env order, initial/terrain setup and warmup. Check actual stage0 states/terrain hashes for pairing; report any mismatch rather than claiming identical conditions.
- Use 10 environments per simulator process initially to bound synchronous actual-LIO prefix replay overhead. Preserve dt=.005/control=.05/DEM=.25/replan2s. This is selection evaluation, not another training dataset collection.
- Goal forward3m and predicted tilt filter60deg, no other selection constraints. Failed/terminal runs retained; no rescue policies.
- Current old bags are incompatible20x20; user reports a new4x4m bag with non-FAST-LIO odometry. Await path/source, audit input compatibility, label odometry substitution as distribution mismatch; never relabel as FAST-LIO or formal matched-candidate accuracy.
