# P1 G3 Path crossover v1: execution preparation (offline)

## Scope and frozen inputs

Protocol `eatnav_p1_g3_path_crossover_v1` tests whether Direct, Left +20°, and Right −20° commands produce distinguishable motion, progress, tracking, contact, and joint-load responses. A difference in local endpoint arrival or tracking alone is **not** navigation-quality evidence because the endpoints differ. Common-goal progress and body response remain separate outputs. No Isaac runtime, Path rollout, Evaluator training, or full collection was performed for this preparation.

- Assignment: [`p1-g3-crossover-v1-assignment.json`](p1-g3-crossover-v1-assignment.json), 18 sequential processes and 72 trials; 2 StateGroups × 3 seeds × 3 periods × 4 env. For every StateGroup/seed/env ID, Direct, Left, Right occur once. Period execution order cycles across seeds. There are 12 Direct duplicate pairs in periods 0/1; period 2 has none.
- Contract: [`p1-g3-crossover-v1-manifest.json`](p1-g3-crossover-v1-manifest.json), SHA-256 recorded below. It fixes checkpoint, follower, bridge, Path and source-input hashes, 20-step settling, visible clone, shared DEM v2, paired noise, exact-stamp 20 Hz control, 40 s episode, 600-step censor and 40-step arrival hold (640 maximum), existing numeric gates, and resource caps.
- The source of the previous 48 Direct trials remains `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-direct-repeat-v1-attempt-0/`. Its 12 sealed batches, raw traces, old manifest, and aggregate were read only. The earlier canonical material reanalysis in [`p1-g3-path-crossover-v2-analysis.json`](p1-g3-path-crossover-v2-analysis.json) remains a separate analysis version.

The assignment uses the already frozen source-contract Path SHAs: Direct `4ec126da…`, Left +20° `01c33ca3…`, Right −20° `b0a72176…`. All three are 8-point Paths; the common goal is the source-contract 0.75 m forward point. The first future pilot is `batch-first-differential-0-period-{0,1,2}` in that order, 3 processes/12 trials. It counts toward the 72 only if each process is protocol-valid. A protocol change creates a new version, with no mixing.

## Pre-Path and live-material gate

Each process starts four production-equivalent exact-stamp followers, performs the fixed 20-step settling, clones the visible representative state, and distributes the immutable env-0 four-frame DEM as a shared experimental input. Native DEM and native contact/Critic observations remain separately identified. No extra physics step occurs after clone and before Path publication.

The simulator saves `live-material.json`, `prepath-crossover-snapshot.json`, and `prepath-ready.json`. The host checks their SHA and the within-process physical/shared DEM/Actor gates **before** writing `prepath-approval.json`. It compares each later period against the sealed period-0 snapshot and requires the immediately preceding period to be sealed. Any strict mismatch stops before Path publication.

Strict cross-period fields are StateGroup, seed, canonical terrain mesh/material and effective robot PhysX-material hashes; each env ID's root state, joint position/velocity, applied torque, Actor pre-noise, post-noise/pre-delay and actual post-delay input, Actor action, noise sample, command, four shared-DEM frame SHAs, episode length and reset count. Numeric fields require finite values and exact equality. Native contact current/history and Critic observation/value are saved and compared as **diagnostic differences**. They are not asserted equal and never silently promoted to Path effect. The old Direct duplicate post-Path gate is unchanged and is applied only to identical Direct pairs. Its Fail is a valid repeatability outcome, not a protocol failure.

The live audit reads the composed terrain collision-mesh physics-material binding and material attributes, plus all four robot envs' collision bindings and material prim attributes. It compares terrain's canonical live value fingerprint with the declared value. It also reads the effective robot PhysX per-shape material tensor, requires the frozen startup `(static friction 0.8, dynamic friction 0.6, restitution 0.0)` within `1e-5`, and fingerprints it. USD bindings and the effective tensor are both saved; the tensor is the effective robot-friction source if startup randomization overrides authored USD values. An unresolved binding/attribute/tensor is a pre-Path failure. Actual USD material availability and robot tensor shape remain **runtime unverified**; no candidate rollout may proceed if these values cannot be read.

After approval, the simulator publishes each env's assigned Path once. The host verifies Path SHA/role, four follower readiness, clock/TF/command/bridge/physics stamp order, same episode, arrival/censor and hold, trace completeness, native exit, GPU attribution, memory/storage/time/heartbeat caps, cleanup, host/container SHA, and seal. Diagnostic errors fail the host even if the shell exits 0. The seal is created after writers close.

## Offline analysis contract

The analyzer reads only 18 sealed protocol-valid batches. It keeps arrival time separate from right censoring, and distinguishes pre-arrival exposure (`exposure_steps`, tracking/recovery/contact/torque) from the fixed first-40-step body-load window. It reports 2-second local endpoint progress and 2-second common-goal progress separately. Geometry descriptors include Path length, maximum absolute discrete curvature, and endpoint-to-common-goal distance. Candidate−Direct differences are signed within the same StateGroup/seed/env ID. StateGroup summaries aggregate the four env differences within a seed before computing across-seed variation; env trends remain descriptive. Six seed bundles, three per StateGroup, are the independent conditions. The 48 old Direct trials remain a separate secondary repeatability baseline.

## Static verification and frozen SHA

`PYTHONPATH=src:scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kojima/anaconda3/envs/tinyvla/bin/python -m pytest -q` on the selected P1 G1/G3/runtime/crossover tests passed **126/126**. New synthetic tests cover role/period balance, strict versus diagnostic pre-Path signals, nonfinite input, effective material mismatch, host approval before publication, geometry and common-goal metric separation. Python compilation passed. The stage tool built 18 offline input roots under `/tmp/eatnav-p1-g3-crossover-v1-final-static-staging/`; every batch input ledger and listed file SHA was rechecked. This is a static staging artifact, not a runtime output. Manifest SHA-256: `6a87ec0f05fb334b7968435d6c7db871ebe0ed2c5786ecb2d2be616bcba80162`; assignment SHA-256: `46ad229c69bb2024cb4b87d183204d93a1ae14bb3fdfdda00a550aa6733624c1`.

Frozen source SHAs are in the manifest. The source files are `eat_nav_p1_g3_crossover_sim.py`, `_host.py`, `_stage.py`, `_analyze.py`, and `p1_g3_crossover_runtime.py`; the original direct-repeat runner and production follower were not changed.

## Approved future pilot procedure and resource planning

The future pilot requires separate approval. At that point, stage the frozen assignment/manifest into a **new** persistent root such as `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-crossover-v1-attempt-0/`; confirm that its container mirror is `/workspace/isaaclab_onix/.eatnav-p1-long-horizon-v1/g3-crossover-v1-attempt-0/`. Use the external router's current read-only PID/start-tick/ZID/log directory. Run the host for `first-differential`, batch 0, execution indices 0, 1, 2 **sequentially**; verify each seal before starting the next. Do not rerun a protocol-valid process. The next 15 processes are outside that pilot's approval.

Example future invocation, after stage, mirror and router verification:

```bash
PYTHONPATH=src:scripts /usr/bin/python3 scripts/eat_nav_p1_g3_crossover_host.py \
  --root /home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-crossover-v1-attempt-0/batch-first-differential-0-period-0 \
  --remote /workspace/isaaclab_onix/.eatnav-p1-long-horizon-v1/g3-crossover-v1-attempt-0/batch-first-differential-0-period-0 \
  --anchor-id first-differential --batch-index 0 --execution-index 0 \
  --router-log-dir /path/to/current/external/router/logs
```

Historical 48-Direct processes took 49.4–90.2 s each, with peak GPU device-total 5,195.25 MiB, container anon+shmem 3.57 GB, and active storage 24.58 MB/process. The G2 600-step process took 306.0 s and 63.8 MB. The 3-process crossover pilot is therefore budgeted at approximately **3–16 minutes** before startup/cleanup margin, approximately **4–20 minutes** with 20% margin. Estimated final storage is **0.04–0.20 GB**, plus live logs and copies; it must be re-estimated from period 0. Hard caps remain 600 s/process, GPU device-total 7,680 MiB, attribution gap 256 MiB, container anon+shmem 6 GiB, active storage 1 GB/process, heartbeat stale 90 s, and persistent free floor 2 GB. These are planning estimates, not runtime measurements.

## Unresolved runtime questions

The composed live USD material attributes, robot effective material tensor, exact 4-follower behavior under mixed Paths, cross-period state equality, and actual pilot resource use have not yet been exercised. If any fails, stop at the corresponding gate and preserve the new attempt; do not infer candidate quality. Local execution-quality identifiability and eventual navigation quality remain unassessed until valid Path outcomes exist.
