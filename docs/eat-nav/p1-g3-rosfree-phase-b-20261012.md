# P1 G3 ROS-free follower Phase B — offline implementation review

## Scope and decision

`eatnav_p1_g3_rosfree_direct_core_v1` is a **new experiment series**, independent of ROS G3 v2/v3/v4. This phase implements the direct C++ core connection and verifies it without starting Isaac, publishing a Path, operating the external router, training, or collecting new outcomes. The first runtime process remains separately gated. Existing sealed trials and the production ROS node were read only.

## Frozen offline preparation

- Manifest: `docs/eat-nav/p1-g3-rosfree-direct-v1-manifest.json`, SHA-256 `ae81bd56910b5a725e5dfee9df90834322d6d4ede84a0de128098fe74cdd2fea`.
- Persistent, **runtime-disabled** evidence root: `/home/kojima/eatnav-experiments/p1-g3-rosfree-direct-v1-prep/`. Its `stage-complete.json` binds the manifest, four-env direct assignment, and 197 staged source/input file SHA values. The assignment SHA is `6bf378d7288e023ed7460b79b6694d58a647d186e6be2af00255e7ce33776619`; the input ledger SHA is `4b053166d8bce25b4d1038087d6f15216de700b1c7076877bc39c2759e8955a2`.
- Simulator source SHA: `2225fb3892165598cb9c2a6a11a3e741d54bf237724235775381fd9b62bd6dd8`; host source SHA: `4e0a80ef6128118ef554ca41e6a2a48490a3ae4e3d3d508a476db44ee608806f`. Additional module and C++ core SHA values are in the manifest. The first assignment is `first-differential`, seed `2026101101`, env 0–3, one identical direct Path per env; Path SHA `4ec126da6c09e7ebc02f2febba35666ef57a334cec9745301554a13aa1d0155e`.
- `runtime_authorized=false` and the authorized process list is empty. The host rejects this preparation root. A runtime approval would require a **new** persistent attempt root and a newly sealed manifest with explicit authorization; no existing preparation or ROS evidence should be edited in place.

## Direct control path

`eat_nav_p1_g3_rosfree_sim.py` retains frozen Path assignment/geometry, `local_pose_from_root`, `path_in_map`, visible-state clone, shared four-frame DEM, native DEM diagnostics, paired observation noise, `CommandOverride`, Actor/Critic capture, `env.step`, same-episode gate, exposure/terminal capture and raw traces. After 20-step settling and pre-Path approval, it compiles the existing Phase A C++ core and constructs **four distinct `PureFollower` instances**. Each receives the branch tile-local pose and Path once at activation. The new control state machine enforces one call per env at each 50 ms logical stamp and the order `pose → core → command injection → Actor → physics → next pose`. A duplicate stamp, skipped stamp, wrong frame, wrong command injection, reset or episode discontinuity fails closed.

The existing environment generates the next observation inside `env.step` before the next pose-dependent Path command exists. The new adapter replaces **only** the generated-command terms in that already sampled Actor/Critic observation after direct injection. It uses the saved prior raw command term and the environment's latency fraction to keep the Actor's post-delay semantics; it updates the latency buffer for the following step. It does not generate a second noisy observation. Pre-noise, post-noise/pre-delay and Actor-delivered values are kept separately. Other observation terms and native contact/Critic differences remain measured; shared DEM is labeled `shared_experimental_history`, not native sensor history.

The ROS-free host starts only the simulator owner wrapper and GPU/resource monitors. It does not launch or require a ROS follower, bridge, TF service or Zenoh router. Core construction, Path activation SHA, exact stamp and causal order, injected command, Actor/Critic command slice, Goal zero command, exposure/hold/censor, same episode, material/pre-Path gates, native exit, cleanup, host/container SHA and host seal replace ROS transport readiness/acceptance evidence. A ROS acceptance record is never used as proof for this protocol.

## Verification

- Phase A read-only replay rerun: **16 existing output groups, 11,268 rows, zero command/diagnostic mismatches**. The original 3,360-command test and G2 replay tests remain included in `test_p1_follower_core.py`.
- Static/synthetic: `22 passed` for Phase A core plus new preparation/evidence tests; `6 passed` in the torch-enabled environment for the control adapter, including a prior-step latency-buffer test. The system-Python run has `12 passed, 1 skipped` because torch is unavailable there. `py_compile` and `git diff --check` pass.
- Offline stage SHA recheck: all 197 staged source/input files, manifest, simulator and host source match. No Isaac run or closed-loop comparison was performed.

## First runtime gate and ROS comparison

The first separately authorized runtime should use a fresh persistent root, this exact first-differential/seed-0 Direct-only assignment, 4 env and 1 process. Require 20-step settling, physical/shared DEM/Actor/material pre-Path gates, four independent cores, one Path activation, one core call per env per 50 ms stamp, 40 s episode cap with 600-step censor/40-step hold and maximum 640 steps. Reuse the frozen 600 s/process wall cap, 7,680 MiB device-total GPU cap, 256 MiB attribution gap, 6,442,450,944 B container anon+shmem cap, 1 GB active storage cap, 90 s heartbeat and 2 GB free-space floor. As a sizing reference only, the prior ROS direct pilot used 52.30 s, 5,195 MiB GPU peak, 3.51 GB anon+shmem and about 12 MB final storage; ROS-free runtime has **not** been measured. Stop at any protocol/resource/seal failure and do not continue to curved candidates automatically.

`config/eat_nav/p1_g3_rosfree_bridge_comparison_v1.json` (SHA-256 `42d89be4320784d1ca71ca5801c0015ea693e91ad16b8ee0ef8a45c22e90369c`) freezes the paired reference, exact identity fields, offline pose replay tolerances, and a **strict descriptive** closed-loop screen before runtime. Compare the new Direct-only run with the sealed ROS `first-differential / batch 0 / repeat 0` process at the same StateGroup/seed/env ID/Path, verifying both seals and input provenance. The command at the *same recorded pose and stamp* must match within `1e-6` (Phase A). Closed-loop comparison records command, Goal/arrival, recovery, Actor/action, XY, tracking, local/common-goal progress, contact and torque step by step. The frozen duplicate tolerances are a screen, not a statistical equivalence guarantee. The ROS and ROS-free outcomes remain separate protocol series even if the screen passes.

## Remaining risks

Isaac has not yet verified the new observation-term patch against the live 4-env manager, native core compilation inside the container, GPU ownership/finalization, or closed-loop trajectory parity. Native contact/PhysX variation can produce different motion even when visible state and Actor input match. ROS traces contained no measured recovery episodes, so **measured recovery parity is unverified**; synthetic boundary tests do not replace it. The direct-only gate cannot establish curved Path quality or evaluator identifiability.
