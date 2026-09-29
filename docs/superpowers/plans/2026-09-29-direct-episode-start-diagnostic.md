# Direct episode-start replay diagnosis and conditional seven-terrain continuation

**Authority:** User instruction of 2026-09-29 approving State 42 bridge/duplicate gate and requiring direct-start comparison before any candidate rollout. The user supplied the design and conditional stop rules; no separate design approval is needed. Execute inline, sequentially, in the existing isolated `/tmp/eatnav-vnt` worktree.

## Scope and immutable inputs

- Leave the earlier seven-state selection, State 42 diagnostic, candidate bank, model checkpoints, follower controller, and prior output unchanged.
- No candidate Path rollout until direct replay comparison passes for both short state 42 (episode 0) and boxes state 7 (episode 3).
- Use only the fixed nominal Path for the comparison. Do not use same-process snapshot restore.
- Direct mode may advance per-env reset streams through earlier episode counts without physics steps. It must not read or set candidate outcomes or copy an anchor state from recorded data.
- A direct mismatch stops all rollout; the alternative is a new cost-aware selection v2 from complete-only train, without any candidate results, and then an approval hold.
- Formal 56 train/20 validation, Phase B Evaluator, NoMaD, Phase D/E, full collection, real robot remain stopped.

## Frozen diagnostic method

1. Full mode uses the established 100-env runner from global step 0, with the original actions and actual `simple_trajectory_follower` bridge. Run one fresh nominal trace for state 42 and state 7. Existing State 42 clean retest may serve as full reference only after source/Path/checkpoint hashes match.
2. Direct mode creates the same 100-env terrain scene with base seed 123 and the same checkpoint. It calls the normal initial reset, advances the **target env only** through `episode_id` reset operations with no intervening physics, then replays original recorded actions only from the target episode's first global step through its anchor. For state 42, episode_id=0 and direct mode is the no-skip control. This is a deliberately limited seed/episode-only reconstruction: global RNG and command-stream state are not copied from the full run. Their differences are diagnostic signals.
3. Both modes capture the complete anchor via the existing `detailed_anchor` function, including root/joint state, four DEM frames, command, action/observation delay buffers, Actor pre-noise, post-noise/pre-delay, post-delay input, latent/action, Critic observation/value. Both then run the **same** 40-step nominal Path through the real follower, RL Actor and physics. Save command/action/XY/contact/DEM/reward/termination per step and all source hashes.
4. Compare first the anchor signals in a fixed order, then the 40-step series. Exact match is required for identity, tensor shapes, DEM masks and hashes, Path, episode-local index, termination kind. Numeric limits are fixed before either run: root position 1e-3 m, orientation 1e-3 rad, joint position/velocity 1e-4, DEM height RMS 1e-3 m, command/action/observations/latent/Critic value/reward 1e-4, delay state numeric 1e-4, XY RMS 0.0105 m, contact force 1e-3 N. Any nonfinite value or mismatch fails direct replay. Report first signal and step with actual maximum/RMS, even when later signals also differ.
5. If both states pass, resume the **same** frozen seven states and five Paths using direct mode. Begin with boxes #7: three fresh nominal repeats then five frozen candidate Paths. Analyze runtime, duplicate RMS (<=0.0105 m), effect, command divergence, termination. Stop before other terrains if unexpected runtime, duplicate failure, or follower timeout. Otherwise continue the existing seven-state order and stop before 56/20 formal rollout.
6. If either comparison fails, do not run any candidate. Create selection v2 from `CompleteOnlyView(EATDataset(..., split='train'))`. Pre-fixed eligibility: goal radius >=0.30 m, `local_step <=80` (the existing feasible-set maximum), one episode per state, deterministic SHA256 rank seed 20260929; one state per terrain. If a terrain has no eligible state, mark `NA / no state within runtime budget`. Save IDs, full ranked eligibility, prefix histogram and source/input hashes. Do not execute v2 rollouts; await approval.

## Files and gates

- New `scripts/eat_nav_direct_episode_inputs.py`: prepare episode-local action windows and hash-checked metadata; tests `tests/eat_nav/test_direct_episode_inputs.py`.
- New `scripts/eat_nav_direct_episode_run.py`: full/direct fresh-process simulation with common capture and nominal bridge path; tests for mode/step contracts where runnable outside Isaac.
- New `scripts/eat_nav_direct_episode_compare.py`: pure ordered comparison and first-divergence report; tests `tests/eat_nav/test_direct_episode_compare.py`.
- New `scripts/eat_nav_direct_episode_stage.sh`: detached tmux, atomic per-stage markers/status, source SHA, logs, no false success on Isaac cleanup exit zero.
- Conditional new `scripts/eat_nav_cost_selection_v2.py`; tests `tests/eat_nav/test_cost_selection_v2.py`; separate output root.
- Output root `docs/quality-identifiability-gate/feasible-set/direct-episode-start/` and container mirror `/tmp/eatnav-quality-identifiability-gate/feasible-set/direct-episode-start/`.

## Task order

1. Freeze comparison config, source SHA and local windows for state 42 and boxes #7. Confirm episode boundary and complete source actions.
2. Implement pure comparison, test RED→GREEN, and establish immutable numeric gates.
3. Implement full/direct runner and contract tests. Verify State 42 short control in separate fresh processes.
4. Execute boxes #7 full/direct nominal comparison in detached tmux. Preserve all completed stages and failed attempts; do not run candidates before comparison result.
5. Conditional pass: run boxes #7 duplicate + five candidates in direct mode, analyze, then remaining seven-state pilot if its gates pass. Conditional fail: freeze v2 selection and stop.
6. Save report, runtime/storage, plots where supported, hashes, and final status. Verify no forbidden processes ran.

**Review focus:** seed-only direct mode does not silently load recorded start state; `episode_id=3` reset-only calls do not count as prior physics replay; old checkpoint/Path/bank SHA remain fixed; first divergence is reported before aggregates; candidate rollout process cannot start on comparison failure; v2 cannot overwrite earlier selection; empty terrain support is NA, not backfilled.
