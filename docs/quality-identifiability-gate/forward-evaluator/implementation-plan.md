# Phase B forward evaluator implementation plan

**Goal:** Test whether a logged candidate trajectory predicts future relative outcomes on the frozen aligned windows.

**Architecture:** One SHA-gated, detached runner audits the inputs, trains capacity-matched neural variants and simple baselines on train/validation, then evaluates the frozen checkpoints once on held-out. Actor-observation and latent variants are unavailable on this dataset and remain explicitly untested.

**Data:** Frozen split seed 123; 49,576 / 6,164 / 6,208 aligned windows. The 29 terminal endpoint windows remain excluded. Inputs are DEM history, relative goal, 8×2 local trajectory increments, and optionally privileged V0. Targets and normalization come from the existing train-only aligned labels.

**Execution:** `/tmp/eatnav-quality-identifiability-gate/forward-evaluator/` in detached `tmux` session `eatnav-forward-evaluator`, with `nohup` fallback. New code and reports go under `scripts/` and this directory; prior artifacts remain untouched.

## Tasks

- [x] Audit source hashes, timing, trajectory frame and delta representation; record actor/latent absence and Critic privilege.
- [x] Add schema and leakage tests, including zeroed-branch capacity equivalence.
- [x] Train mean, linear and G/GT/GV0/GTV0 baselines and evaluators with three fixed seeds, common schedule, scratch starts and validation-only checkpoint selection.
- [x] Evaluate each fixed model once on held-out; compute raw and standardized metrics, episode cluster intervals, terrain/common-support and seed results.
- [x] Reuse frozen Phase A geometry for cross-episode, disjoint matched-pair ordering. State that this is a ranking proxy, not a counterfactual test.
- [x] Write gate decision, runtime and artifact hashes; verify outputs without changing Phase A or starting any rollout.

## Review focus

- Anchor timing: logged pose and Critic are offset by one control step; use only the audited aligned windows.
- Future leakage: no future reward, pose, Critic or terminal field enters any encoder.
- Missing actor observation: never substitute twist or action for the exact policy observation.
- V0: evaluate only as privileged upper bound, never as deployable evidence.
- Held-out use: load it only after checkpoint and model selection are frozen.
