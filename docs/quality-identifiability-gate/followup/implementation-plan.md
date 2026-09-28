# Quality identifiability follow-up implementation plan

Approved user brief: attachment `1a8cd52a-f11d-42e4-8575-01f60d2cf87d/Pasted text.txt`. Existing gate artifacts remain immutable. Run in the existing `/tmp/eatnav-vnt` worktree and `isaac-lab-onix` Docker image. Follow-up runtime artifacts are under `/tmp/eatnav-quality-identifiability-gate/followup`.

## Task 1: Frozen audit and subsets

- Add `quality_followup.py` primitives to define complete windows, enforce same-episode/split and exact held-out count, and select nested B1/B8/B64 from the frozen 64 pairs.
- Add tests first for partial exclusion, episode/split failure, deterministic nested selection, duplicate-window and episode caps.
- Add worker audit command to emit counts, IDs, and pair manifest before training. Compare split/manifest/source hashes with prior gate.
- Verify with the Docker EAT test suite and audit smoke; save outputs in follow-up only.

## Task 2: Complete-only Gate A

- Add resumable scratch Probe 0/Q/shuffle training using existing deterministic model and unchanged settings, with complete-only train/validation data, aggregate validation checkpoint choice, nonfinite guards, and epoch checkpoint state.
- Add tests first for fixed q, resume equivalence and complete-mask filtering. Reuse the existing held-out evaluator with a complete-only option, recomputing mean trajectory from complete train.
- Verify smoke on a small complete subset, then stage full training/evaluation. Preserve partial-inclusive gate result as secondary.

## Task 3: Gate B ladder

- Add resumable scratch B1/B8/B64 training, 1000-epoch cap and predeclared loss/plateau stops, fixed noise bank and per-epoch optimizer/RNG checkpoints. Route low/high targets through canonical pair helper and assert identity.
- Add tests first for stop rules, resume, pair counts/targets and finite guards. Evaluate each stage using paired noise and unchanged original Gate B decision metrics.
- Verify smoke with B1/B8/B64, then run full ladder.

## Task 4: Detached orchestration and reporting

- Add host runner with seven required markers, atomic status updates, input/config/output SHA checks, stale-marker rerun, and container worker stage invocation. Science fail/inconclusive does not stop sequence; technical violations do.
- Add tests first for status atomicity, marker reuse invalidation, stage failure, and reporter completeness. Run detached as `tmux` session `eatnav-quality-followup`, stdout/stderr to host `runner.log`.
- Archive checkpoint/log SHA, copy machine reports and figures to the tracked follow-up docs folder, render final Markdown/JSON, run full tests and artifact integrity checks. No full collection or model/loss change.

## Rulings

- User's explicit instruction to implement and execute the attached design supplies authorization for this follow-up; no additional design approval is needed.
- The primary complete-only Gate A evaluator must filter train, validation and held-out consistently. The prior partial-inclusive result is read-only secondary evidence.
- Host runner owns status and markers; Docker worker owns model training. Container checkpoints persist across SSH loss and resume by epoch after a stopped worker. Host copies final artifacts at stage boundaries.
