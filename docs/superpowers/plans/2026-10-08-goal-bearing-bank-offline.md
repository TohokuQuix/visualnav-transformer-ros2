# Goal-bearing Candidate Bank Offline Review Plan

**Goal:** Freeze eight supplied R3 anchors, generate and audit two seven-Path banks at five goal bearings, then propose a bounded rollout without running one.

**Inputs:** The first-seed inventory, eight-pose pilot and transition reserve; the second-seed 448-pose scan and eight-pose pilot. Read only from the dated `/home/kojima/eatnav-backups/2026-10-08-pre-shutdown` backup. Base commit `ecadcd3fa9f7cc57ecab092777f387d14d8d98d9`.

**Outputs:** Code/tests and small review artifacts on `exp/eatnav-goal-bearing-bank-review`; all 560 candidate definitions, dense geometry and SHA ledger under a new `/tmp/eatnav-goal-bearing-bank-review-20261008` root.

**Constraints:** Offline only. No simulator, follower, bridge, router, Outcome, reward, Critic, Evaluator, training or robot control. Do not modify prior selection, eligibility or thresholds.

## Tasks

- [ ] Validate source files and seals, select exactly one physically supplied anchor per class per seed by frozen rank, and freeze pose/DEM identity before generating Paths.
- [ ] Add deterministic cubic Bézier Bank A and Bank B generation, goal/map transforms, eight-waypoint arc resampling, hard geometry checks and SHA tests.
- [ ] Add offline DEM profile/corridor and production-config first-command diagnostics, with known/unknown and geometry/command claims kept separate.
- [ ] Screen only the given amplitude and offset pools, choose one global three-level set per bank using the specified outcome-blind priority, and generate all 560 definitions.
- [ ] Produce bank/bearing/class/seed comparisons, 12 figures, split contract, prospective rollout gates, process/resource forecast, risk register, decision and SHA ledger.
- [ ] Run focused tests and reproducibility check, review artifacts, then commit code and small reports while retaining raw definitions outside Git.
