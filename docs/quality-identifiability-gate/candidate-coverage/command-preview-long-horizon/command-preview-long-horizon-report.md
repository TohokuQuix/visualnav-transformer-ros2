# Command-preview and long-horizon coverage: stopped pilot report

All inputs were frozen before analysis. Stage A used the existing 91 candidates; Stage B started only the approved 20-run schedule and stopped after run 8 when the second state duplicate gate failed. No unapproved candidate, training, formal validation or real-robot control was started.

## Stage 0

- Branch exp/eatnav-candidate-coverage, starting HEAD cb6d1beab003c473cd578afbf2e8b65f8476792f, starting worktree clean.
- NoMaD/GT checkpoint, train-only scaler, pilot Path/trace/outcome and production follower source/binary/config SHA values are in source-provenance.json.
- Production follower: lookahead 0.5 m, goal tolerance 0.2 m, turn-in-place threshold 0.6 rad, linear speed 0.3 m/s, angular limit 1.6 rad/s; control 20 Hz; GT 8 points at 0.25 s.

## Stage A: online-available follower command preview

- Preview used only frozen Path, anchor pose/yaw, production follower parameters and a 0.05 s kinematic pose update. Recorded commands and outcomes were loaded only after preview markers were saved.
- All 91 candidates completed in 7.5 min. Seven production first-command fixtures matched within 1e-5. State-macro same-command-cluster rate was 0.275, below the 0.75 gate.
- Closed-loop was no worse than open-loop in 5/7 states. First command-threshold exceedance had median step 16.
- Frozen GT(C-hat): top-1 0/7, top-3 0/7, pairwise 0.488, oracle regret 0.120.
- Reference GT(P), GT(C), GT(R) pairwise were 0.467, 0.669, 0.534. Recorded C and realized R remain privileged diagnostics.
- Command Preview Gate: fail. Current preview and frozen Evaluator cannot be used for online candidate selection.

| State | same cluster / 13 | GT(P) pairwise | GT(C-hat) pairwise | GT(C) pairwise |
|---|---:|---:|---:|---:|
| coverage-000 | 0/13 | 0.475 | 0.131 | 0.754 |
| coverage-001 | 0/13 | 0.629 | 0.514 | 0.514 |
| coverage-002 | 1/13 | 0.233 | 0.753 | 0.808 |
| coverage-003 | 0/13 | 0.507 | 0.627 | 0.893 |
| coverage-004 | 12/13 | 0.522 | 0.362 | 0.420 |
| coverage-005 | 0/13 | 0.471 | 0.324 | 0.662 |
| coverage-006 | 12/13 | 0.437 | 0.704 | 0.634 |

## Stage B: stopped at duplicate gate

- Completed 8/20 runs in 21.5 min; run storage 20.2 MB, below the 90 min / 250 MB caps. Simulator CPU RSS peak 5502 MiB and GPU used peak 5549 MiB.
- One approved retry was used for long-005 after a Kit crash at startup with uptime 0 s and no simulation step. The original log and SHA remain in startup-crash-review.json and runner.log.
- coverage-001 nominal 3 repeat XY RMS was 0 m. Its 2 s and 6 s exploratory comparisons are below; the bank winner is constraint-invalid, and one state does not establish feasible coverage.
| coverage-001 | nominal Delta V | NoMaD Delta V | bank Delta V | bank minus NoMaD |
|---|---:|---:|---:|---:|
| step 40 | 0.949 | 0.999 | 1.018 | 0.020 |
| step 120 | 1.024 | 0.950 | 1.042 | 0.092 |

- coverage-003 duplicate XY RMS 0.03526 m exceeded the frozen 0.0105 m threshold. The first difference was follower command at step 39. One nominal repeat reversed linear speed from +0.293 to -0.296 m/s after a production Linear stuck detected event; the other two repeats matched each other for 120 steps.
- The follower reported commanded distance 5.929 m, actual 0.165 m, ratio 0.028 against its 0.25 threshold. The 20 s motion window uses ROS elapsed time; a scheduling boundary with slow simulator progress is the leading explanation, not a proven unique cause.
- Candidate comparisons for coverage-003 are invalid. coverage-004 and coverage-006 were not started. The planned four-state long-horizon coverage gate is measurement-invalid; no generalization or formal pass is claimed.

## Stage C: proposed work only

- First restore duplicate reproducibility by measuring ROS/simulator clock and follower motion-window state without changing the production control law. The proposed three-run diagnostic requires new approval.
- Separately, since Stage A fidelity failed, design an offline state ablation using current body velocity/yaw rate, roll/pitch, controller state and Actor latent. Future simulator information remains excluded from the predictor.
- No further rollout, retraining, Phase D/E, full collection or real-robot control was started.

## Execution and integrity

- Stage A command: detached scripts/eat_nav_command_preview_detached.sh; frozen GT inference: scripts/eat_nav_command_preview_score.py.
- Stage B command: detached scripts/eat_nav_command_preview_long_horizon_detached.sh; startup retry was source-SHA reviewed and limited to the same run.
- Raw traces, checkpoints and large logs remain outside Git. Small JSON, figures, config, source, tests and this report are saved in the dedicated branch.
| Artifact | SHA-256 |
|---|---|
| source-provenance.json | 7092cd61d1660a81061873705a59ce345c6f94209d9e1721baf4851ce0323e70 |
| preview-command-fidelity.json | dcb7b58a873690ab6c492e61bcb6e0006ce9f4f1dd932671a39e3b67b63a4ec1 |
| preview-evaluator-results.json | 7ef61a854c8abc9ddaac1ab2728501bd6d642254308d234ee859cf45dfef8071 |
| preview-gate.json | 8b67a8383337aa49f0c362a6608f09d6e1b489b13aa9f9db45cc9748f7a66171 |
| long-horizon-run-identities.json | 1148d8a48002f221a9414ad1c1941b7a3ad22fda0662bd6049cfe2a816c9d9ab |
| duplicate-failure-diagnostic.json | 7403c69e3c81b1e20e9d8f28be8f42a3a9725389d7047082cf40081cbc6a13bf |
| long-horizon-results.json | 62973d2dba245dc5b0e22f5cadafa4e677da049c783b7f13526e1a82f258cf8a |
| long-horizon-decision.json | 18747c626bae213dc1cdefc3a99aee22ffc00ddd588f95c9bd3c68f1cbccb512 |
| next-design-decision.json | 3b0c0c9c39f0186b333c67105c98d11b4e56530bc11c66053807eeab543e38fc |
