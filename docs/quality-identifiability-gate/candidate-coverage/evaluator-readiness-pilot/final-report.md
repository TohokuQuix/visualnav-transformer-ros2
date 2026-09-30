# Counterfactual Evaluator readiness pilot

Train-only, seven-terrain, fresh-process, exact-stamp diagnostic. Each of 7 states has 3 nominal repeats, 8 frozen NoMaD seeds, and the unchanged 5-path bank. The 7 anchor checks are separate from 112 rollouts.

## Decisions

- **candidate_coverage:** `insufficient`
- **relative_outcome_dataset:** `ready_for_scaled_collection`
- **safety_dataset:** `insufficient_failure_coverage`
- **evaluator_input_schema:** `incomplete`
- **next_action:** `measurement_remediation`

## State-level evidence

| Terrain | NoMaD valid precision | NoMaD realized modes | valid 6s ΔV range | feasible-union Recall@8 |
|---|---:|---:|---:|---|
| boxes | 1.00 | 1 | 0.17431640625 | True |
| hf_pyramid_slope | 1.00 | 1 | 0.1878509521484375 | True |
| k_rail_field | 1.00 | 2 | 0.23925018310546875 | True |
| k_rail_step_field | 1.00 | 1 | 0.3431053161621094 | False |
| pyramid_stairs | 1.00 | 1 | 0.34039306640625 | True |
| pyramid_stairs_inv | 1.00 | 1 | 1.0276603698730469 | True |
| random_rough | 1.00 | 1 | 0.11051559448242188 | True |

## Measurement protocol

The first 100 runs use the frozen original protocol. The remaining 19 use a fresh bridge and eight fresh followers per run after three outcome-blind equivalence replays passed. The old command-delivery failure remains mechanistically inconclusive; its observed boundary is between follower publish return and bridge adoption. Failure attempt 0, fault injection, diagnostic replay, and equivalence replays are excluded from all outcome estimates.

The feasible union contains nominal, constraint-valid NoMaD K=8, and constraint-valid bank candidates. Recall@K compares NoMaD prefixes against this union with ε=0.01. Seven terrain states are descriptive support, not formal generalization.

P/C/R two-second OOD counts over frozen train P95: {'P': 12, 'C': 0, 'R': 8}. The Phase B GT(P) model is frozen; its results are secondary and never selected candidates. Its macro metrics are in `final-decision.json` and raw score artifact `/tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol/unified/analysis/frozen-evaluator.json`.

Runtime: 5.81 h, median 164.9 s/process, peak GPU 5551 MiB, saved 1086.6 MB. Raw traces, callback logs, and run markers remain under `/tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol/unified` with SHA ledger in `artifact-sha-ledger.json`.

### Limits and next approval

Input readiness reflects the fields actually captured at the anchor. The current trace does not include a separate joint-effort field, so `incomplete` means that deployable schema parity remains unproven. Early termination censors six-second continuous labels but remains a safety label. The seven states are all train split; no validation, held-out, retraining, production follower edit, full collection, or real-robot test was performed.

A scaled train/validation/held-out counterfactual collection needs a separate design and approval. At the measured sequential median, 70 states × 16 rollouts would require roughly 51.3 h before startup/retry reserve; storage scales to roughly 10.9 GB. A candidate split is 50 train, 10 validation, 10 held-out states with terrain-seed and episode separation, contingent on resolving the stated gate results and validating parallel throughput.

Reproduce status: `cat /tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol/unified/status.json`; `cat /tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol/unified/postprocess-status.json`; `tail -n 100 /tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol/unified/runner.log`; `tmux ls`; `tmux capture-pane -pt eatnav-readiness-full`.
