# Feasible coverage and Evaluator transfer: train-only diagnostic

All Stage 0–3 analyses used the seven frozen train states and 91 existing 40-step rollouts. No new simulator rollout, training, formal validation, or real robot control was performed. Candidate pairs share state; the statistical unit is seven states.

## Source and execution

- Source audit: 91 candidate ID/seed/Path/trace links; NoMaD and three GT checkpoint SHA values matched. See `source-provenance.json` and `candidate-identity-table.jsonl.gz`.
- Phase B held-out GT reproduction: max absolute reported-metric difference 0 (tolerance 1e-05). The held-out split was used only for this reproduction and distribution comparison.
- Teacher search: 49,598 complete-only train windows; no validation/held-out threshold choice.
- Commands: `PYTHONPATH=src:.:scripts /usr/bin/python3 scripts/eat_nav_feasible_transfer_stage0.py`; Stage 1A same with `...stage1.py`; Docker `/isaac-sim/python.sh` scripts `eat_nav_feasible_transfer_teacher.py`, `eat_nav_feasible_transfer_phaseb_reproduce.py`, `eat_nav_feasible_transfer_evaluator.py`, `eat_nav_feasible_transfer_ood.py`; host `eat_nav_feasible_transfer_finalize.py`.
- Observed GPU peak: Phase B reproduction 97.5 MiB, Evaluator transfer 9.9 MiB, OOD 32.6 MiB. Stage 1A elapsed 2.0 s, teacher 29.0 s, Evaluator 28.6 s, OOD 28.5 s.

## NoMaD short-horizon coverage

- State-macro joint observed-screen precision: NoMaD 0.964; fixed bank 0.771. These are 2 s screens, not proof of long-horizon physical feasibility.
- Observed-screen feasible command-mode Recall@8: 0.869 state macro; realized-mode Recall@8: 0.929. Per-state numerators, denominators, and IDs are in `nomad-feasible-coverage.json`.
- Four bank-winning states: 0 have a bank winner that is simultaneously joint-screen valid and command/realized distinct from nearest NoMaD. `hf_pyramid_slope` has a distinct command winner but it fails the existing constraint/OOD screen; three other bank winners share both modes with nearest NoMaD.
- K32 adds a planned Path cluster in 2/7 states, without rollout evidence for its feasibility. All four bank winners have independent-episode teacher shape/direction matches.

## Evaluator transfer

| Input | top-1 / 7 | top-3 / 7 | state-macro pairwise | state-macro oracle regret |
|---|---:|---:|---:|---:|
| GT(P) | 0 | 2 | 0.467 | 0.305 |
| GT(C) | 3 | 3 | 0.669 | 0.035 |
| GT(R) | 1 | 1 | 0.534 | 0.100 |

P is planned Path, C integrates recorded follower commands, and R is the realized simulator trajectory. GT(R) uses privileged outcome information and cannot select online. GT(C) is a diagnostic, not an approved selector. The same original GT checkpoints and train-only target scalers were used for all three.
- Frozen train action p95 OOD: P 8/91, C 0/91, R 2/91. OOD alone does not explain all P ranking errors.
- Phase B held-out goal/end-point exact-match fraction: 0.802; pilot bank 35/35 exact, NoMaD 0/56 exact. Endpoint substitution changed P score spread little in most states.
- GT(P) top-1 0/7, worse than nominal fixed 2/7 and goal/end-point distance 2/7. Per-state baselines, calibration, uncertainty and generator bias are preserved in JSON.

## Gates and next step

- Gate N: **inconclusive**. High 2 s precision and command variation do not establish long-horizon feasible coverage or a useful constraint-valid bank mode that NoMaD misses.
- Gate E: **fail**. Exact Phase B reproduction rules out the audited implementation mismatch, but planned Path ranking remains below chance and below simple baselines. Do not use this Evaluator for online candidate choice.
- The proposed 20-run, 120-control-step long-horizon experiment is described in `decision.json`; it has **not** been started and requires separate approval. Formal validation and model retraining also remain stopped.

## SHA-256

| Source artifact | SHA-256 |
|---|---|
| source-provenance.json | `c75493cf93e69aaa484ecb25ce48f1a50149005a340f225134c9a357d39d62aa` |
| nomad-feasible-coverage.json | `a44c9bddf9254a445eec5067db0adee82eb930b9c2c8ec23742282d0329d0f9c` |
| nomad-oracle-regret.json | `f2767e3e8a6ee38330a79d13f515fcc4e480cb986310eb3b738ea363c0ab05da` |
| teacher-support-analysis.json | `843296440ae2ad6e5ef1fc8223f62c954dd6a0be3e566a874be00ffe818f40e4` |
| k32-offline-analysis.json | `65a9fe97d5d78b260d72996395ac0b48cb6faa863593c1bbdd666addde3bccae` |
| phase-b-reproduction.json | `8ecb985650b4c676cc53c993e29e03f666752e87ce0457319fedd921c4fa229f` |
| evaluator-input-audit.json | `c1cc74724e2fcc9aa95f6e24ec1ec97bd625077913c770f6ca64c56a58d08420` |
| planned-command-realized-comparison.json | `108b3517dfa0c92c4443fc4d50a7ba442953990e2dbe8e7c9564ac4e2cdeb2d8` |
| evaluator-counterfactual-results.json | `1f6600809401cc7879060edec4141aa28653ff8f5cc17916de132f9ed61b474e` |
| trajectory-ood-comparison.json | `a5770cb0b53ddd02563dc8d2a7e92d3f9339ccbc3f1c7fd3ee7cc545822895e4` |

| Derived artifact | SHA-256 |
|---|---|
| cause-classification.json | `c1110d36ead84ab8cbd69560ddb48ba2a90e37693cfeeda9bc011b2ed8506ee1` |
| decision.json | `fb0356040c108b83e62a38d9b3d8c8f684cd384e70ab1e2927f98431639b3d67` |
| evaluator-strata-and-uncertainty.json | `5e99e64b27d3cbb200028f158f2204e2f30f15f7c3d91e8be4f33300a9e56fb0` |
| figures/bank-winner-nearest-nomad.png | `848312b6cd06c86ad54fbc43a8497b0ce9455afe4f9c7592e9a9f2d65a12fe26` |
| figures/feasible-mode-recall.png | `67533c3acb65582bbc9d3a6a0cb100e64b3786ff46b4243cbf9a5232c4b9960b` |
| figures/feasible-precision.png | `cde7446ba901e931e7e73df29eee4ea855914a1352758b400b7d8061c471e51b` |
| figures/generator-bias.png | `bd575ee2f2b26a571e9b12fa9fcf236cfa2e778e7e67310d72729f276d74e5ce` |
| figures/gt-pcr-performance.png | `a219c2298355ef21402bf9d9f126ed2cd3cb6c315c68bb8336819a92d0640ac8` |
| figures/oracle-regret.png | `1ed572c0bcb981636d74361a41b6e8aaae2a247ee547880c90f797f145eb1647` |
| figures/phaseb-pilot-goal-endpoint.png | `8d095945b0e91895a7627b958a92545333542f9db639ae4ab7e5b2e666b9bdc7` |
| figures/three-space-clusters.png | `4d367947ed0448d72219a7b6c7f8df73726d619ced9a985d5581efd5c0a9bf6e` |
| figures/true-vs-predicted.png | `bae907ad18959fceed2906c55e72a08fbb66a2cf84aa8985262524b5850221fc` |
| figures/uncertainty-vs-misrank.png | `b8f35ed818c820c18e819a1b5eb4d8b473b0f1093f5a1b08a4c9c6bf59365781` |

## Verification

Focused host tests: `PYTHONPATH=src:. /usr/bin/python3 -m pytest -q tests/eat_nav/test_feasible_transfer.py tests/eat_nav/test_candidate_coverage_oracle.py tests/eat_nav/test_candidate_coverage_evaluator.py` (11 passed). The Stage 0 identity and SHA gate and Phase B exact reproduction also passed.
Original pilot traces, checkpoints, and logs remain outside Git under `/tmp/eatnav-candidate-coverage/` and `/tmp/eatnav-quality-identifiability-gate/`.
