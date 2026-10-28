# R4* / R5 code and checkpoint transfer (2026-10-28)

R4* and R5 use exactly the same 12 trained Evaluator checkpoints. They differ in candidate scoring and exclusion settings, not in weights. Both train and evaluate with the original 14 circular candidates (7 forward, 7 reverse). This branch builds on the previous R4 code/checkpoint snapshot and retains its 24 earlier checkpoints.

## Select a model and an evaluation profile

The new checkpoint index is `checkpoints/eat_nav/fall_stall_r4star_r5_index.json`. It lists SHA-256, field, training seed, and the two profiles. K-Rail has six model seeds; the other six fields have seed123 each. Example:

`docs/eat-nav/r4-star-fourteen-training-six-evaluation-20261026/models/k_rail_field/seed123/checkpoint.pt`

The historical directory name contains `six-evaluation`; the completed manifest and implementation specify **14 evaluation candidates**.

- R4*: `eat_nav_r4_star14_eval.R4Star14Guard`, risk weights `[0, 0, 1, 0]`.
- R5: `eat_nav_r5_14.R5FourteenGuard`, risk weights `[0, 0, 1, 0.5]`.
- Both disable measured 45-degree STOP and 35-degree release, tip and near-fall prediction use. Other dwell/direction/STOP confirmation settings are retained.
- Head order is tip-over, near-fall, linear-stall, turn-stall. R5 requires supported linear and turn heads; R4* requires the linear head.

`config/eat_nav/r4star_r5_profiles.json` records the exact settings. It is a reference manifest for integration; the existing simulator runners do not parse this file as a new CLI option. Select the corresponding guard implementation, not only a checkpoint or a risk-weight vector: R5 also applies turn-stall candidate exclusion. Use `eat_nav_command_free_learning.load_checkpoint` and its matching `predict` with the checkpoint's embedded normalization. Add `src:scripts:.` to PYTHONPATH.

## Real ONIX2 integration boundary

This snapshot provides the Evaluator, candidate bank and tested simulation guards. It does not implement or certify a ROS real-robot bridge. Real deployment still needs observation-schema/units/frame/timing parity, the actual topics, and the existing path follower / RL Actor interface. The checkpoint alone does not select R4* versus R5 behavior. No real-robot performance is claimed.

## Datasets and experiment evidence

The 14-candidate field datasets are located on the collection PC at:

`docs/eat-nav/r4-star-fourteen-training-six-evaluation-20261026/datasets/`

Training also uses the existing common dataset at:

`docs/eat-nav/additional-development-batch02-cumulative-20261021/cumulative/`

Each field dataset contains samples.npz, sample-index.json, split.json, coverage and added-candidate support plus its hash ledger/seal. Transfer the complete dataset directories separately using rsync or Google Drive. Raw collections are in the R4* root's collection-runtime directory. Dataset tensors and raw traces are not included in this Git transfer.

Completed original manifests, aggregate reports, model training histories and metrics are copied as evidence. Absolute paths retain their original provenance. The full original audit trees and original model-directory seals are not packaged, so do not pass these partial roots to seal-verifying historical resume runners. Copy complete sealed roots separately for full experiment reproduction.

K-Rail results in the R5 paired report use 12 cases (2 physical rollout seeds x 6 optimization seeds), versus 11 common cases in the old R4/R4* comparison. Do not treat optimization-seed repeats as 12 independent terrains. STOP seconds are commanded protective STOP time, not measured stall duration. Reserved test data remained unopened.

## Verification

All 12 new checkpoints passed SHA-256 and matching-loader checks; both guard implementations passed a 14-candidate smoke check. The portable snapshot passed 31 self-contained tests. In the complete original artifact workspace, the 8 related modules passed all 76 tests. Running all 8 in this dataset-free snapshot instead gives 64 passes, 9 failures and 3 errors because historical dataset/baseline trees are intentionally omitted. See `r4star-r5-transfer-verification.json`.
