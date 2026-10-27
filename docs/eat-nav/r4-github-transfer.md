# R4 code and checkpoint transfer

This branch preserves the implementation snapshot used through R4 and the current follow-up code. It includes 24 completed Evaluator checkpoints, their embedded normalization/provenance and an SHA-256 index, and the already packaged RL Actor. R4* collection is ongoing; R5 is queued awaiting R4* completion. No newly trained R4*/R5 model is claimed.

Checkpoint index: `checkpoints/eat_nav/fall_stall_r4_index.json`. Model roots:

- `docs/eat-nav/all-fields-learning-20261023/models/`:12 field models used for R1/R2/R3/R4.
- `docs/eat-nav/command-free-learning-20261021/artifacts/`:6 command-free variants/seeds.
- `docs/eat-nav/current-dataset-learning-20261021/artifacts/`:6 earlier variants/seeds.
- `checkpoints/eat_nav/rl_actor/`: packaged RL Actor and environment config already in repository history.

Checkpoint normalization, configuration, support flags and training provenance are embedded in each `.pt`; use the matching v1/v2 loader rather than mixing observation schemas. Prediction JSONs, metrics, original model-directory seals and raw datasets are not included in this code/checkpoint transfer. For complete experiment replay, transfer the original sealed artifact trees separately without modifying them.

Datasets and raw rollout/evaluation logs are transferred separately using rsync or Google Drive. In particular, retraining needs `docs/eat-nav/additional-development-batch02-cumulative-20261021/cumulative/` and `docs/eat-nav/all-fields-learning-20261023/datasets/`. Historical runners also verify complete experiment roots/registries; checkpoints alone do not recreate the full audit tree. Preserve the original directory layout and any absolute source-path mappings.

For Isaac GUI reruns, recreate the Isaac/Onix Docker setup and bind-mounted sources used by the experiment, in addition to installing the Python package. See the corresponding experiment specs. The current local source checkout predates three commits on the existing remote GUI branch; this is a separate snapshot branch so the remote branch is not overwritten or merged.

Validation:207 focused tests passed; all24 checkpoints match their SHA and load using their corresponding model version. Broader test collection was blocked by missing `diffusers` in9 existing test modules; this transfer does not claim the entire repository suite passed. See `r4-github-transfer-verification.json`.
