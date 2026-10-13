# Baseline Evaluator pilot: implementation and training result

## Decision and scope

The ROS-independent Baseline Evaluator can build a derived dataset from sealed P1 evidence, train a small CNN+MLP, save/reload a checkpoint, and score variable numbers of candidate Paths. This is a student handoff pipeline smoke. It is **not** evidence of unseen-terrain generalization, useful Path selection in closed loop, or DEM-based embodiment awareness. No ROS2 GUI demo, Isaac rollout, second-graded 24-trial crossover, or full collection was run.

## Repository and evidence

- Repository: `visualnav-transformer-ros2`; implementation branch `exp/eatnav-baseline-evaluator`; isolated worktree `/tmp/eatnav-vnt-baseline-evaluator`, based on teacher-capture commit `77ccad4`. Main checkout and unrelated uncommitted files were untouched.
- Implementation commit: `f6de13968de357e7a84ced740ab8f8dd98b605e2`.
- Raw input: three sealed ROS-free crossover period roots at `/home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-attempt-{1,2,3}`. The existing audit confirms 12 protocol-valid trials, balanced Direct/Left/Right, exact cross-period pre-Path state, and one independent StateGroup×seed condition. The teacher capture Direct-only root was separately adapted to four samples; it was not mixed into the training set.
- Derived dataset: `/home/kojima/eatnav-experiments/baseline-evaluator-pilot-20261013/crossover-dataset-v2/` (12 samples; `samples.npz` SHA `bcf94bec2e7578532fd85a0ba64fabb6f018f9b5ff8d92ae73dedc883ffb5a8a`). Separate Direct teacher adapter result: `teacher-direct-dataset-v2/` (4 samples). All source roots stayed read-only.
- Split: 12 train, 0 validation, 0 test. Paired candidates are kept in one group. A group-level validation/test split would be empty, so no held-out metric is reported.

## Model and input contract

The model has three convolution blocks over 8 DEM channels (four height/mask frames), a flattened eight-point Path MLP, a 22-value pre-Path Actor-state projection, and a fusion MLP. Parameter count: **31,441**. DEM/state features are shared across candidates. Inputs are `[B,4,2,20,20]` DEM history, `[B,22]` pre-noise Actor observation, `[B,K,8,2]` body-local Path, and optional `[B,K]` candidate mask; output is `[B,K]` scores. Invalid candidates receive `-inf`; all-invalid rows raise a named error. The pilot score is common-goal progress in meters, higher is better.

The adapter verifies seals, SHA, assignment, material/pre-Path/control gates, frame metadata, and exact eight-point Path provenance. It converts tile-local Path XY into robot-base-yaw XY using the pre-Path root and tile origin. DEM is already a scalar-height-offset, robot-relative four-frame history with an explicit validity channel. Normalization was fit only on the 12 train samples and stored in the checkpoint. No future reward, contact, Critic value, or post-Path motion is used as an input.

## Label availability

| Label | Crossover available | Teacher Direct available | Pilot training |
|---|---:|---:|---|
| Negative normalized command-to-motion error | 12/12 | 4/4 | No |
| Fixed 2 s common-goal progress | 12/12 | 4/4 | **Yes** |
| Fixed 2 s realized RL reward, γ=1 | 12/12 | 4/4 | No |
| Negative normalized Pain | 0/12 | 0/4 | No |

The teacher Direct source additionally preserves resolved reward term logs; the older crossover has only total reward. Pain is missing because mechanical allowables and combination weights lack engineering provenance. Synthetic formula tests do not change that status.

## Training and inference

- Frozen config: `config/eat_nav/baseline_evaluator_pilot.json` SHA `5b06da3cf26a11489705958cdefe021ec5220ddd2c1691f3f347a8961935b838`; label common-goal progress, seed 123, Adam LR 1e-3, batch 12, 300 epochs, CPU, no scheduler.
- Pilot environment: Python 3.10, PyTorch `2.0.1+cu117`, CUDA unavailable in that environment. The repository declares `torch>=2.3` in `pyproject.toml`; the pilot's actual software version is recorded in its checkpoint and should be rechecked in the eventual student environment.
- Standardized train MSE: `1.025710` at epoch 1, `0.289528` at epoch 50, `0.278809` at epoch 100, and `0.278654456` at epoch 300. Identical pre-Path inputs with different env outcomes impose a measured MSE floor of `0.278654454`; the model reached it. Training took 0.895 s on CPU. Full history is in `baseline-evaluator-pilot-result-20261013.json`.
- Checkpoint: `/home/kojima/eatnav-experiments/baseline-evaluator-pilot-20261013/train-common-goal-v1/checkpoint.pt`, SHA `329341fbf56ccb6bbc2f05b3db6b143cc1a1e9553ba2796e3dec3c785f30e470`. It contains weights, optimizer, model/input/label schema, train-only normalization, label transform, split, source/dataset SHA, seed, config, loss history, source SHA, Git commit, and PyTorch version. Checkpoint round-trip and schema mismatch rejection passed.
- Train-set-only ranking diagnostic: four env displays, Direct selected in all four and Direct was the observed best in all four. Mean pairwise agreement `0.9167`; mean regret `0`. The model emits the same three scores for every env because the recorded pre-Path DEM/state/Path inputs are identical. Therefore these numbers are a pipeline diagnostic, not independent candidate-ranking performance. Mean CPU latency over 100 single-candidate calls was 0.371 ms in the pilot environment.

## Verification and compatibility

Focused preprocessing/dataset/model/checkpoint tests: 12 passed, including ROS/Isaac-free core import, mask/nonfinite handling, crop, body-frame transform, variable K, candidate permutation, all-invalid rejection, synthetic Pain, group split leakage, sealed-data mutation detection, forward/backward, and checkpoint compatibility. Existing DEM/dataset/teacher tests: 25 passed. A legacy `tests/eat_nav/test_model.py` collection attempt could not run in the available `tinyvla` environment because its `diffusion_policy` dependency is absent; no legacy model file was changed. GPU inference was unavailable in this Python environment.

## ROS2 demo boundary and next work

The future ROS2 adapter is specified in `baseline-evaluator-ros2-demo-design.md`; it is not implemented. It must supply a time-aligned GridMap DEM, robot pose, pre-Path Actor state, and candidate Paths in the verified frame, then use the common preprocessing and checkpoint loader. It must stop safely on missing/stale inputs, TF failure, no valid candidate, timeout, or schema mismatch, and reuse the C++ follower core rather than reimplementing its control law.

For a research evaluation, first collect additional **independent** StateGroups, terrain seeds, and candidate outcomes with the same input and label contract. Then create a group-held-out validation/test split, compare geometry-only and state/DEM ablations, evaluate the choice of teacher signal, and test closed-loop selection separately. A trained score from this one-condition pilot must not be treated as a calibrated real-robot decision rule.
