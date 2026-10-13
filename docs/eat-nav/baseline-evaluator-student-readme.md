# Baseline Evaluator pilot

This package scores candidate local Paths without importing ROS2, Isaac Lab, or PhysX. It is a student handoff baseline for changing labels, inputs, and network structure. The pilot checkpoint is a **train-set smoke test on one StateGroup and one seed**, with no independent validation or generalization estimate.

## Input and output contract

`Evaluator.predict(dem, robot_state, candidate_paths, candidate_mask=None)` returns `[B,K]` scores, where larger is better. All-invalid rows raise `NoValidCandidateError`; invalid individual candidates receive `-inf`. K is variable and candidate permutation changes only the corresponding output order.

| Field | Raw shape | Frame / meaning |
|---|---|---|
| DEM | `[B,4,2,20,20]` | Four chronological robot-base-yaw frames; channel 0 scalar-reference-relative height in m, channel 1 binary validity mask; 0.1 m/cell |
| State | `[B,22]` | Pre-Path, pre-noise RL Actor observation; not Critic input or future state |
| Path | `[B,K,8,2]` | Eight XY waypoints in the same robot-base-yaw frame, m |
| Mask | `[B,K]` | Boolean candidate validity |
| Score | `[B,K]` | Selected teacher label units; larger is better |

Stored P1 `map_xy` waypoints are tile-local. The adapter subtracts the terrain tile origin from the pre-Path world root and rotates Path points by negative root yaw to produce robot-local XY. This conversion is shared with future inference. DEM source frames are the sealed `shared_experimental_history`, and native sensor history remains a separate diagnostic. Invalid DEM cells stay masked and never become valid zero heights. Training normalization is fit only on the train split and saved in the checkpoint.

The four selectable label names are `negative_motion_error`, `common_goal_progress`, `realized_reward`, and `negative_pain`. The first uses mean absolute forward residual divided by 0.3 m/s plus mean absolute yaw residual divided by 1 rad/s, then negates it. Progress is fixed 40-step distance reduction to the common goal. Reward is the sum of observed step rewards (γ=1 by default), without multiplying by dt again. Pain requires versioned mechanical allowables and explicit joint/body combination weights with engineering provenance; absent allowables remain missing. Raw torque and body contact load remain available for later work.

## Data and pilot commands

The adapter reads existing P1 roots and verifies host seals, SHA ledgers, frozen assignments, full control traces, pre-Path gates, DEM frame provenance, and balanced 3-period pairing. It never writes into a sealed root. Rebuild a derived dataset in a **new** output directory:

```bash
PYTHONPATH=src:scripts python scripts/eat_nav_baseline_prepare.py --source crossover \
  --root /home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-attempt-1 \
  --root /home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-attempt-2 \
  --root /home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-attempt-3 \
  --output /path/to/new/crossover-dataset
PYTHONPATH=src:scripts python scripts/eat_nav_baseline_train.py \
  --dataset /path/to/new/crossover-dataset \
  --config config/eat_nav/baseline_evaluator_pilot.json \
  --output /path/to/new/pilot-train
PYTHONPATH=src:scripts python scripts/eat_nav_baseline_evaluate.py \
  --dataset /path/to/new/crossover-dataset \
  --checkpoint /path/to/new/pilot-train/checkpoint.pt \
  --output /path/to/new/ranking.json
```

The Direct-only teacher capture can be adapted separately with `--source teacher --root <teacher-capture-root>`. Do not merge it with the older crossover merely because fixed-window progress agrees for some envs. The sources have different logger protocols and must retain distinct provenance.

The current crossover has 12 trials (Direct/Left/Right for each of 4 env IDs) but only **one independent paired StateGroup×seed condition**. The train split contains all 12, while validation and test are empty. Identical pre-Path DEM/state and the same Path can have different downstream outcomes across envs, creating an irreducible loss floor for this input set. A low train loss or ranking agreement here does not show terrain generalization, robust candidate ranking, or navigation benefit.

## Student change points

- Labels: `evaluator/labels.py`; keep observed, fixed-window outcomes and masks separate from candidate inputs.
- Inputs and normalization: `evaluator/preprocessing.py`; preserve frame, resolution, mask, and train-only statistics.
- Model: `evaluator/model.py`; keep `Evaluator.predict` and checkpoint schema compatibility, or version the interface.
- Dataset: `evaluator/dataset.py`; preserve group-level splits and source SHA/seal checks.
- Pilot config: `config/eat_nav/baseline_evaluator_pilot.json`. Changing the label creates a separate checkpoint; missing Pain must fail closed.

The existing NoMaD/ViNT modules and production follower are untouched. `pyproject.toml` declares `torch>=2.3`; the local pilot environment used for this implementation must be reported alongside its actual torch version. The full student environment setup and GUI demo instructions are deferred until those workflows are verified.
