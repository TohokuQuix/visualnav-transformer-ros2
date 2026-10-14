# EAT-Nav data, Evaluator pilot, and GUI experiment pipeline

## What is runnable today

The GUI demo is an online **inference** pipeline using the sealed MI-B Pilot
Full seed-456 Motion Predictor checkpoint. It predicts 2 s body-frame XY
motion; the GUI converts that prediction into global-goal distance
reduction and chooses the largest score. This selector is experimental:
MI-B's held-out motion accuracy does not establish online Path-selection
benefit. The older **Baseline Evaluator** GUI checkpoint was a
one-StateGroup×one-seed train-set smoke test for 2 s common-goal progress,
with empty validation/test splits. Its preparation/training commands below
remain for historical reproduction.

```mermaid
flowchart TD
    G[RViz2 Goal, map] --> F[ROS2 Jazzy frontend]
    F <-->|localhost JSON IPC| S[Isaac Lab / Isaac Sim]
    S --> T[Terrain + root/joint state]
    T --> D[4-frame robot-relative DEM]
    G --> P[Direct / Left / Right candidate generator]
    D --> E[MI-B Pilot Motion Predictor CNN + MLP]
    T --> E
    P --> E
    E --> C[Selected Path, map to tile-local]
    C --> PF[C++ PureFollower per env]
    PF --> CM[CommandManager]
    CM --> AO[Actor observation: noise + latency]
    AO --> RL[RL Actor]
    RL --> PH[Isaac physics]
    PH --> T
    S --> F
```

ROS2 transports goals and state visualization. The production ROS follower,
ROS bridge, and external Zenoh router are **not** in the simulator control
path. The router carries the frontend's ROS topics. The actual 20 Hz
simulator loop is in `scripts/eat_nav_demo_isaac.py:main`: root pose →
`PureFollower.step_diagnostic` → `CommandOverride.set` →
`reconcile_command_observation` → `policy.act_inference` → `env.step`.
The next post-physics state is sent to `DemoFrontend.publish_state` in
`scripts/eat_nav_demo_ros_frontend.py`; `/clock`, TF, odometry, DEM, Paths,
scores, and status use simulation stamps.

| Component | Source/API | Input → output | Time/frame |
| --- | --- | --- | --- |
| ROS frontend | `scripts/eat_nav_demo_ros_frontend.py:DemoFrontend` | `/goal_pose` or `/eat_nav/cancel` → validated map goal/Cancel over IPC; simulator state → ROS display topics | Goal stamp within 0.5 s of simulation `/clock`; `map` and `base_link` |
| Terrain/state | `demo_isaac_env_cfg.py:EATTerrainGenerator/EATTerrainImporter` | seed, type, row → mesh, tile SHA, origin, robot state | world/map frame; selected row held fixed on reset |
| DEM | `scripts/eat_nav_demo_isaac.py:dem_from_sensor`, `dem.py:robot_dem` | raycaster → `[4,2,20,20]` history | robot-base-yaw, 0.1 m/cell; four chronological frames at 250 ms intervals; actual sensor update stamp is recorded separately |
| Path bank | `demo_contract.py:candidates_for_goal` | current map pose + map goal + frozen bank → `[K,8,2]` robot-local candidates and map Paths | current pre-Path pose; 0.75 m local radius, Direct/Left +20°/Right −20° |
| Motion Predictor | `demo_motion_predictor.py:DemoMotionPredictor.predict` | DEM, pre-noise state, three Paths → predicted body XY `[3,2]` → global-goal progress scores `[3]` | pre-Path input; largest predicted 2 s distance reduction is selected |
| Follower | `p1_follower_core.py:PureFollower` | selected tile-local Path + tile-local pose + logical time → `[vx,vy,wz]` and diagnostics | one call per 50 ms control step; one instance for the selected Path |
| Command/Actor | `phase_c_sim.py:CommandOverride`, `p1_g3_rosfree_control.py:reconcile_command_observation` | follower command → observed command after noise/latency → Actor action `[B,6]` | body velocity command; Actor observation `[B,22]`, Critic `[B,267]` |
| Physics/log | `scripts/eat_nav_demo_isaac.py` | action → next robot state, reward, done, trace | 50 ms control boundary; new trace row after physics |

### Current GUI Motion Predictor contract

The launcher stages sealed `training/full-seed456/best.pt` from the MI-B
Pilot evidence root; seed 456 has the lower aggregate validation loss of the
two Full initializations. `demo_motion_predictor.py` verifies the checkpoint
SHA and training-model source SHA, applies train-only DEM/Actor normalization,
and reverses the target normalization to return three predicted body-frame
XY displacements in metres. The three GUI Paths occupy the forward slots
of the six-candidate model; the remaining slots are masked. For each Path,
the selector transforms the full map-frame global goal into the current body
frame, then scores the reduction in distance to that global goal after the
predicted displacement. The Path bank still uses a local subgoal no farther
than 0.75 m. The `/eat_nav/scores` topic reports
`predicted_global_goal_progress_m`.
No goal is fed into the Motion Predictor itself.

The MI-B training Paths were forward/reverse Direct and ±30° Arcs, whereas
the GUI keeps Direct and ±20° Arcs rotated/scaled toward the RViz goal.
This distribution difference and the new goal-based selection rule make
the GUI an exploratory demonstration, not a measured MI-B online result.

### Historical Baseline Evaluator input and checkpoint contract

| Tensor | Raw shape | Meaning |
| --- | --- | --- |
| DEM | `[B,4,2,20,20]` | four frames; height relative to scalar reference and binary valid mask |
| Actor state | `[B,22]` | **pre-Path, pre-noise** Actor observation; not Critic or future state |
| Candidate Paths | `[B,K,8,2]` | robot-base-yaw XY in metres |
| Candidate mask | `[B,K]` | Boolean validity; all invalid fails closed |
| Scores | `[B,K]` | predicted teacher units; highest is selected |

`evaluator/preprocessing.py:prepare_inputs` turns four DEM frames into eight
channels for the CNN; it does not encode the image with a plain MLP. The model
in `evaluator/model.py:BaselineEvaluator` has a three-layer convolutional DEM
encoder, separate Path and state MLPs, and an MLP fusion head. The saved
train-only normalization and label transform are part of the checkpoint.
`evaluator/inference.py:load_checkpoint` checks schema, input dimensions,
label type, dataset SHA, and normalization. The historical GUI additionally
pinned checkpoint SHA and `common_goal_progress` label. The current GUI pins
the MI-B checkpoint SHA and model source provenance.

The old P1 stored Path `map_xy` is tile-local. `evaluator/dataset.py` subtracts
the tile origin and applies `body_local_path` with the pre-Path root yaw.
During the GUI demo, `candidates_for_goal` creates robot-local paths from
the frozen bank and transforms them into map Paths; the selected Path is
converted back to tile-local for the C++ follower. No post-Path outcome enters
the Evaluator input. The current demo logs the selected candidate and scores,
but its `trace.jsonl` is **not** automatically a new teacher dataset.

## MI-A / MI-B offline evidence and reproducibility

MI-A v3.1 formally measured Path-dependent 2 s body-frame XY motion:
192/192 processes and 576/576 candidate trials were protocol-valid across
12 terrain seeds, four terrain types and Front/Rear initial flipper poses.
The frozen exploratory MI-A Gate passed. Terrain type, level and seed are
coupled, so this does not establish transfer to unseen terrain. The
[formal result report](motion-predictor-vnext/mi_a_v31_formal_result_report.md)
retains the Gate, bootstrap and resource details. Its sealed aggregate is
`/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-a-v31-formal-20261014-03/`;
see `formal-analysis-sha-ledger.json` and `formal-analysis-seal.json`.

The MI-B Pilot imported those 576 trials read-only and formed 96 paired
examples. Its raw tensor contract is DEM `[B,4,2,20,20]`, pre-noise Actor
state `[B,22]`, candidate Path `[B,K,8,2]`, direction `[B,K,1]`, mask
`[B,K]`, and 2 s body-frame XY output `[B,K,2]`. Train, validation and
test each contain four distinct terrain seeds, 32 paired groups and 192
candidate trials; normalizers use train only, checkpoint selection uses
aggregate validation loss, and the held-out test is used only for evaluation.
The small overfit test passed. Held-out Arc−Direct XY RMSE was
`0.0713/0.0724 m` for Full versus `0.0834 m` for the train-only
Direction mean. No Path did not improve on that baseline; No DEM performed
similarly to Full, while the Actor-state contribution and Front/Rear
interaction prediction remain unclear. Reverse +30° and `random_rough`
retain notable error. The predeclared exploratory MI-B Pilot criterion
passed; **a formal MI-B Gate is not defined**. See the
[MI-B report](motion-predictor-vnext/mi_b_pilot_report.md),
[fixed contract](motion-predictor-vnext/mi_b_pilot_contract.json), and
`/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-b-pilot-20261014-01/`
(`audit.json`, `overfit-result.json`, `pilot-results.json`,
`pilot-subgroups-v2.json`, `artifact-ledger.json`, `artifact-seal.json`).

These source roots are read-only evidence, not output directories. The
[student roadmap](research_roadmap.md) states the current open question and
next-meeting assignment. Phase C collection, MI-C and deployment are outside
the present approved work.

## Existing offline preparation and training commands

These are the implemented scripts and their historical source roots. Use a
Python environment satisfying `pyproject.toml` (`Python 3.10–3.12`,
`torch>=2.3`, `numpy<2`, and the listed packages); the simulator container's
`/isaac-sim/python.sh` has Python 3.11/PyTorch 2.7. Create new output paths;
never write into the sealed source roots or the existing pilot checkpoint.
Running the training command makes a **new train-only pilot**, not a
validated model.

```bash
PYTHONPATH=src:scripts /isaac-sim/python.sh scripts/eat_nav_baseline_prepare.py \
  --source crossover \
  --root /home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-attempt-1 \
  --root /home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-attempt-2 \
  --root /home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-attempt-3 \
  --output /path/to/new/crossover-dataset
PYTHONPATH=src:scripts /isaac-sim/python.sh scripts/eat_nav_baseline_train.py \
  --dataset /path/to/new/crossover-dataset \
  --config config/eat_nav/baseline_evaluator_pilot.json \
  --output /path/to/new/pilot-train
PYTHONPATH=src:scripts /isaac-sim/python.sh scripts/eat_nav_baseline_evaluate.py \
  --dataset /path/to/new/crossover-dataset \
  --checkpoint /path/to/new/pilot-train/checkpoint.pt \
  --output /path/to/new/ranking.json
```

The commands above are **script interfaces**, not a newly executed training
run in this handoff. `/isaac-sim/python.sh` is inside the container; an
equivalent compatible host interpreter can be used if it can access the
source roots. The source roots may need bind mounts or a host path adjustment.
Verify their existence before running. The [baseline pilot guide](baseline-evaluator-student-readme.md)
and the existing persistent dataset/checkpoint under
`/home/kojima/eatnav-experiments/baseline-evaluator-pilot-20261013/`
record the actual prior run.

| Stage | Implemented source, input → output | Validation/failure check |
| --- | --- | --- |
| Candidate execution | `scripts/eat_nav_p1_g3_rosfree_sim.py` and its historical manifests: pre-Path state/Path/seed → sealed trace and outcome | The old 12-trial crossover is one independent StateGroup×seed; it is not a general collector |
| Label calculation | `evaluator/labels.py:labels_from_trace`: 40 observed 50 ms rows → four label slots and masks | Requires same-episode complete 2 s; failure/censor remains explicit |
| Dataset | `scripts/eat_nav_baseline_prepare.py`, `evaluator/dataset.py`: three sealed crossover periods → `samples.npz`, `metadata.json` | Verifies seals, Path assignment, pre-Path equality, DEM SHA, and group split; 12 samples, train only |
| Training | `scripts/eat_nav_baseline_train.py`, `config/eat_nav/baseline_evaluator_pilot.json` → `checkpoint.pt`, `result.json` | Checks dataset SHA, group split, normalization from train only, finite loss; no independent validation |
| Offline ranking | `scripts/eat_nav_baseline_evaluate.py` → ranking JSON | Train-set diagnostic only; check source and checkpoint SHA |
| MI-A motion evidence | `scripts/eat_nav_mi_a_v31_analyze.py`, sealed formal root → 192 process / 576 trial result | Frozen exploratory MI-A Gate Pass; no unseen-terrain claim |
| MI-B offline Motion Predictor | `scripts/eat_nav_mi_b_pilot.py`, `src/visualnav_transformer/eat_nav/mi_b_pilot.py`, sealed MI-B root → grouped dataset, checkpoints, held-out metrics | Small overfit and exploratory Pilot Pass; formal MI-B Gate undefined; Full seed-456 checkpoint now drives exploratory GUI scoring |
| GUI inference | `eat_nav_demo_launch.py`, `eat_nav_demo_isaac.py`, ROS frontend → new trace, score, GUI/ROS topics | Check input SHA, `startup-ready.json`, `complete.json`, `failure.json`, host seal, finite Actor/action, physical root motion |

Labels currently defined in `evaluator/labels.py` are negative normalized
motion residual, common-goal progress, accumulated observed RL reward, and
negative Pain. Motion residual compares executed forward/yaw motion to the
follower command. Reward sums the observed per-step reward over the fixed
window (γ=1 in the pilot). Pain is unavailable without versioned joint/body
allowables, weights, and engineering provenance; raw simulated torque and
contact are not themselves a validated Pain label. The current checkpoint
uses **only common-goal progress**. Motion residual, reward, and Pain are
research alternatives, not trained heads in the current checkpoint.

## What must be built for a research dataset

The **Baseline Evaluator adapter** described above deliberately requires exactly one independent
StateGroup×seed and train-only split. A future general Evaluator experiment
would require additional independent StateGroups/terrain seeds with paired Path
conditions, a versioned collector and dataset adapter, group-level splits,
sealed full traces, train-only normalization, validation checkpoint selection
and untouched test evaluation. The current
demo's short, selected-Path-only trace does not contain counterfactual
outcomes for unselected Paths. No script in this handoff automatically
creates that broader dataset or trains a generalizable Evaluator.
This historical limitation does not describe the separate MI-B offline
Motion Predictor dataset. Additional collection is not approved for the
current student assignment.
