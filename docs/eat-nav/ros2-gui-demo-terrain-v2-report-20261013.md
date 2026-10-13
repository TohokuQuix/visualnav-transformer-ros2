# One-robot ROS2 + Isaac GUI demo: terrain v2 verification

## Scope and provenance

Implementation branch `exp/eatnav-ros2-gui-demo-terrain` started from the
unchanged GUI demo commit `d7df373ed711ee7a8a998e1879d87ac78070266b`.
The original four-env GUI run and its seal were read only. The pilot Evaluator
SHA is `329341fbf56ccb6bbc2f05b3db6b143cc1a1e9553ba2796e3dec3c785f30e470`;
the RL Actor SHA recorded by the new simulator is
`b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d`.
No model, Path geometry, follower control rule, observation definition, noise,
latency, or terrain subgenerator parameter was changed.

## Gates and observations

The [Stage A audit](ros2-gui-demo-terrain-stage-a-20261013.md) verifies the
sealed old boxes four-env run: 170/170 steps, 0.7915 m physical travel,
0.5814 m net common-goal progress, nine candidate selections, and zero
follower command after local Path Goal. This establishes the existing
Goal → candidate → Evaluator → C++ follower → Actor → physics path.

Four envs were fixed sizes in the demo runner and its observation adapter,
not a checkpoint input-batch requirement. The new default is one env; four envs
remains an explicit compatibility choice. An initial one-env attempt exposed
the stock `ignore_sprocket_self_collision` startup USD stage edit, which only
executes for a single-env scene and invalidated the PhysX tensor view. The demo
disables this edit to match the existing four-env effective collision setup;
reset events, robot dynamics, and policy are unchanged. A second pre-Path
attempt found a fixed four-row reconciliation tensor. The adapter now uses
the observation batch dimension, with a regression test for one and four envs.
Both failed attempt roots remain preserved separately.

The sealed [one-env boxes level-0 run](/home/kojima/lib/onix_isaac_lab/.eatnav-ros2-gui-demo-terrain-v2-20261013-173938/)
completed 260/260 physics/control steps. Startup shapes were Actor `[1,22]`,
Critic `[1,267]`, action `[1,6]`; the DEM and Actor were finite. A ROS goal was
received, six selections were scored (Direct 3, Right 3), and the robot moved
1.195 m. Common-goal distance changed from 0.752 m to a minimum of 0.063 m;
the final distance was 0.342 m. Median Evaluator inference took 0.825 ms.
Cancel was received at step 190, and all 70 subsequent follower commands were
zero. **The robot still traveled 0.262 m after Cancel.** This is physical
residual motion under the trained RL Actor/physics, so zero command is not a
physical stop guarantee. The run had no episode termination. This limitation
must be visible in a student demo and must not be represented as a safety stop.

The terrain selector retains the existing Isaac generator and creates five
curriculum rows of the requested terrain type. The robot is assigned to the
requested row, including after reset; `num_envs=4` uses four distinct columns.
The Onix rough config range `(0,1)` gives these row intervals; Isaac samples
the actual difficulty within each interval from the recorded seed:

| CLI level | Actual difficulty interval |
| ---: | --- |
| 0 | `[0.0,0.2)` |
| 1 | `[0.2,0.4)` |
| 2 | `[0.4,0.6)` |
| 3 | `[0.6,0.8)` |
| 4 | `[0.8,1.0)` |

All seven existing terrain types (`k_rail_step_field`, `pyramid_stairs`,
`pyramid_stairs_inv`, `random_rough`, `boxes`, `hf_pyramid_slope`,
`k_rail_field`) passed a candidate-free **terrain generation and origin**
smoke at all five levels: 35/35. This is a generator/selection check, not
35 complete ROS/robot runs. The persistent
[smoke root](/home/kojima/lib/onix_isaac_lab/.eatnav-demo-terrain-smoke-v2-20261013/)
contains 35 condition records, a 36-file SHA ledger, and a host seal.
Runtime `terrain-provenance.json` records the terrain type, selected row,
sampled difficulty, seed, tile geometry SHA, terrain origin, and spawn pose;
`terrain-config.json` stores the generator config and its SHA.

## Representative closed loops

All rows below received a ROS goal, performed finite Actor inference and
physics steps, scored candidates, and passed their individual SHA ledger and
host seal. Travel is integrated map-frame root XY motion; goal distances are
computed in that same map frame. Each run was 160 steps, except the GUI run
which was 200 steps. These are pipeline checks, not navigation performance
estimates.

| Terrain, level | Actual difficulty | Travel m | Goal initial → minimum → final m | Selections | Median score latency ms |
| --- | ---: | ---: | --- | --- | ---: |
| boxes, 0 | 0.0965 | 1.195 | 0.752 → 0.063 → 0.342 | D3/R3 | 0.825 |
| random_rough, 2 | 0.5921 | 1.132 | 0.747 → 0.091 → 0.107 | D3/R3 | 0.777 |
| pyramid_stairs, 1 | 0.2718 | 0.921 | 0.747 → 0.055 → 0.109 | D3/R1 | 0.811 |
| k_rail_step_field, 2 | 0.5921 | 0.872 | 0.750 → 0.191 → 0.196 | D1/R12 | 0.809 |
| k_rail_field, 2 | 0.5921 | 0.503 | 0.750 → 0.549 → 0.549 | D1/R12 | 0.790 |
| pyramid_stairs_inv, 1 (GUI) | 0.2718 | 1.101 | 0.750 → 0.037 → 0.115 | D3/R6 | 0.769 |

The first four non-box runs are listed in the persistent
[representative summary](/home/kojima/lib/onix_isaac_lab/.eatnav-demo-terrain-representative-v2-20261013.json)
(SHA `5108a0890cca131294e43b2b48b4651007b11905cb764e95e2db7deb8736d5e6`).
The GUI [inverted-stairs run](/home/kojima/lib/onix_isaac_lab/.eatnav-ros2-gui-demo-terrain-v2-20261013-175143/)
has a verified 7-file simulator seal and an
[Isaac window screenshot](/home/kojima/lib/onix_isaac_lab/.eatnav-ros2-gui-demo-terrain-v2-20261013-175143/one-env-isaac-window.png)
(SHA `3c2dc90cd2167fe0db2d9a86c4ce3a67f0e36e6ee95c0a112b2f34555438ff83`).
The screenshot is auxiliary evidence captured after the simulator seal. The
old four-env root has an RViz screenshot; the new one-env runs sent the goal
through the same ROS topic using the existing sender. A fresh manual RViz
click was not separately repeated for the one-env run.

Optional four-env boxes level-0 completed 160/160 steps with Actor `[4,22]`,
Critic `[4,267]`, and the same env-0 tile seed, sampled difficulty, and
geometry SHA as the one-env boxes run. Its env-0 trace and seal verify. The
pre-Path body and DEM diverged after settling as the env count changed, so
Evaluator scores, commands, and trajectories are not claimed to be exactly
equal. Both modes ran the same checkpoint and GUI demonstration pipeline.

Across the new representative runs, observed whole-device GPU peak was
5.2–7.5 GiB; the runs' minimum host available RAM was 79.3–81.6 GiB.
The logged simulator plus frontend wall spans were about 29–71 s per run.
These are device-total and host-available measures, not per-process GPU/RAM
attribution. All audited runs had native completion, seven sealed simulator
files, and no `done` step. The screenshot and outer launcher logs are outside
the inner seal.

## Limits and next use

The default one-env demo is usable for viewing the closed loop and selecting
any of the 35 terrain conditions. Exact wheel/body rest after local Goal or
Cancel is **not verified**: the command becomes zero, while physical motion
continues. `k_rail_field` showed limited goal progress, which is an outcome
observation rather than a pipeline failure. All runs use one seed and short
horizons. No claim of terrain generalization, robust arrival, or safety
stopping follows from these checks. No Evaluator training, full collection,
or real-robot action was performed.
