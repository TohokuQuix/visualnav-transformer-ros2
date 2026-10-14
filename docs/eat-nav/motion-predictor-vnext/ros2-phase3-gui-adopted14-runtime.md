# ROS-separated adopted Motion-14 GUI integration (bounded check)

## Scope and provenance

- Branch: `exp/eatnav-ros2-adapter-design`; baseline GUI Motion-14 integration cherry-pick: `b62a97d4`.
- Adopted Stage C checkpoint: `motion-14-training-20261014-04/training/full-seed123/best.pt`, SHA-256 `83a565f0c1b662c278da20ce851297d890f974f39a86df73b971fd0a2c0d22fa`. Stage C adoption decision is `pass`; normalization is train-only.
- The model runs in the ROS-separated Model Worker. Isaac publishes measured DEM and Actor state, receives the selected frozen body-frame Path, runs the existing pure C++ follower, and applies only the Supervisor-gated command to the RL Actor. The production follower, robot topics, and external router were not modified.
- The bounded check used a private ROS domain (227), one GUI Isaac environment, `boxes` level 0, seed 2026101101, 20 Hz control, the frozen 14-Path bank, and an automatically published map goal 2 m ahead of the initial XY position.

## Verification

The direct Stage C model and ROS Model Worker produced matching 14-candidate predictions on a sealed input group. The relevant ROS CPU tests passed (13), and the direct/worker parity test passed (1). The 14 Path order, wire SHA, selection generation and lease, and Path bytes are checked before Isaac activates the Path.

The final bounded GUI attempt is at:

`/home/kojima/lib/onix_isaac_lab/.eatnav-ros2-phase3-evidence/eatnav-phase3-gui-20261014-222550/`

It completed 160/160 control steps with one goal, 12 ROS-selected Paths, 112 nonzero movement commands, no episode reset, and no owned residual process. The robot moved 1.062 m in XY; distance to the test goal decreased by 0.922 m. The selected roles were forward direct (7), reverse −30° (3), forward −30° (1), and forward −90° (1). The resource series contains 102 samples over 52.4 s; peak device memory was 7,425 MiB. The inner simulator seal verified 7 files; the outer host seal verified 16 files. The external router PID/start tick remained the same during the attempt.

Two earlier attempts remain separate failed evidence: the first reached ROS selection but the frontend could not JSON-serialize a NumPy float32 Path value; the second passed Path activation and four gated control steps, then received unrelated goals on the shared ROS domain. Their roots were preserved. The final attempt uses an isolated domain and a new root.

## Interpretation and limits

This establishes a functioning **short ROS-separated Isaac GUI closed loop** using the adopted checkpoint. It does not establish good goal-directed navigation. The observed tendency to take large arcs or continue straight remains a model/selection behavior to investigate separately. This check covered one terrain, one seed, and eight logical seconds. It did not exercise Cancel, DEM/TF loss, timeout, or fall reset, and it did not collect a complete live worker-input tensor trace. The full Phase 3 stop and parity gates remain open; no real robot connection follows from this result.

## Reproduce or inspect

From this branch/worktree, with the existing external Zenoh router running:

```bash
/usr/bin/python3 scripts/eat_nav_phase3_gui_launch.py --max-steps 160 --terrain-type boxes --terrain-level 0
```

The launcher writes a fresh persistent attempt under `onix_isaac_lab/.eatnav-ros2-phase3-evidence/`. For an interactive RViz goal, use `--manual-goal` and run RViz on the same private domain (`ROS_DOMAIN_ID=227`, `RMW_IMPLEMENTATION=rmw_zenoh_cpp`) with `config/eat_nav_phase3_gui.rviz`. The selected Path and all 14 candidates are visualized from the ROS-separated selection; the displayed height image is for inspection and does not replace the metric DEM sent to the model.
