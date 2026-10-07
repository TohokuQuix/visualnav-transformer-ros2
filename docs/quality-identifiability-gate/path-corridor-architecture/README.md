# Path-corridor architecture comparison

This directory contains the frozen offline A1–A4 design and config. The experiment reads the 28-StateGroup source at `/tmp/eatnav-evaluator-independent-expansion-20261006-v7` and writes numeric artifacts under `/tmp/eatnav-path-corridor-architecture-20261007`. It does not invoke Isaac, ROS, NoMaD, the follower, or the robot.

Run order:

1. `scripts/eat_nav_corridor_audit.py` checks all 252 record/label/trace identities, 14 anchor-only and 189 new live process artifacts, source hashes, frame sequence, Path/goal/pose identity, DEM reconstruction, and 40 fold boundaries.
2. `scripts/eat_nav_corridor_prepare.py` aligns the four source DEMs to the anchor pose, extracts full grids and 16×7 Path corridors, and records Outcome-blind support classes under the frozen config SHA.
3. `scripts/eat_nav_corridor_train.py` fits A1–A4 for all 40 folds and seeds 123/456/789. One fit is sealed by a prediction NPZ SHA and JSON marker; a rerun skips only matching sealed fits. CPU shards partition folds without changing their order or content.
4. `scripts/eat_nav_corridor_report.py` verifies 480 prediction artifacts, computes StateGroup-level metrics and paired confidence intervals, and writes figures, decisions, and a final SHA ledger.

The A1 flatten branch shares anchor-aligned DEM preprocessing with A4, so it is an architecture reproduction of the historical B4 baseline rather than byte-identical historical B4 preprocessing. All four variants retain the historical 64-point Path DEM profile in their common input. E-goal shares anchors and E-terrain is diagnostic-only. Layers with fewer than three StateGroups are marked `insufficient_support`.

The optional geometry-only collection proposal is separate from this experiment. No data collection or downstream learning is authorized by these artifacts.
