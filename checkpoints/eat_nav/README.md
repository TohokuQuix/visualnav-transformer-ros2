# EAT-Nav demo checkpoints

These files are pinned inputs for the GUI demos. The `.pt` and `.npz` files
are stored as ordinary Git blobs because the public fork cannot accept new
Git LFS uploads. Stage C provenance artifacts are used by
`validate_adoption_artifacts()` at startup. Run
`/usr/bin/python3 scripts/eat_nav_verify_checkpoints.py` before launching.

The RL Actor checkpoint is the Onix rough `model_10000.pt` from
`2026-07-24_18-20-16`. The Motion-14 checkpoint is the Stage C selected Full
seed-123 `best.pt` from `motion-14-training-20261014-04`. Neither file was
retrained or converted for this package. Their SHA-256 values are frozen in
`manifest.json`.

The `motion14_heading/h2a-seed123/best.pt` checkpoint is a separately trained
two-second yaw head over the frozen Stage C XY model. The ROS-separated GUI
uses it to orient the candidate prediction arrows in RViz and, when heading
scoring is selected, to reward predicted terminal alignment with the global
goal. Its SHA-256 and
base-checkpoint provenance are verified at startup.

No separate download is needed after checkout. If an external artifact store
is used later, keep the same relative paths and verify every file against
`manifest.json` before running the demo.
