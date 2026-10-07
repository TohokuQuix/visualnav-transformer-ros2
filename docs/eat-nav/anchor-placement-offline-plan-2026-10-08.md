# Anchor placement offline comparison plan

**Scope.** Reuse the completed first-seed 112-pose radial inventory without modifying it. Build a new 112-environment scene from the frozen terrain seed and generator, and use its existing `height_scanner_eat` RayCaster without any control or physics step after scene initialization. Never infer physical stability from geometry.

1. Freeze provenance and the parity tolerance before any virtual-radius scan: height RMS ≤ 1e-5 m, maximum absolute error ≤ 1e-4 m on jointly finite valid cells, mask IoU = 1.0, and exact frame, axis, resolution, shape, seed, terrain and logical pose identity. A parity failure stops the scan.
2. At each recorded pose's last available DEM frame, set the robot root to the saved frame-time root pose, call `sim.forward()` and force a RayCaster update with zero elapsed time. Compare source-grid origin, raw ray heights/mask, transformed DEM, yaw and env origin against the immutable frame. The physically terminated pose uses its last available frame and remains anchor-ineligible.
3. If all 112 reconstructions pass, query the same generated terrain with the same RayCaster at the 448 frozen virtual XY/yaw placements (four radii × 16 directions × seven terrain families). Keep the future measurement root-z rule separate from an offline geometry query; never run settling or a policy action.
4. Classify each virtual DEM with the unchanged `classify_dem` code and thresholds. Record coverage, height/profile/onset metrics, footprint proxies and structural invalidity. Compare R0–R3 using deterministic global rules with no reward, Critic, Outcome or terrain-specific manual ranking. Keep interpolation radii separate.
5. Produce the pilot proposal, source-family support and second-seed decision, figures, report and SHA ledger. Stop before any new physics rollout, candidate Path, follower, Outcome or learning.

**Fail closed.** Missing source identity, inability to reproduce the scanner raw geometry, nonfinite geometry, or a parity miss prevents the virtual scan. The stopped run still saves first failing signal, measured errors and source SHA.
