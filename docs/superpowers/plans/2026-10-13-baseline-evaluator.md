# Baseline Evaluator implementation plan

1. Audit the latest committed EAT-Nav branch, sealed ROS-free crossover, teacher capture, DEM and Path contracts. Keep all source roots read-only and use an isolated worktree.
2. Define a versioned pre-Path input schema: four robot-relative DEM frames with validity masks, 22-dimensional pre-noise Actor observation, and eight body-local XY Path points. Implement shared preprocessing and fail-closed validation.
3. Implement a sealed-evidence adapter that verifies host seals and source hashes, reconstructs fixed 40-step teacher labels, preserves paired groups, and prevents group split leakage. Keep old crossover and new teacher-capture protocols separate.
4. Implement a small CNN plus Path MLP plus state projection, an inference API with variable K and candidate mask, and checkpoint compatibility checks.
5. Train a reproducible pilot on the 12-trial, one-group ROS-free crossover using common-goal progress. Treat training as an overfit/pipeline smoke; no independent validation or generalization claim. Save checkpoint, history, provenance, and ranking diagnostic.
6. Test preprocessing, dataset, label availability, model, checkpoint, ROS/Isaac-free import, and existing EAT behavior. Document student handoff and future ROS2 demo connection without implementing the demo.
