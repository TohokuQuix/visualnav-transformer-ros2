# ROS2 + Isaac Sim GUI demo connection design (not implemented)

The Evaluator core remains a plain Python/PyTorch component. A future ROS2 adapter owns subscriptions, TF lookup, stale-data checks, candidate generation, score publication, and Path publication. The adapter must call the exact same `prepare_inputs` and checkpoint loader used by offline evaluation.

## Proposed flow and contracts

1. Subscribe to a local GridMap DEM with height and validity, robot pose/TF, the robot's pre-Path 22-value Actor observation, and a goal `geometry_msgs/PoseStamped` from RViz2. Each observation needs a frame and timestamp. Use the existing measured GridMap-to-`SourceGrid` mapping and `robot_dem` conversion; verify 20×20 at 0.1 m, axis direction, height scalar offset, and unknown mask.
2. Resolve robot pose and goal into the local map frame at a common stamp. Reject stale or unavailable TF; never treat tile-local XY as body XY. Transform each candidate Path to robot-base-yaw XY before scoring.
3. Begin with the frozen P1 Direct/Left/Right bank for a demo that stays within the pilot input distribution. A goal-centered arc generator can later be a separate module, but its endpoint/curvature distribution differs from the P1 bank and needs new data before interpreting its scores as calibrated quality.
4. Call `Evaluator.predict` with K candidate Paths and a boolean validity mask. Publish diagnostic scores, Path IDs, checkpoint SHA, selected index, and candidate geometry for RViz. If all candidates are invalid or inference times out, publish a stop decision.
5. Publish only the selected `nav_msgs/Path` through a dedicated selector output. The existing follower core should remain the sole control-law implementation; the ROS node should not copy its controller logic. Whether ROS wraps the C++ core directly or the simulator uses its pure binding is a deployment choice requiring separate integration testing.

Suggested namespaced topics are `~/goal` (`PoseStamped`), `~/local_dem` (GridMap), `~/candidate_paths` and `~/selected_path` (`nav_msgs/Path` plus a documented candidate ID mapping), and `~/scores` (a versioned score message or diagnostic array). Final topic names, QoS, and GridMap layout must be frozen against the actual `onix_policy_bridge` and simulator launch configuration before implementation.

## Safety and evidence gates

On missing DEM, invalid mask/state, stale TF, no valid Path, checkpoint/input schema mismatch, inference timeout, ROS interruption, or follower failure, the adapter must stop publishing movement commands and issue the existing safe-stop Path/command contract. It must not continue an old trajectory. Checkpoint load is fail-closed on schema and label mismatch. Candidate Path publication should be traceable by score, Path SHA, stamp, and selector decision.

Before any GUI closed-loop claim, verify the live ROS DEM/Actor adapter against sealed preprocessing fixtures, TF frame conversions, Path publication exactly once per decision, follower command timing, RL Actor observation semantics, and fail-safe stop behavior. The ROS-free pilot checkpoint is not a validated safety or navigation policy. No ROS node or GUI demo is provided in this phase.
