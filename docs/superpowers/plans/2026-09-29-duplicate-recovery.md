# Feasible-set duplicate recovery plan

Scope: preserve failed 0.0237 m case, compare with verified 0.0035008 m source and logs, repeat the same frozen nominal Path from independently fresh 100-env processes with logged action prefix, diagnose first divergence, use recorded command control only if live duplicate fails, then at most three fixed maneuver states if the duplicate gate passes. Never start the 56+20 rollout in this task.

1. Freeze all existing failure outputs and hashes, including exact state ID, Path, source, config, trace, bridge log and logged 100-env action prefix. Mark absent baseline fields explicitly.
2. Audit prior and current source, config and logs field by field; save verified differences and uncertainty.
3. Implement fresh-process episode-prefix runner without snapshot restore. Capture detailed anchor state, 4-frame DEM history, command/Actor/Critic, latency/action manager, contact/raycast and 40-step live follower trace; keep source and output hashes per run.
4. Run three independent detached duplicate processes, compare anchor equality and signal first divergence. If gate fails, run recorded command control in fresh processes.
5. Apply frozen duplicate gate. If passed, run at most three fixed-bank maneuver smoke states; otherwise report measurement_invalid and stop.
6. Save machine-readable results, plots and concise report. Leave full 56+20 counterfactual rollout stopped pending a later decision.
