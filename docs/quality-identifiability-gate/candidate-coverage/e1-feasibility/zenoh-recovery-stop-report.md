# E1 external Zenoh recovery: attempt-1 stopped at cleanup gate

## Decision

**`attempt_1_measurement_invalid_stop`.** The external-router readiness gate passed. The first direct repeat (attempt-1) reached a 120-step simulator trace and a SHA-consistent `run-complete.json`, but the per-PID monitor failed its required cleanup marker verification. The frozen schedule stopped immediately. No second direct repeat, lateral candidate, or second StateGroup was started. The attempt-1 trace is preserved for diagnosis but excluded from the valid E1 process count and Outcome comparison. Attempt-0 remains `measurement_invalid_pre_path` in its original output root.

This is neither an E1 candidate-generator failure nor an E1 feasibility pass. The duplicate gate and P→C→R→Y analysis are **NA**. No automatic retry or model/control/threshold change was made.

## Recovery prerequisite and prelaunch evidence

The canonical ROS Jazzy `rmw_zenohd` binary was run outside the E1 runner in the dedicated detached tmux session `eatnav-e1-external-router`. It used the same command pattern and packaged default router config as previous successful simulator work. It listened on `0.0.0.0:7447`; the host bridge/follower and host-networked Isaac container used `rmw_zenoh_cpp`, ROS domain 0, and localhost port 7447. The launch command, binary/config SHA, PID/start, host/container environment and endpoint are in `router-launch-provenance.json`. The router was owned by this experiment, not by the E1 runner.

The candidate-free diagnostic used the real bridge and eight production follower processes. Only empty Paths were published; all eight followers received and ignored the clear messages. Three consecutive snapshots showed one subscriber per branch (eight total), all followers alive and publishing status, all TF frames ready, an advancing logical `/clock` from 2,300,000,000,000 to 2,300,100,000,000 ns, and zero accepted pre-Path commands. Router/bridge logs had no connection warning. The first diagnostic command failed before starting ROS because its shell replaced the ROS Python path; the corrected candidate-free smoke passed. Neither smoke is an E1 candidate run.

The new output root is `/tmp/eatnav-e1-zenoh-recovery-20261004/`. Frozen input copies, the 20-row original manifest, 18-row continuation manifest and order, 14 map Path SHA, G0-E anchor SHA, checkpoint/reference config, source/binary SHA, failure preservation, recovery commits and clean launch HEAD are in `recovery-provenance.json` and `recovery-launch-provenance.json`. The recovery launch HEAD was `9fbdf73a97e473b4c9a883e7ff31bb0043ffa9e9`. The earlier `source-freeze.json` was written during prelaunch code preparation; the launch provenance records the actual clean committed state. Forecasts were 2,926 s for 18 processes, about 5,548 MiB GPU peak and 99,961,280 bytes total storage, within the frozen 60-minute, 6,656 MiB and 800 MB limits.

## Attempt-1 measurements and first failing signal

| Item | Evidence |
| --- | --- |
| Anchor and four-frame DEM gate | Both pass |
| Path/command protocol | 120 steps; independent exact-stamp inspection passes |
| Simulator/bridge exit | Both exit code 0; no owned residual processes |
| Run marker | Present; all listed output SHA-256 match |
| GPU/storage | Peak 5,667 MiB; recovery root about 8.7 MB at audit |
| Monitor | `simulator_PID_ownership_lost`; `timing-monitor.done.json` status `failure` |
| Lifecycle | Monitor failure at wall ns 1791043111044148185, 1.600 s before Docker exec returned at 1791043112644108246 |

The monitor's current condition treats disappearance of the simulator process tree **while the host Docker exec process still exists** as ownership loss. Here Isaac Python had completed and Docker exec had not yet returned. The run marker was written later, so the monitor had already stopped before it could verify that marker. This is a cleanup-state race, not evidence that the 120-step simulator or ROS command path failed. The runner correctly required `timing-monitor.done.json` to report `complete`; it stopped when that requirement failed.

The generated `recovery-stop.json` mechanically reports `completed_valid_processes: 1` because it counts `run-complete.json` files. That count does **not** incorporate the failed monitor gate. The authoritative final decision is **zero protocol-valid E1 processes, one diagnostic trace acquired**. It must not be used as a duplicate repeat or Outcome.

The 60-minute continuation clock began at 2026-10-04 00:55:50 JST and the runner exited at 00:58:33 JST, 163.1 s later. The attempt-1 process took 149.2 s. The previous failed attempt-0 took 50.5 s separately. Router startup and recovery preparation began around 00:43:41 JST and are outside the continuation clock; they are not hidden in the 163.1 s measurement.

## Cleanup and scope

After the stop, the dedicated router PID 2046175 and its tmux session were terminated. Port 7447 had no listener; the simulator, bridge, follower and continuation processes had no owned residuals. Cleanup evidence is saved in `cleanup-evidence.json`. No unrelated process was signaled. No E1 runner router lifecycle, candidate, Path, state, seed, follower control law, simulator dynamics, model, threshold, or downstream learning was changed.

The next decision requires a separately reviewed cleanup-monitor protocol that distinguishes normal Isaac Python exit from ownership loss while Docker exec and marker flush are in progress. The saved attempt-1 result and its SHA allow that question to be assessed without rerunning it, but this report does not reclassify it as valid or authorize further E1 rollout.

## Artifact locations

The new root contains `router-readiness.json`, router and follower logs, `attempt-0-preservation.json`, `recovery-classification.json`, both continuation start markers, frozen input/source provenance, the attempt-1 raw trace and SHA markers, per-PID resource log, `attempt-1-lifecycle-audit.json`, `final-decision.json`, `cleanup-evidence.json`, and `artifact-sha-ledger.json`. Large traces and logs remain outside Git. The E1 source and focused tests passed **37/37** immediately before launch.
