# E1 external Zenoh router prerequisite

The E1 runner does not start or stop the Zenoh router. Start the canonical ROS Jazzy router **before** the E1 readiness smoke, in a detached session that survives SSH/VS Code disconnects. The dedicated session for this recovery is `eatnav-e1-external-router`:

```bash
tmux new-session -d -s eatnav-e1-external-router "bash -lc 'source /opt/ros/jazzy/setup.bash; exec /opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd >> /tmp/eatnav-e1-zenoh-recovery-20261004/router.log 2>&1'"
```

This is the same `rmw_zenohd` binary and no explicit router config used by prior successful simulator pilots. Its packaged default router config listens on `0.0.0.0:7447`. The host bridge/follower use `rmw_zenoh_cpp`, ROS domain 0, and the packaged default session endpoint `tcp/localhost:7447`. The host-networked Isaac container uses `/etc/zenoh/ros2-client.json5`, which specifies `tcp/127.0.0.1:7447`. The router PID, binary/config SHA, process start, endpoint, environment, and log are recorded in `router-launch-provenance.json` under the recovery output root.

Readiness requires the router PID and endpoint, a clean router startup log, status messages from all eight actual production follower processes, one subscriber per branch Path publisher (eight total), advancing `/clock`, TF readiness in every branch, three stable subscriber samples, empty-Path clear, and zero accepted pre-Path commands. The diagnostic uses the real bridge/follower ROS graph and publishes **only empty Paths**. It is separate from candidate rollout and contributes no E1 Outcome. A router connection warning or any missing condition stops the run before attempt-1.

The previous attempt-0 is retained under `/tmp/eatnav-e1-feasibility-20261003/` with its SHA ledger and remains `measurement_invalid_pre_path`. The new recovery output root is `/tmp/eatnav-e1-zenoh-recovery-20261004/`. The E1 runner does not own the dedicated router process; an experiment operator may stop only the router identified in `router-launch-provenance.json` after the recovery run has ended. Existing routers of unknown origin must not be signaled.
