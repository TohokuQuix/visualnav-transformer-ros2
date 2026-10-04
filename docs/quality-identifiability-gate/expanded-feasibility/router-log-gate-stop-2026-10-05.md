# Expanded feasibility continuation: X1 evidence stop

The previous `X17_STOPPED_ROUTER_UNAVAILABLE` root remains unchanged and contains zero new process attempts. A separate resume root was created at `/tmp/eatnav-expanded-feasibility-resume-20261005`.

X0 read-only re-audit passed: 18 old sealed E1 runs, corrected addendum, five frozen complete-only train StateGroups, 35 robot-local Paths, 50 ordered process rows, source/config/input SHA, and the runtime/storage forecast. The original RL checkpoint, reference config and collection manifest SHA were rechecked in Docker. The adapted E1 simulator source changes only expanded StateGroup/scope validation and the anchor-only map Path creation path; no dynamics, reset, follower, Actor or Critic change was made. No adapted simulator has been executed.

An external canonical `rmw_zenohd` process is running with PID `2784060`, start tick `176326379`, start time 2026-10-05 04:44:55.790 JST, and command `/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd`. Host and Isaac container both connected to `127.0.0.1:7447`; router binary, router config, host client config and container client config SHA match the previous E1 provenance. No simulator, bridge, follower or monitor process was found.

**X1 cannot pass yet:** the router's stdout/stderr points to interactive `/dev/pts/0`, no persisted router log was found, and the PID has no journal entries. Therefore startup bind, connection and config warnings cannot be audited as required. The operator has been asked to provide the router terminal output or saved log location. Candidate-free X2 readiness, pilot wall clock, anchor, duplicate and candidate runs have not started. This is not a candidate attempt or retry.

The router remains externally owned. The runner did not start, stop or signal it. No automatic retry, state change, Path change, threshold change, collection, training, validation or robot work occurred.
