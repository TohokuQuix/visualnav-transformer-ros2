# Expanded E1 pilot: X2 pass, first anchor monitor stop

Date: 2026-10-05 JST. The new output root is
`/tmp/eatnav-expanded-feasibility-live-20261005`. The original
`X17_STOPPED_ROUTER_UNAVAILABLE` root and the subsequent X1 log-evidence stop
root remain unchanged. Their SHA audits passed before this attempt.

## Gates and fixed inputs

- X0: 5 fixed complete-only train StateGroups, 35 local Bezier Paths, fixed
  50-process order, original 18 E1 runs and coordinate addendum rechecked.
- X1: the operator supplied the full visible startup output of the external
  router. It showed ZID `64b3083dfa0bbd40d69d62530fc6b269` and no
  connection, bind, or config warning. Canonical router PID `2784060`, start
  tick `176326379`, binary/config SHA, host and container connectivity at
  `127.0.0.1:7447`, and zero pre-existing measurement processes passed.
- X2: actual bridge plus 8 production followers passed 3 consecutive stable
  snapshots. Each branch had one Path subscriber, status, TF, and advancing
  clock. Only empty Path clear messages were sent; pre-Path commands were 0.
  Follower warnings were only the expected `Received empty path, ignoring.`;
  no connection warnings appeared in the captured bridge/follower logs.
- Prelaunch commit: `fef19e5af8c27396aaabc8b5cb7d3e0110fd74e7`.
  Expanded adapter SHA: `0e8ea4639f0496ab45b4b833ab1bc0e8c6320d3419194e42ff2b8e84e0454e2e`.
  The 13 targeted tests passed before launch.

## Actual run and stop

The four-hour clock began immediately before the first anchor-only process,
`expanded-train-000-anchor_only-00` (boxes). This was one anchor attempt and
**zero candidate processes**. The anchor did not reach target global step 20;
the last monitor observation was step 9. There is no passed anchor, fixed map
Path, candidate trace, duplicate gate, or Outcome from this attempt. The run is
`measurement-invalid`; no retry or later StateGroup was started.

The anchor monitor exited with `cgroup memory exceeded frozen 8 GiB cap`.
At its final heartbeat it recorded `memory.current = 9,683,791,872 B`,
simulator RSS `5,707,522,048 B`, and device GPU `5,809 MiB`; the runner
recorded peak GPU `5,810 MiB`, below the frozen `6,656 MiB` limit. The
prelaunch cgroup sample contained `5,452,697,600 B` of file cache. The
runner's maximum observed cgroup `anon` was `2,508,648,448 B`.

The frozen E1 safety config specifies `anon + shmem > 6,442,450,944 B` as
the CPU-memory stop condition. The anchor used the older
`eat_nav_goal_subgoal_timing_monitor.py`, which instead compared the broader
`memory.current` to its default 8 GiB cap and did not record contemporaneous
`anon + shmem`. The immediate stop is therefore a **monitor protocol mismatch**.
The data do not prove whether `anon + shmem` would have stayed below the
frozen limit for the entire anchor; that remains unmeasured in this attempt.

The simulator was terminated during cleanup and exited 0. Its successful
shutdown is not a successful anchor. The run's final status records zero owned
residual processes. The external router retained PID/start tick
`2784060/176326379` after the stop. The failed run folder is sealed with SHA
`a9ce0c9b2b4f456b104b85be80f5921d2aa73ff5e7255f6a7615909288834d93`.

## Offline correction and disposition

After stopping, the timing monitor gained an optional `anon + shmem` metric
using `/sys/fs/cgroup/memory.stat`; the expanded runner passes the unchanged
cap from `e1-safety-stops.yaml`. The legacy total-memory behavior remains the
default for other callers. New unit tests model the measured file-cache-heavy
condition and a real `anon + shmem` excess; both pass after the change.
The expanded targeted suite passed 15 tests. This corrected path has **not**
been run against the simulator, and no failed output was reclassified.

The frozen retry count is zero, so this pilot remains stopped. A new attempt
requires an explicit decision and a separate output root. The state, goal,
Path, seed, order, E1 thresholds, follower control, Actor/Critic, and dynamics
were not changed.

## Requested outcome questions

New anchors reproduced: 0/5. External router remained the same process for
X1, X2, and this failed anchor; continuity across all 50 runs is untested.
Duplicate gate, P→C command modes, Actor response, realized XY/body modes,
Outcome difference, ranking consistency, and failure-positive support are
all **NA / no valid candidate runs**. Candidate-generator, Evaluator ranking
collection, and Safety-head readiness cannot be decided from this attempt.

## Evidence

- `external-router-provenance.json`, `router-operator-terminal-output.txt`,
  `x2-readiness-gate.json`, `router-readiness.json`, `prelaunch-freeze.json`
- `pilot-started.marker.json`, `status.json`,
  `anchor-attempt-0.measurement-invalid.json`
- `runs/expanded-train-000-anchor_only-00/timing-monitor.log`,
  `timing-stall.failure.json`, `simulator.log`, `final-status.json`,
  `sealed-tree-manifest.json`
