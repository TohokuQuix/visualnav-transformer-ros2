# Expanded E1 pilot: X2 pass, anchor config contract stop

Date: 2026-10-05 JST. Output root:
`/tmp/eatnav-expanded-feasibility-approved-resume-20261005`.
The earlier monitor-stop root remains separate and excluded from outcomes.

## External router and readiness

The specified `external-router/router.stdout.log` records a new operator-owned
router, ZID `23466e323413a4e4c65e887017a57092`, PID `2852834`, start tick
`178039512`. Its stderr log is empty. Both logs matched their recorded
SHA-256 in the final file inspection. Host and container endpoint checks passed
at X1 and X2.
The runner did not start or stop the router.

X2 passed three stable snapshots of the actual bridge and eight production
followers: one Path subscriber per branch, ready status and TF, advancing
clock, and zero accepted pre-Path commands. No candidate Path was published.
The frozen five states, 50-row schedule, candidate paths, seeds, order, and
runtime thresholds were retained.

## First anchor attempt

One `expanded-train-000-anchor_only-00` process reached global step 20.
Its preliminary anchor diagnostic reported position error 0 m, orientation
error 0.000593 rad, velocity error 0, DEM RMS 0 m, mask IoU 1.0, and prefix
command difference 0. The raw DEM history SHA differed; the intended DEM gate
uses numeric RMS and mask IoU. These values are **not a passed formal anchor**.

Before formal gate completion, `compare_anchor()` raised
`KeyError('velocity_max_abs')`. All five frozen expanded state configs lack
`velocity_max_abs` and `command_max_abs`; the original E1 anchor config has
these keys with limits `0.0001` and `1e-05`, respectively. The other four
anchor limits match the original E1 config. This is an input/config contract
defect discovered before map Path generation or publication, not an observed
numeric gate failure. The simulator returned 0 during cleanup. The run folder
was sealed; the runner stopped without an automatic retry.

Peak GPU was 5,838 MiB, below 6,656 MiB. Peak cgroup anonymous memory was
2,483,965,952 bytes; cleanup recorded zero owned residual processes.
No 120-step trace, formal anchor gate, map Path freeze, duplicate gate,
candidate process, or candidate outcome exists in this root. Counts are one
new anchor attempt, zero accepted anchors, zero candidates, zero completed
measurement processes. Outcome, ranking, feasibility, and evaluator-readiness
questions are `NA / no valid candidate runs`.

## Offline prevention and disposition

The runner now checks every state config against the SHA-verified original E1
anchor-limit map before any simulator launch. A test reproduced the missing
keys; the guard now rejects the current frozen root offline with
`missing=['command_max_abs', 'velocity_max_abs']`. The guard also rejects a
changed numeric threshold. It does not repair or rewrite inputs.

The failed root and its SHA ledger remain intact. No threshold, model, Path,
seed, state, order, follower, or dynamics were changed. The authorized attempt
was consumed and the frozen retry count is zero. A further attempt would
require an explicit decision to create a fresh output root with the two
missing config keys restored to the original E1 values, followed by a new
input SHA audit and preflight. No candidate rollout is authorized by this
report.

## Evidence

- `external-router-provenance.json`, `x2-readiness-gate.json`,
  `router-launch-provenance.json`, `prelaunch-freeze.json`
- `pilot-started.marker.json`, `status.json`,
  `anchor-attempt-1.measurement-invalid.json`
- `runs/expanded-train-000-anchor_only-00/timing-events.jsonl`,
  `failure-0.marker.json`, `simulator.log`, `final-status.json`,
  `sealed-tree-manifest.json`
- Original E1 `inputs/state-000/config-120.yaml`, SHA-256
  `d89e7853cc72ce0e0090623f6a4e300637cd6f35efa2cd7058ddf3d1720be605`
