# E1 feasibility pilot: pre-Path infrastructure stop

## Decision

**measurement_invalid; stop.** The first of 18 authorized fresh candidate processes stopped before any Path was published. No E1 candidate outcome, direct duplicate comparison, or P→C→R→Y result exists. The two previously completed G0-E anchor-only runs remain valid historical artifacts; they were not rerun. No later E1 run was started and no automatic retry was made.

The dedicated worktree is `/tmp/eatnav-vnt-e1-feasibility`, branch `exp/eatnav-e1-feasibility`. The frozen implementation commit at launch was `ac09473bb418326e7b71127dc7995d86466bc739`. Raw artifacts are under `/tmp/eatnav-e1-feasibility-20261003/`; this report does not overwrite G0-E artifacts. The launch provenance records a clean committed worktree. `source-freeze.json` was generated earlier, before that commit, and its dirty/HEAD fields describe that precommit preparation snapshot.

## Actual sequence and evidence

| Stage | Result |
| --- | --- |
| G0-E anchor-only reuse | 2/2 SHA-identified prior results; not rerun |
| New E1 candidate process | 0/18 complete; first direct repeat attempted once |
| Anchor gate | Pass; position, velocity, command, action, goal and current DEM within frozen limits |
| Four-frame DEM history | Pass; all four RMS 0 m, mask IoU 1.0, frame identity/timestamp/shape/axis/resolution matched; raw SHA also matched |
| Pre-Path bridge readiness | Fail: 8/8 Path subscriber counts remained zero through timeout; all 8 follower processes were alive; TF ready and `/clock` present |
| Path and 120-step suffix | No Path published; zero measured suffix steps; trace is `[]` |
| GPU and cleanup | 5,667 MiB device peak, below 6,656 MiB cap; no owned residual processes; run elapsed 50.49 s |
| Saved storage | 3,369,708 bytes at audit, below 800 MB cap |

The bridge log explicitly says `rmw_zenoh_cpp` could not connect to a Zenoh router. The follower callback logs remained in `no_path` mode. A router was absent when cleanup was inspected. These observations strongly support missing Zenoh discovery/connectivity as the immediate readiness failure. They do not establish whether router absence was the only cause; no router-restored E1 run was authorized or performed. The implementation's top-level `docker cp .../live-0.json` error is a secondary effect: the simulator's pre-Path timeout produced no live result. The monitor's `simulator_PID_ownership_lost` notice likewise arrived after the simulator stopped; it is not evidence of a primary GPU or memory failure.

The failure marker's partial trace SHA-256 is `42db769bfbdb80f12e2b64c9d6f537c3df16731c0bab67c6959673bc03213041`; the copied host trace matches and decompresses to an empty list. The complete input SHA list is in that marker. `anchor-gate.json`, `bridge.log`, `simulator.log`, follower logs, timing/resource logs, and `artifact-sha-ledger.json` are retained under the output root. The runner left an atomic `E1_STOPPED` status and exited; detached tmux and owned simulator/bridge/follower processes are no longer running.

## E1 questions that remain unanswered

Both direct duplicate gates are **NA**. The number of command or realized trajectory modes, candidate-versus-duplicate effect, P→C→R→Y retention, Outcome ranking/spread, high-curvature negative examples, and stack/tilt/termination support are **NA**. This infrastructure-invalid attempt must not be treated as a negative candidate Outcome. The E1 generator is neither supported nor failed by these data. StateGroup `g0e-train-001` was not started.

## Boundary and next decision

The pre-existing frozen safety rules prohibit automatic retry, so the pilot remains stopped. A future continuation would need a separately authorized recovery protocol that verifies Zenoh router readiness **before** launching a fresh candidate process, preserves the original state/Path/seed/order and failed attempt record, and defines how a replacement first direct repeat is counted. No threshold, model, follower control law, bridge exact-stamp protocol, or candidate definition was changed after observing this failure. E1 candidate rollout, scaled collection, and downstream learning remain unapproved.

## Verification

Focused prelaunch tests: **46 passed** (`/tmp/eatnav-e1-feasibility-20261003/focused-test.log`). Artifact integrity: copied failure trace SHA matches the simulator marker; input/provenance checks passed before launch. The cleanup check found no owned residual processes. P→C→R→Y figures were not generated because no valid Path rollout occurred.
