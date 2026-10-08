# P1 r5 attempt 2: external-router X1/X2 and pre-candidate stop

**Primary decision: `source_or_artifact_integrity_invalid`.** X1, candidate-free X2 and both GPU baselines passed. The 28-process pilot did not start: zero candidate Paths were published and zero candidate control steps occurred.

## Router provenance

The external router log directory was fixed to `/home/kojima/eatnav-router-logs/p1-r5-20261008-attempt2/` and read only. At X1, `router-launch.pid` identified launch PID **98913** (start tick **1646269**); its `rmw_zenohd` child was PID **98923** (start tick **1646280**). Both log ZID entries matched `3110e0eea0702bccfcc86507623b3971`. PID 98923 owned the `0.0.0.0:7447` listener, and host plus existing `isaac-lab-onix` container connected to `127.0.0.1:7447`. No disallowed warning appeared. Codex did not start, stop, restart or write to the router or its log directory.

The separate **X1** and **final** SHA snapshots are in `external-router-provenance-x1.json` and `external-router-provenance-final.json`, respectively. Each captures SHA, size and timestamp for `router.stdout.log`, `router.stderr.log`, and `router-launch.pid`. In this attempt the files did not append between snapshots, so each file's X1 and final SHA happens to be equal; these remain two separately timed measurements. Final PID/start tick, ZID and listener matched X1.

## Readiness and resources

The fresh pre-X1 GPU baseline passed: RTX 5070 Ti UUID `GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26`, device use **487.313 MiB**, known other-process use **422.625 MiB**, unresolved 0 bytes, bracket gap **64.688 MiB**, finite temperature 48 °C. Known graphics memory was included in device total under the corrected v2 contract.

Candidate-free X2 used the actual bridge and eight production followers. The follower binary and parameters matched their frozen SHA. All three snapshots showed eight single Path subscribers, live followers/status, ready TF, advancing `/clock`, empty Paths and zero accepted pre-Path commands. The eight follower processes cleaned up with zero owned residual. The post-X2 GPU baseline also passed: device use **513.313 MiB**, unresolved 0 bytes, bracket gap **64.688 MiB**. The existing Isaac container was started for container connectivity and left running; it was not recreated. The external router remained operator-owned.

## Why the pilot stopped before the first row

The frozen P1 manifest's `simulator` SHA identifies `eat_nav_e1_sim.py`. That source accepts only a `g0e-train-*` state with `episode_key`, `step`, `local_step`, recorded DEM and a **100-env episode prefix** of shape `(step+1, 100, 6)`. The four selected P1 radial anchors are settled-pose records and have none of those episode/prefix fields. The manifest also directs run outputs to the preserved r3 root rather than this attempt's root. All 21 frozen source SHA checks passed, so this is a real input-contract mismatch, not a changed source file.

Creating synthetic episode-prefix fields or altering the frozen simulator contract would change the experiment definition. No candidate process was launched and `pilot-started.marker.json` was not written. Before another attempt, a versioned radial-anchor execution contract must specify deterministic anchor reconstruction and DEM/Actor equivalence, an output-path overlay, and the exact unchanged physics/follower/Actor interfaces. The anchor, Path, model, threshold and Outcome definitions remain frozen.

## Artifacts

The focused P1/GPU/router suite passed 28 tests. A separate historical E1 smoke-fixture suite had six failures because its old `/tmp` fixture is absent after reboot; those tests do not exercise this X1/X2 attempt.

The authoritative root is `/tmp/eatnav-local-primitive-p1-r5-20261008-attempt2`. Small JSON evidence and this report are also committed in the r5 worktree. Raw X2 follower/bridge logs remain in the authoritative root. The SHA ledger records both copies and the raw log hashes. Historical r1–r4 roots were not modified.
