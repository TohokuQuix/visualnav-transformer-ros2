# Readiness-100 command-delivery recovery: stopped inconclusive

## Decision

The primary command-delivery cause is **inconclusive**. The single permitted diagnostic replay completed all 120 steps, but the old bridge did not record callback entry, executor stack, ROS transport state, or resource counts at the missing stamp. A fresh bridge passing the same input does not establish lifecycle accumulation. No root-cause correction is justified. D6 recovery gate, measured rerun, remaining 18 processes, and postprocessing remain blocked. The original 100 completed runs and the failed attempt-0 are preserved.

The secondary cleanup failure was observed: the simulator printed `RECOVERY_FATAL` for the three-second fresh-command timeout, then remained alive until the outer 900-second limit. The original run did not log the entry/exit of `simulation_app.close()`, so the exact blocked cleanup stage is unknown. The successful diagnostic run closed normally; this does not validate cleanup on the exception path.

## Evidence and chronology

- Frozen branch and HEAD: `exp/eatnav-candidate-coverage`, `21fe8cec2cc265978118af80549d67449dba4095`; the worktree was clean before diagnostic changes.
- Original status: 100/119 processes complete, comprising six anchors and 94 rollouts. All six completed duplicate gates passed with zero difference in the checked signals.
- All 100 `*.done.json` entries match the schedule and `run-complete.json`; every listed output SHA was recomputed and matched.
- Failed input remains `readiness-100`, state `readiness-005`, `pyramid_stairs_inv`, `bank-strong_left`, logical start `2000000000000`, 120 control steps, frozen base Path SHA `32b14748eb6f0704730981a9136a1400962b62f55a08808c61c219f1e3dca23d`.
- Original follower logs show all eight branches reached `command_pre_publish` and then `tracking` at stamp `2001300000000`; control values were finite and the C++ call returned past `publishVelocityCommand`. The stamp changed from the prior stamp, so the source's duplicate-stamp suppression does not explain this event.
- At the same stamp the old bridge reported command sequence 26 for every branch, callback count 0 for every branch, no rejection or protocol error, all followers alive, one Path subscriber per branch, and TF ready. This proves failure to adopt a new command; the old logs do not distinguish ROS delivery failure from an executor or callback stall.

| Logical stamp (ns) | Follower publish sequence, all 8 | Follower return, all 8 | Old bridge adopted sequence | Old bridge callback entry |
| --- | ---: | --- | --- | --- |
| 2001050000000 | 22 | yes | null | null |
| 2001100000000 | 23 | yes | null | null |
| 2001150000000 | 24 | yes | null | null |
| 2001200000000 | 25 | yes | null | null |
| 2001250000000 | 26 | yes | 26 at later timeout; per-stamp value unavailable | null |
| 2001300000000 | 27 | yes | 26 (unchanged) | null |

The full 48-row timeline records unavailable fields as `null`. The unresolved visual USD reference warning occurred 1,984 times in the preceding successful run and 1,982 times in the failed run. It is not a failure-specific signal.

## Diagnostic replay

The diagnostic-only bridge journal, follower binary copy, and simulator copy added event logging. The production follower source, binary, and parameter SHA remained unchanged. Before the replay, ten targeted diagnostic tests and eight existing protocol/readiness tests passed. These checks did not fully exercise all ten D4 requirements against the live Docker/ROS lifecycle, so D4 was only partially satisfied when D5 ran. This ordering deviation does not authorize recovery. A later targeted run passed 19 tests in total.

One `diagnostic-attempt-1` replay used the frozen state, Path, prefix, suffix, checkpoint, and logical time. It completed 120 steps in 152.2 seconds, with 121 fresh bridge responses including reset, no rejected commands, no stamp/TF/clock mismatch, and exactly one accepted command per branch per stamp. At the original missing stamp, all eight callback entries, queue inserts, waiter wake, and TCP response occurred. The first 27 stamp command values across all eight branches matched the original follower logs exactly (maximum absolute difference 0).

The fresh bridge had one follower reset, versus 95 in the old run. During the fresh run, sampled FD count was 47–64, thread count 13–16, socket count 5–14, and RSS was 17,574–18,730 pages. Historical old-bridge FD/thread/socket/RSS and executor heartbeat were not saved; those values cannot be inferred from the fresh run. The original router log contains no diagnostic transport event. H1 executor starvation, H3 transport failure, and H4 lifecycle accumulation therefore remain unresolved; H2 lock deadlock and H5 candidate-specific follower failure have weaker support but are not formally excluded.

## Gates and artifacts

D0 evidence freeze, D1 offline timeline, D2 hypothesis matrix, D3 instrumentation, and the single D5 diagnostic replay were completed. D4 has passing targeted tests but its full live-lifecycle matrix remains incomplete. D6–D9 were not started because the root cause was not established. There was no measured retry, remaining-run resume, or candidate-coverage analysis. Host pilot plus recovery artifacts and Docker pilot plus recovery artifacts totaled about 928 MB at stop, below the 1 GB cap; a complete-resume storage forecast would need rechecking before any future run.

Raw evidence and SHA ledgers are under `/tmp/eatnav-candidate-coverage/evaluator-readiness-recovery/`: `evidence-freeze.json`, `offline-timeline.json`, `hypothesis-matrix.json`, `diagnostic-artifact-sha.json`, `analysis.json`, and `diagnosis-status.json`. Failed attempt-0 remains under `/tmp/eatnav-candidate-coverage/evaluator-readiness-pilot/runs/readiness-100/`. No production follower, model, candidate, Path, seed, or threshold changed.
