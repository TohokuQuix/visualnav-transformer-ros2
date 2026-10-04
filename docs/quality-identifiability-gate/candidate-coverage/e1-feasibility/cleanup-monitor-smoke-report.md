# E1 cleanup monitor: one-shot lifecycle smoke

Decision: `cleanup_protocol_smoke_fail`. No E1 attempt-2 or candidate continuation started.

The saved attempt-1 evidence supports `normal_finalization_misclassified_as_pid_ownership_loss`: the 120-step trace, simulator marker and run-complete SHA, Docker exec exit 0, bridge exit 0, and residual count 0 agree. The first observed simulator PID absence was 1.600 seconds before Docker exec returned. Attempt-1 remains protocol-invalid and is not promoted.

The monitor now separates runtime safety from finalization, with a fixed 30-second finalization grace and a final `protocol-valid.json` requirement. The 15 requested cases and additional lifecycle/router cases passed in 57 related tests. The old PID-only end condition is removed. Frozen behavior and resource thresholds were not changed.

The external Zenoh readiness gate passed: three stable samples had eight follower subscribers, advancing `/clock`, TF ready, empty Path clear, and zero pre-Path commands. A single non-E1 `readiness-002` 120-step smoke was then started. It failed the frozen 90-second heartbeat rule. Its last written `timing-progress.json` was global step 75, while the simulator log shows step 97 entered; the telemetry source calls `timing_progress` in its prefix loop but not in its suffix loop. The monitor therefore correctly stopped an unobservable live run at 90.134 seconds. There is no completed 120-step trace, no `run-complete.json`, and no protocol-valid marker. Bridge exit was 0 and owned residual count was 0.

This fixture cannot validate `RUNNING → FINALIZING → COMPLETE` without a new smoke run using a source that emits suffix heartbeat. The E1 simulator source has a suffix `timing_progress` call, but it was not run because the approved smoke failed and the instruction forbids automatic retry. The stopped fixture is excluded from every E1 Path/Outcome/oracle count.

Output root: `/tmp/eatnav-e1-cleanup-fixed-20261004`. It contains the monotonic attempt-1 timeline, root cause, old artifact SHA pointers, frozen manifest/input provenance, 14 unique Path SHA identities, external router readiness, smoke logs and SHA ledger, and `cleanup-fixed-final-decision.json`. Counts: E1 completed traces 0, E1 protocol-valid processes 0, E1 measurement-invalid processes 0; the separate lifecycle smoke has one measurement-invalid attempt.

Next decision requires review of the smoke fixture telemetry and explicit authorization for another one-shot lifecycle smoke. Attempt-2 and the 18-process continuation remain blocked by this gate.
