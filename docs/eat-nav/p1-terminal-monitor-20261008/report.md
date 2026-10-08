# P1 terminal-aware monitor: pre-candidate stop

Decision: `p1_partial_measurement_review_required`. T3, T4, T5, X1 and X2 passed. The frozen 28-process P1 rollout did not start; candidate Paths, candidate control steps and Outcomes are all zero.

The terminal-aware lifecycle accepted the valid pre-close completion marker followed by normal owner PID disappearance. The zero-physics T3 ended with native exit 0. The shard-B T4 completed 200/200 physics steps for four poses, 16 DEM frames and eight Actor records, with native exit 0 and no owned residual. Its 33 GPU brackets peaked at 5883.562 MiB device use (7,680 MiB cap), 122.398 MiB gap (256 MiB cap), 3.64 GiB anon+shmem (6 GiB cap), and 24.716 MiB sampled storage (1 GiB cap). The focused suite passed 72 tests.

The external operator router X1 passed with live PID/start-tick identity, ZID, listener and log snapshots. Candidate-free X2 passed with the actual bridge and eight production followers, three readiness snapshots, advancing clock, TF and empty Paths, with no adopted pre-Path command. The router was neither started nor stopped by this run. The post-X2 baseline and follower cleanup passed.

Before writing `pilot-started.marker.json`, the 28-row overlay was checked against the frozen simulator and anchor contracts. The manifest still identifies `eat_nav_e1_sim.py` (SHA `9e2a40f0b283d4581219302c3f2f0782b00ca825e07653bbe3924dd5c47cec07`), which requires a `g0e-train` episode key and 100-env recorded action prefix. All four selected P1 anchors are radial settled-pose records and lack those fields. The current terminal-aware host runner supports only preflight and candidate-free smoke. The prior r5 prelaunch contract gate independently records the same incompatibility. Therefore no candidate process was launched; synthetic episode metadata or a changed simulator would alter the frozen experiment.

A separately versioned radial candidate simulator/launcher with the same reset, physics, bridge/follower, Actor and Outcome interfaces is needed before another 28-row attempt. Its input contract and terminal-aware candidate execution path need offline tests and a candidate-free end-to-end validation before the pilot can be started. No retry, training, expanded collection or robot control was performed.

Authoritative artifacts: `/tmp/eatnav-p1-terminal-monitor-20261008-r1`. Prior roots and the operator router logs were left unchanged.
