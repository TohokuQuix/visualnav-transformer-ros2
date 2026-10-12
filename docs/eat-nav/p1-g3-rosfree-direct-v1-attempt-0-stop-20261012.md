# P1 G3 ROS-free Direct runtime attempt-0: pre-Path stop

## Decision

**Fail — pre-Path startup path-resolution defect.** Isaac/Kit launched and Python main began, but the single authorized process stopped before settling, core construction, Path activation, Actor control, or physics stepping. There are **0/4 candidate trials and zero closed-loop Outcome rows**. Do not use it for ROS-vs-ROS-free closed-loop comparison. There was no automatic retry; the approved one-process runtime scope is exhausted for this attempt.

## Frozen provenance and storage

The runtime-disabled preparation root `/home/kojima/eatnav-experiments/p1-g3-rosfree-direct-v1-prep/` remains unchanged. A separate attempt root was created at `/home/kojima/eatnav-experiments/p1-g3-rosfree-direct-v1-attempt-0/`, backed by the existing Isaac bind mount. The attempt manifest explicitly authorizes only `batch-first-differential-0-period-0`; its SHA-256 is `4e56a7873670d991eeef95f99372f7256a2807276998a8572e9730e0915403d7`. Assignment SHA is `6bf378d7288e023ed7460b79b6694d58a647d186e6be2af00255e7ce33776619`. Simulator source SHA is `2225fb3892165598cb9c2a6a11a3e741d54bf237724235775381fd9b62bd6dd8`; host source SHA is `4e0a80ef6128118ef554ca41e6a2a48490a3ae4e3d3d508a476db44ee608806f`. The attempt records preparation and stage-marker SHA values and rechecks all 197 staged source/input files. The Direct Path SHA remains `4ec126da6c09e7ebc02f2febba35666ef57a334cec9745301554a13aa1d0155e`.

The attempt was created by `scripts/eat_nav_p1_g3_rosfree_authorize.py` (SHA-256 `dd13ca40075861ba07bb868da81e513a4b1a292688b380374f7f7ac7a047cd48`). The detached outer launcher SHA is `fe63c00d4cd7b069766cb8e7e412431584e40fb8a7075f375c733d2a298dd58e`. It invoked only the ROS-free host and the existing Isaac owner wrapper. No follower/bridge/TF service/router was started by this attempt. The command and host prelaunch evidence mark external-router use false. Because the simulator stopped before control, live ROS-free closed-loop operation itself is not demonstrated.

## First failure and scope reached

`source.py:382–385` retained an inherited guard requiring the output path to be under `/workspace/isaaclab_onix/.eatnav-p1-long-horizon-v1`. The new persistent attempt is under `/workspace/isaaclab_onix/.eatnav-p1-g3-rosfree-direct-v1-attempt-0`. This is a source-level path-contract mismatch, not a measured Path, physics, follower-core or observation-adapter failure. The simulator wrote `output/diagnostic-error.json` with `ValueError: G3 output must be under the host-backed persistent root`. The host correctly converted the diagnostic marker into failure despite native exit 0; pre-Path approval was never reached. `path-activation.json`, four traces, and `complete.json` do not exist.

The fix should be a new source/protocol version that validates the actual bind-backed attempt root against the frozen manifest, rather than accepting arbitrary output paths. It requires a new source SHA, fresh attempt root and separate runtime approval. This failed root and seal must remain unchanged.

## Resource, cleanup and seal

The host recorded 9 GPU brackets; all 9 attribution checks passed, 5 contained an owned GPU sample. Peak device-total memory was **701.31 MiB**; peak attribution gap **89,812,992 bytes** (<256 MiB); peak container anon+shmem **805,908,480 bytes**; peak active root **1,318,867 bytes**; minimum host available RAM **92,396,306,432 bytes**. Host elapsed time was **5.79 s**. No resource limit caused the stop.

Native exit was **0**, outer launcher exit **1**, owned residual processes **0**, cleanup signals **0**. All simulator, heartbeat, resource and log writers were closed before the failed/inconclusive host seal. Read-only seal verification passed for **230 files**, seal SHA-256 `06a6b5fe846c1f537767823ecd7ff26c07c2ca90e476728f8dbad5e45a183610`. Host and container SHA match for source, attempt contract and `diagnostic-error.json` (error SHA `806b9b2e9523e3972190d6814e5309abbac31a179856adc0ea6b6a41e222b945`). The original ROS reference run's seal was verified read-only as protocol-valid (297 files); it was not altered.

## Consequences

Pre-Path physical/Actor/shared DEM and live material gates were **not reached**. Per-step command, observation latency, Actor, Goal/recovery, episode, arrival/hold/censor and closed-loop ROS comparison are **not assessable**. Measured recovery parity remains unavailable. Direct/curved crossover, additional repeats, ROS v4 continuation, evaluator training and collection are not justified by this run.
