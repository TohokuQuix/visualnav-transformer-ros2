# P1 G3 crossover v3: first remaining process stop

## Decision

The only authorized runtime process, `batch-first-differential-1-period-0` (seed `2026101102`, Left/Right/Direct/Direct), ran once and stopped **protocol-invalid**. No retry or subsequent process started. Its four traces remain sealed failure evidence and are not added to the 12-trial conditional crossover dataset.

The first decisive host failure was the **loss of the externally owned Zenoh router launcher PID 6019**: final `router_audit()` raised `FileNotFoundError: /proc/6019/stat`. The router child PID 6026 and port 7447 were also absent when checked after the run. The router's termination mechanism is not established. No router process was started, stopped, or reconfigured by this run.

## Frozen v3 inputs

- Manifest SHA-256: `3f9c588bf002d9967632425f66cf32a57a6efdd0c7a795b0e993eba8e260b426`.
- V3 host SHA-256: `5786fd00b6c8c84d4b83a0e82702ec43529c19d1f2e466648d517338a95b8372`.
- Shared assignment validator SHA-256: `fa81c20f1d6ead95adc42f1de0c2dca5a7eb8fc698ab4d5fb9844e39794dcac3`.
- Simulator SHA-256 unchanged from v2: `9c4a8838f0fa6fc1a86ca5c5244bb0b4ebd9a76378fc47253fb6148ca57b21b3`.
- Staged input ledger SHA-256: `ac793568ca95e00297741f478467483c2514065a773cab4b17f2d09b506eeb08`.
- The external router's original stdout/stderr paths had been removed outside this run, but the live launcher file descriptors still contained the original bytes. A new read-only identity snapshot outside the batch was made from those descriptors. Its stdout SHA `745de975...d8ff8dbd9e2de4773220913e9a9a0b` matched the old period-0 router record. Before launch, PID/start ticks were `6019/306981` and `6026/306994`, ZID `87abc71b81b6bdf4a3557eaec78222aa`, and both endpoints were reachable. The snapshot is derived evidence, not a recovered original log path.

## Runtime evidence

- Pre-Path approval, material gate, and within-process strict gate passed; strict failures were zero. Each branch published its assigned Path once.
- Simulator wrote `complete.json` and exited 0. Four trace files have **233 rows each** and match their marker SHA. This verifies preservation only; the post-Path host validator was skipped after the router failure, so exact-stamp, same-episode, arrival/hold, and Outcome gates are **not certified** for this batch.
- A final resource sample after simulator completion recorded `owned_start_tick_mismatch,simulator_disappeared_in_bracket`. The host classified this at the terminal boundary, but the resource series retains one attribution failure. GPU device-total peak was **5,212.06 MiB**, container anonymous plus shared memory peak **3,500,916,736 bytes**, persistent batch peak **29,288,165 bytes**, and host elapsed **101.85 s**. There is no evidence of a GPU/RAM/storage/wall cap exceedance; GPU monitoring evidence is incomplete at the final sample.
- Native exit 0; owned simulator and bridge residuals zero. The outer launcher exited 1. The batch host seal remains `G3_crossover_batch_failed_or_inconclusive`, SHA-256 `8f804d1e786afd44841a705dd202467226c130339d9bf477ed11e92cc19163f5`, and the read-only post-launcher seal verification passed for 308 files.
- Previously sealed period 0/1 host-valid evidence and the older period-2 failed evidence were reverified unchanged. The new failed batch is not promoted into either evidence class.

## Decision for the remaining 14

Do **not** start them under the current state. The external router must be available with durable PID and stdout/stderr evidence through finalization, and a new runtime authorization must establish how to handle this Path-published, protocol-invalid attempt. The v3 assignment validator itself was not implicated by the observed host failure. The frozen remaining assignment, Path geometry, simulator, follower, checkpoint, material and numeric gates remain unchanged.

Related static/synthetic tests: **58 passed**. The repository-wide suite has unrelated collection errors in this offline environment from missing `diffusion_policy` and ROS `geometry_msgs`; they are not the v3 runtime gate result.
