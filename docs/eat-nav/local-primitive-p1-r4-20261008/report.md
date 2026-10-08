# P1 r4 prelaunch result — 2026-10-08

**Primary decision: `p1_measurement_or_protocol_invalid_stop`.** Subreason: the operator-owned external Zenoh router and its required provenance were unavailable at X1. The corrected GPU contract and fresh pre-X1 baseline passed. X2, the post-X2 baseline and the 28 candidate processes did not start; no candidate Path was published or candidate control step taken.

## Preserved experiment inputs

The r4 branch/worktree is `exp/eatnav-local-primitive-p1-r4` at `/tmp/eatnav-vnt-local-primitive-p1-r4`, based on r3 commit `3e9bd9db0f0ab65ce12e939600564e7a996fa257`. All 18 r3 artifact-ledger entries and 21 source paths were SHA rechecked. The r3 selected anchor artifact, 28-row manifest, Path definitions and input freeze were copied byte-for-byte to the new r4 root. Anchors remain `first-differential`, `second-differential`, `first-graded`, `second-graded`; seed 123, the five local Paths, global goal `[0.75, 0.0]`, row order and retry count zero are unchanged. r1–r3 roots and the r3 failed decision were not altered.

## Corrected GPU resource contract

`p1_local_primitive_resource_contract_v2` counts known non-owned PID/start-tick graphics memory in **device total**, without treating it as the 256 MiB uncertainty allowance. Unresolved GPU memory remains required to be zero, the bracket gap must be at most 256 MiB, and the device-total hard cap remains 7,680 MiB. Before X1, device total must additionally be at most 1,024 MiB, with unchanged GPU identity and no unexpected compute workload.

The fresh accepted baseline measured the audited RTX 5070 Ti UUID `GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26`, 16,303 MiB total, driver `580.173.02`, NVML `13.580.173.02`, MIG `[N/A]`, device use **416.813 MiB**, known other-process use **370.547 MiB**, unresolved **0 bytes**, bracket gap **46.266 MiB**, and temperature **48 °C**. Four other processes had known PID/start-tick and graphics classification; no compute process was present. The supported full `nvidia-smi -q` health query succeeded without a recovery action. The 7,138.8 MiB forecast and 541.2 MiB headroom were confirmed from the historical 5,949 MiB peak, with no second 20% margin. Wall, storage and anon+shmem forecasts remain under their frozen caps.

A first r4 diagnostic root (`...-r1`) recorded a false `gpu_health_error` because `nvidia-smi -q -d ERROR` is unsupported on this driver. This happened before Path publication or control steps. One authorized pre-control remediation cycle fixed the health-query invocation with RED→GREEN tests and used a fresh authoritative root (`...-r2`). The invalid diagnostic root was not promoted.

## X1 stop and unmeasured stages

At X1, no `rmw_zenohd` PID/start-tick was available, `127.0.0.1:7447` refused connection, the historical router log files were absent, and no Isaac container was running. The runner did not start, stop or restart the router or container. Consequently, router ZID/log integrity and host/container connectivity cannot pass. X2 and all rollout/Outcome/duplicate/relabel artifacts are **not measured**, rather than failed candidate results. `pilot-started.marker.json` does not exist.

## Verification and artifacts

The focused P1/GPU suite passed 23 tests. The r4 tests cover the r3 baseline reclassification, compute/PID ambiguity, unresolved bytes, sampling gap, device cap and health-query correction. The independent final review found no Critical or Important defect in the executed pre-X1 scope. A later attempt must remap the frozen manifest output directories from the r3 root to its own fresh run root, while preserving row order and Path SHA, before any candidate starts; this was not implemented because X1 failed. The SHA ledger, frozen inputs, corrected contract, raw NVML table, X1 observation, stage status and final decision are saved in this report's directory and in `/tmp/eatnav-local-primitive-p1-r4-20261008-r2`. The relevant figure is `gpu-contract-v2-baseline.png`.
