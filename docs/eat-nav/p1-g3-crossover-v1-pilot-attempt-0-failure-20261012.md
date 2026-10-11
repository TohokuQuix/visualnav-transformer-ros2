# P1 G3 crossover runtime pilot attempt 0: pre-Path stop

## Decision

`first-differential / batch 0 / period 0` stopped before Path publication. Period 1/2 were not launched. Candidate trials: **0/12**; execution-quality comparison is **not assessable**. No old P1/G1/G2/G3 evidence, production follower, external router, checkpoint, Path, seed, physics, or thresholds were modified.

Frozen manifest SHA-256: `6a87ec0f05fb334b7968435d6c7db871ebe0ed2c5786ecb2d2be616bcba80162`; assignment SHA-256: `46ad229c69bb2024cb4b87d183204d93a1ae14bb3fdfdda00a550aa6733624c1`. Simulator source SHA-256: `a0e07d4cd043eab42d44629377d2a91f89a7b8ddd0de4a0b26ca457a824307f5`.

Raw attempt root: `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-crossover-v1-pilot-attempt-0/batch-first-differential-0-period-0/`. The existing attempt, its failed host seal, and the original source/input files are preserved. Period 1/2 stage inputs exist but neither has an `output/` directory or host seal.

## First failing signal

The simulator started under an owned PID and reached `read_live_materials()` after `env.reset()`. Composed terrain collision material rows were traversed in memory, then binding resolution of a robot collision prim failed:

`/World/envs/env_0/onix_chassis/chassis_fl_body/collisions/mesh_0/cylinder`

The exception was `ValueError: live collision material binding missing`. It was saved in `output/diagnostic-error.json` with the traceback. The simulator's shell-native exit was 0, but the host correctly classified the diagnostic marker as failure. `output/path-publication.json`, pre-Path snapshot/approval, traces, and Outcome do not exist; Path publication count in the diagnostic marker is 0. Therefore live robot material, pre-Path state equality, follower/bridge timing, and all post-Path metrics remain untested. The binding failure is consistent with the possibility that startup material randomization supplies effective PhysX per-shape values without a composed USD material binding on every robot collider. This is a source-based hypothesis, not a measured effective tensor result: the current auditor throws before writing `live-material.json` or reading/saving that tensor.

The frozen manifest explicitly requires composed robot USD binding. Treating an unbound prim plus a PhysX tensor as sufficient would change that contract's gate meaning. No such change or retry was made. A later protocol revision should first decide whether each robot collider must have a USD binding, or whether an explicitly recorded `unbound` USD status together with a complete effective PhysX tensor is the intended criterion. Robot asset/material authoring changes could alter physics and require separate review.

## Host lifecycle, resources and preservation

External router before/after audits agree: PID `6026`, start tick `306994`, launcher PID `6019`, start tick `306981`, ZID `87abc71b81b6bdf4a3557eaec78222aa`, reachable 7447 endpoints, identical stdout/stderr SHA. Host status reports 14 GPU bracket samples, 10 owned GPU samples, no monitor failures, no owned simulator or bridge/follower residual, and all host writer-state fields true. Peak measured GPU device-total was **5,084 MiB**; container anon+shmem **3,078,905,856 B**; active root **1,642,664 B**. Host elapsed time was **19.58 s**, within all frozen caps. The failure seal's stage is `G3_crossover_batch_failed_or_inconclusive`.

There is a **separate finalization defect in the outer tmux launch script**. The host sealed the root, then its traceback was appended to `pilot-host-console.log`, and `pilot-host-exit-code.txt` was created. `verify_seal()` therefore raises `post_seal_mutation:files`. Read-only ledger comparison found exactly one changed recorded file (`pilot-host-console.log`) and exactly one added file (`pilot-host-exit-code.txt`); no recorded source, input, resource, diagnostic, or cleanup file differed from its ledger SHA. The attempt remains invalid, and its seal/ledger/files were not edited to conceal this defect. For any newly approved attempt, external console/exit bookkeeping must be placed **outside** the sealed batch root or all such writers must finish before host sealing. The current frozen attempt is not reusable or promotable.

## Next decision

Stop at period 0 as required by the pilot protocol. The 3-process pilot did not provide candidate comparison data. A new attempt needs an approved material-audit contract resolution and a fresh root/source SHA, plus a corrected outer launch/finalization path. No periods, candidates, Evaluator training, or full collection were started after this failure.
