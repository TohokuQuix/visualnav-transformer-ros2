# P1 G3 period-2 continuation: stop report

## Decision

The one authorized period-2 continuation ran, but its host seal is `G3_crossover_batch_failed_or_inconclusive`. Preserve periods 0/1 as **2 protocol-valid processes / 8 valid trials**. Do not integrate the four period-2 traces into the 12-trial crossover analysis. The remaining 15 processes, Evaluator training, and full collection were not started.

The period-2 failure is a second host validator assumption about within-period Path diversity. The frozen balanced assignment deliberately has `Right, Left, Right, Left` in period 2: two distinct Path SHA values. `validate_g3_output()` nevertheless required exactly three distinct SHA values and raised `ValueError: crossover Path publication contract failed` after simulator completion. This is separate from the prelaunch assignment bug fixed before this continuation. Because Path was published and the batch sealed invalid, no retry or outcome promotion was attempted.

## Frozen provenance

| Artifact | SHA-256 |
|---|---|
| Old period-0/1 manifest | `1099036714055ccf424bf71ab00ce8f65dea597a8d2407f062dcaef8fe01c623` |
| Old host source | `ef988448e66dda2e4cc5b6611a185caa4fce45003045957d72aad1212bcc0712` |
| Old period-0 seal | `43481f2ee59c58a352ef2e41a777d6009b11b47a5ce7e6ead10468a81d612e90` |
| Old period-1 seal | `50e02055999d7acd7c1d973befa9a297f17b9bbb9937d993c8e8077349dc3a80` |
| Old period-0/1 pre-Path snapshot | `22b8721e7a3498c6fee55502eefd21db8670be36bfa4271fb6463fa5f57d59c9` |
| Continuation manifest | `d79be08e596b0cfd0fd1fc7972e98b21cdb26c8edb0b3c46f1a7d65fd8e47bc3` |
| Period-2 runtime contract | `a3a4d63aff1746682a3f836a8619317121d981cf23449a4a3ba7404fbf08b81a` |
| Assignment (all periods) | `859e5d4f31125ca975a1c0f5cb79bc627b0dd2e6c234c77c8a62b01106af54f8` |
| Corrected period-2 host source | `a5b5b580e31633c23d7fdd599b5f82de02252dc67d139d614ab97739461ddf63` |
| Simulator source, unchanged across periods | `9c4a8838f0fa6fc1a86ca5c5244bb0b4ebd9a76378fc47253fb6148ca57b21b3` |
| Assignment-validator source diff | `f0aed0a659f871c446bace51cfd930b344a3468e7962e713d915f7ff568bfb05` |
| Period-2 failed host seal | `e150351dce413ee70b37211e3c9f9cd5bb7ceb87e8286dc852dc509f6b8ca047` |

The old roots were referenced through read-only aliases. Before launch, the continuation checked both old seals, source/manifest/snapshot SHA, assignment rows, and Path roles. The source diff against the old host contains only the four-branch assignment validation change. A single-source execution across all periods is **not** claimed.

## Runtime evidence

- Cross-attempt pre-Path gate: **Pass**. Period-2 snapshot SHA is exactly `22b872...57d59c9`, the same as sealed period 0. Strict failures: 0; contact/Critic diagnostic differences: 0. Material fingerprint and live material JSON SHA also match period 0 (`84157d...bc`).
- Live material gate: **Pass**. Terrain composed binding matched the declared value; 1,084 robot colliders mapped to effective PhysX tensor `[4,271,3]` with the resolved startup material values.
- Path publication: one publication per branch; roles `Right, Left, Right, Left`. Four traces reached 227 control steps and the simulator produced `complete.json`, with native exit 0.
- Host post-Path validator: **Fail** at `scripts/eat_nav_p1_g3_crossover_v2_host.py:511`, where `len(unique Path SHA) != 3` rejects the frozen period-2 assignment's two unique Paths. The trace/Outcome remains measurement-invalid under the current seal.
- Resource/cleanup: 133.26 s host time; GPU device-total peak 5,223.94 MiB; container anon+shmem peak 3.291 GiB; batch storage peak 30.29 MB. 138 owned GPU attribution samples; monitor failure `null`; owned residual 0; bridge residual 0.
- Finalization: failure seal is SHA-valid (`308` files), host exit 1, and the external launcher verified the unchanged seal after exit. Console/exit records are outside the sealed batch.

The raw evidence is under `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-crossover-v2-period2-continuation-attempt-0/`. The valid prior batches remain under `g3-crossover-v2-pilot-attempt-2/`. The sealed period-2 batch and old batches were not modified after sealing; the outer launcher wrote its console and exit records outside the sealed batch.

## Next decision

The intended three-period crossover still lacks a protocol-valid period 2. A future recovery would require a separately frozen post-Path validator contract that compares publication roles and SHA directly with the assigned four slots, without imposing a per-period count of three unique Paths. It would also need an explicit decision on whether the sealed four traces may be audited offline and reclassified, or whether a new run is required. Neither action is authorized by this continuation; no 12-trial paired inference or transition to the remaining 15 processes is justified.
