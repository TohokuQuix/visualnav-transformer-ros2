# P1 G3 candidate-free material diagnostic and gate proposal

## Decision

The one approved **Path-free** runtime diagnostic completed and was sealed protocol-valid. It does not authorize a candidate rollout. Existing crossover v1 and its invalid period-0 attempt remain unchanged. The v1 material gate cannot pass on this asset because all robot collision prims lack a composed USD physics-material binding. Its hard-coded robot `(0.8, 0.6, 0.0)` expectation also differs from the **resolved OnixRoughEnvCfg startup event and live PhysX values**, both `(1.0, 1.0, 0.0)`.

The proposed, **unapproved** gate is [`p1-g3-material-gate-v2-proposal.json`](p1-g3-material-gate-v2-proposal.json). Terrain retains required live USD binding and composed-attribute equality. Robot USD binding status is recorded per prim; its complete post-startup effective PhysX tensor is the primary physical-material evidence. Missing/nonfinite tensor, failed link/shape coverage, unresolved material assignment, or disagreement with the frozen resolved startup event fails closed. All other crossover inputs and gates remain unchanged. A new protocol version, source SHA, manifest, root, and separate rollout approval are required before repeating the 3-process crossover pilot.

## Runtime evidence

Persistent diagnostic root: `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-material-diagnostic-v1-attempt-0/`. Material JSON SHA-256: `9b2c1329c9179a4f64a49aa5509b3a9b5eb5d3dbefdef45fcd0247a5ab98991f`. Contract SHA-256: `97ee77a7c077209a141891119d0a5101fa3de2b513cd9f61f66fdf54220dbe79`. Simulator source SHA-256: `dbf727b9935b07b99c38a1c3c6af37b1dc48861a4746da87a7be44541cfc07b8`.

| Signal | Measurement |
| --- | --- |
| Terrain CollisionAPI prim | 1, bound to `/World/ground/terrain/physicsMaterial` |
| Terrain composed material | Static 1.0, dynamic 1.0, restitution 0.0; multiply combine modes, compliant stiffness/damping 0.0. Canonical live = declared SHA `ff1c75072f159375d7f3fd74cd3f7e2f445df9430dede14d7ca8184b18023836`. |
| Robot CollisionAPI prims | 271 per env × 4 = **1,084**; 272 Cylinder and 812 Cube total |
| Robot USD physics-material binding | **0 bound / 1,084 explicit `unbound`** |
| PhysX articulation material tensor | `[4, 271, 3]`; all 1,084 triples `(1.0, 1.0, 0.0)` |
| PhysX links and shape coverage | 60 links/env; per-link USD CollisionAPI count matches per-link rigid-body PhysX shape count and tensor slice length; 0 unmapped collision prims, 0 unmatched tensor shapes, 0 ambiguous nonuniform links |
| Per-env material fingerprint | The same SHA for env 0–3: `b5dbaa680b862eeb2382a8d31552b2ba6a9419d11730e22d17fe9dcdde608747` |
| Resolved startup event | Static range `[1.0,1.0]`, dynamic `[1.0,1.0]`, restitution `[0.0,0.0]`, 128 buckets, `make_consistent=true`; live tensor agrees |

Isaac Lab's installed `randomize_rigid_body_material` event uses `root_physx_view.link_paths` order and per-link rigid-body view `max_shapes` to slice and set the tensor. This source behavior supports the link-to-tensor mapping. The current public tensor view exposes link paths and shape counts but no individual shape-path order. Of the 1,084 material assignments, 4 are single-shape links with exact tensor index; 1,080 are in count-matched multi-shape link sets whose tensor values are **uniform**, so each collision prim's effective *material triple* is unambiguous even though its exact tensor index is not established. The proposal accepts this uniform-link proof for material-value coverage and fails closed if a multi-shape link ever contains different values without exact shape mapping. It does **not** claim exact index-to-prim identity.

Source audit: `velocity_env_cfg.py` declares base startup friction `(0.8,0.6)`; `config/onix/rough_env_cfg.py` overrides static/dynamic to `(1.0,1.0)` for the actual OnixRoughEnvCfg. The staged collection config does not revert that override. The live measurement matches the resolved config. The original v1 auditor's `(0.8,0.6,0.0)` expectation came from the base declaration, so it cannot be used as this scene's post-startup material condition.

No follower, bridge, Path publication, or control step ran. The frozen checkpoint SHA was checked and unchanged. Host native exit 0, 14 GPU bracket samples with 0 attribution failures, GPU device-total peak **5,082.75 MiB**, container anon+shmem peak **3,116,134,400 B**, active root peak **2,577,925 B**, host elapsed **9.20 s**, and owned residual 0. Host/container material JSON SHA matched. Host seal verified immediately after writers closed: 216 files, stage `G3_material_measurement_protocol_valid`.

## Outer launcher finalization

The new outer launcher writes console and `launcher-result.json` to the **separate sibling** `g3-material-diagnostic-v1-attempt-0-launcher/`, outside the sealed batch. Its read-only post-host verification again returned the same valid 216-file seal after launcher exit. Synthetic tests cover both late console/exit bookkeeping without batch mutation and a deliberate post-seal batch write, which is rejected. [`eat_nav_p1_g3_crossover_outer.py`](../../scripts/eat_nav_p1_g3_crossover_outer.py) applies the same external-log/after-exit verification pattern to a future separately approved crossover runner, checking the frozen host source SHA and period root. It has **not** run a candidate process.

## Limits and next approval point

This single diagnostic verifies material-value coverage under the uniform-link inference and records the exact-index limit. It does not test cross-period equality or candidate behavior. The v1 manifest, source, failure seal, and attempt-0 have not been repaired or promoted. A 3-process crossover can be reconsidered only after the v2 material-gate criterion (including acceptance of link-set uniform proof), corrected expected startup triple, new manifest/source SHA, external launcher, and a fresh attempt root are reviewed. No candidate rollout, remaining crossover process, Evaluator learning, or full collection was started.
