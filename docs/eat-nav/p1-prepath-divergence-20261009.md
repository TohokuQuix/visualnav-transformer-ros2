# P1 pre-Path divergence diagnosis — 2026-10-09

## Decision

**2. Root cause partially isolated; one additional minimal diagnostic required.** The frozen P1 batch pre-Path gate remains failed. No candidate Path was published, and no mixed-candidate rollout was run.

Phase A found a concrete **geometry mismatch before the first physics step** for the `first-differential / boxes` StateGroup. The other three groups still fail both historical-golden and within-batch comparisons at step 200. Their earliest causal boundary cannot be recovered from v2's step-200-only batch records. We stopped after the candidate-free terrain probe, without another physics run.

## Frozen evidence and provenance

| Evidence | Seal SHA-256 | Verification |
|---|---|---|
| P1 28-run golden backup | `2cd38096e8b74efafb89ad28329f5695915ca80cf13b8b98cf8ef2816e596808` | 2,919 ledger files verified |
| Batch v1 | `a7ce044c7abaea96a6478af938cba303ae1aa81c43bc04fdcfec0b99cb54a79e` | 15 ledger files verified |
| Batch v2 | `b0b65a417ad59eabfcc99920078890ace949c0cba1095f66349d9744640a1246` | 16 ledger files verified |
| New static audit | `/tmp/eatnav-p1-prepath-divergence-20261009-phase-a/seal.json` | read-only; no simulator |
| New terrain probe | `/tmp/eatnav-p1-prepath-terrain-probe-20261009-v1/` | frozen manifest SHA `19cb0818a3825f2b12535af085a95c7c30e801a7a8bb55be7f5fbfdf27019cb6`; source SHA `78724386cd8725f3f3d36c9ade57265bf243633952341f6e87fc73a23b602149` |

The terrain probe generated the two source-seed scenes and one mixed batch scene using the original `16 × 7` layout, terrain configs, tile seeds, difficulty `0.5`, and generator algorithms. It performed **zero physics steps**. The runner returned exit 0, emitted a complete marker, and its PID had no residual process. Host and container SHA values match for source, manifest, geometry output, and complete marker. It sampled 814 MiB device memory and 83 GiB available host RAM; a continuous GPU peak was not recorded. Output is under 1 GiB. The prior golden/v1/v2 roots were read only and remain unchanged.

## A. Terrain geometry and initial-state audit

All 28 target terrain **names, seeds, and difficulties** match the plan. Those fields alone did not establish mesh identity. The canonical tile-local vertex/triangle SHA comparison yielded:

| StateGroup | New source-versus-target geometry | Unique meshes among seven targets | Collision/material config |
|---|---:|---:|---|
| first-differential, `boxes` | **0/7 equal** | **7** | config equal |
| second-differential, `k_rail_step_field` | 7/7 equal | 1 | config equal |
| first-graded, `k_rail_step_field` | 7/7 equal | 1 | config equal |
| second-graded, `k_rail_step_field` | 7/7 equal | 1 | config equal |

The `boxes` source tile and every corresponding batch tile differ in canonical tile-local geometry, including the three nominally identical direct clones. Thus world translation alone cannot explain **their mesh hashes**. The terrain settings use the same collision/material configuration, but the old v2 run did not save its live collision mesh, so exact historical PhysX collision-representation identity is still unproven. The new source scene has the same vertex and face counts as the saved golden source scene, but its whole-scene SHA does not equal the saved golden SHA. The regenerated source mesh is **not** certified as the historical golden mesh; comparisons with the historical mesh remain limited to the saved whole-scene hash and live outcomes.

The code gives a specific reason for the `boxes` mismatch: the configured `boxes` terrain is `random_grid_terrain`, whose cell heights are sampled by `torch.Tensor.uniform_` on CUDA. Both P1 and batch terrain generators set `cfg.seed` and `np.random.seed` per tile, but neither restores or sets a per-tile **PyTorch CUDA RNG state** before terrain generation. Sequential tile order therefore changes the `boxes` mesh even with identical named terrain seed. This is supported directly by the generator code and by seven different target mesh hashes. It is not an assertion about PhysX nondeterminism.

The four source prephysics fixtures match their sealed `reset-pre-physics.json` records exactly. In batch v2, the code writes tile-local root XY/yaw, source joint positions/velocities and flipper latch, then replays the 20 source commands. It **retains batch reset root Z and velocity** and does not restore or record ActionManager state, flipper previous target, actuator internals, latency buffers, observation RNG, or contact solver state. The actual batch state immediately after placement was not saved. These fields must be classified as *unobserved*, not equal or unequal. Field-level entries and source values are in `phase-a-static-audit.json`.

## B. Earliest divergence and propagation

| Group | Earliest established signal | Magnitude available | Later measured differences |
|---|---|---|---|
| `boxes` | Different tile-local mesh at **scene generation, before physics step 0** | Different SHA for all seven targets; vertex-wise distance was not saved | DEM frame at physics step 50 differs; direct-clone frame-0 RMS 0.0178–0.0251 m; settled root-local max difference 0.0091–0.0322 m |
| three `k_rail_step_field` groups | Regenerated canonical geometry equal; actual batch step-0 physical/control state **not recorded** | First numerical difference step unknown | Direct-clone frame-0 DEM RMS 0.000204–0.00325 m; at step 200, root velocity max difference 0.00437–0.0477 and joint-position max difference 0.00217–0.0454 rad |

The first **gate-relevant** observation difference is no later than DEM frame 0 at physics step 50 for all four groups. Actor values were recorded only at settling step 200. No batch per-physics-step trace exists. Therefore the first nonzero root/joint/control difference, first crossing of each frozen threshold, and order of physical versus observation-noise propagation cannot be stated from the old runs. A mesh SHA mismatch establishes geometrical nonidentity but does not provide a vertex-distance magnitude.

## D1. Historical golden reproducibility

Under the unchanged `config/eat_nav_p1_batch_parity.json` tolerances, **0/28 targets pass**. Failures: root velocity 28/28, joint position 28/28, joint velocity 28/28, four DEM frames 28/28 each, Actor input 28/28, root position 3/28, orientation 10/28. Terrain seed/type and no-termination checks pass 28/28. Root-local position error ranges 0.000487–0.03399 m; Actor max-absolute error ranges 0.05596–2.0. These are settled step-200 comparisons; they do not identify the first causal step.

## D2. Within-batch direct-clone reproducibility

We separately compared all three pairs among the three direct clones in each StateGroup, in each tile's local frame. **0/12 pairs meet all applicable frozen pre-Path criteria.** The pairwise metrics are diagnostic; this is not a substitute pass for D1. The four groups' pairwise root-local max differences span, respectively, 0.00914–0.03218 m, 0.000809–0.001865 m, 0.001659–0.006485 m, and 0.000450–0.001406 m. All 12 pairs fail the root-velocity, joint-position, joint-velocity, four-DEM-history, and Actor comparisons. The three direct golden repeats per group had exactly matching settled root/joint values, but were sequential runs at one original environment location; the batch direct clones occupy distinct tiles simultaneously.

## C. Hypothesis matrix

| Hypothesis | Supported evidence | Evidence against / unresolved | Next minimal discriminator |
|---|---|---|---|
| H1 terrain identity | `boxes` 0/7 source-target mesh matches; seven target hashes differ; `torch.uniform_` is not tile-seeded | Other three groups' regenerated geometry matches; historical live collision mesh was not saved | Save canonical mesh per tile at generation in a future controlled run; no additional full scene is needed to establish the `boxes` issue |
| H2 world relocation | Direct clones are at different world coordinates; their step-200 states differ | `boxes` has a prior mesh confound; no same-mesh step-0/stepwise controlled relocation measurement | One deterministic `k_rail_step_field` source/relocated pair, same physics/control state, step-0 and per-step logs |
| H3 simultaneous environment count | Golden direct repeats were sequential, batch clones simultaneous | No controlled small-versus-112 comparison; H2/H4/H5 confound | If H2 control does not explain, repeat the same deterministic-terrain pair in minimal and 112-env scenes |
| H4 incomplete dynamic state | Batch v2 did not restore or capture ActionManager, flipper target, actuator, latency or solver state; it retained batch reset Z/velocity | Their actual step-0 values and first command/target were not logged | Capture source and batch post-reset/placement state and first applied target/physics step, before any Path |
| H5 observation noise | Actor errors can contain noise; environment IDs and RNG streams changed | Root/joint/DEM differences already exist and cannot be noise-only | Log pre-noise, post-noise/pre-delay, post-delay Actor values with per-env RNG state after physical-state alignment |

## Recommended next action and limits

Freeze a **separate candidate-free, instrumented minimal diagnostic** for one `k_rail_step_field` source/relocated pair and one same-scene direct pair. Record root/joint/targets/latency/action state immediately after reset and placement, then at every physics step through the first divergence; record DEM/Actor only at actual update points. Do not change the original gate or inject golden observations. If this cannot reproduce the old 112-env discrepancy, a single instrumented 112-env candidate-free run is the next discriminating measurement. The `boxes` geometry issue must be resolved separately in a future approved design or by recovering exact historical mesh/RNG state; a matching terrain label and seed are insufficient.

This diagnosis does not authorize Path evaluation, follower/production changes, Outcome collection, StateGroup expansion, or training. The original frozen gate remains binding.
