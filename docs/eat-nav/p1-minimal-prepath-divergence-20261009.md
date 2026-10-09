# P1 minimal pre-Path physics divergence diagnosis — 2026-10-09

## Decision

The three-environment, candidate-free test reproduced the early direct-clone divergence. The first large baseline difference is caused by an inconsistent flipper action term previous target, despite matching root pose, root velocity and joint state before physics. Restoring only that flipper control state moves the first frozen-threshold crossing from physics step 1 to step 38. At step 38 the raw PhysX contact force first becomes nonzero. The residual contact-onset divergence is **not explained yet**; neither historical golden reproducibility nor direct-clone equivalence passes. Keep the full 112-environment pre-Path gate stopped.

No candidate Path, Outcome, mixed-candidate rollout, follower modification, data collection or training occurred. The separate `boxes` CUDA terrain RNG issue was not revisited.

## Provenance and test definition

- Historical reference: sealed [P1 golden backup](/home/kojima/eatnav-backups/2026-10-09-p1-radial-28-analysis), `p1-01-second-differential-b+00-r00`, pose `b3773891777-k_rail_step_field-r0.60-a00`, historical source env 6. Golden backup seal SHA: `2cd38096e8b74efafb89ad28329f5695915ca80cf13b8b98cf8ef2816e596808`. The prior v1/v2 and Phase A seals were checked before the simulator runs and their contents left unchanged.
- Frozen criteria: [eat_nav_p1_batch_parity.json](/tmp/eatnav-vnt-p1-radial-candidate/config/eat_nav_p1_batch_parity.json), SHA `433853aa7b3f83d17ff76d1fe49fd86eabdc51837181f4ec0758431546bab8c2`. In particular root position 0.02 m, root velocity `1e-4`, joint position/velocity `1e-4`, DEM RMS `1e-5` m, DEM max `1e-4` m, mask IoU 1.0 and Actor max `1e-4`. No tolerance was changed.
- Deterministic terrain `k_rail_step_field`, difficulty 0.5, terrain seed `2525527592`, batch seed `3773891777`, source pose local XY `(0.60, 0)`, yaw 0. A 1×3 scene uses env 0 as source-like, env 1 as relocated target and env 2 as a second direct clone. The historical scene had 112 envs, with direct targets 9 and 10. The smaller scene has different tile allocation and world placement; its D1 result is diagnostic, not a historical 112-env gate result.
- All three regenerated tiles have the same canonical tile-local mesh SHA `ace2a5c9aac2f18cfff79c56a36fbf05c8ad0598e688ccfe03bfefbb203c28e8` in every valid run. The historical source tile's mesh bytes were not archived, so cross-scene collision mesh identity cannot be asserted. In R4 the *same* three tile meshes and control are translated globally by approximately `(-60,+32)` m; no local geometry changes.
- Four valid runs each completed 200/200 physics steps, 4 finite DEM frames at steps 50/100/150/200, 20 finite post-delay Actor records, native exit 0, owned residual 0, Path publications 0 and a host/container SHA-verified completion seal. Each had 5166 MiB peak GPU, under the 7680 MiB cap; elapsed simulator/monitor time was 14.8–17.0 s. RAM, storage, heartbeat and timeout monitor gates did not fire. The valid roots and their frozen manifest SHA are:

| Run | Root | Manifest SHA |
| --- | --- | --- |
| Baseline v2 restoration | `/tmp/eatnav-p1-minimal-prepath-20261009-v3-baseline` | `752344ba04faadb1b766803f629f7366f48595607ad81f4b362f186211396c21` |
| R2 flipper previous target only | `/tmp/eatnav-p1-minimal-prepath-20261009-v3-r2-flipper` | `9ef33169c98ad0ea77ccf70607c367e16ca28e544a7aea307bc07e80bbc81b55` |
| R3 raw contact observation | `/tmp/eatnav-p1-minimal-prepath-20261009-v4-r3-contact` | `25a70b1ff8d44a0582414d7cf334ec37e0ef473c9eaddefdb8f764192e2d7e85` |
| R4 world-translation control | `/tmp/eatnav-p1-minimal-prepath-20261009-v5-r4-shift` | `61089055deeab2e41305af405e50e1a8d095cb7d46dbfaff713f16f7d5f5fd45` |

The earlier `v1-baseline` and `v2-baseline` are retained as **measurement-invalid**, not promoted: the Actor logger indexed an RL TensorDict as a plain tensor. The first has a partial diagnostic physics trace but no completion marker; the second records the exception. This instrumentation error was fixed in a new versioned source, with no control or physics change. No old root was overwritten.

## Initial state and intervention

At `before_first_physics`, after source-state placement, all three roots have the same tile-local XY `(0.60,0)`, world Z `0.4993465245` m, identity orientation, zero linear/angular velocity and matching source joint positions/velocities. This Z matches the historical source fixture. Flipper latch is 1 for all. The settling command is the same for all three. Therefore R1 root Z/velocity restoration would be a no-op and was **not run**. The source-like label does not mean the historical 112-env physics state was completely restored.

| Field | Baseline observation | R2 handling |
| --- | --- | --- |
| Root Z, orientation, linear/angular velocity | Equal across three before physics; source fixture Z/velocity equal | Unchanged |
| Joint positions/velocities and flipper latch | Equal across three; latch 1 | Unchanged |
| ActionManager flipper `_previous_targets` and `_pre_clip_processed_actions` | Env 0/1 `[π,π,0,0]`; env 2 `[0,0,π,π]`, despite equal joints | Set all three to the source fixture `[0,0,π,π]` |
| Latency substeps | `[5,3,5]`, with different previous observation/buffer contents | Unchanged; not claimed equal |
| ActionManager current/previous action | Recorded in the boundary artifacts | No general ActionManager restoration |
| Joint target, applied torque | Recorded at boundary and every physics step; initial applied torque differs | Allowed to evolve naturally after only the flipper target change |
| Solver/actuator hidden state | Not fully observable/restorable | Not overwritten or called equal |

R2 changes only the accessible flipper previous target/preclip state. It is a diagnostic intervention, **not** a correction to the frozen batch protocol. The source and target root, joints, command, terrain and all other measured control fields were held fixed. R3 adds a read-only 4096-cap raw PhysX contact observer to R2. The R2 and R3 root-state traces are byte-for-byte equal at all 200 steps for all three envs; contact instrumentation did not perturb the observed physics. R4 retains R3 and changes only global world translation.

## Earliest divergence

| Boundary | Baseline direct pair env 1–2 | R2/R3 direct pair env 1–2 |
| --- | --- | --- |
| Before physics | Flipper previous target differs by π on four channels; root/joints match | Flipper previous target matches; root/joints match |
| Physics step 1 | Joint position target differs by `1.57079637` rad; applied torque by 40 N·m; joint position `0.00954739` rad, joint velocity `1.91618` rad/s and root velocity `0.338511` differ | Joint position target equal; root/joint differences only floating-point scale (`~8e-9` root velocity and `~1e-10` joint position) |
| Steps 2–35 | Already divergent | No root/joint frozen-threshold crossing; same command and joint target |
| Step 36 | — | First raw contact candidates in all three envs, reported force zero |
| Step 37 | — | Contact candidates present, force zero |
| Step 38 | — | First nonzero raw contact force; direct pair root velocity difference `0.211096`, joint position `0.00169938` rad, joint velocity `1.17250` rad/s. Source-like env 0 versus target env 1 first root velocity crossing is `0.352550` |
| Step 39 | — | First applied-torque frozen diagnostic crossing after contact (`75.97` N·m direct pair) |

At step 38 in R3, raw contact counts are `130/128/131` and maximum forces `5577/4865/6075` N for env 0/1/2. Contact count/start bounds are valid, none of the 4096 buffers is full and contact data are finite. The standard contact sensor updates at control cadence and first reflects contact at step 40; it cannot locate the causal onset as precisely as the raw PhysX observer.

Thus the flipper state is a controlled causal explanation for the **step-1** mismatch. The remaining threshold crossing is synchronized with first contact impulse, but identical mesh hash plus initial measured equality do not prove identical PhysX solver/contact state. Tiny pre-contact numerical differences exist. Contact ordering, collision prim placement/material state, hidden actuator/solver state, concurrent env coupling and world-coordinate precision remain unresolved. No claim of generic PhysX nondeterminism is made.

## Reproducibility and observations

| Comparison at step 200 / actual update points | Baseline | R2 flipper |
| --- | ---: | ---: |
| D2 direct pair, root velocity max abs | `0.096367` | `0.187814` |
| D2 direct pair, joint position max abs | `0.052910` rad | `0.031380` rad |
| D2 direct pair, joint velocity max abs | `0.177023` rad/s | `0.117484` rad/s |
| D2 direct pair, DEM frame 0 RMS | `0.003952` m | `0.004323` m |
| D2 direct pair, DEM frame 3 RMS | `0.004396` m | `0.023746` m |
| D2 direct pair, post-delay Actor max abs | `0.717961` | `0.937084` |
| D1 historical source versus minimal env 1, DEM frame 3 RMS | `0.051803` m | `0.046780` m |
| D1 historical source versus minimal env 1, Actor max abs | `1.717961` | `2.0` |

The intervention fixes the *earliest* large control discrepancy, but final DEM/Actor equality does not improve monotonically because contact evolution differs. Baseline and R2 D1 each pass `0/3` historical comparisons; D2 passes `0/1` direct pair. These are diagnostic counts for the reduced scene. The previously frozen full-scene results remain D1 `0/28`, D2 `0/12`; they were not rerun. In R2, source-like versus historical physical root/joint velocity first cross their frozen criteria at step 38. DEM difference already exists at its first frame (step 50) despite regenerated tile-local mesh identity, and cannot yet be divided uniquely among pose, collision/placement and observation timing effects.

Only the actual post-delay Actor input was captured. Pre-noise and post-noise/pre-delay observations were not captured and are **not** treated as matching. Latency buffers and prior observation histories were recorded and differ, so final Actor mismatch is not evidence of an independent Actor-model defect.

## World-coordinate sensitivity

R4 globally shifted the same three canonical tiles and robots approximately `(-60,+32)` m, moving env 0 to the historical source tile origin. Commands, local geometry, initial relative pose, flipper restoration and env count stayed the same. The first raw contact candidates and forces still occur at steps 36 and 38. Compared with R3 for the *same env ID*, root velocity first exceeds `1e-4` at step 38 by `0.00212/0.00230/0.00159` for env 0/1/2. The direct-clone gap remains `0.207207` at step 38 versus `0.211096` without the shift. This demonstrates coordinate sensitivity at contact onset, but the controlled translation effect is much smaller than the clone gap and does not isolate env ID or parallel-execution effects.

## Next minimal action

Preserve the flipper previous-target finding as a candidate restoration fix, then make one **candidate-free, contact-onset isolation** comparison before altering the batch implementation: in the same three-env scene and with identical flipper state, compare a relocated target and clone under a contact pair/normal/impulse and collision-prim/material trace at steps 35–39, with per-env solver/actuator state if exposed. Keep the frozen tolerances, seeds and geometry. If those contacts prove equal while dynamics differ, a single 112-env instrumented run can then isolate scene count or concurrent placement; it needs its own preflight and review. Do not run the full batch pre-Path gate or any candidate Path until the residual D2 divergence is explained and a separate validation is approved.

The machine-readable [analysis](/tmp/eatnav-p1-minimal-prepath-analysis-20261009-v3/analysis.json) contains per-step first nonzero and first frozen-threshold crossings, all D1/D2 metrics, contact timing, resource status and SHA references. The durable [backup](/home/kojima/eatnav-backups/2026-10-09-p1-minimal-prepath-physics-diagnosis) contains the report, frozen run roots, invalid attempts, source snapshots, analysis and a SHA ledger.
