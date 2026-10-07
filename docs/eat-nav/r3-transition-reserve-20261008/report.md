# R3 transition reserve: pre-audit freeze, cause audit, one-shard pilot

## Decision

`r3_transition_supply_confirmed_ready_for_second_seed_design`. The two frozen, previously unexecuted transition reserves ran once in one candidate-free 112-environment shard. Both measurements were protocol-valid. R3 rank 3 supplied one eligible, known-onset, virtual/settled `transition` anchor; rank 4 remained numeric-ineligible. Together with the original eight-pose pilot, the first seed now has at least one physical supply for transition, differential, graded and flat control. This supports **designing** the second-seed check; no second seed, Path, follower, Outcome or learning run was started.

## Ordering and frozen inputs

The selection artifact was written at **2026-10-07 22:32:24 UTC** before the detailed eight-pose trace audit. The selection procedure read only the frozen 104-pose `reserve-rank-freeze.json`; the subsequent geometry enrichment read only the frozen radius scans. The prior high-level pilot decision was already known from the earlier review, but no prior per-pose result or trace was an input to selection. A guarded test fails if the selector opens an `analysis` or `trace` path. The frozen reserve-manifest SHA-256 is `4b5e5cf07763d6253d19ca3c73e03ee99a7578a5be87a356afb103f66d46a75a`. The available transition reserve contained exactly two poses, and both were frozen. Their source families differ.

| R3 rank | Pose | Terrain / source family | Radius / direction | Virtual onset | Virtual known corridor | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | `b20261007-k_rail_field-r0.60-a04` | `k_rail_field` / `20261007:k_rail_field` | 0.60 m / 4 | 0.55 m | 0.95 | eligible transition supply |
| 4 | `b20261007-k_rail_step_field-r0.30-a02` | `k_rail_step_field` / `20261007:k_rail_step_field` | 0.30 m / 2 | 0.35 m | 0.95 | angular-speed numeric ineligible |

Requested XY/yaw, terrain seed, virtual DEM SHA, footprint diagnostics and input SHA are in [reserve-manifest.json](reserve-manifest.json). The source base was `dfe81c1c399ef48167a268f5daf5b93524102684`, branch `exp/eatnav-r3-transition-reserve`, worktree `/tmp/eatnav-vnt-r3-transition-reserve`. The prior source/physics/classifier and all thresholds were unchanged. The original 224-file output ledger was rechecked after the new analysis with zero mismatch.

## Read-only cause audit of the original eight

The frozen final40 angular-speed **max** limit is 0.20 rad/s. Rank 1 reached 0.3165 rad/s (13/40 exceedance steps, final 0.2997); Rank 2 reached 0.2459 (2/40 exceedance steps, final 0.1029). Their first transient angular peaks were at steps 42 and 39. Rank 1's contact-force peak was at step 42; Rank 2's at step 40. This timing suggests contact impulses as a candidate explanation for the early transients, not a demonstrated cause of the final40 exceedances. Rank 1's final40 angular-speed trend rose, consistent with unresolved late oscillation under the frozen 1.0 s settling window. Rank 2's final40 trend fell and only two steps exceeded the limit, but the unchanged max gate still makes it ineligible. Contact bounds/nonfinite/full-buffer flags were clear for all eight.

The four saved DEM frames kept one class per pose. Rank 1 stayed `transition` with onset 0.35 m. Rank 2 was `flat-only` in all four frames, whereas its virtual class was `transition` with onset 0.55 m. This is a secondary geometry/classification observation; Rank 2's primary ineligibility remains the angular-speed max. Rank 8 was eligible numerically but changed from virtual `flat-only` to settled `graded`: virtual slope 0.1129 lies just below the frozen 0.12 threshold, while settled frame-3 slope is 0.2210. Its settled interaction onset remained unknown, so it does not establish a new known-onset terrain-interaction supply. The full per-step kinematics, contact category counts and timing, root pose/yaw and height offsets at all DEM frames, threshold margins and first class-switch frames are in [eight-pose-cause-audit.json](eight-pose-cause-audit.json). These observations support class-boundary and corridor-edge sensitivity as design hypotheses. They do not change existing eligibility or prove a contact cause.

## Reserve result and integrity

| R3 rank | final40 max angular | DEM valid fraction | Primary settled class / onset | Eligibility | Supply |
| --- | ---: | ---: | --- | --- | --- |
| 3 | 0.0179 rad/s | 0.90 | transition / 0.55 m | settled anchor eligible | yes |
| 4 | 0.3760 rad/s | 0.79 | NA because numeric-ineligible | angular-speed limit | no |

The rank-3 virtual/settled class **and onset** matched, its virtual unknown-only flag was false, and all four secondary saved DEM frames classified as transition. Rank 4's secondary four frames also classified as transition; they were not promoted to primary eligibility. No physical termination occurred. The simulator used the original 112-environment scene, seed 20261007, difficulty 0.5, normal RL reset and `env_origin.z + 0.35 m`, requested XY/yaw only, zero 6D action, 0.005 s physics, 200 steps, four DEM frames, Actor prephysics/step 200, capacity 4096, and the same classifier and numeric gates.

The one-shot shard completed in 38.8 s wall time (resource samples through 32.5 s). Peak GPU device use was **5,871 MiB** of 12,288; peak container anon+shmem **3.91 GB** of 10 GiB; minimum host available **87.7 GB** versus 8 GiB; peak monitored storage **4.69 MB** versus 500 MB. Native exit was 0. Host/container copies matched; 49 sealed run files and the unchanged 224 original files passed SHA checks. Owned simulator PID `1157494` / start tick `203266977` had zero residual processes. There was no remediation cycle or retry. The exact Docker command is in the raw root `shard-reserve/host-launch.json`; the code and preflight are committed here. Raw trace remains at `/tmp/eatnav-r3-transition-reserve-20261008/shard-reserve/run`.

## Next design review

Freeze the present R3 selection rule for the proposed second-seed design, then audit class robustness under small XY/yaw perturbations *offline* without changing physical eligibility thresholds. Proposed second-seed stages are an offline DEM scan with independent terrain seeds, source-family grouping, and a candidate-free settling pilot using the same termination-aware protocol. The observed one-shard runtime and 4.69 MB monitored peak are only a starting estimate for runtime/storage budgeting; a second-seed manifest should supply its own prelaunch estimate. Goal-bearing and candidate-bank geometry can use only anchors that pass the second-seed supply checks. Any second-seed measurement needs separate authorization.

## Figures and machine-readable artifacts

The eight requested figures are in [figures](figures): angular speed; contact/peak timing; eight-pose DEM classes; virtual/settled threshold margin; reserve placement; reserve eligibility/class; four-class supply; decision flow. [final-decision.json](final-decision.json), [reserve-per-pose.json](reserve-per-pose.json), [resource-lifecycle.json](resource-lifecycle.json) and [integrity-check.json](integrity-check.json) contain the machine-readable outcome. The raw root has the seal, status, host/container SHA ledger, lifecycle log and complete traces. This report ends at design review; no downstream operation was started.
