# EAT-Nav all-terrain radial inventory — first-failure review

## Decision

**Stopped at candidate-free anchor gate.** The one authorized inventory execution recorded the first seed batch: 112 poses (16 directions × 7 terrains). The second seed batch did not start. No candidate Path, follower/bridge, Outcome, 120-step suffix, training or pilot was run.

The simulator completed normally and host/container output SHAs match. Live `scene.env_origins` matched row/column placement for all 112 envs, and each terrain column used one seed and fixed difficulty 0.5 across 16 angles. All records contain four finite 20×20 DEM/mask frames and finite Actor observations. Valid-mask fractions span 0.785–0.820.

**No anchor is promoted as valid.** Contact force was recorded, but contact separation (penetration depth) was not instrumented, so `penetration=None` for all 112. This is `measurement_invalid`, not physical failure. Separately, 16 poses exceed at least one fixed settled-state numeric threshold. No limits were changed after observation.

## Numeric audit by terrain

| Terrain | Class | poses | mask min | max XY error (m) | max yaw error (rad) | numeric exceedances |
|---|---|---:|---:|---:|---:|---:|
| `boxes` | source_flat_radial | 16 | 0.790 | 0.036 | 0.101 | 1 |
| `hf_pyramid_slope` | source_flat_radial | 16 | 0.790 | 0.032 | 0.373 | 2 |
| `pyramid_stairs` | source_flat_radial | 16 | 0.790 | 0.031 | 0.099 | 0 |
| `pyramid_stairs_inv` | source_flat_radial | 16 | 0.790 | 0.018 | 0.018 | 0 |
| `random_rough` | unguaranteed_center_radial | 16 | 0.790 | 0.096 | 0.210 | 6 |
| `k_rail_field` | unguaranteed_center_radial | 16 | 0.790 | 0.105 | 0.081 | 4 |
| `k_rail_step_field` | unguaranteed_center_radial | 16 | 0.785 | 0.077 | 0.138 | 3 |

Fixed limits were XY ≤0.10 m, yaw ≤0.10 rad, speed ≤0.10 m/s, angular speed ≤0.20 rad/s and DEM valid fraction ≥0.70. Detailed per-pose reasons are in `anchor-only-gate-results.json`. The source-flat guarantee applies only to the central-square placement geometry; settled physical stability was not established.

## Provenance and resource

- Prior radial review artifacts, 12 direct inputs and 30 execution-source SHA entries were rechecked before launch; the 1,015 older entries remain stored audit attestations.
- Scene: 16×7 tiles, 112 envs; `r=0.30 m`, `θ=22.5°k`, fixed difficulty 0.5. Root requested at runtime terrain origin plus 0.55 m, then settled for 20 control steps. Candidate-free source and input SHA are frozen in `input-freeze.json`.
- Last retained resource sample: GPU device total 5853 MiB, anon+shmem 3.84 GiB; no cap crossed. The monitor kept its latest sample rather than a complete peak series, so GPU peak is NA.
- The owned simulator exited and cleanup passed. Retry count 0. The input root and 112 raw pose records are preserved.

## P1–P4 and downstream scope

P1 main-track/sprocket projection is the intended primary corridor; P2 adds initial flipper support, P3 is potential articulated flipper reach, and P4 is the full collision envelope for safety. These source-derived proxies are not a measured settled contact footprint. The 0.56 m visible width and 0.438 m collision width are distinct.

The five constant-curvature Paths and common front goal `(0.6,0)` remain **provisional geometry only**. Since no anchor passed, main-track/flipper terrain-corridor profiles, terrain clusters, taxonomy, four-family selection and a pilot manifest are NA. Seven source-seed families were recorded but zero are gate-eligible. Unknown DEM cells were never treated as free.

A future final-collection design proposes goal bearings `{-20,-10,0,+10,+20}°` at 0.6 m, with new forward-tangent constrained curves per bearing. Multiple bearings/angles/candidates from one source seed and environment stay in one split. No goal-bearing Path was executed; bank size or missing mask must be frozen before a later review.

## Stop and next review

The next technical question is how to instrument PhysX contact separation without changing reset events, dynamics or follower control, and how to interpret the 16 settled-state exceedances at the fixed limits. This report does not request automatic retry; a new inventory attempt would need a new output root and explicit review. `small_pilot_ready_for_review=false`, and no later collection/training gate passed.

## Figures

1. [All seven terrain radial poses](figures/01-all-seven-terrain-radial.png)
2. [Source-flat vs unguaranteed classes](figures/02-terrain-class.png)
3. [Settled pose and DEM mask](figures/03-settled-anchor-dem-mask.png)
4. [P1/P2/P4 footprint roles](figures/04-footprint-roles.png)
5. [Five Paths, corridor NA](figures/05-five-paths-corridor-na.png)
6. [Terrain corridor cluster NA](figures/06-terrain-corridor-clusters.png)
7. [Taxonomy supply NA](figures/07-taxonomy-supply.png)
8. [Independent family support](figures/08-independent-families.png)
9. [Front-goal pilot NA](figures/09-pilot-proposal.png)
10. [Future goal bearings](figures/10-goal-bearing-future.png)
11. [Decision flow](figures/11-decision-flow.png)
