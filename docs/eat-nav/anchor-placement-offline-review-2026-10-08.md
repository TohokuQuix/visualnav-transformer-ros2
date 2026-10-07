# Anchor placement offline review — 2026-10-08

## Decision

`anchor_placement_rule_ready_for_candidate_free_pilot`: propose the common **R3 class-balanced max-min** rule for the next *candidate-free* anchor pilot. This means that the first-seed geometry supplies reviewable poses across all four DEM classes. It does **not** establish physical settling, contact safety, outcome value, or seed generalization. No new settling run or candidate Path was started.

The proposed pilot has eight poses: two different terrain/source families for each of transition, differential, graded, and flat control. The machine-readable proposal is `/tmp/eatnav-anchor-placement-20261008/analysis-v9/pilot-candidate-proposal.json`. All eight remain `physical_stability=unmeasured`.

## Input and parity

- Isolated worktree: `/tmp/eatnav-vnt-anchor-placement`, branch `exp/eatnav-anchor-placement`, based on `4399100e1b92a0344d70edaeb2c613f88a6b0f13`.
- First-seed source: 112 measured radial poses in the original 23 shards and termination-aware continuation. Both existing SHA ledgers were read-only verified; all saved frame/source/reset metadata, pose identity, terrain seeds, and DEM shapes were audited. No original classification or eligibility label was changed.
- The 112 saved anchor-time DEM frames were independently reconstructed with the same generated terrain mesh and RayCaster/DEM path: **112/112 passed**. Maximum measured raw/transform height discrepancy was zero, mask IoU was 1.0, and maximum yaw discrepancy was `2.22e-7 rad`. Frozen tolerances were height RMS `1e-5 m`, max `1e-4 m`, mask IoU `1.0`, origin `1e-5 m`, yaw `1e-5 rad`. Simulator control/physics step counter remained zero.
- Frozen terrain mesh SHA: `1689eff80304bdbc318e6662f66883d8bb7a34e729ee0f24535f701bdec17d1e`. Full source/input SHA provenance, including the RL checkpoint, `params/env.yaml`, terrain generator, RayCaster, Onix asset source, and old ledgers, is in `preflight.json`, `scan-freeze-v4.json`, and `analysis-v9/source-input-provenance.json` under the new output root.
- Only first batch seed `20261007`, difficulty `0.5`, seven existing terrain types, 16 fixed directions, and radii `0.30/0.40/0.50/0.60 m` were used. The same terrain mesh and scene origins were maintained. Root-z specification is `scene.env_origins.z + 0.35 m`; this scan did not test dropping or settling from it.

## Offline geometry results

The final scan produced **448/448 geometry-valid, DEM-complete poses** with finite 20×20×2 DEMs. The scene step counter stayed at zero. DEM known fraction ranged from `0.79` to `0.905`; known fraction at the collision-bound corner proxy was `1.0` for all 448. The minimum tile-boundary margin of that proxy was `2.763 m`. These are descriptive coverage measures, not physical feasibility passes.

| Radius | Flat | Graded | Differential | Transition | Known interaction onset within 0.6 m |
|---:|---:|---:|---:|---:|---:|
| 0.30 m virtual | 69 | 20 | 22 | 1 | 25 |
| 0.40 m | 61 | 24 | 26 | 1 | 30 |
| 0.50 m | 52 | 22 | 38 | 0 | 40 |
| 0.60 m | 33 | 23 | 54 | 2 | 62 |

The old **settled** `r=0.30 m` inventory remains 94 eligible, 17 numeric-ineligible, one physical termination, and zero measurement-invalid. Its 94 eligible classes are flat 70, graded 15, differential 9, transition 0. The nominal virtual pose and the settled physical pose are different measurements. Among the 94 eligible old poses, the virtual and settled DEM classes disagree for one pose; the 18 ineligible poses have no settled class. The correspondence table records these without transferring an old stability label to a new radius.

## Common-rule comparison

| Rule | Selected poses | Flat | Graded | Differential | Transition | Interaction onset ≤0.6 m |
|---|---:|---:|---:|---:|---:|---:|
| R0 virtual fixed 0.30 m | 112 | 69 | 20 | 22 | 1 | 25 |
| R1 fixed 0.40 m | 112 | 61 | 24 | 26 | 1 | 30 |
| R1 fixed 0.50 m | 112 | 52 | 22 | 38 | 0 | 40 |
| R1 fixed 0.60 m | 112 | 33 | 23 | 54 | 2 | 62 |
| R2 minimum known onset radius | 29 | 0 | 4 | 24 | 1 | 29 |
| R3 class-balanced max-min | 112 | 36 | 36 | 36 | 4 | 46 |

R2 used its pre-scan criteria: onset 0.10–0.50 m, known centerline, corridor known fraction at least 0.70, known terrain difference, and smallest qualifying radius per terrain/direction. It supplies no flat control and one transition. R3 applies one rule to all terrains: the frozen class priority and 18-component DEM-profile max-min selection with a 112-pose budget. It supplies transition from two terrain/source families, differential from three, graded from four, and flat from six. The four transition poses are related within only two families, so their apparent count is not four independent StateGroups.

The eight proposed pilot poses are:

| Class | Terrain | Radius / direction | Known onset |
|---|---|---:|---:|
| transition | k_rail_step_field | 0.40 m / 2 | 0.35 m |
| transition | k_rail_field | 0.60 m / 12 | 0.55 m |
| differential | k_rail_step_field | 0.60 m / 0 | 0.15 m |
| differential | boxes | 0.60 m / 1 | 0.35 m |
| graded | k_rail_step_field | 0.30 m / 6 | 0.25 m |
| graded | k_rail_field | 0.50 m / 1 | 0.35 m |
| flat control | k_rail_field | 0.50 m / 4 | none |
| flat control | random_rough | 0.60 m / 6 | none |

The complete proposal records requested XY/yaw, terrain seed, root-z rule, DEM SHA, known/unknown coverage, footprint diagnostics, and selection rank. No interpolation-radius sensitivity scan was needed because the base grid supplies all four classes. Such radii remain separate future sensitivity work, not part of this decision.

## Implementation and diagnostic caveats

The first scan attempt stopped before virtual DEM output because USD collision shapes have `purpose=guide`; excluding that purpose made all bounds empty. A second attempt produced matching DEM/classification results but an invalid footprint proxy: cloned Actor root coordinates and the USD template authoring transform were mixed. The final scan rebases the 271 composed `CollisionAPI` bounds by the inverse USD template-root transform. The failed/diagnostic attempts remain in `scan/` and `scan-v2/`; only `scan-v3/` feeds the final analysis. Numerical thresholds, terrain geometry, pose grid, and rule definitions were unchanged. The simulator wrapper returned native exit 0 even for the first marked failure, so the valid result is established by the final completion report and all 448 per-pose checks, not exit code alone.

The collision footprint is an AABB-corner proxy, not a contact-response test. The 0.6 m forward DEM corridor is a two-second **comparison proxy** at an assumed 0.3 m/s, not a proven reach distance. Unknown DEM cells were never treated as free space or counted as terrain interaction. Reward, Critic, candidate Outcome, and earlier Evaluator results were not used for selection.

## Review artifacts and stop point

The new output root is `/tmp/eatnav-anchor-placement-20261008`. `analysis-v9/` contains the R0–R3 JSON, class/source support, 112-pose correspondence, all 448 per-pose metrics, eight-pose proposal, second-seed decision, risk register, ten figures, final decision, and SHA ledger. The large raw DEM arrays remain outside Git. Small review artifacts and the source/tests are saved in the isolated branch.

**Next review:** decide whether to approve the eight-pose candidate-free physics/settling pilot under R3. A second seed should be considered after that common rule is fixed and its physical feasibility is checked. No second seed or downstream rollout is authorized by this review.
