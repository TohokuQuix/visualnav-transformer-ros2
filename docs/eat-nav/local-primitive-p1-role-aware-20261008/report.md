# P1 role-aware terrain gate

Taxonomy SHA: `b787e879bc0417132ab79b429d90ad44031dbdd075297b05d34aa6d798a1c60a`.
Previous gate decision: `local_motion_primitive_terrain_support_fail` (unchanged).
New gate: **fail**.
Only SHA-verified physical anchor, DEM, Path and source records were used; no candidate Outcome was read.

| Anchor | Seed | Source family | Primary role | Secondary flags | Known paths | Differential pairs |
|---|---|---|---|---|---:|---:|
| first-transition | first | 20261007:k_rail_field | none | — | 5/5 | 0 |
| first-differential | first | 20261007:boxes | R-differential | — | 5/5 | 7 |
| first-graded | first | 20261007:k_rail_step_field | R-transition-common | — | 5/5 | 0 |
| first-flat-only | first | 20261007:k_rail_field | R-transition-common | — | 5/5 | 0 |
| second-transition | second | 3773891777:boxes | none | — | 5/5 | 0 |
| second-differential | second | 3773891777:k_rail_step_field | R-differential | — | 5/5 | 9 |
| second-graded | second | 3773891777:k_rail_step_field | R-transition-common | — | 5/5 | 0 |
| second-flat-only | second | 3773891777:pyramid_stairs_inv | R-transition-common | — | 5/5 | 0 |

Missing primary roles: `['R-graded-common', 'R-flat-control']`.
Decision: `role_aware_terrain_support_insufficient`.
P1 simulator and follower were not started.
