# P1 radial-anchor execution contract: candidate-free fixture

Decision: `radial_anchor_execution_contract_resource_attribution_inconclusive`. Physics parity passed 4/4, but the full P1 v2 resource gate is not yet proven. P1 rollout was not started.

## Input and source contract

- Base commit `0fa3cb5`; four selected anchor records (28 source files) and 21 r5 source paths were SHA-verified.
- The three original complete 112-env shard manifests were reused, one scene per original shard/seed. Normal RL reset, requested XY/yaw, origin z +0.35 m, zero 6D actions, dt 0.005 s, 200 physics steps and DEM at 50/100/150/200 were preserved.
- The saved Actor vector is the step-200 `obs[policy]` 22-vector. Separate pre-noise/post-noise vectors were not stored; they are diagnostic-only and are not represented as reconstructed inputs.
- E1 episode/prefix metadata is incompatible with these radial records. No synthetic episode fields were created.
- The E1 numeric-gate provenance path is corrected in `tolerance-source-addendum.json`; the pre-run tolerance values and file are unchanged.

## Fixture and parity

| Anchor | Saved versus reconstructed | Root max error | Actor max error | DEM max height error |
|---|---|---:|---:|---:|
| first-differential | pass | 0.0 m | 0.0 | 0.0 m |
| second-differential | pass | 0.0 m | 0.0 | 0.0 m |
| first-graded | pass | 0.0 m | 0.0 | 0.0 m |
| second-graded | pass | 0.0 m | 0.0 | 0.0 m |

- For every anchor, the pose record, Actor JSON and all four DEM JSON files are byte-identical to the saved originals. All 200 trace physics payloads, including root/joint/contact/termination and episode/reset counters, are identical after excluding wall and monotonic timestamps.
- Thus stored settled eligibility, primary DEM class and known onset are invariant under this fixture. These labels were not recomputed with changed thresholds.
- The 28-row manifest overlay changes only output directories. Run/order/anchor/candidate/Path SHA/seed/goal remain unchanged; no Path was published.

## Resource and lifecycle

- Three sealed processes: 200/200 steps, native exit 0, completion marker, host/container SHA match, owned residual 0. GPU peak 5906 MiB; anon+shmem peak 3.674 GiB; host output 27.1 MB.
- The device-total GPU cap was met. Per-PID GPU samples were not captured, and NVIDIA accounting mode is disabled; the v2 zero-unresolved-memory condition cannot be proven from these runs. No automatic post-control rerun was made.
- An earlier pre-control fixture root failed because the launcher precreated the simulator output directory. The failure root was preserved. A RED→GREEN test covered the launcher correction; one allowed remediation cycle was used.

## Scope boundary

X1/X2, external-router operations, candidate Path publication, follower/bridge, 28-process P1 rollout, Outcome, relabeling and learning were not performed.
