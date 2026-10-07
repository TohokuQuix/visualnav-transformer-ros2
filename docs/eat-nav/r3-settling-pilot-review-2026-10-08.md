# R3 eight-pose candidate-free settling pilot — 2026-10-08

## Decision

`r3_candidate_free_pilot_class_mismatch_review_required`.

Both four-pose shards were protocol-valid: eight of eight poses have complete, measurement-valid eligibility results. Four were settled-anchor eligible and four were numeric-ineligible; none physically terminated. The fixed physical-supply criterion passed for differential, graded, and flat control at **1/2 each**, but transition was **0/2**. One eligible flat-control pose changed from virtual `flat-only` to settled `graded`, so the class-mismatch review category applies. This pilot does not authorize applying R3 to a second seed.

## Frozen experiment and integrity

- Source commit: `c6b14d9d77ba2d424f4e832c51d810923819a7af` on `exp/eatnav-r3-settling-pilot`, separate worktree `/tmp/eatnav-vnt-r3-settling-pilot`.
- New output root: `/tmp/eatnav-r3-settling-pilot-20261008`. The original 112-pose inventory, continuation, offline scan, `analysis-v9`, and proposal were read-only. Their old ledgers were reverified over 1,897 and 912 files. Input source, RL checkpoint, scene configuration, terrain mesh SHA, eight-pose proposal, and virtual DEM SHA were checked before physics.
- Frozen A/B assignment: A uses primary ranks 1, 3, 5, 7; B uses ranks 2, 4, 6, 8. Four poses per shard, once each. The 104 R3 reserve poses were frozen separately and never entered an execution manifest.
- Each shard used the existing 112-environment RL scene, normal reset, requested XY/yaw only, root z `env_origin.z+0.35 m`, zero root velocity and zero 6D action, physics dt 0.005 s, 200 steps, four DEM frames, contact capacity 4096, and Actor observations before physics and at step 200. The existing termination and settled numeric limits were unchanged.
- Shard A and B each had native exit 0, complete 200-step traces, valid contact bounds, no full-buffer/nonfinite indication, host/container artifact SHA match, sealed output, and zero owned residual processes. B started only after A passed the protocol gate. No physical termination or post-reset contamination occurred.
- Total elapsed time was 86.3 s. Peak GPU device memory was 5,857 MiB (<12,288), container anon+shmem 3.92 GB (<10 GiB), minimum host available memory 87.7 GB (>8 GiB), and peak monitored storage 22.1 MB (<500 MB). The final output root occupies about 17 MB.

## Pose-level result

The primary DEM comparison uses frame 03 at physics step 200, matching the first inventory. Numeric-ineligible poses retain their full measured DEM and trace but do not receive a primary settled class for supply.

| Rank | Virtual class | Terrain, radius, direction | Eligibility | Primary settled class | Max angular speed, steps 161–200 | Physical supply |
|---:|---|---|---|---|---:|---|
| 1 | transition | k_rail_step_field, 0.40 m, 2 | numeric-ineligible | NA | 0.3165 rad/s | no |
| 2 | transition | k_rail_field, 0.60 m, 12 | numeric-ineligible | NA | 0.2459 rad/s | no |
| 3 | differential | k_rail_step_field, 0.60 m, 0 | numeric-ineligible | NA | 0.2168 rad/s | no |
| 4 | differential | boxes, 0.60 m, 1 | eligible | differential, class/onset match | 0.0219 rad/s | yes |
| 5 | graded | k_rail_step_field, 0.30 m, 6 | eligible | graded, class/onset match | 0.1951 rad/s | yes |
| 6 | graded | k_rail_field, 0.50 m, 1 | numeric-ineligible | NA | 0.4271 rad/s | no |
| 7 | flat control | k_rail_field, 0.50 m, 4 | eligible | flat-only, class/onset match | 0.0062 rad/s | yes |
| 8 | flat control | random_rough, 0.60 m, 6 | eligible | **graded**, class mismatch | 0.0142 rad/s | no |

The frozen angular-speed limit is 0.20 rad/s. Rank 6 also exceeded the H1 limit: 0.1043 rad against 0.10 rad. All eight DEMs had at least 0.78 valid fraction, above the frozen 0.70 requirement. Rank 8 was physically eligible, so its class change is a genuine offline-versus-settled comparison result under this protocol.

## Cause candidates and limits of inference

- Rank 8's virtual maximum forward slope was 0.1129, close to the unchanged graded threshold 0.12. The settled value was 0.2210. A small pose-dependent DEM change can therefore cross the existing classification boundary. This observation identifies sensitivity; it does not prove which contact, attitude, ray, or height-normalization effect caused it.
- Rank 1 retained a `transition` DEM class in a **secondary** frame-03 inspection, but failed eligibility on angular speed. Rank 2's secondary frame-03 class was `flat-only` even though its virtual class was `transition`; its virtual interaction onset was 0.55 m, near the 0.6 m comparison-corridor edge. Both ranks remain primary `NA` because they were numeric-ineligible. The secondary inspection was not used to change the decision.
- Rank 3 and rank 6 kept their virtual DEM classes in the same secondary inspection but exceeded the settled numeric limits. Rank 6 had the largest decoded penetration statistic, 0.00322 m. Penetration remains a threshold-free descriptive measure and is not classified as a safety violation or Outcome.
- The 0.6 m DEM corridor is an interaction proxy, not a measured two-second reach guarantee. Eight poses across different terrain-specific source families all come from the first batch seed; this is not seed-generalization evidence.

The frozen primary pilot therefore **does not establish transition physical supply**. Review the saved traces and frozen reserve list or reconsider placement geometry in a separate decision. No threshold, pose, radius, reset, DEM classifier, or R3 rule was changed after seeing the results.

## Artifacts and stop point

The output root contains both independent shard subroots, per-pose trace/contact/DEM/Actor data, process identity, resource logs, seals, SHA ledgers, final decision, eight figures, and the secondary diagnostic detail. The compact review artifacts are committed under `docs/eat-nav/r3-settling-pilot-20261008/`.

No reserve pose, second seed, candidate Path, follower, bridge, router, Outcome collection, model training, online selection, or robot control was started.
