# Radial 112-pose inventory: termination-aware continuation review

The 15 previously sealed shards were reused read-only. A new root reran shard-15 and executed shards 16–22 in the frozen order. All 23 shards are protocol-valid; the old shard-15 partial trace remains measurement-invalid and was not promoted.

## Classification

| Class | Poses |
|---|---:|
| A: settled-anchor eligible | 94 |
| B: anchor ineligible by settled numeric limits | 17 |
| C: anchor ineligible by physical termination | 1 |
| D: measurement invalid | 0 |

The single C pose is `b20261007-k_rail_field-a00`, env 5. It terminated at physics step 140 with the frozen `tilt_over` term. The pre-reset root state, 22-component Actor observation, contact aggregate, two available DEM frames, episode/reset counters, and exact term flags were captured. The other four selected poses in that shard completed 200 steps. C is an anchor eligibility result, not a candidate Path outcome or a safety label.

Across all seven terrain families, eligible counts by terrain are: boxes 16, hf_pyramid_slope 16, k_rail_field 11, k_rail_step_field 6, pyramid_stairs 16, pyramid_stairs_inv 16, random_rough 13. Numeric exceedance reasons (a pose can have more than one) are angular speed 14, linear speed 6, XY drift 2, and heading H1 2.

DEM: 111 complete poses have four frames; the terminated pose has the two frames available before step 140. Prephysics Actor is present for 112/112; step-200 Actor for 111/111 completed poses; terminal pre-reset Actor for 1/1 terminated pose. Contact bounds, capacity 4096, finite and no-full-buffer checks pass for 112/112.

## Terrain interaction and independence

Among 94 eligible anchors, the outcome-blind DEM classification is flat-only 70, graded 15, differential 9, transition 0. Ten anchors have a classified interaction within the 0.6 m two-second reach proxy. The provisional shortlist has 10 anchors: 9 differential and 1 graded, drawn from only two terrain source families. Each terrain's 16 directions share one seed/source family; they are related measurements, not 16 independent StateGroups. The missing transition class and source-family concentration require a second-seed or anchor-placement design review before freezing a candidate bank. No second seed was executed.

## Runtime and integrity

The continuation coordinator completed in 374.2 s; summed process time across all 23 shards is 900.5 s. The highest recorded GPU device total was 5,907 MiB, container anon+shmem 3.94 GB, and the lowest host available RAM remained above 87 GB. The new continuation root occupies 71.4 MB on the host; the old root occupies 137.9 MB. The new-run host-plus-container monitored storage peak was 129.0 MB, below the frozen 500 MB cap.

All eight new shards report native exit 0 and owned residual 0. Independent verification found 797/797 host/container run files SHA-identical, all 912 new-root ledger entries matching their files, and the 15 reused seal hashes unchanged. The old failed shard has no seal and its failure marker SHA is unchanged. The worktree source commit is `8261706d5089542c51152b2793a42f705d3ff2da`.

**Decision:** `radial_inventory_complete_with_physical_terminations`. The inventory is complete. Candidate-bank offline review can inspect these measurements, but candidate specification freeze requires review of missing transition coverage and source-family concentration. Candidate Paths, follower, bridge, router, Outcome, training, online selection, second seed, and robot control were not started.

Primary run artifacts: `/tmp/eatnav-radial-termination-20261008/`.
