# R6 fixed-cohort production driver

`config/eat_nav/r6/protocol.json` fixes all seven fields × levels0–4 with 10 train
and 10 validation StateGroups per cell: 700 groups, 9,800 candidate branches.
Five separate supplemental scene cohorts (wall, narrow, one_sided, diagonal,
approach) add 100 groups/1,400 branches. Base counts exclude supplementation.

The frozen parent terrain width is ten columns. Execution chunk size may be
1/2/5/10 after the actual benchmark. Each chunk preserves parent seed, original
tile column, generated tile index (column×5+level), root initialization stream,
and all candidate branches. Parent and derived seeds are disjoint between
collection, validation rollout, and test rollout. All branches of a parent seed
share one role. Cohort membership includes exact derived terrain/initial/command
seed IDs; actual simulator geometry metadata remains the authoritative terrain
record. Vector dynamics randomization can depend on batch width; reducing chunks
preserves terrain/root RNG identity but must not be claimed to reproduce exact
physical starts until independently verified.

Validation has 30 independent parent seeds per field/level and 13 conditions:
old reference, two-second event, ten pose profiles, combined (13,650 planned runs).
Test independently has 30 seeds/cell and six frozen conditions (6,300 planned).
These phases are manifest-only pending model/policy integration; the host driver
runs collection only. Thresholds must freeze before test/bag evaluation.

Prepare only (does not stage or launch Isaac):

```
PYTHONPATH=src:scripts /usr/bin/python3 scripts/eat_nav_r6_run.py --prepare-plan docs/eat-nav/r6-2s-pose-20261030/collection-plan.json
```

After the successful actual benchmark and internal root authorization, supply a
benchmark JSON with `passed:true, selected_num_envs:10` (actual selected size),
and an internal approval JSON with `production_collection_approved:true`:

```
PYTHONPATH=src:scripts /usr/bin/python3 scripts/eat_nav_r6_run.py --plan docs/eat-nav/r6-2s-pose-20261030/collection-plan.json --benchmark BENCHMARK.json --approval-record INTERNAL_APPROVAL.json --output /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/collection
```

Runtime staging uses fresh host/container `/tmp/eatnav-r6-production-*`, copies
all EAT-Nav Python modules plus immutable actor/env/config, and checks every
copied SHA256. Image identity and exact source hashes are saved. No container
restart or dependency installation occurs. Isaac is headless; sensors remain on.
GNU timeout runs inside the container; host timeout has a further 30-second bound.
Partial outputs, simulator logs, exact commands, per-group identities, completion
and failure JSON are retained. Completed and failed groups are skipped on resume;
interrupted running attempts become preserved failures. No automatic retries
select only successful initializations. Collection capacity changes retain frozen
parent tile identities and skip previously attempted groups.

Packing actual outputs, optionally merging the independently relabeled old pool:

```
PYTHONPATH=src:scripts /usr/bin/python3 scripts/eat_nav_r6_run.py --plan docs/eat-nav/r6-2s-pose-20261030/collection-plan.json --output /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/collection --pack-output NEW_DATASET --reuse-dataset /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/reused-dataset
```

Packing takes roles from the frozen group manifest. It rejects role/seed changes,
duplicate group attempts, fresh/reused seed collisions, shape/dtype mismatch and
field/seed split leakage. Actual partial branches retain censored teachers; absent
branches stay masked. A failed stabilization with no initial observation remains
an unavailable group in the audit, rather than becoming a synthetic success.
All-unobserved new collection can merge a reused pool, but the manifest separately
reports zero fresh support. Check planned/attempted/completed and known targets;
never interpret the frozen plan as completed data.
