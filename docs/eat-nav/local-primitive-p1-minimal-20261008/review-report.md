# Minimal four-anchor P1 preflight

## Decision

`p1_measurement_or_protocol_invalid_stop`, subreason `prelaunch_resource_forecast_exceeds_frozen_cap`. This is a **forecast stop before any P1 measurement**, not a failed candidate or invalid historical measurement. The previous five-role decision `role_aware_terrain_support_insufficient` and the earlier `local_motion_primitive_terrain_support_fail` remain unchanged.

The new `minimal_p1_gate_v1` was SHA-frozen (`d5db05e611eba28eeaf5c8e0dd8013bc4b1351bbceff71941ee71aed93b0a05f`) before applying it to the eight measured anchors. The four-slot gate passed using only saved anchor/DEM/Path/source records. No previous candidate Outcome was read. The selected anchors and process manifest are frozen in `selected-four-anchors.json`, `p1-run-manifest.json` and `input-freeze.json`.

| Slot | Selected anchor | Measured primary role | Role evidence | Alternative |
|---|---|---|---|---|
| Seed 1 differential | `first-differential` | R-differential | Max known-pair RMS 0.03408 m; seven separated pairs | None |
| Seed 2 differential | `second-differential` | R-differential | Max known-pair RMS 0.04806 m; nine separated pairs | None |
| Seed 1 transition | `first-flat-only` | R-transition-common | Transition step margin 0.02473 m; onset 0.60 m | `first-graded`: margin 0.02103 m, onset 0.20 m |
| Seed 2 transition | `second-flat-only` | R-transition-common | Transition step margin 0.06000 m; onset 0.60 m | `second-graded`: margin 0.02051 m, onset 0.20 m |

The names `flat-only` refer to the old anchor inventory labels. They are **not** flat controls in the new Path-level classification. Both selected transition onsets lie exactly at the inclusive 0–0.60 m boundary, so they provide boundary transition support only. The prospective selection rule ranks the known transition margin before onset; it therefore chose these anchors over the earlier but weaker alternatives. Four distinct `(seed, source_family)` identities are present. R-graded-common and R-flat-control remain unsupported and untested.

The manifest contains 28 unique physical process IDs: four anchors × (one each at ±20° and ±10°, plus three direct 0° runs). The eight-point local Path SHA is frozen for each row. Each row stores the same robot-local navigation goal `[0.75, 0.0]` separately from its local primitive endpoint. Seed `123`, source SHA values, retry count zero and run order are frozen. Static rehashing verified 28 IDs, 12 direct repeats, 16 non-direct rows, four DEM frames and finite 22-value Actor body state for every selected anchor. Exact simulator anchor reconstruction and coordinate-corrected runtime behavior remain untested because the resource gate stopped before X1.

## Resource stop

The frozen local-primitive review measured 23 fresh-bridge candidate runs from the durable expanded E1 archive. Device-total GPU peaks ranged **5,823–5,949 MiB**. Applying the requested 20% prospective margin to the measured maximum gives **7,138.8 MiB**, exceeding the SHA-frozen expanded E1 hard cap of **6,656 MiB** by **482.8 MiB**. Even the lowest observed peak plus 20% is 6,987.6 MiB, above the cap. The same expanded E1 contract supplies the 4 h schedule cap, 1 GB storage cap, 6 GiB anon+shmem cap, 600 s per-process timeout and 90 s heartbeat. The preserved 28-process forecast is 1.805 h and 0.320 GB with margin; anon+shmem is 4.469 GiB with margin. The GPU device margin is the sole failed prospective resource check.

The earlier `r1` output root was superseded before any router, X2 or candidate execution because its source map depended on a missing historical `/tmp` follower binary. The `r2` root uses the byte-identical SHA-verified binary from durable backup. A targeted test reproduced that path defect; the corrected test passed. The `r1` root was not changed or promoted.

## Execution boundary and artifacts

No router was started, stopped or audited; no X1/X2, bridge, follower, simulator candidate process or Path publication occurred. The 28-run P1 clock was not started. No P→C→A→R→Y, two-second Outcome, duplicate, relabeling or trained Evaluator result exists. Machine-readable not-started markers make those omissions explicit; no figure of unmeasured signals is fabricated.

The output root is `/tmp/eatnav-local-primitive-p1-minimal-20261008-r2`. It contains the source provenance, gate, selection, manifest, static verification, resource forecast, final decision and SHA ledger. The same small artifacts are saved with this report in Git. The frozen prior geometry and follower static screens remain valid, but they do not establish runtime feasibility for these four anchors.

## Verification

`tests/test_minimal_p1.py` and `tests/test_minimal_p1_freeze.py` exercise selection, missing-anchor rejection, four-frame/body-state checks, manifest accounting, resource margin and source integrity. The prior primitive and role ledgers, both radial seeds, settling records, five Path definitions, follower source/config/binary, RL checkpoint audit copy, simulator/bridge and E1 safety contract were rehashed before selection. `artifact-sha-ledger.json` records each new saved file.

No numeric cap or Path was changed to obtain a pass. A later rollout would need a separately approved resource contract or a pre-run implementation that lowers validated GPU use while preserving the frozen experiment, plus completion of the outstanding runtime restoration and X1/X2 gates.
