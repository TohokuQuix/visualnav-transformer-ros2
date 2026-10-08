# P1 r3 prelaunch report (2026-10-08)

**Decision: `p1_resource_contract_invalid_stop`.** The X1 router gate, X2 readiness, and all 28 candidate processes were not started. No candidate Path was published, no candidate control step occurred, and no unrelated process was managed.

## Frozen inputs and geometry

- Isolated worktree: `/tmp/eatnav-vnt-local-primitive-p1-r3`, branch `exp/eatnav-local-primitive-p1-r3`, base commit `ace048acb7ef75310f350aeed46393e05f4daada`.
- Immutable prior r2 decision: `p1_measurement_or_protocol_invalid_stop` due to the old 6,656 MiB resource forecast. Its source, gate, selected anchors, and manifest were SHA verified; no historical decision was changed.
- Versioned gate selected `first-differential`, `second-differential`, `first-graded`, and `second-graded`. The two transition-common alternatives have known onset **0.20 m**, strictly within the 0–0.60 m primary interval. All four were previously measured and physically eligible. No candidate Outcome was read.
- The new manifest has 28 unique, sequential physical run IDs: seven per anchor, comprising twelve 0° direct repeats and sixteen non-direct primitives. The five Path SHAs, goal `[0.75, 0.0]`, seed 123, retry count zero, source SHA values, four DEM frames, and finite 22-value Actor state passed static verification. The global navigation goal and local primitive endpoint remain separate. The frozen map/local transform source SHA was verified; a live transform was not exercised because the resource gate stopped the run.
- The superseded r3 output root `...-r1` was left unchanged after a prelaunch provenance-source correction. The authoritative root is `/tmp/eatnav-local-primitive-p1-r3-20261008-r2`.

## Resource contract and measured stop

The P1-specific cap is **7,680 MiB** on the audited RTX 5070 Ti, UUID `GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26`, total **16,303 MiB**. Driver `580.173.02`, NVML `13.580.173.02`, and MIG `[N/A]` were recorded. Historical measured peak was 5,949 MiB; the 20% forecast is 7,138.8 MiB, leaving 541.2 MiB below the new cap. Preserved forecasts with margin were 1.805 h wall time, 0.320 GB storage, and 4.469 GiB container anon+shmem, all under their respective limits.

The corrected, bracketed NVML baseline immediately before X1 measured **358.125 MiB device total** and **311.859 MiB attributed to other processes**. The latter exceeds the frozen **256 MiB** allowance by **55.859 MiB**. Unresolved GPU bytes were zero; the bracket gap was 46.266 MiB, within its 256 MiB limit; temperature was finite at 39 °C. The only failed baseline condition was `other_process_gpu_over_cap`. The raw sample includes per-PID/start-tick attribution, query timing, GPU identity, and its SHA. No unrelated process was stopped to alter this result.

Because the baseline failed, external router provenance, X2, the second pre-candidate baseline, per-run traces, duplicate tests, P→C→A→R→Y Outcomes, goal relabeling, and downstream figures are **not measured**. `pilot-started.marker.json` does not exist. This is a resource prelaunch stop, not a candidate quality result.

## Verification and artifacts

The focused P1 and GPU-bracket suite passed **16 tests**. Script compilation and the independent 28-row/SHA static audit passed. A broader historical E1 coordinate-addendum suite had **8 failures and 24 passes** because its old `/tmp/eatnav-e1-artifact-recovered-continuation-20261004/continuation-manifest.json` fixture is absent after the environment restart; these tests do not exercise r3. The complete static audit and exact input SHA values are in `prelaunch-static-verification.json`, `source-provenance.json`, `input-freeze.json`, and `artifact-sha-ledger.json`.

Review figures: [anchor onsets](r3-anchor-onsets.png), [primitive Paths](r3-primitive-paths.png), [GPU forecast and baseline](r3-gpu-forecast-baseline.png). The frozen manifest and full measurement are saved beside this report and in the authoritative `/tmp` root.

The next P1 attempt requires a fresh pre-X1 baseline satisfying the frozen other-process allowance; this report does not authorize bypassing that gate or reusing a candidate result.
