# P1 startup ownership handshake and resource smoke

**Decision: `p1_resource_attribution_fail`.** The conditional P1 rollout did not start. X1, X2, bridge, follower, candidate Path and Outcome were not run.

## Provenance and isolation

- New branch/worktree: `exp/eatnav-p1-handshake`, `/tmp/eatnav-vnt-p1-handshake`, from `a2915537c124ea59061061c8625273a4310c3b30`.
- Authoritative run root: `/tmp/eatnav-p1-handshake-20261008-r2`. A preceding `/tmp/eatnav-p1-handshake-20261008-r1` stopped before launching a wrapper because its baseline did not classify already-running desktop graphics processes. It is not a measurement.
- H0 verified 142 file SHA entries, including prior parity, the two failed attribution attempts and their backup, selected anchors, Paths, manifest, source files, and resource contract. The 28-row overlay changed only `output_directory`; overlay SHA is `b76c726cd91a70f254442d8ac0abd5a060951ffbdb18fbc4e7c29337bff9e06a`.
- The historical r3 `input-freeze.json` records resource SHA `978eb72e...`; the immediately preceding attribution attempt independently pinned the current v2 resource file SHA `709e2b0d...`. The current v2 file was verified against that attempt. Numeric caps were not changed.

## Implementation and tests

The simulator wrapper uses only Python standard-library imports before authorization. It writes and reads back `owner-ready.json`, waits for a matching random-token authorization, then executes the original frozen simulator in place. The host resolves one PID using token, container cgroup, `NSpid`, and start tick, saves the complete process table, takes an NVML bracket before authorization, and only then writes authorization. Full process tables are saved in every bracket.

The related offline suite passed **42 tests** after the final H5 validation code was added. The wrapper SHA is recorded in each run's launch intent. The H4 host runner was extended with H5 artifact and resource checks after H4; the H4 wrapper and frozen simulator source did not change. The host runner's exact H4 source SHA was not captured before that extension; this limits source-level replay provenance, but the H4 PID, marker and bracket records remain intact.

The H3 suite covered exact token/NSpid/start-tick mapping, stale and mismatched markers, preauthorization GPU rejection, child lineage, path-only false matches, timeout/rejection, and live-prior-owner blocking. It did **not** include a dedicated nonreturning-App-close fake or exact cleanup-signal fake before the real run. The H3 prerequisite was therefore incomplete; this is a protocol deviation in addition to the H5 terminal bracket failure. Neither H4 nor H5 is used to authorize candidate measurement.

## H4 zero-physics preflight

H4 passed once in r2: registered host PID `162712`, start tick `2131729`; exact token and container identity matched. The preauthorization bracket had **zero owned GPU processes**. The completion marker recorded zero physics and control steps before App close, native exit was 0, and exact owned residual was 0. Seven brackets contain the complete process tables. No scene, robot, Path, bridge or follower was used.

## H5 candidate-free shard-B smoke

The single authorized H5 attempt used original shard-B, batch seed `20261007`, original reset and zero-action protocol. The simulator reached **200/200 physics steps** for all four selected poses with zero physical terminations. Its saved output has 16 DEM frames, eight finite `[22]` Actor records and 800 contact trace rows. Contact capacity was 4096 throughout; every row was finite and bounds-valid, with zero full-buffer suspicions. The remote output and separately archived host copy have **94/94 matching SHA files** including the coordination markers. Native client exit was 0. Exact owned PID `164857` / start tick `2141572` and run token have no surviving process.

Resources stayed within the v2 caps: peak device total **5848.4 MiB** (cap 7680), maximum bracket gap **122.4 MiB** (cap 256), unresolved GPU memory **0**, peak container anon+shmem **3.71 GiB** (cap 6), minimum host available **80.7 GiB**, and maximum sampled storage **24.4 MiB** (cap 1 GB).

The **33rd bracket (index 32)** failed with `owned_start_tick_mismatch` and `simulator_disappeared_in_bracket`. The previous bracket still saw the exact simulator PID/start tick. The failing bracket began after the simulator had exited normally: its before and after process tables lacked that PID, while `sim-finished.json` and heartbeat both record step 200. Thus the first failing signal is an end-of-process monitoring boundary, rather than a resource cap or physics failure. The monitor required the simulator to stay present for a bracket selected while the Docker client was still transitioning to exit. This attempt remains protocol-invalid; its complete physical trace is preserved only for diagnosis and is not promoted to H5 pass.

No H5 retry was started, as required for a postauthorization failure. No X1/router audit, X2, or 28-process candidate rollout began. The prior 4/4 radial physics parity remains the functional parity evidence and was not rerun.

## Artifacts

- H0 overlay/audit and H1 root-cause: `/tmp/eatnav-p1-handshake-20261008-r2/`.
- H4 preflight: `/tmp/eatnav-p1-handshake-20261008-r2/preflight/`.
- H5 failed measurement, 33 complete NVML brackets, owner table, cleanup: `/tmp/eatnav-p1-handshake-20261008-r2/smoke/`.
- Read-only poststop copies and 94-file SHA ledger: `/tmp/eatnav-p1-handshake-20261008-r2/smoke/poststop-archive/`.
- Durable backup: `/home/kojima/eatnav-backups/2026-10-08-p1-handshake-r2` (128 files, 26,018,850 bytes, every SHA matched the output root at backup time).
- Previous attribution attempts and durable backup were not altered.

The next protocol version needs a terminal-state-aware finalization bracket that recognizes a verified normal simulator exit without treating the absent PID as a live ownership mismatch. That change requires its own RED→GREEN tests and a new preflight/smoke root; this report does not authorize or perform a retry.
