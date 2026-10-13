# MI-A v3.1 formal evaluation: stopped at process 43

**Decision: Inconclusive. MI-B is not authorized by this result.** The frozen 192-process schedule was started on a new persistent root. Processes 0–42 passed their frozen per-process protocol, including Gate P and inner/outer seal verification. Process 43 crashed twice during Isaac Sim/Kit startup, before Python main, Path publication or physics. Both failed attempts retain their original invalid decisions and seals. No process after 43 was launched. No 12-seed bootstrap or MI-A candidate-quality verdict was computed from this incomplete schedule.

## Provenance and preservation

- Frozen assignment manifest SHA-256: `5d3ccdc70b45c534ec27f60fe9b2a959c69f41ec9093cc07b9f879fbdbf4ee7a`; 192 rows, 12 distinct seeds, four terrain types, three levels and both initial flipper postures. Scientific source, assignment, Gate P, 0.03 m primary threshold, Actor, follower and physics were not changed.
- First coordinator source SHA-256: `768751ab7bd55a39a4414946b8d2d3ab460f9ec87ab5c246096b8974466ee77f`, commit `2c29851`. First evidence root: `/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-a-v31-formal-20261014-01`.
- Continuation source SHA-256: `28a2ab098d0e83a44da5753179768534c056f4db023d490032579097acadf829`, commit `fd95748`. Second evidence root: `/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-a-v31-formal-20261014-02`. It links the 43 sealed valid attempts read-only and records each original seal SHA. It does not link or promote either failed process-43 attempt.
- The v3.1 Stage 4a/4b evidence remains in `/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-a-v31-20261014-0141/`; no prior evidence was rewritten.
- Crash dumps and Python snapshots were copied byte-for-byte from the container to `/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-a-v31-formal-20261014-crash-diagnostics/`. Host/container SHA values match.

## First failing signal

Process 43 is `pyramid_stairs`, level 3, seed `2026101403`, Rear-folded, Forward +30°. The first attempt had native exit 1 after 1.26 s and a Kit crash backtrace through `libX11.so!XOpenDisplay`. The second fresh-root attempt had native exit 1 after 1.28 s and a Kit startup/crash-reporter backtrace. Both report a segmentation fault before Python main and have no simulator output or candidate trace. The X11 socket and Xauthority mount were present during a read-only check, and the container had no residual simulator process. No blocking launch-command or display-environment difference from the preceding successful process was found. The underlying Kit/X11 failure mechanism remains **unknown**. Because the startup failure repeated, further retries were stopped.

## Valid measurements and resource audit

The 43 valid processes comprise 12 pre-Path captures and 31 candidate processes, or 124 candidate trials. They cover three seeds of `pyramid_stairs` only; the third seed's Rear-folded candidate set is incomplete. Each of the 43 original attempts was reverified against its row identity, Gate P, inner seal and outer seal. This subset is retained as evidence but is not a valid 12-seed formal analysis sample. No terrain/posture/candidate effect or uncertainty from it is presented as a formal MI-A result.

In the first root, the maximum sampled GPU device usage was 5,290 MiB (cap 6,656 MiB), minimum host MemAvailable was 84,364,900 KiB (floor 2,097,152 KiB), largest attempt wall time was 52.78 s (cap 600 s), and largest sampled attempt storage was 262,958 bytes (cap 800 MB). The first root occupies about 46 MB; the continuation root about 244 KB excluding linked data. The failed native exits are recorded, and no owned simulator process remained. Neither stop was a measured resource-cap violation.

## Decision and next condition

MI-A is **Inconclusive** because only 43/192 processes are valid. The frozen terrain-stratified, seed-clustered 10,000-draw bootstrap, 0.03 m Primary XY threshold and 9/12 seed consistency criterion cannot be applied. Twelve independent terrain seeds are planned, but only three were encountered; terrain type, level and seed are coupled by design, limiting any later generalization. MI-B, Motion Predictor training, MI-C and Phase C collection remain stopped.

Before resuming, diagnose the repeated Kit startup segfault without changing scientific inputs or silently reclassifying failed attempts. Any further continuation must use a fresh source SHA and attempt root, preserve both failed seals, and prove the runtime cause or a bounded environmental remedy. DEMO remains Front-folded fixed; collection assignment remains StateGroup-level Front/Rear randomization; MI-A evaluation remains manifest-specified.
