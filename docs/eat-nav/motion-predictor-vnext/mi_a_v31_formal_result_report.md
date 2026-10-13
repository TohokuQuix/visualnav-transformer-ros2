# MI-A v3.1 formal evaluation: Kit startup recovery and final result

**Decision: MI-A Pass under the frozen exploratory gate.** All 192/192 scheduled processes are protocol-valid, all 192 Gate P checks pass, and 576/576 candidate trials have complete 40-step labels. The analysis uses 12 independent terrain seeds; four env tiles per seed are paired replicates. MI-B may be considered as the next phase, but no MI-B training or Phase C collection was started.

## Startup fault and recovery

The original process 43 (`pyramid_stairs`, level 3, seed `2026101403`, Rear-folded, Forward +30°) failed twice during Isaac Sim/Kit startup, before Python main, Path publication or physics. The first native stack reached `libX11.so!XOpenDisplay`; the second reached Kit startup/crash-reporter code. Both original failed attempts and their decisions/seals remain unchanged in `mi-a-v31-formal-20261014-01` and `-02`. Kernel evidence showed a segmentation fault, with no observed OOM or GPU Xid. Docker had `DISPLAY=:1`, a mounted Xauthority file and a live X11 socket; a direct `XOpenDisplay` check passed. No blocking environment or launch-command difference from the preceding valid process was established. The exact native root cause therefore remains unknown.

Minimal Kit startup passed four consecutive launches using the original DISPLAY and one diagnostic launch with DISPLAY unset. The latter was slower and was **not** used for formal evaluation. No scientific input, simulation configuration, headless flag, DISPLAY, Xauthority, Actor, follower, physics, Path, Gate P or resource threshold was changed. The process-43 input was run twice in separate fresh roots with the original launch conditions. Both runs were protocol-valid, and their pre-Path snapshot, motion labels and four trace files were byte-identical. The first valid run became the formal row 43; the second remains an excluded stability control. Neither failed process-43 attempt was promoted. The crash did not recur in rows 44–191.

The final evidence root is `/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-a-v31-formal-20261014-03`. Its `frozen-provenance.json` records the 43 reused original seal hashes, the designated and excluded process-43 seals, and both failed seal hashes. Frozen assignment SHA-256 is `5d3ccdc70b45c534ec27f60fe9b2a959c69f41ec9093cc07b9f879fbdbf4ee7a`; protocol SHA-256 is `ec1a5b43e2770beb724a5ca128c99f5fad0ee62211b2bbf559fbd1371bb84444`. The new continuation coordinator SHA-256 is `eb3514ea74382283e4bf5b5247648b4c858fd7932fcfaa718f884e993d293382` (commit `c971368`). No candidate rollout was automatically retried after Path publication.

## Frozen MI-A analysis

Arc minus same-direction Direct is measured as the 2-second displacement vector in the initial body frame, averaged over four env tiles within each seed. The table gives the norm of the 12-seed mean XY vector, the frozen terrain-stratified seed-bootstrap 95% interval projected onto that mean direction, and the number of seeds whose individual XY effect norm exceeds 0.03 m. The interval is exploratory because its projection direction is estimated from the same data.

| Initial flipper pose | Arc candidate | Mean XY effect (m) | Projected 95% interval (m) | Seeds >0.03 m | Frozen criterion |
| --- | --- | ---: | ---: | ---: | --- |
| Front-folded | Forward +30° | 0.0654 | 0.0552–0.0765 | 12/12 | Pass |
| Front-folded | Forward −30° | 0.0532 | 0.0437–0.0631 | 10/12 | Pass |
| Front-folded | Reverse +30° | 0.0038 | −0.0127–0.0206 | 6/12 | Below threshold |
| Front-folded | Reverse −30° | 0.0811 | 0.0633–0.0996 | 11/12 | Pass |
| Rear-folded | Forward +30° | 0.0353 | 0.0237–0.0450 | 9/12 | Inconclusive |
| Rear-folded | Forward −30° | 0.0744 | 0.0648–0.0830 | 10/12 | Pass |
| Rear-folded | Reverse +30° | 0.0413 | 0.0333–0.0502 | 12/12 | Pass |
| Rear-folded | Reverse −30° | 0.0470 | 0.0411–0.0533 | 10/12 | Pass |

The frozen MI-A rule requires at least one Forward and one Reverse candidate/posture with lower interval above 0.03 m and at least 9/12 seeds above 0.03 m. Six combinations satisfy that rule, hence **MI-A Pass**. Front-folded Reverse +30° is below the threshold by its upper interval; Rear-folded Forward +30° remains inconclusive. The separate Rear-minus-Front Arc-relative interaction mean XY norms are 0.0301 m (Forward +30°), 0.0275 m (Forward −30°), 0.0410 m (Reverse +30°) and 0.0493 m (Reverse −30°). These are posture interactions in measured motion, not a prescribed preferred posture.

Terrain-specific mean XY effect norms (m) are descriptive, with only three independent seeds per terrain:

| Pose / candidate | Pyramid stairs | Inverse pyramid | Random rough | K-rail step field |
| --- | ---: | ---: | ---: | ---: |
| Front / Forward +30° | 0.055 | 0.070 | 0.081 | 0.056 |
| Front / Forward −30° | 0.079 | 0.069 | 0.037 | 0.032 |
| Front / Reverse +30° | 0.017 | 0.017 | 0.029 | 0.016 |
| Front / Reverse −30° | 0.136 | 0.116 | 0.035 | 0.037 |
| Rear / Forward +30° | 0.042 | 0.067 | 0.028 | 0.005 |
| Rear / Forward −30° | 0.125 | 0.104 | 0.033 | 0.038 |
| Rear / Reverse +30° | 0.042 | 0.049 | 0.049 | 0.037 |
| Rear / Reverse −30° | 0.055 | 0.061 | 0.040 | 0.035 |

The machine-readable result preserves every seed mean, terrain mean, yaw effect and bootstrap interval. Across all 23,040 candidate trace rows, there were no terminal records. Peak single-link contact force was 8,687 N and peak absolute applied joint torque was 3,000 in the logged units; these are descriptive extrema, not new gates or safety conclusions.

## Protocol, resources and interpretation

Every row was reverified against its frozen assignment, Gate P, inner/outer seal and native exit before analysis. Native exits were 192/192 zero. The formal root has 192 row links/directories; the first 43 are read-only links to original valid evidence. Peak sampled device GPU use was 5,555 MiB (6,656 MiB cap), minimum host MemAvailable was 79.98 GiB (2 GiB floor), maximum sampled batch storage was 0.25 MiB (800 MB cap), maximum process wall time was 78.92 s (600 s cap), and median process wall time was 25.25 s. No owned residual simulator process was reported by the per-process cleanup checks. The final root occupies about 164 MiB, excluding targets of the linked original attempts. `formal-analysis-sha-ledger.json` and `formal-analysis-seal.json` protect the aggregate results and provenance; all referenced hashes were reverified after writing.

This is an identifiability result for the selected 2-second motions, not Motion Predictor accuracy or a claim of generalization to unseen terrain. Terrain type, level and seed are coupled in the frozen design; each terrain has only three seeds. DEMO remains Front-folded fixed, collection posture assignment remains StateGroup-level Front/Rear randomization, and MI-A evaluation used manifest-specified poses. MI-B can be planned on this evidence, but its training and Phase C collection remain outside this run.
