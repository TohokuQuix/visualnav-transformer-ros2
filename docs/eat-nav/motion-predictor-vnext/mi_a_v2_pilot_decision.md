# MI-A v2 one-StateGroup pilot decision

**Pilot protocol: Pass. Formal MI-A: pending multi-seed evaluation.** The nine frozen fresh processes completed in order: two candidate-free pre-Path snapshots, six Forward/Reverse Direct and ±30° candidates, and one Forward Direct repeat. Each candidate returned four finite 40-step same-episode labels. This is one StateGroup (`pyramid_stairs`, level 2, terrain seed `2026101305`) with four distinct env tiles; it is **not** six independent terrain seeds. No 72-run schedule, Motion Predictor training, goal-selection analysis or Phase C collection was started.

## Paired-state and provenance gates

The second pre-Path process and all seven Path processes matched the first sealed candidate-free snapshot. Maximum absolute difference was **0** for root state, joint state, Actor pre-noise/post-noise/post-delay, four DEM height/mask frames, command, latency and action-delay buffers, contact force/history and Critic observation. Terrain/material fingerprints, episode length, DEM stamps and Python/NumPy/Torch CPU/CUDA RNG SHA also matched. The snapshot intentionally had nonzero settling pose and asymmetric DEM validity masks; Gate M self-reflection failed in all nine runs and remained an independent diagnostic. These runs must not be called mirror-controlled pairs.

All seven Path processes constructed four fresh C++ follower instances with initial state `(0,0,0,1)`. Each env received one Path. First-step `travel_direction` matched Forward/Reverse assignment in all 28 traces. Converting each body-frame Path with the **measured** post-settling pose reproduced the activated tile-local Path to at most `4.53e-8` m. No post-settling pose was overwritten.

## Two-second motion, initial-body frame

The table shows the mean of four env observations. Arcs are compared with the same-direction Direct in each env. Reverse X remains negative for backward travel.

| Candidate | Mean body ΔX (m) | Mean body ΔY (m) | Mean Δyaw (rad) | Mean Arc−Direct ΔX (m) | ΔY (m) | Δyaw (rad) |
|---|---:|---:|---:|---:|---:|---:|
| Forward Direct | +0.5685 | −0.0301 | −0.0757 | — | — | — |
| Forward +30° | +0.5002 | +0.0012 | +0.0333 | −0.0684 | +0.0313 | +0.1090 |
| Forward −30° | +0.5111 | −0.0685 | −0.2145 | −0.0574 | −0.0384 | −0.1387 |
| Reverse Direct | −0.3850 | +0.0141 | −0.0304 | — | — | — |
| Reverse +30° | −0.4151 | +0.0165 | −0.0256 | −0.0301 | +0.0024 | +0.0048 |
| Reverse −30° | −0.2783 | +0.0209 | −0.0812 | +0.1067 | +0.0068 | −0.0508 |

Forward +30° and −30° produced clear, oppositely signed candidate-relative yaw in this StateGroup. Reverse +30° produced little yaw change relative to Reverse Direct; Reverse −30° produced a −0.0508 rad mean relative yaw but substantially less backward X travel. Four-env per-candidate distributions are in `mi_a_v2_candidate_differences.json` in the evidence root. The frozen v1 B-R1 Fail is unchanged; these values do not revise its absolute-yaw gate.

The Forward Direct fresh-process repeat matched its earlier candidate run **exactly** in all four envs: per-env body XY and yaw differences were zero. This demonstrates deterministic repeatability for this seed, environment configuration and candidate; it does not bound variability across independent terrain seeds. The observed candidate effects exceed this local repeat difference, but one seed cannot support a confidence interval, an MI-A Pass, or a claim that all 14 bank candidates are useful. Direction-only vs Path-specific predictor performance and goal-selection utility remain untested.

## Runtime and preservation

Nine host decisions and nine inner/outer seals passed independent read-only verification. Native exit was 0 for all runs. Peak device-total GPU memory was **5,737 MiB** (cap 6,656 MiB); minimum host available RAM was **84,297,488 KiB** (floor 2,097,152 KiB). Longest process wall time was 65.28 s (cap 600 s); summed process wall time was 256.42 s. Total persistent pilot root size, including analysis, was about 9.38 MB. No owned simulator process or pilot tmux session remained. The analysis seal SHA-256 is `b01cbd1cb55f6453da9192d4b182f5d0782038c2a23e9c6299859e1c958d58fd`.

Evidence: `/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-a-v2-pilot-20261013/`. This root contains frozen source/manifest/protocol SHA, per-process raw traces or candidate-free snapshots, resource series, inner/outer seals, `prepath_identity_audit.json`, `fresh_process_repeatability.json`, `mi_a_v2_pilot_results.json`, `mi_a_v2_candidate_differences.json`, `mirror_diagnostics.json` and the analysis SHA ledger/seal. MI-A v1 evidence was not changed; its first outer seal SHA remains `444ffc985991a663f347cd997b026ccb10bed3e05a65564cb8d150f6525ee468`.

## Decision

The v2 **pilot** has enough protocol evidence to propose a separate, frozen multi-seed main schedule. That schedule should use natural post-settling states and Gate P within each seed, retain 2 s as primary, preserve the v2 Gate M diagnostic, and apply the predeclared practical effect thresholds with seed-clustered uncertainty. Its assignment and independent seed count require review before launch. The present instruction explicitly stops after this pilot, so the 72-run schedule was not started. MI-B, MI-C and Phase C remain blocked pending formal MI-A evidence.

Runtime command:

```bash
cd /home/kojima/lib/visualnav-transformer-ros2-motion-predictor-vnext
PYTHONPATH=src:. /home/kojima/anaconda3/envs/tinyvla/bin/python scripts/eat_nav_mi_a_v2_pilot.py --root NEW_PERSISTENT_ROOT
```

The completed root above cannot be used as `NEW_PERSISTENT_ROOT` and none of its nine sealed attempts should be rerun.
