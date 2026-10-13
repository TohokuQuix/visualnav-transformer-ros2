# MI-A v3.1 seed preflight and multi-seed smoke result

Decision: **Stage 4a preflight Pass; Stage 4b smoke Pass; formal MI-A not started**.

The MI-A v3 failed manifest and evidence remain unchanged. Its manifest SHA is `bd496473ba3b6a18ff80dcc5464f342075eb5cad4758345d6dbbc44d505542ed`, and the failed attempt's outer seal SHA remains `14e06cbb79285c8e3b20627b225d9836efe8356f2b7455c87bf6ad3fff8da70a`. MI-A v3.1 is on branch `exp/eatnav-mi-a-v31` at commit `0c9f71e`; its frozen manifest SHA is `5d3ccdc70b45c534ec27f60fe9b2a959c69f41ec9093cc07b9f879fbdbf4ee7a`.

The focused MI-A/MI-FI/DEMO/bidirectional suite passed 58/58 tests. A repository-wide `pytest -q` was also attempted in the `tinyvla` interpreter: collection stopped with 40 unrelated import errors, including missing `diffusion_policy` and unavailable historical diagnostic script modules. These collection errors do not alter the focused test or sealed runtime results; repository-wide test success is **not** claimed.

The twelve distinct seeds use `2026101401 + 100*terrain_index + (terrain_level-1)` for four ordered terrain types and levels 1–3. All seeds are within `0..2**32-1`. The seed validator now checks both bounds; tests cover 0, `2**32-1`, −1, `2**32`, and the former invalid seed. The 192-row schedule, 576 candidate-trial design, terrain types/levels, both flipper postures, six Path candidates, Gate P `1e-6`, primary XY threshold `0.03 m`, Actor, follower, physics and resource caps are unchanged. DEMO source remains Front-folded fixed; the regenerated opt-in collection allocation is 6 Front / 6 Rear at StateGroup granularity.

The first `pyramid_stairs` seed's Front-folded candidate-free process built Isaac Lab, settled for 20 steps and produced four finite pre-Path snapshots with DEM shape `[4,4,2,20,20]`, complete Actor/Critic/contact/latency arrays and valid inner/outer seals. Its pre-Path NPZ SHA is `a19d37838729dc0210f3a55bac4485557c069c0928f14e92cfb1613d58cbf04c`. This sealed process was reused exactly as smoke row 0.

The smoke covered `pyramid_stairs` seed `2026101401` and `k_rail_step_field` seed `2026101701`, both at level 1. Each seed used Front and Rear, two candidate-free pre-Path processes per posture, and Forward/Reverse Direct and ±30° candidates. **32/32 processes and 96/96 candidate trials were protocol-valid; Gate P passed 32/32.** Each condition had 16 processes. All candidate traces had 40 control rows per env, no candidate termination, native exit 0, and valid per-attempt inner/outer SHA/seals. The preflight outer seal matched smoke row 0. No owned Isaac simulator or smoke tmux process remained afterward.

Resource observations: 851.3 s summed process wall time (median 25.8 s, max 64.5 s), GPU device peak 5,473 MiB versus 6,656 MiB cap, minimum host MemAvailable 84,463,504 KiB versus 2,097,152 KiB floor, maximum per-attempt storage 1,338,348 bytes versus 800,000,000-byte cap. The evidence root used about 36 MiB after summaries. The top-level summary seal SHA is `a487622cbd4328a8aba0b0b29bf5090a0430add315c50ebffcf2b0976576837c`.

The two-seed descriptive motion file records candidate-minus-Direct body-frame XY and yaw for every terrain/posture/candidate. For example, on `pyramid_stairs`, Front-folded Forward −30° changed XY by approximately `(−0.0965, −0.0083) m` and yaw by `−0.0503 rad` relative to Forward Direct; Rear-folded Reverse +30° changed XY by `(0.0365, −0.0204) m` and yaw by `0.0831 rad` relative to Reverse Direct. On `k_rail_step_field`, the corresponding Rear-folded Reverse +30° yaw difference was `0.1964 rad`. These are **descriptive smoke observations from only two independent terrain seeds**, not a formal identifiability or generalization result. MI-A Pass/Fail/Inconclusive, MI-B readiness and ±60°/±90° expansion remain undecided.

Evidence: `/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-a-v31-20261014-0141/`. The root contains `preflight-result.json`, `mi_a_v31_smoke_test_results.json`, `mi_a_v31_smoke_motion_diagnostic.json`, `mi_a_v31_smoke_resource_summary.json`, a top-level summary SHA ledger/seal, and 32 individually sealed process subroots. The smoke was stopped after analysis. **No formal 192-process run, Predictor training or full collection was launched.**

Reproduction of the completed gates uses the frozen worktree and a fresh root:

```bash
cd /tmp/eatnav-mi-a-v31
export PYTHONPATH=src:scripts:.
/home/kojima/anaconda3/envs/tinyvla/bin/python scripts/eat_nav_mi_a_v31_run.py --root NEW_PERSISTENT_ROOT --phase preflight
/home/kojima/anaconda3/envs/tinyvla/bin/python scripts/eat_nav_mi_a_v31_run.py --root NEW_PERSISTENT_ROOT --phase smoke
/home/kojima/anaconda3/envs/tinyvla/bin/python scripts/eat_nav_mi_a_v31_analyze.py --root NEW_PERSISTENT_ROOT --phase smoke
```

Do not reuse the completed evidence root for reproduction.
