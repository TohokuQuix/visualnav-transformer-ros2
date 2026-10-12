# ROS-free G3 crossover v1: period 0 attempt 1

## Decision

The single authorized period-0 process is protocol-valid. Periods 1 and 2 were not started. This process establishes only the first period of one StateGroup and one seed; candidate-minus-Direct paired effects require the remaining periods.

## Prelaunch repair and provenance

The successful Direct core v3 host and crossover host both resolve `eat_nav_p1_owner_wrapper.py` next to the running host script using `__file__`. The failed crossover attempt staged `host-runner.py` without that sibling. The staging script now copies the unmodified owner wrapper beside the staged host, and the authorization and host checks verify both bytes against repository source and manifest SHA. The original failed attempt-0 remains unchanged.

The period-0 frozen roles are `Direct, Direct, Left, Right`; period 1 is `Left, Right, Direct, Direct`; period 2 is `Right, Left, Right, Left`. Each env receives every role once, giving four assignments per role across the three periods. The assignment SHA is `8c910f1204502cdf749c2d265887c3b29026bcdef4647674b8548d49a6277d29`.

The fresh preparation manifest SHA is `04ace04b702459bc28a55ad285ec5297edb1d138d5a3336719163dbf499e7300`; the authorized attempt manifest SHA is `9b4b3f21ff9266b9883e5e4b23fa560215ecddf7619c6bc269addaa0d2e2df92`. The staged host SHA is `0d4f754b1beb632086a94fc7233b01626f1881a8f8b918ae7683272d292603f3`, wrapper SHA `021b3117eef205b88cafce72240a2cb9b7f9c44255abb543de0cf74c5ee134ce`, and simulator SHA `0132a4e4bcfc6fa651f27fba13420ec5286a43aae34e7f4be553fc08347c80f5`.

Offline preflight passed from the actual staged host path: sibling wrapper resolution and SHA, its prelaunch imports and dependent paths, runtime authorization, persistent root, Docker bind, container source bytes, and sealed-root rejection. The preflight report is outside the sealed batch root at `/home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-attempt-1/preflight.json` (SHA `47878c864e7db75ce0772850584835bb0d6a83e92eac1545492b0846b9013fff`). Related static/synthetic tests: 54 passed.

## Runtime evidence

The four-env pre-Path physical, shared DEM, Actor, and live material gates passed. Actor direct-input maximum absolute difference was 0 against its `1e-4` limit. Native DEM history remains separately diagnostic and did not pass equality. The shared four-frame input had byte-identical copies. The four independent ROS-free follower cores were called once per logical stamp; the validated trace shows `pose → core → command injection → Actor → physics → next pose`, with 145 control rows per env, 20 Hz stamps from 2.00 to 9.20 s. No ROS follower, ROS bridge, TF service, or Zenoh router was part of this runtime control path. All four Paths were activated once, reached Goal, and completed the 40-step hold. No reset or termination occurred.

| Env | Path | Arrival (s) | Local progress 2 s (m) | Common-goal progress 2 s (m) | Tracking mean (m) | Recovery | Contact mean, fixed 2 s (N) | Torque mean abs, fixed 2 s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Direct | 2.20 | 0.494 | 0.494 | 0.022 | 0 | 8.21 | 173.38 |
| 1 | Direct | 2.05 | 0.537 | 0.537 | 0.021 | 0 | 9.66 | 256.30 |
| 2 | Left +20° | 5.20 | 0.320 | 0.361 | 0.078 | 0 | 7.86 | 238.34 |
| 3 | Right −20° | 3.00 | 0.348 | 0.368 | 0.035 | 0 | 7.23 | 185.65 |

Path lengths were 0.65625 m for Direct and 0.66956 m for each curve. Period 0 assigns different Paths to different env IDs, so these numbers alone are not a paired Path-quality ranking. Local arrival is not common-goal success.

The host reported native exit 0, no owned residual, 77 owned GPU attribution brackets, GPU device peak 5,188 MiB (cap 7,680 MiB), container anon+shmem peak 3.25 GiB (cap 6 GiB), 53.48 s wall time (cap 600 s), and final persistent root size 13.68 MB (cap 1 GB). The 305-file host seal remains valid after read-only analysis; host/container SHA checks passed.

The frozen v1 offline analyzer failed on a schema mismatch: it required `follower_diagnostic.mode`, while the ROS-free core emits explicit Goal/recovery flags. A separate read-only v2 analyzer derives a display mode from those flags without changing raw trace bytes or the sealed batch. Its [machine-readable result](/home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-attempt-1/period-zero-analysis-v2.json) has SHA `44ab12d0574ace8b9baf9b3fa0eae2eaae7ce45ad8b5f50444733d03456f7cc8`. The runtime protocol verdict does not depend on this presentation fix.

## Next decision

Period 0 is suitable as the first member of the planned balanced crossover. Periods 1/2 remain unexecuted and require a separate execution decision. No additional seed, Evaluator training, or full collection was started.
