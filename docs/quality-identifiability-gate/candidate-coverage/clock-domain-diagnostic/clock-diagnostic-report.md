# coverage-003 clock-domain duplicate diagnostic

The approved nominal Path was replayed in three fresh simulator processes for 120 control steps each. No candidate run or unfinished Stage B run was started.

## Frozen inputs and instrumentation

- Starting branch `exp/eatnav-candidate-coverage` at `6f186c04a3f902b0008eeb10e3da6d84e4abaf53`; the worktree was clean before implementation.
- Path SHA-256: `c15ccddd1be40f576289a0eb1a9a302c1088157503b87e06ad2a9291a6eeed43`.
- Production follower uses `create_wall_timer` at 20 Hz (`simple_trajectory_follower.cpp:154-155`); its motion history uses `get_clock()->now()` from the control callback (`:373`) and integrates `abs(vx) * dt` (`:716-748`). With `use_sim_time=false` and no `/clock` publisher, that elapsed time follows wall time.
- A diagnostic-only follower copy added logging for Path, TF, callback sequence, clock, motion-window state and command. No control branch, parameter, timer, publisher or subscriber was changed. Seven frozen production bridge command fixtures matched exactly.
- The bridge and simulator runner copies added timestamp and sequence fields. Source and patch SHA values are in the audit JSONs; the production source and binary SHA values still match D0.

## Three nominal repeats

| Repeat | 120-step wall time | Response wall time for command step 39 | Motion window for command step 39 | Max window | Ready callbacks | Bridge seq |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 132.2 s | 33.44 s | 19.950138487 s | 19.999999987 s | 0 | 689 |
| 1 | 131.8 s | 33.11 s | 19.950015198 s | 19.999999965 s | 0 | 682 |
| 2 | 134.3 s | 33.30 s | 19.999933074 s | 19.999999895 s | 0 | 686 |

- The bridge response during `env.step(38)` supplies the command recorded in trace step 39; sampled response values match the next trace row after float32 rounding. Nominal simulation time before command step 39 is 1.95 s. The follower had accumulated about 5.91–5.93 m of commanded distance around that response, matching roughly 0.3 m/s over about 20 s of wall-clock motion history.
- All three new repeats had identical command, Actor input/latent/action, Critic, root, contact, DEM, and reward over all 120 steps; duplicate XY RMS was 0 m against the frozen 0.0105 m threshold.
- All new first-40-step traces matched the original 40-step pilot exactly. Repeat 1 differed from its original failed 120-step trace first at step 39, as expected because the new run did not reverse.
- No new `Linear stuck detected` event occurred. The motion window never became ready; its measured maxima were a few tens to hundreds of nanoseconds below 20 s. The original event cannot be dismissed as resolved: the logging build changes scheduling, and the original internal callback times were not recorded.

## First divergence and causal assessment

- In the original long-horizon triplet, run `long-006` first diverged at follower command step 39. Actor input/action and reward changed at the same step; root and contact first changed at step 40. Anchor, Path and prior control/physics signals matched the other nominal repeats.
- The original follower warning reported 5.929 m commanded, 0.165 m actual, ratio 0.028. Integrating the frozen simulator commands for the first 39 × 0.05 s before that command gives 0.576 m; 0.165 m divided by this is 0.287, above the 0.25 stuck threshold. This is an illustrative time-domain comparison, since the follower's exact TF motion samples from that old event were not logged.
- H1 clock-domain mismatch is supported as the causal mechanism. The wall-driven 20 s history and distance accumulation advance much faster than simulation physics. The controller prunes samples older than 20 s and calls the window ready at >=20 s, making the trigger sensitive to an exact callback boundary.
- H2 callback scheduling is a possible trigger cofactor. New repeats differed in callback count and wall timing but did not differ in command or physics. The exact old callback boundary is unobserved. H3 state contamination and H4 physics-first divergence are not supported by the available traces.

## Stop and proposed validation

- The production controller, thresholds, timer, `use_sim_time`, and `/clock` setup were not changed. The bridge and follower diagnostic processes were stopped after the third run.
- Before any fix or candidate comparison, obtain approval for a separate clock-domain protocol: synchronize simulator `/clock`, bridge/follower node clocks and TF/Path stamps; verify command freshness and the original production fixture, then repeat the frozen duplicate and 40-step parity checks. The proposal does not authorize implementation or further rollout.

- Raw alignment and callback trace SHA-256: `ba5d98e80b1101ecf5518b51c867be7a69ce7dd3190ac45e837b3a4acd916c66` and `972206d6a68834dfa9e880b2c359ed91cf3a13925e37e949f28cd012d83b903f`; stored under `/tmp/eatnav-candidate-coverage/clock-domain-diagnostic` outside Git.
- Three simulator repeats total 398.3 s; archive size before report 52.2 MB.
