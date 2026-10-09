# P1 curved-Path forward-loss causal diagnostic

**Decision:** The production follower **does** reduce forward command on curved Paths. Its source uses `0.3 × max(0, 1 − |heading_error|/π)` below a 0.6 rad turn-in-place threshold and commands zero at or above that threshold. The measured moving commands fit this formula to floating-point precision. The controlled intervention gave **partial, condition-dependent recovery**, not a general explanation of the P1 progress deficit. Do not retune the follower or expand P1 collection based on this six-block experiment. The smallest next test is a paired-noise, fixed-state command-to-motion check focused on the curved branch's turn-in-place transition and the `second-graded` downstream residual.

## Protocol and instrumentation

- New experiment `v6` contains 2 post-hoc selected StateGroups × 3 predefined independent block seeds × 5 conditions = **30 trajectories**, each 120 steps at 20 Hz. It is separate from the frozen 56-trial P1 pilot. The independent unit is StateGroup × block; five slots within a block are not five independent samples.
- Conditions were direct normal (A), ±20° curved normal (B), and the same curved Paths with only the positive final forward command replaced by a **predeclared constant 0.3 m/s** (C). The 0.3 m/s production hard maximum was retained. Follower stop, goal, turn-in-place zero, reverse, lateral, and angular commands were retained. Thus C does **not** remove turn-in-place stops.
- The production follower binary, RL checkpoint, Actor inference mode, observation-noise configuration, terrain mesh per StateGroup, anchor, Path definitions, 2 s Outcome, and resource caps were preserved. The only follower parameter difference was diagnostic log period 0.5→0.1 s; the binary and control calculations were unchanged. An identical monitor Path/pose ran in a sixth production follower and its command matched the direct condition at each step to 1e-6. Five simulator envs were used to respect the 30-trajectory bound; the canonical terrain mesh matches P1, while the composed 5-env collision-stage SHA necessarily differs from P1's 7-env scene. Collision/mesh fingerprints were constant across the three new blocks within each StateGroup.
- Every trace contains pre-noise and Actor-input observations, Actor output, original follower command, applied command, root state/velocity, flipper joint position/velocity, applied torque, reward, and the existing 60-body contact sensor net forces. Body velocities were rotated into the body frame for command comparison. Sampled production follower logs contain pose, carrot, lookahead, heading error, remaining Path, command, and recovery flags. No extra contact sensor or physics view was added.
- All 30 traces reached 120 steps without termination. Initialization, Path SHA, command-component isolation, finite values, resource/native exit, seal, and owned-process cleanup passed. Peak device memory was 5,185 MiB against the unchanged 7,680 MiB cap. Total six-process measured wall time was 358 s. The external Zenoh router was left unchanged. Failed pre-Path `v4` and older `v5` measurements are excluded from `v6` Outcomes.

## Controlled contrasts

Numbers below are the **mean of three matched block contrasts**, in metres unless stated. B−A is the curved normal total contrast, C−B is the forward-command intervention contrast, and C−A is the remaining contrast to direct. These are diagnostic contrasts under unpaired observation noise, not estimates of natural mediation or population effects.

| StateGroup / curved side | B−A: 2 s progress | C−B: 2 s progress | C−A: 2 s progress | C−B applied forward command (m/s) |
| --- | ---: | ---: | ---: | ---: |
| first-differential / +20° | −0.173 | +0.002 | −0.171 | −0.009 |
| first-differential / −20° | −0.044 | +0.066 | +0.022 | +0.004 |
| second-graded / +20° | +0.022 | −0.023 | −0.001 | +0.030 |
| second-graded / −20° | −0.040 | −0.006 | −0.046 | +0.026 |

The frozen practical recovery threshold was 0.025 m, half the P1 0.05 m goal-progress margin. Only `first-differential / −20°` exceeded it in **all three blocks** (C−B +0.105, +0.064, +0.030 m). However, one of those blocks improved despite a *lower* mean applied forward command in C than B; this exposes remaining observation/physics variability and prevents a clean quantitative mediation claim.

For `first-differential / +20°`, B−A is consistently about −0.173 m. The follower commands zero forward speed during 8–9 of the first 40 steps in B; C retains zero during 13–16 steps. Consequently the intervention did **not** match direct forward commands and did not recover distance. This is evidence for a follower turn-in-place mechanism, but the downstream behavior under a fully matched positive-speed command was **not tested** because preserving the stop rule was part of the fixed intervention.

In `second-graded`, C's applied forward command approximately matches A, yet progress does not consistently improve over B. For −20°, C−A is −0.008, −0.067, and −0.063 m across blocks; two exceed the frozen 0.05 m residual threshold. Mean body-frame forward-speed tracking error is lower by about 0.018 m/s versus A. Actor actions change, and the measured contact-force sum rises in all three C−A −20° blocks, but the sensor sums forces across 60 bodies and does not identify ground traction or slip. Flipper torque differences vary. These observations support a **downstream Actor/physical response** under turning demand; they cannot partition neural-policy response from contact mechanics.

The 2 s discounted reward remains a separate Outcome. Its C−B contrast changes sign across blocks for most conditions and does not provide a consistent locomotion-quality explanation. The post-hoc 6 s goal-progress contrasts also vary by block and side; for example, mean C−B is +0.045 m for first-differential +20° and +0.021 m for second-graded −20°, both with mixed individual signs. These diagnostics do not replace the approved 2 s endpoint.

## Interpretation and next decision

1. **H1 is directly established at the command level:** the follower's curvature/heading response reduces positive forward speed and can force zero during turn-in-place. It contributes to the tested first-differential behavior, with only partial intervention recovery.
2. **H2/H3 remain unresolved:** when forward commands were matched in second-graded, actual forward motion sometimes remained lower and measured contact/actuator/Actor signals changed. Unpaired observation-noise streams and contact/Actor coupling prevent unique attribution. A contact-force sum is not proof of slip.
3. **Next smallest experiment:** hold the same two anchors and ±20° Paths, explicitly pair or verify the *actual* post-delay Actor observation-noise streams across conditions, and predeclare a focused intervention that separately tests the follower's soft speed scale and turn-in-place zero while retaining the hard 0.3 m/s cap. Record per-step follower target/status and contact-body identities. Freeze the intervention safety rule before running. Do not expand the P1 dataset or retune production control yet.

## Artifacts

- New root: `/tmp/eatnav-p1-causal-20261009-v6/`; design, source/config SHA, assignments, seeds, schedules, all 30 sealed raw traces, and six follower log sets are preserved there.
- Read-only analysis: `/tmp/eatnav-p1-causal-20261009-v6/analysis/` contains `derived-steps.jsonl.gz`, `per-trial.json`, `matched-contrasts.json`, `follower-internal-samples.json`, `run-ledger.json`, `provenance.json`, and four PNG figures.
- Historical P1 v2 source and Outcome roots were not modified or pooled into this diagnostic. No full collection or Evaluator/Actor training was started.
