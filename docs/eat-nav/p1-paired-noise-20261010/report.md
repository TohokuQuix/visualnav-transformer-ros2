# P1 paired-noise follower diagnostic — 24-trial result

## Decision

**Integrity gates passed: 24/24 trials, six independent paired-noise blocks, all 120/120 control steps.** The first-differential +20° case shows a reproducible ordinary turn-stop contribution to its 2 s progress deficit. Soft-scaling removal alone has a small, inconsistent effect. In the second-graded −20° case, turn-stop intervention does not change the first 2 s forward command and the progress contrasts vary in sign across blocks. The present data do not establish a consistent downstream locomotion deficit. Keep the production follower unchanged. This small mechanism study supports moving the P1 design review toward longer-horizon Path evaluation and label design, with residual response flagged as uncertain; no further experiment was launched.

## Provenance and integrity

- Base Git branch `exp/eatnav-p1-causal`, HEAD `c28d45ca7a734c796d4cd065c51524d5d131c496` before the diagnostic implementation. The implementation and this report are new files in the isolated worktree `/tmp/eatnav-vnt-p1-statistical`. The unrelated preexisting `ros-decoupling-audit.md` remains untouched.
- Frozen design SHA `22076b2afc1b3a5d9ab23e4a3c513f8f9cd6ae5b6c2d1578bbb82e96af5d959a`; assignment SHA `033b2056ef0e56aa1cadb1ee3f3b2ced66eb1060fb2012bbbb45c756eb0d9c80`. The sealed P1 historical results, StateGroups, paths, Actor checkpoint, intervention rules and thresholds were unchanged.
- External router audit used `router.pid` (actual `rmw_zenohd` PID 1744855, start tick 23096119), parent `router-launch.pid` (PID 1744849, start tick 23096108), expected cmdlines and parent relation, live state, ZID `552d5892d82875243b8c4b3a06056398`, and reachable port 7447. X1 and final log SHA snapshots are in `router-identity-freeze.json` and each run’s `router-after.json`. The router was not restarted or reconfigured.
- Gate 0 branch review and Gate 1 Path-free baseline equivalence were completed before this continuation. Baseline maximum observation-noise difference was 1.19e-7; post-delay Actor input/action matched exactly. Production ROS follower and exact-stamp TF path were used.
- Every block has one Path publication, four SHA-matching traces of 120 consecutive steps, exact 50 ms logical timestamps, matching follower/TF/bridge stamps, 120 accepted commands per condition, no terminal or recovery activation, native exit 0, resource pass, and zero owned residual processes. All four conditions have exactly equal injected noise tensors at every corresponding term/step; the six block streams have distinct SHA-256 values. The differing post-noise observations and latency histories are expected closed-loop consequences.
- The first pilot preparation root `pilot-first-differential-b0-run-v1` stopped **before launch/Path** because its default diagnostic follower YAML did not match the frozen YAML SHA. The fresh `run-v2` used the already frozen YAML. It is not an additional trial. No Path-published run was retried.

## Run matrix

Each row represents one independent noise block; A/B/S/T are paired within that row. The table gives 2 s global-goal progress in metres. Full 24-trial telemetry, per-condition run statuses, and exact artifact paths are in `trials.json` and `ledger.json`.

| StateGroup | Block | A | B | S | T | Root |
|---|---:|---:|---:|---:|---:|---|
| first-differential | 0 | 0.446 | 0.352 | 0.342 | 0.509 | `pilot-first-differential-b0-run-v2` |
| first-differential | 1 | 0.558 | 0.360 | 0.397 | 0.538 | `pilot-first-differential-b1-v1` |
| first-differential | 2 | 0.544 | 0.371 | 0.382 | 0.520 | `pilot-first-differential-b2-v1` |
| second-graded | 0 | 0.058 | 0.121 | 0.102 | 0.079 | `pilot-second-graded-b0-v1` |
| second-graded | 1 | 0.259 | 0.100 | 0.078 | 0.096 | `pilot-second-graded-b1-v1` |
| second-graded | 2 | 0.085 | 0.083 | 0.094 | 0.117 | `pilot-second-graded-b2-v1` |

## Paired contrasts

Differences are condition on the left minus condition on the right. Values are block 0 / 1 / 2, followed by their mean. The independent sample count is **3 blocks per StateGroup**, not 120 steps or follower callbacks.

| StateGroup | Contrast | 2 s goal progress Δ (m), blocks → mean | 6 s goal progress Δ mean (m) | Applied forward command Δ mean (m/s) | Body forward velocity Δ mean (m/s) |
|---|---|---|---:|---:|---:|
| first-differential | B-A | -0.095 / -0.198 / -0.172 → -0.155 | -0.049 | -0.090 | -0.077 |
| first-differential | S-B | -0.010 / +0.038 / +0.011 → +0.013 | -0.029 | +0.003 | +0.007 |
| first-differential | T-S | +0.167 / +0.141 / +0.137 → +0.148 | +0.146 | +0.090 | +0.075 |
| first-differential | T-A | +0.063 / -0.020 / -0.024 → +0.006 | +0.069 | +0.004 | +0.005 |
| second-graded | B-A | +0.063 / -0.159 / -0.002 → -0.033 | +0.064 | -0.023 | -0.027 |
| second-graded | S-B | -0.020 / -0.022 / +0.011 → -0.010 | +0.014 | +0.027 | +0.009 |
| second-graded | T-S | -0.023 / +0.019 / +0.022 → +0.006 | -0.034 | +0.000 | +0.002 |
| second-graded | T-A | +0.020 / -0.163 / +0.032 → -0.037 | +0.044 | +0.004 | -0.016 |

The complete metric matrix is `pairs.json` and `aggregate.json`: 2 s discounted reward, 2/6 s progress, forward displacement and command integral, command-to-body tracking error, zero-forward steps, angular command, heading, lateral motion, Path error, contact-force sum, flipper torque and roll/pitch. The 2 s reward and global-goal progress remain separate labels.

## Mechanism and safety interpretation

- **First-differential (+20°):** B−A progress is negative in all three blocks (−0.095, −0.198, −0.172 m). S−B is small and changes sign; T−S is positive in all three (+0.167, +0.141, +0.137 m). T removes 12 zero-forward steps on average relative to S in the first 2 s. This supports a large total effect of removing ordinary turn-in-place stopping in this case. The intervention changes subsequent trajectory and observations, so this is not a natural mediation estimate.
- **Second-graded (−20°):** B−A varies from +0.063 to −0.160 m; S−B and T−S also change sign. B has no zero-forward steps in the first 2 s; T−S changes applied forward command by 0.000 m/s on average. This case cannot attribute its variable progress to ordinary turn stopping. T−A retains a mean 2 s deficit of −0.037 m despite similar commanded forward velocity (+0.004 m/s); body-forward velocity averages −0.016 m/s. The three block results vary substantially, and angular command, state, terrain contact, Actor action, and physical response remain entangled.
- **Safety/load:** T deliberately permits forward travel during a large heading error. First-differential T−S increases mean heading change by +0.039 rad and Path error by +0.009 m; 2 s discounted reward decreases by −0.041 despite progress recovery. Contact-force sum and flipper torque changes have mixed signs across the two cases. All runs stayed within the frozen command/state/resource caps and had no termination or recovery activation. These observations do not establish safety or desirability for a production controller.
- Total contact force is a load diagnostic only. It does not identify traction or slip. The interventions are closed-loop total effects; S−B and T−S should not be treated as an additive causal decomposition of B−A.

## Limits and next decision

There are only three independent noise blocks per StateGroup and two selected cases. This is a targeted mechanism check, not a terrain-general population estimate. Second-graded residual command-to-motion behavior remains uncertain. If that residual becomes a blocker, the smallest follow-up is a pre-registered same-command/controlled-angular-command diagnostic with additional independent blocks; it needs separate approval. No full collection, Evaluator training, physical-robot trial, or longer-horizon Path rollout was started.

## Resources and artifacts

- Six valid run roots, 24 trials; peak device-total GPU 5270 MiB; per-block wall time 61.5–61.6 s; diagnostic root storage approximately 75 MiB.
- `trials.json`: complete per-trial summaries and trace SHA; `pairs.json`: each block contrast; `aggregate.json`: three-block means/ranges; `ledger.json`: per-run seal SHA/resource; `paired_noise_stream_sha256.json`: noise identity and independence. Raw traces and host/container SHA seals are under the run roots.
