# EAT-Nav G40 boundary sensitivity and Evaluator contract freeze

**Decision: `split_reward_terminal_value_teacher_required`.** This is an offline review of the same 7 train StateGroups and 63 sealed, protocol-valid live traces. No simulator rollout, new candidate, training, collection, formal validation, or robot test was run.

## Source-confirmed facts and B0 freeze

The dedicated branch `exp/eatnav-g40-boundary-freeze` starts from the clean prior G40 commit `87d1ef3e1e617bf452949a7f48658718af51723e`. The previous G40 ledger, 63 trace SHA values, checkpoint, saved Actor/Critic and environment configs, observation functions, exact-stamp E1 source, candidate identities, corrected frame contract, reference config, and existing input/output/preference contracts were hashed and verified before Critic inference. [Source provenance](source-provenance.json) and [analysis config](boundary-analysis-config.yaml) record the source and thresholds. Config fixed CPU float32 replay tolerance at `1e-5`, `gamma=0.99`, 40 transitions, and canonical command `(0.3 m/s, 0, 0 rad/s)` before any conditional result was inspected. The 0.3 m/s command is within the saved RL range (`x: [-1,1]`, `y: [0,0]`, `yaw: [-0.5,0.5]`) and equals the production follower's declared default linear speed.

Isaac Lab's ObservationManager concatenates Critic terms in saved config order. The first three terms are 3-D base linear velocity, 3-D base angular velocity, and 3-D projected gravity; `velocity_commands` is the next 3-D term. Its **exact indices are `[9:12]`**, function `generated_commands -> command_manager.get_command(base_velocity)`, no scale, clip ±100 after term construction, and no configured Critic corruption. The full schema totals 267 values; the last 187 come from the 17×11 height scan. There is no Critic command-history term. Existing last action, flipper mode, contact, latency and height scan remain the *observed terminal state* during conditional command queries. These are conditional Critic diagnostics, not physically realized continuations. [Field schema](critic-observation-schema.json) and [command contract](terminal-command-contract.json) include dimensions, frames, units, clip and timing.

## B1: offline Critic replay

The fixed checkpoint was loaded strictly into the same `rsl_rl.ActorCritic` class with saved 1024/512/256 ELU Critic architecture and disabled observation normalizer. CPU float32 `evaluate` reproduced **all 189 values** (`V0`, `V39`, `V40` for each of 63 traces): maximum absolute error **7.62939453125e−6**, mean absolute error **1.5541359230324073e−6**, both within the pre-fixed `1e−5` gate. The observed command slice equaled each recorded command exactly at all three steps. The JSON report includes every value. [Replay equivalence](critic-offline-replay-equivalence.json).

At step 39, the command in `trace[39]` conditioned the Actor/Critic and transition 39. `trace[40]` is after transition 39 and after the command manager's next update. `trace[40]` was literally zero for **57/63** traces; six retained a nonzero command. Therefore the name `zero_recorded` denotes the *recorded Path-end boundary condition* and is not a uniform zero-command intervention across all 63 traces. Every conditional terminal vector kept all 264 noncommand float32 elements byte-identical to its saved `trace[40]` observation; only `[9:12]` changed. The direct-reference command is repeat 0 at step 39 within each StateGroup, chosen without using candidate outcomes.

## B2–B4: boundary sensitivity

The 63 per-candidate records contain observed `Rγ40`, saved `V40`, conditional hold/canonical/direct-reference values, corresponding returns and advantages, task/body metrics, and masks. The recorded-boundary return recomputes the previous G40 exactly. [Per-candidate values](terminal-value-per-candidate.jsonl.gz), [group comparisons](bootstrap-teacher-comparison.json), and [gate result](bootstrap-rank-stability.json) are the authoritative numeric artifacts.

| Comparison | 7-group macro Kendall | top-1 agreement | mean top-3 overlap | macro direct-sign agreement |
|---|---:|---:|---:|---:|
| recorded vs hold | 0.918 | 7/7 | 3.00/3 | 0.976 |
| recorded vs canonical | 0.714 | 5/7 | 2.57/3 | 0.881 |
| **hold vs canonical** | **0.769** | **5/7** | **2.57/3** | **0.905** |
| canonical vs reward-only | 0.646 | 5/7 | 2.57/3 | 0.905 |

The pre-fixed hold/canonical gate required macro Kendall ≥0.8, top-1 agreement ≥6/7, top-3 mean ≥2, direct-sign agreement ≥0.9, and no complete reversal within one terrain. Kendall and top-1 **failed**; the latter three passed. Boxes and pyramid_stairs_inv changed top candidate under hold vs canonical. There was no terrain with all 21 candidate pairs reversed. Pairwise signs are descriptive within each group and were not treated as independent samples.

Across the 63 runs, observed discounted reward averages **14.14** and the canonical discounted bootstrap contribution averages **25.56** (descriptive run means, with direct repeats). The 40-step return is therefore materially shaped by the terminal Critic condition. `Rγ40` is the observed 2 s reward; terminal value is a separate conditional estimate. A single `G40^canonical` embodiment-quality target is **not frozen**. The zero/recorded condition is an implementation diagnostic, not a receding-horizon deployment default. The 6 s Outcome remains untouched.

## Teacher and output contract

[Teacher contract](teacher-contract.yaml) freezes `discounted_reward_2s` as the observed reward head and keeps `terminal_value_zero`, `terminal_value_hold`, `terminal_value_canonical`, and optional same-group direct-reference value separate. For the six nonzero recorded boundaries, a literal-zero label is **NA** in this dataset; it must not be silently filled with recorded nonzero value. Combined G40 variants remain diagnostics. How to use a terminal-value head for selection, including any weight or precedence, requires future independent validation. Task heads remain separate and continuous: goal-distance reduction, progress ratio, endpoint error, endpoint encounter probability, and cross-track P95. Drawdown, tilt and uncertainty remain separate diagnostics/heads. No model or loss was changed.

The [Safety head contract](safety-head-disabled-contract.json) explicitly sets `enabled_for_training=false`, `loss_weight=0.0`, and `inference_use=false` because failure-positive support is zero. Infrastructure failure, invalid measurement, manual stop and censored timeout cannot become unsafe positives. This is stronger and clearer than the earlier `train_mask:false` shorthand.

## Portable input and task constraints

The [portable input contract](portable-evaluator-input-contract.json) includes 4-frame 20×20 DEM height/mask history, relative goal, 8×2 base-frame candidate Path, deterministic Path geometry, a 64-sample DEM/Path profile and coverage mask, projected gravity, base linear/angular velocity, 52-joint position and velocity, previous 6-D action, and masked pre-Path controller state. Each field specifies dimension, unit, frame, anchor timing, normalization, clip, simulator and physical availability, parity status, missing mask and source API. The DEM-only model is a separate baseline. Raw effort is excluded: all prior runs reach the simulated 3000 per-joint actuator limit and physical parity is unverified. Post-Path Actor latent, future command/action, realized trajectory, reward, Critic, contact outcome, termination, candidate ID/family/amplitude and absolute environment origin are excluded from primary inputs.

The current ROS2 bridge payload contains IMU orientation/angular velocity and selected joint values, but **does not contain base linear velocity**, a verified full 52-joint vector, or previous ActionManager action. Physical GridMap/IMU/TF parity is also pending. Missing primary body fields must block inference rather than be filled with zero. A 256-D fixed-checkpoint Actor latent is allowed only as a separate anchor-only pre-Path ablation; it is not a portable primary input. [Optional latent contract](optional-latent-input-contract.json).

[Task contract](task-head-and-hard-constraint-contract.json) keeps endpoint/progress/tracking as continuous predictions. Neither progress ratio 0.50, goal-distance reduction 0.20 m, endpoint tolerance 0.20 m, nor cross-track threshold is an immediate hard rejection: curved Paths are longer, the follower does not time-sync waypoints, and a safe detour may need more than 2 s. The 0.20 m radius remains the production follower's diagnostic goal tolerance. Tilt 0.52 rad is not a hard gate until backed by a real safety limit or existing termination setting. Nonfinite state, verified physical termination/stack, clear reverse motion, and nonpositive goal reduction under forward command are only *prospective* hard-failure candidates; future realized outcomes cannot be used as online gates without a validated predictor.

## Next collection and decision

The [next-collection schema](next-collection-logging-schema.json) specifies scalar and post-`dt` reward components with weights/config SHA; Path activity and horizon/replanning flags; pre/post command-update stamps, source and last nonzero command; terminal pre-reset observation and Critic inputs; zero/hold/canonical diagnostic values; full synchronized body/joint/action/controller state; and explicit validity/termination/censor masks. A future parity gate must compare logging enabled/disabled duplicate runs to prove no change to command, Actor/Critic observation, action, physics, reward, timing or termination. This logging was **not** implemented now.

The multi-head teacher/input *design contract* is frozen for review, but real-input integration and parity checks remain. The next approval should cover **input source/parity preparation and prospective teacher/task-threshold review** before hard-state selection design. This review does not authorize hard-state rollout, additional collection, or Evaluator training. [Freeze decision](freeze-decision.json).

## Figures

1. [Recorded/hold/canonical terminal values](fig-01-terminal-value-conditions.svg)
2. [Within-StateGroup teacher ranks](fig-02-g40-ranks-by-group.svg)
3. [Kendall/Spearman by boundary condition](fig-03-rank-correlations.svg)
4. [Top-1 and top-3 changes](fig-04-top1-top3-change.svg)
5. [Direct-vs-candidate sign reversals](fig-05-direct-sign-reversals.svg)
6. [Observed reward versus terminal contribution](fig-06-reward-terminal-contribution.svg)
7. [Portable Evaluator input](fig-07-portable-input.svg)
8. [Continuous task heads and prospective hard constraints](fig-08-task-vs-hard.svg)
9. [Future logging timeline](fig-09-next-logging-timeline.svg)

## Evidence classes

- **Source-confirmed:** saved checkpoint/config SHA, Critic schema and command index, saved follower default speed, E1 trace order, exact reward timing, and prior Path/coordinate identities.
- **Offline counterfactual diagnostic:** 189-value Critic replay and 252 conditional terminal values from unchanged observed terminal states. These do not imply physically feasible command continuation.
- **Teacher contract:** split observed reward from conditional terminal values; disable Safety head; keep task heads continuous.
- **Input contract:** portable anchor body state with explicit physical parity blockers and missing masks.
- **Prospective proposal:** future event thresholds, abstention/selection calibration and logging parity gate.
- **Future hypothesis:** multi-head selection on harder states will improve outcome; this requires new independent support and validation.
