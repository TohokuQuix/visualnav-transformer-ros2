# EAT-Nav 0–2 s bootstrapped-return teacher audit

**Decision: `multihead_teacher_ready_single_score_not_ready`.** This is an offline design review of 7 StateGroups and 63 sealed, protocol-valid live traces. It does not authorize a rollout, collection, or Evaluator training. The statistical unit is a StateGroup. Candidate pairs and direct repeats are descriptive observations within a group.

## Source facts and integrity

The dedicated worktree starts from clean commit `1159cd35a1d8d4c8e0805eadd77b507b119f903b`. The existing E1 source auditor passed: 7 group identities, 63 trace hashes, completion and protocol markers, candidate Path identity, direct repeat gates, corrected map-frame geometry, checkpoint/reference-config hashes, and saved 6 s outcomes. Each recomputed full reward and Critic sequence SHA also matches its prior record; termination-sequence SHA is recorded per run. Five anchor-only runs and all diagnostic/measurement-invalid/GUI runs are excluded. No input artifact was modified. SHA details are in [source-provenance.json](source-provenance.json) and [g40-per-candidate.jsonl.gz](g40-per-candidate.jsonl.gz).

The checkpoint is `model_10000.pt`, SHA `b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d`. Its saved `params/agent.yaml` in the same training-run directory specifies **gamma 0.99**, **GAE lambda 0.95**, empirical normalization false, clipped value loss true (`clip_param=0.2`), and no RND. The saved `params/env.yaml` specifies physics `dt=0.005 s`, decimation 10, 20 s training timeout, and infinite-horizon treatment of time limits. Thus policy/control frequency is **20 Hz**, physics frequency **200 Hz**, and 40 policy transitions are **2.00 s**. At 20 Hz, `gamma^40=0.6689717586`, the geometric-sum effective horizon is **5.00 s**, half-life **3.448 s**, and weights at 1/2/6 s are **0.8179/0.6690/0.2994**. These are discounts per policy step, not per physics step.

`OnixManagerBasedRLEnv.step` performs 10 physics steps, computes termination, then `reward_manager.compute(dt=self.step_dt)`, updates command, and computes the next observation. Isaac Lab `RewardManager.compute` multiplies each active reward function by its configured weight **and `dt` once**. The scalar `reward` saved by E1 already includes that 0.05 s factor; multiplying it again would be wrong. No aggregate reward normalization or clipping was found in the saved config. Value clipping refers to PPO's training loss, not an inference-time clamp on the saved Critic value.

The available `isaac-lab-onix:latest` image was built 2026-08-06, after the 2026-07-24 training run. Its PPO source uses GAE and adds a time-limit correction involving the pre-step value, but its bytes have **not** been proven to be the exact July package. The checkpoint contains iteration 10000 and no additional `infos` metadata. This limits claims about the original PPO timeout implementation. It does not affect the nonterminal 40-transition formula calculated from saved scalar rewards and pre-step values. For future timeout samples, the terminal pre-reset observation/value must be logged and the training-version rule verified before using a bootstrapped label.

The saved Critic config includes current velocity command, base linear/angular velocity, gravity, joint state and effort, last action, latency, contact flags, and a height scan. It does **not** include the candidate Path, relative goal, or complete future Path geometry. It estimates RL return under the current state/command, not the outcome of an entire Path. Actor and Critic are evaluated from the same pre-step observation capture; policy input has the additional latency path.

## Trace index contract

Each trace row `k` is captured before `env.step`: `V_k`, Actor input/action and current command are pre-transition values. After that step, the row receives `r_k` and `post_root_state_(k+1)`. `trace[k+1].root_state` equals that post-root state. All **7497** adjacent boundaries across 63 traces match exactly, all timestamps advance by 50 ms, and no reward is missing. Corrected map-frame 2 s endpoint error, cross-track P95, and along-path progress recompute exactly against prior saved geometry.

```text
t_k: follower command -> observation_k -> V_k / Actor_k -> env.step(action_k)
     -> reward r_k, post-root_(k+1) -> next command/observation_(k+1)

G40 = sum_{k=0}^{39} 0.99^k r_k + 0.99^40 * trace[40].critic_value
A40 = G40 - trace[0].critic_value
```

The Path controls transitions 0–39. At the **t=2.00 s bootstrap boundary**, 57/63 `trace[40]` commands are zero. This is a real command/Path-horizon boundary effect, not an index error. It makes `V40` a value of the newly updated command state for those runs. No alternate bootstrap index was substituted after looking at outcomes. Details are in [trace-index-contract.json](trace-index-contract.json).

## Recalculated 0–2 s results

For every run, the per-candidate artifact records `R^gamma_40`, undiscounted `R^1_40`, `G40`, `A40`, `deltaV40`, 41 Critic values, 40 rewards, TD residuals, Critic drawdown/recovery timing, corrected map-frame task metrics, tilt/contact/action/joint/effort diagnostics, and old 6 s labels. All 63 40-step labels are finite and nonterminal. Direct repeats have **zero observed variation** for the audited metrics; no epsilon was added and effect/noise ratios are NA when the denominator is zero.

| Terrain | `G40` top candidate | Candidates above direct in `G40` | `G40` winner's 2 s progress deficit from best (m) | Endpoint-error excess from best (m) |
|---|---|---:|---:|---:|
| boxes | left_1 | 5 | 0.041 | 0.008 |
| hf_pyramid_slope | right_1 | 2 | 0.089 | 0.014 |
| k_rail_field | direct | 0 | 0.179 | 0.022 |
| k_rail_step_field | right_2 | 3 | 0.000 | 0.018 |
| pyramid_stairs | direct | 0 | 0.142 | 0.053 |
| pyramid_stairs_inv | right_3 | 6 | 0.000 | 0.0003 |
| random_rough | direct | 0 | 0.039 | 0.005 |

`G40` versus the **old 6 s ΔV** has StateGroup macro Spearman **−0.122**, pairwise agreement **0.463** (descriptive pairs), top-1 agreement **0/7**, and mean top-3 overlap **1.0/3**. Against old 6 s H-step advantage, macro Spearman is **0.214**, top-1 **3/7**. Against 2 s progress, macro Spearman is **0.031**, top-1 **2/7**. Against 2 s endpoint error, macro Spearman is **0.158**, top-1 **0/7**; several endpoint differences are millimeters, so the top-1 count alone should not be overread. Against 2 s cross-track P95, macro Spearman is **0.255**, top-1 **2/7**. These are seven-group descriptive comparisons without an independent-pair significance test. Full per-terrain reversals and direct comparisons are in [g40-stategroup-summary.json](g40-stategroup-summary.json) and [teacher-rank-concordance-2s.json](teacher-rank-concordance-2s.json).

`A40` is `G40 - V0`. If all candidates share exactly one pre-step state and command, subtracting `V0` preserves ranking; in saved runs, slight candidate-specific pre-step values make the rankings differ (top-1 agreement **6/7**, macro Spearman **0.918**). The old 6 s H-step advantage covers 120 transitions and a different post-Path command regime. `deltaV40` omits reward and is diagnostic only; its `G40` macro Spearman is **0.597**.

The 0–2 s TD residual `r_k + gamma V_(k+1) - V_k` has mean **0.0627**, standard deviation **0.1725**, mean absolute **0.0956** across 2520 transitions. Mean absolute residual rises from **0.0831** in steps 0–29 to **0.1461** in steps 35–39; per-step mean at steps 38/39 is **0.184/0.135**. For next-command changes above numerical threshold `1e-6`, mean absolute residual is **0.0992**, versus **0.0808** without a change; jump magnitude and absolute residual have Spearman **0.032**. Thus a change-associated difference exists, but jump size alone explains little. The Path-end rise is consistent with the command boundary and/or Critic approximation error; it does not prove calibrated value estimates. Terrain-wise means and 0.5/1/2 s horizon residuals are in [bellman-consistency.json](bellman-consistency.json). The later command at `V40` is a material teacher-definition issue to carry into prospective review.

## Reward and body audit

Active saved reward weights include forward linear-velocity tracking **+5**, angular-velocity tracking **+2**, XY tracking **+1**, standstill **+1**, support margin **+0.5**, flipper contact **+0.4**, flipper angle **+0.25**, action rate **−0.01**, angular body velocity **−0.08**, flipper torque **−2e−6**, and physical termination **−500**. There is no explicit relative-goal progress or endpoint reward. Because traces store only the scalar reward, and reward-evaluation component tensors and complete previous sensor/action state were not logged, **exact component cumulative values, `G40` contribution fractions, candidate ranking contributions, and terrain dominance are NA**. Configured weights alone do not establish which term dominated an observed rank. In particular, we cannot prove command tracking was dominant, though a nonprogressing candidate can in principle score well from the reward definition and the observed task-rank conflicts. This prevents interpreting `G40` as a standalone local Path-quality label.

All 63 runs reach maximum absolute **raw per-joint applied torque 3000** in the first 2 s. Saved actuator config sets `effort_limit_sim=3000` for track joints. The observation function takes individual flipper effort plus a **signed sum** of track-joint efforts, then the observation term clips to ±100 and scales by 0.01. E1 logging separately saves the pre-step `robot.data.applied_torque` tensor without a logging clip. The exact 3000 matches a configured per-joint actuator limit; it is **not** the track sum or observation clip. The tensor and config do not by themselves prove achieved physical torque saturation. Raw effort remains a diagnostic, not a ranking teacher.

No physical termination, verified stack, failure-positive or censoring event occurs in the 63 traces. The future contract separates physical termination (no post-terminal bootstrap), time-limit truncation (requires verified pre-reset terminal value and training-version rule), censored timeout, infrastructure failure, measurement-invalid, and manual stop. Infrastructure and invalid measurement never become a low robot score. Tests cover early physical termination, timeout with/without terminal value, measurement-invalid, and the 40-step off-by-one boundary.

## Recommendation and limits

The preferred prospective design is a **multi-head Evaluator with lexicographic selection**: first safety exclusion, then task feasibility, then `G40` among feasible candidates, with drawdown/tilt/tracking and uncertainty separate. A single weighted sum would mix units and permit reward to compensate for a safety or task failure. The Safety head is masked until failure-positive labels exist. The proposed 0.20 m endpoint radius, 0.50 progress ratio, 0.20 m goal-distance reduction, and 0.52 rad tilt guard are **prospective review values**, not thresholds tuned on these seven groups; they are not frozen by this report. No tie threshold can be estimated from observed zero repeat variation. Prefer direct when it is safe and task-feasible; otherwise use a safe lower-curvature candidate, or replan/controlled hold with an independent safety supervisor. Holding position on a slope must itself be verified.

The next approval should be **teacher definition and prospective safety/task threshold freeze**, including the `V40` command-boundary semantics and exact timeout bootstrap source, before selecting harder StateGroups or collecting additional labels. This review alone does not authorize additional collection or Evaluator training. The seven terrains provide descriptive support only; they do not constitute held-out validation or a generalization claim.

## Figures

1. [StateGroup G40 distributions](fig-01-g40-by-stategroup.svg)
2. [G40 versus old 6 s ΔV](fig-02-g40-vs-six-second-deltav.svg)
3. [Rank comparison](fig-03-teacher-rank-comparison.svg)
4. [Reward contribution NA with configured weights](fig-04-reward-contribution-na.svg)
5. [Critic trajectories and drawdown](fig-05-critic-drawdown.svg)
6. [TD residual by time](fig-06-td-residual-time.svg)
7. [0.5, 1, 2 s horizon diagnostic](fig-07-horizon-bellman.svg)
8. [Task feasibility versus G40](fig-08-task-vs-embodiment.svg)
9. [Multi-head lexicographic design](fig-09-multihead-selection.svg)

## Evidence classes

- **Source-confirmed:** saved training config gamma/step/weights, reward manager `dt` multiplication, current observation schema, E1 trace capture order, actuator limit, and frozen protocol identities. July PPO package bytes remain unproven.
- **Trace-recalculated:** 63 finite `G40` values, map-frame 2 s metrics, rank concordance, zero direct-repeat variation, TD residuals, and effort tensor statistics.
- **Design recommendation:** task constraints plus `G40`, separate heads, abstention and prospective threshold review.
- **Future hypothesis:** this multi-head selection will improve hard-state outcome; Safety and uncertainty heads can be learned and calibrated once appropriate support exists.
