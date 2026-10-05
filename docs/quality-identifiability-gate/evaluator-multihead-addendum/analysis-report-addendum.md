# EAT-Nav Evaluator multi-head teacher: M11–M13 addendum

**Final decision: `multihead_teacher_contract_ready_for_input_integration`.** This is an offline design decision. It authorizes the next *design* work on portable ROS2 inputs, field parity, future logging, additional independent StateGroups, and a small-training plan. It does not authorize Evaluator training, hard-state rollout, scaled collection, online selection, Safety training, formal validation, or robot control.

## Source and change boundary

This addendum starts at the clean M0–M10 commit `74bf5029c3818a7335f6b1396cc10c725b84b2a2` in a separate branch/worktree. The [M0–M13 consistency audit](m0-m13-consistency-audit.json) verified all 22 prior ledger files, all 63 raw trace SHA values and process identities, the seven groups of nine runs, 63 saved labels, and 147 saved unique-candidate pairs. The prior report explicitly recorded that the first attachment ended during M10. The supplied M11–M13 attachment resolves that missing tail. No prior label, margin, candidate, split, checkpoint, raw trace, or report was changed; no M0–M10 numerical analysis was rerun.

The prior [M0 provenance](../evaluator-multihead-freeze/m0-provenance.json), [labels](../evaluator-multihead-freeze/multihead-labels.jsonl.gz), [pair table](../evaluator-multihead-freeze/pairwise-labels.json), [input contract](../evaluator-multihead-freeze/portable-evaluator-input-contract.json), [output contract](../evaluator-multihead-freeze/model-output-contract.yaml), [loss contract](../evaluator-multihead-freeze/loss-and-normalization-contract.yaml), and [report](../evaluator-multihead-freeze/multihead-design-review.md) remain the reproducibility baseline. Two interpretations in that earlier design are superseded **only in this versioned addendum**: `endpoint_error_2s` moves from equal-weight primary regression to auxiliary calibration/reporting; Bradley–Terry ranking becomes an auxiliary loss subordinate to continuous Huber regression. The saved endpoint labels and pair labels remain valid and unchanged.

## M11: teacher, baselines and prospective evaluation

Phase A's central teachers are observed `discounted_reward_2s`, within-group `direct_relative_reward_2s`, `goal_distance_reduction_2s`, and separate reward/goal pair preferences. Because fixed start/goal geometry makes endpoint error and goal reduction have identical pairwise order on these groups, giving endpoint error an independent equal primary weight would double-count the same task signal. Endpoint error, encounter, Path progress and cross-track remain auxiliary task labels. Terminal value and Critic residual hold/canonical are Phase B ablations only. The fixed Safety head remains `enabled_for_training=false`, `loss_weight=0`, `inference_use=false`; infrastructure failures, proxies, timeouts and manual stops are not unsafe positives.

All seven direct-repeat groups have zero observed reward/goal/endpoint difference. The unchanged `1e-6` tie margin is **only a floating-point comparison floor for deterministic simulator ordering**. It is not a physical or practically meaningful indifference region. Pairwise labels are correlated within StateGroup, are not independent samples, and do not justify weighting binary ranking more heavily than continuous regression. A minimum meaningful effect must be frozen *before* collecting new data; it is not inferred or adjusted from these 63 outcomes.

The 66 reward/goal conflicts among 147 descriptive pairs show why locomotion reward and navigation progress remain separate objectives. [Multi-objective contract](multiobjective-evaluation-contract.json) specifies group-level conflict/agreement, Pareto recall, selected reward and goal reduction, direct-relative deltas, trade-offs, abstention coverage/performance, and uncertainty versus misranking. No weighted scalar coefficient is fit on the 63 outcomes. [Conflict/Pareto figure](fig-01-conflict-pareto.svg) uses the saved candidate summaries only.

The [prospective gate contract](prospective-evaluation-gates.yaml) fixes StateGroup/source trajectory/environment instance/terrain-seed split, train-only scaler fit, group bootstrap CI, LOSO as diagnostic only, per-head regression/calibration/collapse checks, and group-macro ranking/top-k/oracle regret metrics. The [baseline and ablation contract](baseline-and-ablation-contract.json) lists all 11 required comparators, from constant/direct/geometry through DEM, Path, portable-body, optional anchor latent and Phase B auxiliary variants. Auxiliary heads are rejected if independent validation shows harm to primary task performance. Terrain generalization requires a true terrain-held-out test. With only seven train groups, numeric effectiveness thresholds and `online_selector_ready` are not asserted. The allowed interim states distinguish pipeline smoke, small overfit, held-out signal, insufficient support, auxiliary harm and selector-not-ready. [Head structure](fig-02-phase-heads.svg) and [evaluation flow](fig-03-evaluation-flow.svg) summarize the proposed sequence.

## M12: future collection and logging

The [future logging schema](future-collection-logging-schema.json) defines **56 fields** with shape, order, unit, frame, one of five step phases, normalization, clip, missing mask, simulator API, proposed robot API, and parity status. The phases are: anchor before Path publication; before command update; after command adoption and before `env.step`; after transition/reward; and terminal before reset. It separates scalar reward and pre-/post-dt component values with weights, version and config SHA; Path SHA/publication/end/replan events; command publication/adoption and exact-stamp TF; synchronized Actor/Critic inputs and body/joint/action state; and termination, censoring, infrastructure and failure-label eligibility. No historical component value was fabricated from the scalar-only 63 traces. [Timeline](fig-04-logging-timeline.svg).

The [logging parity gate](logging-parity-gate.yaml) requires matched logging-on/off duplicate runs before future collection. It compares Path identity, logical stamps, command, Actor/Critic input, action, root/joints, contact, DEM, total/component reward, termination/reset and realized XY; it records the first divergence and per-signal errors. Identity/event order must match exactly. Numerical equivalence tolerances must be frozen per signal **before** the first parity run; they cannot be relaxed after results. A missing signal or behavioral change blocks collection. This review defines the protocol; it does not execute it.

## M13: decision and evidence limits

The [final decision](final-decision.json) follows ten separate checks: source integrity, label alignment, deterministic pairwise support, evaluation contract, logging contract, input contract, Safety disabled, endpoint double-count prevention, split/leakage contract, and scope compliance. All passed. The current 63 traces support reproducible label generation, duplicate consistency, and a *design* for future evaluation. They do **not** support generalization, a physical indifference margin, Safety probability, tuned weights, a trained selector, or a terrain-held-out claim. Physical input parity remains a documented integration task; the contract is complete even though the current bridge lacks base linear velocity, ordered full joints and previous ActionManager action.

The next step is **portable input integration and parity design**, followed by logging implementation/parity proposals, independent-StateGroup collection design and a small-training plan for separate approval. [Decision flow](fig-05-decision-flow.svg). The [SHA ledger](artifact-sha-ledger.json) covers only this addendum's small code, contracts, figures, tests and report.

## Final-question checklist

| Requested point | Answer |
|---|---|
| M0–M10 reused/unchanged; tail resolved | Prior 22 ledger files and 63 raw traces unchanged; supplied M11–M13 text closes the previously missing tail. |
| Phase A primary; endpoint double weight | Observed reward, direct-relative reward, goal reduction, and separate pair preferences; endpoint error auxiliary. |
| Pair floor and ranking priority | `1e-6` is numerical only; Huber continuous targets lead, BT ranking assists. |
| 66/147 conflicts | Evaluate group-level agreement, Pareto recall and two-objective trade-offs, without scalar coefficient fitting. |
| Split/bootstrap; Phase B | StateGroup/source/seed split and group CI; auxiliary heads require independent ablation without primary harm. |
| Safety | Disabled for training and inference; no pseudo-positives. |
| Reward and Path boundary logging | 56-field schema includes raw/weighted pre-/post-dt components and command/replan/end exact-stamp events. |
| Body schema and outcome validity | Each field carries metadata/mask/source/parity; invalid, infrastructure, censor and verified failures are distinct. |
| Logging parity | Matched duplicate protocol is frozen as a future pre-collection gate. |
| Current claim and next work | Offline design/pipeline smoke only; next is input integration, not training. |
| Final decision | `multihead_teacher_contract_ready_for_input_integration`. |
