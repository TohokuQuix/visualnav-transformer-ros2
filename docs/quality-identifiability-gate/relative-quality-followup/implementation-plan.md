# Relative Quality follow-up implementation plan

## Fixed brief

Use only the existing 360,000-record extended pilot and the frozen relative-quality aligned windows. Leave the split, normalization, RL/EAT checkpoints, previous diagnostics, model architecture outside diagnostic Probes, and simulator unchanged. Compare seven missing Probe inputs with the previous five, then audit recorded actor inputs before any forward evaluator. Stop forward work if the recorded actor observation cannot be aligned to the anchor window. Save each stage and its input/output SHA in a detached runner.

## Ordered stages

1. Verify prior artifact hashes, raw manifest, RL checkpoint, aligned split counts (49,576 / 6,164 / 6,208), train-only scalers, 40-step reward alignment, terminal 29 exclusion, and previous checkpoint/seed provenance. Produce a machine-readable audit before training.
2. Add tests for scalar/multi Probe input routing, fixed q=0.5, parameter delta, train-only normalization, split/episode separation, and held-out evaluation. Extend only the diagnostic Probe's input width; use exactly the prior encoder, Transformer/head, optimizer, data order, and seeds. Freeze thresholds and config before a smoke run.
3. Train the seven Probe variants in a detached resumable process with validation-selected checkpoints. Compare to existing Probe 0, q, advantage, drawdown, and multi on identical windows. Record paired episode bootstrap, per-terrain support, permutation 10/100, train curves, parameter counts, and static matched controls (train thresholds only).
4. Audit raw record and collector for actor observation/latent. The raw record must contain a complete, time-aligned actor observation to run E-state variants. If latent is absent but observation exists, use observation only. If observation is absent and cannot be reconstructed exactly from saved source fields, record the gate failure and stop before forward evaluator training. Do not substitute twist/command or infer missing joint signals.
5. If the actor-input gate passes, implement and test a light shared-capacity forward evaluator with state, trajectory, combined, and reproducible latent variants. Fit normalization on train only. Compare multi-head held-out predictions to fixed baselines, paired episode CI, terrain support, calibration and seed variability. Do not claim candidate ranking from on-policy one-trajectory states.
6. Generate reports, decision, plots, stage SHA markers, status, and tests. Commit only small outputs; keep checkpoints and bulky intermediates in /tmp.

## Stop rule

The attached request explicitly lists unknown actor observation/window correspondence as a stop condition. Phase A is independent and can finish first. A failed actor-input gate prevents Phase B and its claimed evaluator result; the report must state the missing raw fields and exact alternative data required.
