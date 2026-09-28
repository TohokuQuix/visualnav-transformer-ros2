# Phase C first-stage execution plan

Approved scope and frozen settings: see phase-c-config.yaml and the 2026-09-29 user conditions.

1. Complete-only train/validation view; save ordered identities; verify 49,598/6,171, all masks and source episode integrity; fit action normalization on complete-only train.
2. Run one full timing epoch with the frozen model and hyperparameters. Record epoch wall time, GPU peak, process RAM and checkpoint bytes. The probe loss cannot change the recipe.
3. Start scratch reference training at seed 123, batch 64, LR 1e-4, max 100, patience 10, min delta 0; no LR scheduler. Select best EMA by aggregate validation diffusion loss; atomic latest checkpoints carry RNG and sampler state.
4. Run G/GTE/GTS/GT shortcut diagnosis on frozen Phase B data/checkpoints.
5. Implement real ROS2 follower bridge, exact observation capture, synchronized branch state, duplicate control and resume tests.
6. Run a few-state simulator smoke and issue second preflight report. Formal rollout awaits separate approval.
