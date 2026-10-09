# P1 Minimal Pre-Path Physics Divergence Plan

**Goal:** Locate the first causal difference for a deterministic `k_rail_step_field` StateGroup without publishing a Path.

**Scope:** New, versioned candidate-free diagnostics only. Keep golden, batch v1/v2, Phase A, production follower and frozen gates unchanged.

1. Verify all reference seals and select the frozen second-differential source pose with two direct target clones. Freeze three-env tile geometry, local pose, joint/flipper fixture, settling command sequence, tolerances and resource caps. Explicitly record that the three-env world placement and env count differ from the historical 112-env runs.
2. Implement and test source-state capture/comparison helpers. Distinguish observed-equal, observed-different and unavailable fields. Cover root Z/velocity, joint/targets, ActionManager, flipper state, latency buffers, contact, and actual DEM/Actor update points.
3. Create one candidate-free three-env simulator implementation with config mode `baseline` or `R1`. Baseline follows v2 root placement exactly; R1 adds only source root Z and velocity restoration. Freeze each mode's source/config/manifest SHA before launch. Record immediately after reset, immediately after placement, before first physics, then every physics step through 200; terminate on native/resource/heartbeat failure.
4. Run baseline once, inspect step-0 fields and first divergence versus source and direct clones. If root Z/velocity differ, run R1 once and quantify its incremental effect. Only if needed and observable, make separate R2/R3 interventions with new source/config/output roots. Never change a previous mode after viewing its outcome.
5. If minimal scene cannot reproduce the v2 divergence, report that limitation and consider one instrumented 112-env candidate-free measurement only if it can distinguish an unresolved environment-count effect. No repeated full-scene runs.
6. Seal per-mode results, first-divergence timeline, D1/D2 separation, resource/ownership evidence and report. Stop before any Path, Outcome or full batch validation.
