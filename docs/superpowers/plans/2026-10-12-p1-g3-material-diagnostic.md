# P1 G3 Material Diagnostic Plan

Goal: explain the pre-Path material audit failure with a candidate-free, read-only Isaac runtime measurement. Preserve crossover v1 and attempt-0.

1. Add pure shape-to-PhysX mapping logic and synthetic tests. Use the installed Isaac Lab startup event's documented `link_paths` order and per-link rigid-body view shape count. Require complete counts and uniform material within any multi-shape link when exact shape index is unavailable.
2. Add a dedicated simulator source that constructs the frozen four-env terrain/config, resets once, reads terrain composed USD binding, enumerates all robot collision prims and USD binding status, and reads the post-startup PhysX tensor. Persist all raw values, shape mapping diagnostics, declared event ranges and per-env canonical fingerprints. Never create follower, bridge, or Path.
3. Derive the G1 owner/GPU host launcher for this diagnostic, with a dedicated input contract, diagnostic marker validation, failure propagation, resource caps, cleanup and seal. Run console/exit bookkeeping outside the sealed batch root. Test post-seal immutability synthetically.
4. Stage a new persistent diagnostic root and source/input SHAs. Verify host/container visibility. Run one candidate-free diagnostic only. Preserve failure evidence, stop regardless of result.
5. Report material counts, USD binding states, effective tensor and coverage, startup event agreement, v2 material-gate proposal, launcher/seal result, and whether a later separately approved crossover pilot is ready.
