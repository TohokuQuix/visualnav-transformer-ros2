# P1 statistical Path comparison v2: bounded pilot stop

**Decision:** `protocol_invalid` for final comparison because one of eight required processes did not produce a measurement. This is a simulator startup failure, not a failed initialization or an observed Path effect. The frozen 56-trial design remains unchanged. No retry, replacement trial, Outcome comparison, Path ranking, or replication estimate was made.

## Completed gates

- The v2 static gate confirmed four StateGroups, two batch seeds, seven assigned slots per batch, 56 planned trials, steps 0–39 as primary, and steps 40–119 as diagnostic. The original assignment SHA is `709fa5ccc98c152e75dc24cbd659dc8ea249c740efeb14bdabe31b5cc5affc69`.
- Eight candidate-free processes passed: each emitted seven finite, contiguous 120-step records with Path publication count 0. The analysis CLI accepted their 120-step record structure; unit tests covered short-terminal censoring. Root, joint, flipper, actuator, latency, and observation-history initialization checks passed. Canonical terrain mesh and composed collision fingerprints matched across both batches for each StateGroup.
- Seven Path processes passed the one-shot host seal, producing 49/56 valid 120-step trajectories. None terminated. All seven had native exit 0, trace SHA agreement, no resource violation, and zero owned bridge/follower residual. Their outcomes remain unanalysed because the eighth batch is absent.
- Actor actions used `act_inference()`. Actor observation corruption was enabled in the collection configuration. Uniform additive noise applies to base angular velocity (±0.2), joint position (±0.01), joint velocity (±1.5), and projected gravity (±0.05); IsaacLab draws through `torch.rand_like`. Terrain generation isolates and restores the PyTorch CPU/CUDA RNG. The global observation RNG is not paired by environment or Path, so no matched-noise claim is made.

## First failure

The eighth process, `second-graded / batch 1`, crashed during Isaac Sim/Kit startup before its Python `main`, Path publication, or any control step. The native process exited 1. The first failing signal is a segmentation fault in the `libX11`/`libxcb` `XOpenDisplay` path from `libomni.platforminfo.plugin.so`; the host's missing-output Docker copy error is secondary. The failed process used 349 MiB GPU at its peak, had no resource-cap violation, and left zero owned residual processes. The external Zenoh router remained running and was not modified.

Across all v2 resource samples, peak GPU use was 5,433 MiB (cap 7,680 MiB), minimum available host RAM was 82.07 GiB (floor 10 GiB), and maximum active per-process storage was 0.012 GiB (cap 1 GiB). The v2 output root occupies about 144 MB on the host. The seven sealed Path processes took 432.2 seconds in total.

## Interpretation and next decision

The 49 sealed trials show the corrected 120-step execution path is operational for seven batches. They do not complete the balanced two-batch design for all four StateGroups. Direct-reference variability, Path-minus-direct differences, confidence intervals, slot/batch effects, and a replication requirement are **NA** for this incomplete pilot. Do not infer a population Path effect or a definitive ranking from these artifacts.

The next decision is whether to authorize a separate, explicitly bounded recovery of the single startup-failed batch. Its failed attempt and all seven sealed batches must remain immutable; no recovery was attempted under this instruction. Full collection, new StateGroups, and Evaluator training remain unstarted.

## Provenance

- Output root: `/tmp/eatnav-p1-statistical-v2-20261009/`
- v2 config SHA-256: `20ba502a9e4f3f7de61f05bbc04d7de0d9a5654ae0032be9e2fb16e662e5dec5`
- Path simulator source SHA-256: `60b53d58593f304dabafa5be3ee37b70c4706340ea83c932177631fcc189a2c1`
- Stop summary SHA-256: `a5fb9e2795f3cef7d4e31fa5d0886cfcd1e5b7ebdd13fa15987eb4b945a1f86a`
- Durable archive: `/home/kojima/eatnav-backups/2026-10-09-p1-statistical-v2-stop/artifacts.tar.gz`
- Archive SHA-256: `636abb509465b275905ab0102fe40213164ddc00e709261116b2b81708c3b523`
- Relevant tests before execution: 19 passed. The original v1 80-step artifacts and invalid status were not modified or included in v2 analysis.
