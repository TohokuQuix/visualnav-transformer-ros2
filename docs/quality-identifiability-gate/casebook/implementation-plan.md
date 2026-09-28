# Held-out trajectory casebook implementation plan

1. Freeze hashes, thresholds, seeds, and cohort rules in `casebook-config.yaml` before model inference.
2. Audit raw command, dataset goal, split, complete windows, and terminal proximity. Reconstruct goal and teacher XY from raw poses.
3. Select A/B/C/D cohorts deterministically from held-out metadata and frozen Gate A2 pairs; save identities before inference.
4. Run a three-window inference and figure smoke test; verify paired noise, `eval`, finite tensors, frame labels, runtime and VRAM.
5. Use a detached, SHA-checked stage runner for full inference, metrics, figures, report, and artifact hashes. Preserve prior artifacts.
6. Run focused tests, inspect summary and plots, and report static-inference limits without changing the model or starting collection.

The interpretation stage normalizes CSV line endings to LF without changing fields or values; final SHA markers cover the normalized artifact.
