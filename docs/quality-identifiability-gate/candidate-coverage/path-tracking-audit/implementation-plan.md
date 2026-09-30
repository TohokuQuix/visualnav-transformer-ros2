# NoMaD Path Tracking Audit Implementation Plan

**Goal:** Analyze only the 112 completed six-second readiness rollouts to locate candidate diversity loss and quantify geometric Path tracking.

**Architecture:** A pure NumPy geometry module defines reversible coordinates, segment projection, tracking metrics, and cluster transitions. A CPU runner validates frozen identities/SHA and writes atomic stage outputs; a separate plot/report pass consumes those outputs. Existing rollout, checkpoint, follower, and threshold files remain read-only.

**Tech Stack:** Python, NumPy, SciPy, Matplotlib, pytest; no Isaac or GPU.

**Spec:** User attachment `3202bbce-c474-4a04-b830-b0353cf76df3/Pasted text.txt`.

## Global Constraints

- 112 completed live rollouts only: 21 nominal, 56 NoMaD, 35 bank; seven anchors excluded.
- Each trace has 120 steps at 20 Hz. Prediction Path has eight future points at 0.25 s spacing; 2 s Path and 6 s rollout are distinct.
- Existing Path/command/realized cluster thresholds and follower 0.2 m endpoint tolerance are frozen.
- State is the statistical unit. No new rollout, retraining, Path selection, threshold tuning, or prior artifact overwrite.
- Outputs: small Git files in `docs/quality-identifiability-gate/candidate-coverage/path-tracking-audit/`; raw/intermediate files in `/tmp/eatnav-candidate-coverage/path-tracking-audit/`.

## Review Focus

- Env-local Path versus world root coordinates: derive and verify origin and yaw for each state before tracking.
- Trace row records pre-step state and command; post-step root belongs to the next 0.05 s sample.
- Self-intersections and zero-length segments must retain deterministic projection and ambiguity counts.
- Nominal repeats and candidate pairs must never inflate the seven-state macro support.
- Missing joint effort must remain missing, not inferred from the Actor vector.

### Task 1: Source and identity gate

- [ ] Add failing tests for SHA mismatch, excluded attempt, nominal identity, and step/stamp mismatch.
- [ ] Implement read-only source audit and `rollout-identity-table.jsonl.gz`, `source-provenance.json`, `coordinate-time-audit.json`.
- [ ] Verify all 112 live entries and seven excluded anchors; stop on mismatch.

### Task 2: Geometry and per-run tracking

- [ ] Add failing tests for local/map/increment round trips, segment projection, cross-track/progress, self-intersection, zero-length, overshoot, and endpoint arrival.
- [ ] Implement pure geometry and 2 s/6 s tracking metrics with explicit projection tie rule.
- [ ] Write atomic `path-tracking-metrics.json` and `endpoint-reach-summary.json`.

### Task 3: Diversity and outcomes

- [ ] Add failing tests for P→C→R transitions and seven-state macro support.
- [ ] Apply frozen cluster thresholds to NoMaD, bank, cross-generator, nominal, and all-candidate sets.
- [ ] Write distance, transition, collapse, and tracking-outcome artifacts with leave-one-state-out sensitivity.

### Task 4: Evaluator, schema, figures, report

- [ ] Reaggregate frozen GT(P) score by prespecified tracking and collapse strata; audit online inputs without imputation.
- [ ] Generate the eight requested figures with NoMaD/bank/nominal separated where paths overlap.
- [ ] Write final decision/report and SHA ledger, then verify artifacts, tests, resource use, and Git scope.
