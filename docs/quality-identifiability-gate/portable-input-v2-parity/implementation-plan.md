# Portable Evaluator input v2 offline implementation plan

**Goal:** Freeze three comparable input schemas, implement read-only source fixtures, audit existing physical bags, and propose logging parity and small training without running either.

**Scope:** No changes to production Actor, bridge, follower, checkpoints, robot control, or old artifacts. Missing fields remain unavailable; no synthetic values or fitted 2S-to-24J correction.

1. **I0 provenance:** Freeze the prior 63 trace and label identities; rehash source, contract and selected physical bag files; inventory the wider bag metadata. Stop on mismatch.
2. **I1 schemas:** Define Core, Core+2S and simulator-only Core+24J fields with order, units, frames, time, masks and hashes. Verify 2S constants against the prior frozen definition.
3. **I2 fixtures:** Write failing tests for name-based extraction, duplicate/missing/nonfinite/stale rejection, clip/scale/order, field hashes and 24J/2S saved trace reproduction. Implement pure Python extractors and a full-tensor assembler that returns unavailable when required fields are absent.
4. **I3–I4 source parity:** Audit flipper axes/default offsets and IMU topic selection, extrinsic, covariance sentinel and timestamp handling. Record uncertain physical calibration as unverified.
5. **I5 bag parity:** Read the selected bags without modification, quantify topic/sample coverage and timing, run body-state fixture where possible and record absent DEM/Path/goal as NA. Define minimum future bag schema.
6. **I6–I8 proposals:** Define arm-stowed applicability, a prospective 120-step logging parity pair, and a StateGroup-based small-training ablation comparison. Do not execute either.
7. **Verification:** Run fixture tests and relevant existing tests, verify all artifact SHA values, then report each gate and stop.
