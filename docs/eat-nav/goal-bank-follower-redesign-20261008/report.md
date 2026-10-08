# Follower-compatible candidate bank offline redesign

**Decision:** `candidate_bank_compatibility_still_insufficient`. No Path was executed.

## Frozen inputs and source parity

- Prior output ledger: 35/35 entries verified; prior source provenance: 87/87 files verified.
- The 18 R3 baseline tests pass with an environment-selected frozen artifact root.
- Production follower C++ source, path geometry C++ and YAML match backed-up SHA copies.
- Three prior production C0 fixtures (straight, curved, short goal extension) match the pure first-command calculation within 1e-5 using the same source/config SHA.
- The screen remains 1.236 m⁻¹ as an offline compatibility gate, not a physical limit.

## Bank comparison

| Metric | Bank A | Bank B blended |
| --- | ---: | ---: |
| Selected parameter triple (m) | [0.04, 0.16, 0.2] | [0.06, 0.1, 0.12] |
| Hard-valid groups | 40/40 | 40/40 |
| Compatible candidates, old → new | 120 → 120 | 40 → 40 |
| Groups with ≥5 compatible candidates | 0/40 | 0/40 |
| Groups with 7 geometry modes | 40/40 | 40/40 |
| Known terrain-separated pairs | 229 | 164 |
| Screen-compatible known terrain-separated pairs | 12 | 0 |
| Both-seed interaction-class terrain gate among compatible candidates | False | False |

All ten ordered triples per bank were evaluated using the frozen prospective priority.
The previous selector already ranked follower-compatible count before geometry; the new priority changed tie-breaks, not the attainable compatible supply.
Candidate-level failure flags, initial command estimates and coverage are in the machine-readable artifacts.
A small positive amplitude is not automatically compatible: the maximum Bézier curvature is screened over the whole Path.
For the fixed curve rules, Bank A has only the 0.04 m left/right pair below 1.236 m⁻¹; every Bank B non-direct offset in the new pool exceeds that screen.
Bank A level counts (out of 80 per level): [('0.04', 80), ('0.08', 0), ('0.12', 0), ('0.16', 0), ('0.20', 0)]; Bank B: [('0.04', 0), ('0.06', 0), ('0.08', 0), ('0.10', 0), ('0.12', 0)].
Thus parameter triple selection alone cannot supply the required direct + two compatible side pairs.
Terrain differences involving screen-out Paths are descriptive only. The rollout readiness gate counts common-known terrain separation among screen-compatible pairs.

## Conditional pilot options

- L1: five defined candidates, omitting the large pair; P1 28 processes and P2 56. Its medium pair is also screen-out under the present selected geometry.
- L2: seven defined candidates; P1 36 processes and P2 72. Its medium and large pairs are screen-out under the present selected geometry.
- These process counts are comparisons only. Neither L1 nor L2 is eligible for execution after the failed 5/7 gate.
- Prospective P1→P2 numeric thresholds are copied from the previously frozen E1 gate and saved with its SHA; no P1 data was used to choose them.
- P1 L1 point/P95+20%: 1.18/1.45 h.
- P1 L2 point/P95+20%: 1.51/1.87 h.

## Limits

- The pure first-command calculation is source-equivalent only for a fresh controller at a fixed pose; later commands and realized motion are unmeasured. The production follower requests latest TF, while an upstream bridge can impose its own exact-stamp protocol.
- Bank B has a distinct local endpoint and a structural residual to the common global goal. It remains a separate ablation objective.
- Candidate pairs and bearings from one source family are correlated. Flat control is not required to show terrain separation.
- No follower, bridge, router, simulator, Outcome, direct repeat or training process was started.
