# P1 G3 shared DEM v2: implementation and static verification

## Decision and scope

Option 2 is implemented as protocol `eatnav_p1_longhorizon_g3_shared_dem_v2`. This is a **new controlled Path-comparison input**, not a claim that four native raycaster histories match. No Isaac simulation, Path candidate trial, Evaluator training, full collection, production follower change, router change, or existing artifact modification was performed. The new manifest freezes **candidate-free 20-step diagnosis only**. A candidate trial would require a separate execution-mode freeze and approval.

The original G1/G2/G3/DEM diagnostic seal checks remain valid, including the stopped G3 attempt and candidate-free DEM diagnostic. Those roots were read only. The old G3 simulator, host, stage, `direct_dem_gate` function and all numeric limits were not changed.

## Implementation

| Component | New file | Contract |
|---|---|---|
| Common input and verification | `src/visualnav_transformer/eat_nav/p1_g3_shared_dem.py` | Copies env 0's measured, encoded robot-base `[2,20,20]` height/mask frame at settling steps 4/9/14/19 into four separately allocated, nonwriteable arrays. Validates frame ID, resolution, shape, order, `(s+1)*50 ms` stamp, sensor last-update age, per-frame source SHA, and identical bytes. Durable raw `.npy`, metadata, native comparison, and SHA ledger are separate artifacts. Host verification compares the shared source back to env 0's saved native array and metadata. |
| Simulator | `scripts/eat_nav_p1_g3_shared_dem_sim.py` | Keeps 20 settling steps and existing visible clone. No `env.step` occurs between clone and Path reset. Saves all four native histories, compares env 0 with envs 1/2/3 (12 frame pairs), then saves common input before applying the **unchanged** numeric `direct_dem_gate` to shared copies. Physical state equality is a separate gate. Captures actual pre/post noise and post-delay Actor observation, Critic observation, current contact and contact history in distinct arrays/diagnostics. Path publication and each trace row reference the assigned shared frame files and SHAs. Diagnostic mode returns before Path publication. |
| Host | `scripts/eat_nav_p1_g3_shared_dem_host.py` | Uses the existing owner/GPU/resource/cleanup/seal machinery. Requires the new protocol and source SHAs, staged input ledger, physical and shared gate records, 16 native and 16 shared raw frames, native comparison, branch-input records, and their SHAs. Verifies host/container provenance files. A simulator error is a host failure even with native exit 0; candidate-free completion uses its own marker. Later pilot validation retains the original post-Path direct-duplicate gate. |
| Staging | `scripts/eat_nav_p1_g3_shared_dem_stage.py` | Prepares only a fresh persistent root with `batch-0`, copied frozen G2 inputs/old G3 assignment, new sources, manifest and an input SHA ledger. No run command exists in this stage script. |
| Manifest | `docs/eat-nav/p1-g3-shared-dem-v2-manifest.json` | Freezes source/host/module and assignment SHA, original evidence references, 40 s diagnostic episode, 50 ms control interval, original 600/640-step limits and direct-duplicate thresholds. Mode is `candidate_free_20_step_dem`. |

The common DEM is **not injected into the RL Actor, RL Critic, Isaac native raycaster, or production ROS follower**. It is the G3 experimental observation covariate for each branch's Path-comparison record and a prospective Evaluator input. The Actor's checkpoint disables height-scan input; the Critic retains its own scanner and contact terms. Shared DEM parity therefore cannot establish Critic, contact, physics or follower parity. Their differences remain separately visible, and the pre-Path Actor gate plus post-Path direct-duplicate gate remain necessary.

## Independent pre-Path decisions

1. **Physical state:** existing `require_prepath_gate` checks tile/initial state under unchanged tolerances; clone report lists states not copied.
2. **Shared experimental input:** strict metadata, source SHA and byte checks, followed by the original `direct_dem_gate` with RMS `1e-5 m`, max `1e-4 m`, mask IoU `1.0`. A pass means four branches were assigned identical encoded input, **not** that their sensors agreed.
3. **Measured native sensor history:** all 12 env-0-vs-other frame comparisons, including height/mask/ray/pose diagnostics, are saved with their own `pass` field. Failure is retained as a diagnostic and cannot be replaced by the shared-input result.
4. **Actor/Critic/contact:** raw pre-Path arrays and labeled direct/all-branch max differences are saved. Actor post-delay input has the existing `1e-4` direct gate. Critic/contact differences do not become fabricated equalities. Candidate interpretation remains conditional on their observed size and post-Path duplicate reproducibility.

The logical frame stamps now read 250, 500, 750 and 1000 ms. Actual raycaster last-update stamps stay in separate fields. Shared frame provenance is `source_env=0`, `history_role=shared_experimental_history`; saved native arrays retain their measured sensor identity. The host seal is generated only after writer shutdown and includes the raw evidence and SHA ledgers.

## Static and synthetic verification

- `20 passed` for new shared DEM tests plus existing G3 host/protocol and DEM-capture tests.
- The old and new `direct_dem_gate` function ASTs are identical. Static source check finds no physics step between clone and Path reset.
- Tests cover immutable copies, unequal native histories with equal shared input, saved native-to-shared source binding and tampering, wrong order/stamp, sensor update age, separate Actor/Critic/contact differences, nonfinite rejection, native exit 0 with simulator error, and input staging/SHA tampering.
- Real frozen G2 inputs and old G3 assignment were staged **only into a temporary directory** and verified: 13 staged files, input ledger SHA `9083e0b65cbff3d6195762945873c5716add8f4672bce1c965ad4d5e744cdfcf` for that temporary bundle. No persistent run root or simulator process was created.
- Python compilation succeeded for all new Python modules. Read-only `verify_seal` passed on G1 attempt-2, G2 attempt-0, G3 attempt-0/batch-0, and G3 DEM diagnostic attempt-0/batch-0.

## Runtime uncertainty and next gate

Runtime has **not** verified that the new four-frame copies, source/native SHA binding, actual sensor age, physical state gate, Actor/Critic/contact diagnostics, GPU attribution, or host seal pass together in Isaac. The physics divergence mechanism remains unknown. The next proposed step is one 4-env, first-differential/block-0, 20-step candidate-free diagnostic in a new persistent root, with zero Path publications and no candidate outcome. Stop on any gate, ownership, resource, cleanup or SHA failure; do not automatically proceed to candidate trials.
