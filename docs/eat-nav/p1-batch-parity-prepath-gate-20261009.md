# P1 28-run golden reference: batch pre-Path gate (2026-10-09)

## Decision

**`pre_path_gate_fail_stop`**. No candidate Path was published, and no mixed-candidate batch rollout, Outcome comparison, added StateGroup pilot, collection, or Evaluator training was started. The 112-env simulator completed 200 physics settling steps in both candidate-free diagnostics with native exit 0, but 0/28 relocated target envs matched the frozen P1 start-state gate.

The original 28 P1 runs are SHA-sealed in `/home/kojima/eatnav-backups/2026-10-09-p1-radial-28-analysis/`. The first failed P1 process is excluded by the integrated manifest. Four direct triplicates have identical 120-step trace SHA within each original StateGroup. The golden backup verifier rechecked all 2,919 files before this attempt.

## Implemented

- A ROS-independent C++ follower core with a Python `ctypes` binding and explicit cloneable per-env state was added. It leaves the production ROS node, follower source, config and binary untouched. Its current supported contract is the single-Path P1 six-second protocol; recovery behavior outside that protocol is explicitly rejected.
- The core reproduced **3,360/3,360** golden commands across 28 traces within `1e-6` absolute tolerance. The largest observed difference was `1.1851e-7`.
- A frozen 16×7 scene plan assigned the 28 target envs to distinct tiles: four StateGroups × five Path choices plus two additional direct repeats per group. The original P1 runs use two batch seeds and reuse original env index 48, so root XY is compared in each env's tile-local frame; P1 Path geometry, terrain type, per-tile terrain seed and difficulty 0.5 are held fixed.
- Numeric tolerances and resource limits were saved in `config/eat_nav_p1_batch_parity.json` before either simulator result. The pre-Path DEM RMS, mask, root, joint and Actor bounds were inherited from the earlier radial reconstruction gate. They were not relaxed after failure.

## Candidate-free measurements

| Attempt | Difference in start-state replay | 200 physics steps | Pre-Path pass | Path publications | Native exit |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 | root XY/yaw placement | 28/28 | 0/28 | 0 | 0 |
| v2 | frozen source prephysics joints, flipper latch and 20-step settling command series added | 28/28 | 0/28 | 0 | 0 |

In v2 the command fields in the settled Actor observation agree with P1 for all four source groups. The remaining v2 failures are:

| Frozen check | Target envs failing |
| --- | ---: |
| root velocity | 28/28 |
| joint position / velocity | 28/28 each |
| four DEM frames | 28/28 at every frame |
| Actor input | 28/28 |
| root orientation | 10/28 |
| tile-local root position | 3/28 |

The first recorded DEM frame at physics step 50 already differs in every target env. The first exact divergence step before 50 was not instrumented. At step 200, the largest tile-local root position error was 0.0340 m; maximum DEM height RMS across frames was 0.0521 m, with seven mask mismatches over 112 frames. The maximum Actor component difference was 2.0. The three direct copies of one Path per group also differ **before Path publication**: maximum pairwise local root position differences were 0.0322, 0.00187, 0.00648 and 0.00141 m for the four groups. This violates the direct-clone start-state gate.

The exact cause of all residual differences is not yet established. Candidate causes are PhysX/contact sensitivity to moved tile coordinates and simultaneous envs, remaining per-env action/latency or random state not captured by the P1 start fixture, and terrain mesh identity beyond the metadata seed/type check. Observation noise differs when source env IDs and batch seed are changed. No noise or delayed Actor input was replaced with golden values to mask live physics differences.

## Protocol and artifacts

The final v2 input plan SHA is `fe676b4d4c8182870ad452bdd8f33111fb69df9eb82a25ce978aefa44cd44e43`; source fixture SHA is `8912dd3986f445955349c6755f1cacb2f265df9be7b9704717a469614136b8f8`. The v2 source SHA is `97e7177f9ae58b483c89fd4b35db9879b93ee93319b399369bdd5083504e6f12`. The output audit SHA is `eda355b44e8c7874d1a869f80ea3cd968a25285203729b086abd1d3014f44dd8`; the atomic seal SHA is `b0b65a417ad59eabfcc99920078890ace949c0cba1095f66349d9744640a1246`.

Both output roots are retained separately:

- `/tmp/eatnav-p1-batch-parity-20261009/`
- `/tmp/eatnav-p1-batch-parity-20261009-v2/`

Each root has an atomic decision, SHA ledger and seal. Host and container SHA for the five v2 simulator outputs match. The v2 observed GPU device-total peak was 5,907 MiB against the 7,680 MiB cap; minimum host available RAM was 81.5 GiB against 10 GiB. Monitored active storage peaked at 1.22 MiB, and final host artifacts total 5.76 MiB, against the 1 GiB cap. The owned simulator PID exited, and GPU memory returned to 438 MiB. The 90-second heartbeat and 600-second process caps were not reached.

## Next gate

The batch rollout is blocked until a candidate-free 112-env run matches **all 28** settled root/joint, four DEM frames and Actor inputs under the unchanged pre-Path tolerance. The next diagnostic should first save per-tile mesh SHA and per-step root/joint/action-manager/latency/observation-noise state during settling, then locate the first divergence. That diagnostic must retain the sealed v1/v2 attempts and freeze a new source SHA before running. No 8–12 StateGroup expansion proposal is issued because mixed-candidate parity has not passed.
