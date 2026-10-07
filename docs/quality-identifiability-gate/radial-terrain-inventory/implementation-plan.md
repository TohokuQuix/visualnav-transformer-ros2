# Radial Terrain Inventory Implementation Plan

**Goal:** Define one all-terrain radial placement rule, run one candidate-free inventory, and produce an Outcome-blind corridor/pilot/goal-bearing design review.

**Architecture:** Isolated branch `exp/eatnav-radial-terrain-inventory` at `35026a9`. Pure Python contracts/tests run outside Isaac. A one-shot Isaac script creates one env per terrain×angle per seed batch, captures settled states and four DEM/mask frames without Path/ROS/follower; a host monitor enforces resource and lifecycle limits. Offline analysis reads only sealed inventory outputs. Results live under a new `/tmp/eatnav-radial-terrain-inventory-20261007` root.

**Reference:** `/home/kojima/.codex/attachments/319a0d00-c616-4305-9be8-02e2cb986875/Pasted text.txt` and prior `/tmp/eatnav-radial-boundary-review-20261007/design-review-report.md`.

## Fixed invariants

- 7 terrains; `r=.30 m`; 16 headings `22.5°k`; no Outcome-based selection or post-observation repositioning.
- Two seed batches, each with 112 unique envs arranged 16 rows×7 terrain columns. A pose never shares its env with another angle; no state restore.
- Same terrain type/difficulty/seed within each terrain column across 16 headings; seed batches independent. Actual generated identity and env origins must be verified before accepting records.
- Zero candidate Path, bridge, follower, reward/critic collection, and 120-step suffix; no ROS router required.
- First failure stops; zero retry. Unsealed partial outputs are never pilot data.
- GPU device total cap 12,288 MiB; anon+shmem 10 GiB; host MemAvailable floor 8 GiB; total runtime cap 2 h; per-batch startup cap 12 min; heartbeat 90 s; storage cap 1 GiB. Runtime baseline GPU is recorded before launch. These are inventory resource limits, not outcome thresholds.
- No candidate pilot, training or hardware.

## Tasks

1. Freeze old source/artifact SHA, scene/config, target map, pose order and stop limits in new output root. Add test that checks 7×16×2 identities, transforms, class labels, shared per-column seed and no Path fields.
2. Implement pure P1/P2/P3/P4 footprint and radial transform contracts. Test zero-joint bounds, all 16 rotations, source-flat classification and world origin composition.
3. Implement one-shot candidate-free Isaac inventory with deterministic 16×7 terrain scene per batch, preflight metadata checks, root placement, settling, four frame DEM/mask capture, Actor finite capture and atomic per-pose records. Test static no-Path/no-follower contract and record validation with synthetic fixtures.
4. Implement host lifecycle/resource monitor and one-time detached execution. Test first-failure stop, no retry, cap handling, input/output SHA and cleanup ownership.
5. Execute one candidate-free inventory only after tests and static freeze pass. Audit all outputs, pose transforms, DEM masks, resets and resource samples. If any stop condition fires, preserve partials and stop dependent work.
6. If inventory passes, run Outcome-blind five-Path main-track corridor and flipper reach scan, classify taxonomy and count independent source families. Test unknown propagation and split grouping.
7. Write conditional four-family pilot proposal, front-goal manifest proposal, future goal-bearing design, resource forecast, 11 figures, risk register and individual final decisions. Verify SHA ledger and report links.

## Review focus

- Tile origin/row/column mismatch must fail before pose acceptance.
- One source seed used at 16 headings is one family, not 16 independent groups.
- Missing DEM mask means unknown; it cannot produce a differential label.
- Settling step reward computation internal to env is not recorded as Outcome.
- Any candidate Path publication or ROS bridge invocation violates scope and must be absent from launcher and sim source.
