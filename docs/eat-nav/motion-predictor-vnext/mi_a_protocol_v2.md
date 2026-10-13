# MI-A v2: paired-state pilot protocol

## Scope

`mi_a_protocol_v2` is an independent replacement for the failed **mirror-control requirement**, not a relabeling of MI-A v1. MI-A v1 remains Inconclusive with 0/72 candidate rollouts. B-R0/B-R1 and Phase C status remain unchanged. The robot, terrain generator, material, Actor checkpoint, C++ follower, bank geometry, noise, latency and physics/control timing are unchanged. No post-settling pose correction or DEM symmetrization is allowed.

The sole runtime scope here is **one StateGroup, seed 2026101305, pyramid_stairs level 2, four envs**, nine fresh processes: two candidate-free pre-Path snapshots; Forward Direct, Forward +30°, Forward −30°, Reverse Direct, Reverse +30°, Reverse −30°; and one further Forward Direct repeat. Each candidate process executes 40 control steps at 50 ms. A new persistent attempt root is required per process. The 72-run multi-seed schedule is not authorized by this protocol.

## Independent gates and order

1. **Gate P** compares the two candidate-free fresh-process pre-Path snapshots. No Path activation or follower step is allowed. Fail/inconclusive stops all candidate processes.
2. Each candidate is compared against the first sealed snapshot *before* Path activation. Gate P failure stops the schedule. Candidate run must then satisfy 40/40 same-episode finite label, source/input identity, resource, native exit, inner/outer SHA seal and cleanup. Any failure stops subsequent runs.
3. The final Forward Direct fresh-process repeat has the same Gate P and execution requirements. It estimates process-to-process motion variability. A candidate effect is not inferred from the initial-state gate alone.
4. Review the pilot. Do not start a 72-run schedule, predictor training or goal-selection evaluation automatically.

Gate P exact identity fields are terrain seed and four tile fingerprints, effective material tensor SHA, DEM frame/resolution/stamps, joint names, episode length, and Python/NumPy/Torch CPU/CUDA RNG state SHA. Numeric arrays use maximum absolute tolerance **1e-6**, inherited from the prior cross-candidate gate: world root state (position/orientation/linear and angular velocity), joint position/velocity, Actor pre-noise/post-noise/post-delay observations, four-frame DEM height **and mask**, command, latency state, contact force, and saved action-delay/contact-history arrays when present. Missing required fields fail closed. Different settling pose, native DEM validity mask or contact state **within one snapshot** is permitted; differences between paired candidate snapshots are not. A freshly constructed C++ follower must have state `(0,0,0,1)` before its first call. No controller history is restored across processes.

Gate M records each snapshot's pose and DEM self-reflection metrics independently; it never controls Gate P or candidate validity. Failure does not establish an actual mirror pair. Left/right Arc comparisons in this pilot are paired candidate comparisons, not mirror-controlled comparisons.

## Coordinates and outcomes

The measured root pose at Path activation after 20 settling steps is the Path/body-frame origin. Body Path waypoints are transformed to tile-local using that pose and the env origin. At each 50 ms stamp the fresh follower is called exactly once, its command is injected, the Actor observation is reconciled without another noise sample, Actor inference runs, then physics advances. The raw root state on step 0 and post-root state on step 39 define world displacement rotated by negative **step-0 yaw**. Yaw change is wrapped. Reverse X is not sign-flipped.

Forward arcs subtract Forward Direct; reverse arcs subtract Reverse Direct within the same env. Report raw Δbody XY, Δyaw, candidate-relative XY/yaw, env distribution, command/Actor/heading/tracking/termination and the Forward Direct repeat difference. Pilot one-seed contrasts are descriptive; no confidence interval or formal MI-A Pass is claimed from four envs or two process repeats. The eventual six-independent-seed MI-A decision keeps practical minima 0.05 rad yaw or 0.03 m lateral difference and a 10,000-draw seed-cluster bootstrap; mirror support is **not** a required gate in v2. Any full-schedule decision must be separately reviewed before execution.

## Resource and preservation

Per process caps remain GPU device-total 6,656 MiB, host available RAM at least 2 GiB, host batch storage 0.8 GB and wall 600 s. The pre-Path reference and every candidate require host/container source and input SHA, 40-step trace or zero-step snapshot marker, native exit, complete marker, resource series, inner/outer SHA seals and read-only verification. Native exit 0 with `failure.json` is a failure. No retry after candidate Path exposure. Existing MI-A v1 raw, seal, manifest and decision are read-only. Stage 1–3 outputs and any stop reason are preserved in a separate root.
