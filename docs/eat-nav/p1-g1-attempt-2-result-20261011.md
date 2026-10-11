# P1 long-horizon G1 attempt-2 result (2026-10-11)

**Decision: G1 pass.** This was the one authorized new candidate-free G1 attempt. G2 and G3 were not run. The immutable measurement root is `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g1-attempt-2/` (host path; it resolves to the mounted onix Isaac Lab directory). The attempt-0 and attempt-1 host seals retained their original SHAs, respectively `c7a823e051860454b6934aa63dfd62ee641fc74d4ea13e722d9626af7703b7c4` and `7f33c57538fccb27ef3d0b83d43cd7627f44239f980462cb982e80ac7a749291`.

## Ownership and execution

The new root copied only frozen `source.py`, `input/`, and `src/` from attempt-1. Source SHA `a2fd5519e460ba6baccdaddd3a6ccbb82709e802b1b8683ec604f1de699255f4` and freeze SHA `78ddf5e8181b75f630241beb613817bc4a333a9791147157507aa35575166edd` match attempt-1. Container `392c15ae67cf8462af6d744cce5582944766ff47ca1ecc6af6881199352d5dd4` was already running; no container, router, driver, follower, bridge, checkpoint, or resource contract change was made.

The paused wrapper emitted `coord/owner-ready.json` for container PID **790**. The host resolved it to PID **39119**, start tick **1253055**, with matching container cgroup, namespace PID chain, command token, source/config SHA, and output root. A preauthorization GPU bracket passed before `owner-authorized.json` released the wrapper. `coord/exec-transition.json` then records `execv` of the frozen simulator source under the same PID/start tick. The gate did not infer a missing PID.

The simulator completed **20/20 contiguous physics steps** (0–19), candidate-free with zero Path publications. Trace fields are finite. The four Actor policy vectors are `[22]`, DEM masks are `[400]`, and contact force arrays are `[60,3]` in every trace row. The simulator completion marker, native exit **0**, and lifecycle `COMPLETE` agree. No G2 timing or Path-quality conclusion is drawn from this probe.

## GPU, other resources, and finalization

The host recorded **22** resource/GPU brackets; all passed. **18** contain owned GPU PID 39119, classified `owned_simulator_tree` with the exact start tick. The other GPU PIDs were known graphics processes; no GPU-using simulator child was observed. Any such child would have required an owned process-tree and NVML PID match. GPU UUID was `GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26`; peak device-total memory was **5236.56 MiB** versus the unchanged **7680 MiB** cap. Maximum attribution gap was **108.85 MiB**, below **256 MiB**. Minimum available host RAM was **83.44 GiB**; peak container anon+shmem was **3.18 GiB**; peak monitored root storage was **4.16 MB**. Host elapsed time was **14.14 s**. No resource, heartbeat, or time cap failed.

Cleanup required no signal. `cleanup-evidence.json` records no owned residual and confirms simulator, heartbeat, resource, and log writers closed. An independent `docker top` after the run showed only the pre-existing container shell. The host seal was generated after writer closure. A separate full-file verifier passed for **251** files, and detected no post-seal changes or additions. SHA ledger SHA is `8838bc92091ee4d29c955ec7c7bc678100726af536cfbe7f43b50862d348eb66`; host seal SHA is `c5712eb6bc816f49222f01218bb593f116a38a00f8ce07eb79bd5b637e420609`.

## Audit note

The one owner-poll row was labeled `ambiguous_or_other_identity_mismatch` by an informational diagnostic helper even though the strict `resolve_owner` accepted exactly one full identity match. This was a reporting-label defect, not a relaxation of ownership checks or a measurement failure. A synthetic RED→GREEN test corrected the helper to label one exact match `matched` for future runs. The sealed attempt-2 poll log and all original measurement files remain unchanged.
