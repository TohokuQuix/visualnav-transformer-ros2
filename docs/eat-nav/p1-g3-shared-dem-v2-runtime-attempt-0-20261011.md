# P1 G3 shared DEM v2 candidate-free runtime attempt 0

## Decision

**Inconclusive — prelaunch host import error.** The single approved invocation exited before the host runner's `main`, owner wrapper, GPU monitor, Isaac Python process, Path publication, or physics. No automatic retry was made. The candidate-free v2 runtime gate cannot be evaluated from this attempt.

## Frozen input and attempt evidence

- Protocol: `eatnav_p1_longhorizon_g3_shared_dem_v2`, mode `candidate_free_20_step_dem`, first-differential / noise block 0, four envs.
- New persistent root: `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-shared-dem-v2-attempt-0/`; the `batch-0` output directory was not created.
- Staged 13-file input ledger SHA: `9083e0b65cbff3d6195762945873c5716add8f4672bce1c965ad4d5e744cdfcf`. Source and contract were visible inside the existing `isaac-lab-onix` container; simulator source SHA `c496de4b7aaa139088bee71869cc72fb3b753ec47eaa8b8999151e2194c12db9`; contract SHA `5ac8311f08ccf864e9b276e901e2e1dbc4ad991d16419a561f6b7e540ed50ab0`.
- Persistent failure evidence: `attempt-0-prelaunch.stderr.log`, `attempt-0-prelaunch-status.json`, and `attempt-0-prelaunch-sha-ledger.json` at the new root. All four ledger references were read back and SHA-verified.
- Existing G1 attempt-2, G2 attempt-0, G3 attempt-0/batch-0, and G3 DEM diagnostic attempt-0/batch-0 original seals were verified read-only. No original artifact was changed.

## First failure and classification

The shell resolved `python3` to `/home/kojima/anaconda3/envs/isaac/bin/python3` (Python 3.14.4). At module import, `p1_g3_shared_dem.py` raised `ModuleNotFoundError: No module named 'numpy'`. The host process exited 1. A separate read-only check confirmed this interpreter lacks NumPy, while `/usr/bin/python3` has NumPy 1.26.4 and can import the G3 v2 host module. This is a **host interpreter/dependency mismatch before launch**, not a simulator or candidate measurement failure.

The staging helper's symlink-aware parent check was corrected before the invocation, with tests passing. This helper is outside the frozen simulator/host/module source SHA contract. No source, manifest, input, threshold, Path or control rule was changed after the attempt.

## Gate outcomes

| Required result | Attempt-0 evidence |
|---|---|
| Physical state equality | Not measured |
| Shared DEM shape/metadata/bytes/SHA/frozen numeric gate | Not measured |
| Native DEM 12 pair comparisons | Not measured |
| Actor pre-noise, post-noise/pre-delay, post-delay; `1e-4` direct gate | Not measured |
| Critic and contact current/history differences | Not measured |
| Sensor logical/update timestamps and age | Not measured |
| Clone-to-diagnostic extra physics | Runtime not reached; zero physics steps executed |
| Path publications | 0; Path processing not reached |
| GPU attribution, peaks, RAM/storage/runtime cap | Monitor not started; no execution peaks available. Read-only preflight saw GPU UUID `GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26`, driver `580.178.04`, about 370 MiB idle device usage, about 93.8 GB host available RAM, and 1.4 MB staged-root storage. |
| Native exit and cleanup | Simulator not launched, so native exit is NA. No owned simulator/bridge/wrapper process was launched; residual owned processes 0. |
| Host/container output SHA and host seal | No runtime output or host seal; prelaunch evidence SHA ledger valid. |

## Next decision

The current frozen root and failed invocation must remain as evidence. A further runtime diagnostic would be a **new attempt requiring approval**, ideally in a new root with an explicit `/usr/bin/python3` host interpreter and an import-only check before launch. Do not promote this attempt to a runtime pass or reuse it as candidate evidence. No G3 candidate trial, Evaluator training, or full collection was started.
