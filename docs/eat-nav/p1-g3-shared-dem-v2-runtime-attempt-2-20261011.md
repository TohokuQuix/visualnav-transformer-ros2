# P1 G3 shared DEM v2 candidate-free runtime: attempt 2

## Decision

**Pass for the candidate-free v2 runtime gate.** First-differential / noise block 0 completed 20/20 settling steps across four envs. The physical-state and shared-input gates passed. Native raycaster histories remained unequal and were retained as a separate diagnostic. Path publications: **0**. This result does not establish post-Path duplicate reproducibility or candidate outcome identifiability; no candidate trial was started.

Attempts 0 and 1 remain unchanged. Attempt 0 failed before launch because the selected host Python lacked NumPy. Attempt 1 reached the pre-Path diagnostic, but its sealed output records a keyword mismatch in the branch-input diagnostic helper. The exact simulator capture dictionary was added as a failing synthetic test, then the helper parameter names were corrected. The simulator source, physical experiment inputs, thresholds, Actor checkpoint, terrain, seed, 20-step settling and Path rules were unchanged. Attempt 2 was staged in a **new** root with a new helper/manifest SHA; 20 relevant synthetic/static tests passed before launch.

## Provenance and evidence

- Protocol: `eatnav_p1_longhorizon_g3_shared_dem_v2`, `candidate_free_20_step_dem`.
- Persistent root: `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-shared-dem-v2-attempt-2/`.
- Host interpreter: `/usr/bin/python3` (3.12.3), NumPy 1.26.4; import-only preflight passed for all required host modules. `attempt-2-preflight.json` records Docker, GPU, source and input checks.
- Simulator source SHA `c496de4b7aaa139088bee71869cc72fb3b753ec47eaa8b8999151e2194c12db9`; shared-helper SHA `6c45e38f709630e5c1a5b9f6f703191f440ffcf75b1c4658aa4a5fe5b5049f5b`; manifest SHA `9632d7c4ced83b6088b5c17f056f6e71a95459b71095b423cf8171fdade78e82`; 13-file staged input ledger SHA `84c39f73b79feaccd18d4a0bdd194029e666e633f6c7ccacee43340560fc7dfe`.
- Output native DEM ledger SHA `50a820f879a9eeed0bd65040ad0eea6886df97b2b81cda9e8a389dc3f5f3b197`; shared DEM ledger SHA `4593ea9c94796793ee27e3e76ac000d106ea34992449e157ed8e481079d527c6`.
- Host/container SHA comparison passed for the required native/shared arrays, provenance, metrics and marker. Host seal is valid with 277 files; seal SHA `9f803caacd6afe7ecfe2f5d8d5733df14567899dbd235c5d2494d395f4b3216b`.

## Gate measurements

| Check | Result |
|---|---|
| Physical state equality | **Pass.** Tile-local root, joint position/velocity and command max differences are 0 after the visible clone. Clone report explicitly lists PhysX contact warm-start, contact sensor history, raycaster history, per-env RNG and follower state as unreplicated. |
| Shared DEM experimental input | **Pass.** Four immutable env-0 frames were assigned to each branch, with 16 shared raw arrays verified against source native bytes, metadata and SHA. Frozen direct DEM gate: all four frame RMS/max differences 0 m, mask IoU 1.0, raw SHA equal. This is shared-input parity, not native sensor parity. |
| Native DEM, 12 env-0 comparisons | **Unequal, diagnostic-only.** Env 1: max frame RMS 0.0912 mm, max absolute 0.641 mm; env 2: 0.157 mm / 1.105 mm; env 3: 0.0822 mm / 0.578 mm. Mask IoU remained 1.0. The native result is recorded as `pass=false`; it was not overwritten by the shared-input pass. |
| Actor observations | Pre-noise, post-noise/pre-delay and actual post-delay direct and all-branch max differences are **0** for `[4,22]` arrays. Existing post-delay direct duplicate `1e-4` gate passes at 0. Instrumented noise/action baseline max error is `1.19e-7`, below the frozen `2e-6` limit. |
| Critic observation | `[4,267]`; direct max difference **1.0**, RMS **0.212**. Twelve direct differences exceed `1e-6`, all within indices 26–73, consistent with the configured 48 roller-contact-binary terms. The first 26 values match; values after index 73 differ by at most `2.98e-7`. Critic input was not equalized. |
| Contact current/history | Shapes `[4,60,3]` and `[4,1,60,3]`; direct max difference **176.8 N**, RMS **16.28 N**. The largest direct body differences are `chassis_r05_sprocket` 176.8 N, `chassis_l02_sprocket` 71.1 N and `chassis_rr_subroller_01` 51.0 N. Contact history was not cloned or relabeled equal. |
| Sensor time | Logical stamps after settling steps 4/9/14/19: **250/500/750/1000 ms**. Env-0 actual last-update stamps: 0.249999896 / 0.499999672 / 0.749999464 / 0.999999225 s. Update ages: 0.104 / 0.328 / 0.536 / 0.775 µs. All four sensor buffers reported current. |
| Physics after clone | 20/20 settling heartbeat, then `gate1-probe-complete` before the Path code. Frozen source has no `env.step` on that code path after clone; no Path or rollout trace exists. There is no separate post-clone physics counter in the artifact, so this is a source/control-flow and completion-marker check. |

## Host and resources

Native exit **0**; host failure null; owner lifecycle `COMPLETE`. GPU UUID `GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26`, driver 580.178.04 at preflight. Device-total peak **5,188 MiB** versus 7,680 MiB cap; 17 owned GPU samples among 21 brackets. Maximum attribution gap **104 MiB** versus 256 MiB cap. Container anon+shmem peak **3.40 GB** versus 6 GiB cap. Minimum host available RAM **89.3 GB** versus the frozen floor. Final staged/root storage **2.07 MB** versus 1 GB cap. Host runtime **13.89 s** versus 600 s batch cap. Monitor failures 0. Simulator, heartbeat, resource and log writers were closed before sealing; owned and bridge residuals 0. The seal verification passed after finalization.

## Limit of this pass

The native DEM, Critic contact binaries and contact-force histories differ despite equal Actor input and shared experimental DEM. Those differences remain potential sources of post-Path branch variation. The approved next decision is whether to run a separately authorized **direct-duplicate-only Path reproducibility control**. This report does not authorize candidate trials, Evaluator training, or full collection.
