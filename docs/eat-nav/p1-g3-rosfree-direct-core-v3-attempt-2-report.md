# P1 G3 ROS-free Direct core v3, attempt 2

## Decision

The single authorized Direct×4 process is **protocol-valid**. It ran without a ROS follower, ROS bridge, TF service, or Zenoh router. All four branches reached the local Path goal, completed the 40-step stop hold, and ended within the same episode. No additional process was started.

The frozen ROS bridge comparison remains **identity-inconclusive**: the old ROS Direct reference and this run have different stored terrain material SHA values, and the old reference has no live-material audit. Its sealed traces were compared descriptively, without changing the identity gate or claiming closed-loop equivalence.

## Root cause and fix

`env.step()` is called inside `torch.inference_mode()`. Its returned Actor/Critic observations and `ObservationCapture` tensors therefore may be inference tensors. The prior adapter wrote their command slices in place outside inference mode; its latency buffer could also have that ownership. `reconcile_command_observation()` now returns cloned, writable Actor/Critic and diagnostic tensors, changing only their command slices. When the latency buffer is an inference tensor, it is cloned into a normal tensor before the one command-slice update. Other observation terms, dtype, device, paired noise, and the previous-raw latency formula are preserved. The caller uses the returned tensors for the next Actor call and trace.

## Provenance and tests

- Protocol: `eatnav_p1_g3_rosfree_direct_core_v3`
- Persistent attempt: `/home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-direct-core-v3-attempt-2/`
- Runtime-disabled prep manifest SHA-256: `77fc7e56e00c5effd89ecccc0e7bde22ef470d974aec24d78afb6c8cbce84659`
- Authorized attempt manifest SHA-256: `f229c374585f469c5f837af8acfc32097d388b5c46b3e8be93a56425ca66aeb8`
- Attempt input ledger SHA-256: `d2d6677cbed0281614f03faad0b71f57210cbc5d33e239833abe309786e20a51`
- Simulator source SHA-256: `2b92e7964ec046951697230f54afbe170551b64a4283cd05738935da0e06e8bd`
- Observation adapter SHA-256: `44a6c72afe3d0dbfb40c9ea9951a1df5a3c71772943bab4a662166a449e99594`
- The focused regression reproduced the inference-tensor error before the fix. Afterwards, 85 relevant Torch-enabled static/synthetic tests and 23 host tests passed. Phase A replay: 16 groups, 11,268 rows, zero command/diagnostic mismatches.
- Full repository collection under `tinyvla` has 12 unrelated import errors, including missing `diffusion_policy` and ROS `geometry_msgs`. This did not affect the targeted G3 suite or the Isaac runtime.

## Runtime

- Physical, shared DEM, Actor, and live-material pre-Path gates passed. One Direct Path activation per branch; four independent C++ core instances.
- 86 control steps (logical stamps 2.000–6.250 s), 86 rows per env, native exit 0. Arrival steps: env 0/1/2/3 = 44/41/39/45; all have 40-step stop holds. No reset or termination.
- Across all 344 trace rows, maximum error against the injected command: pre-noise 0, post-noise/pre-delay 0, Critic command 0. Maximum Actor post-latency formula error was `1.49e-8`; the next step's latency `previous_raw` equaled the preceding applied command exactly. The trace validator also checked exactly-once stamp/order and Actor/physics causality.
- Peak GPU device-total memory 5192.875 MiB, peak container anon+shmem 3,508,822,016 bytes, sampled batch storage 9,444,084 bytes, monitored wall time 36.81 s. Resource caps passed. Owned residual processes: zero. Host/container required SHA values matched.
- Host seal SHA-256: `8971d95b68ae42994da7595fbfce07657d360f278a7b0f5cea27c65429bfa905`; read-only verification passed for 302 files, stage `G3_rosfree_protocol_valid`.

## ROS reference, descriptive aligned comparison

Both seals passed verification. Checkpoint, Direct Path SHA, terrain mesh, initial root/joints, 50 ms control interval, episode length, shared DEM source SHA, and first-step paired noise agree. Terrain material SHA differs (`ff1c7507...` new; `fb59d9fd...` ROS), so the exact identity requirement in the frozen comparison contract is unresolved.

The first substantive difference is at **step 0 Actor command input**, not at the follower command: initial follower commands agree within `1e-6`, while the Actor command slice differs by up to `0.09118`; every non-command Actor slice agrees exactly at that step. The ROS trace supplied the previous command (`[0.01475, 0, -0.30393]`) as its pre-noise and Actor command. The ROS-free adapter supplied the newly injected Direct command as pre-noise (`[0.30000, 0, 0]`) and the latency result (`[0.10033, 0, -0.21275]`) to the Actor. Actor action then differs by up to `0.40024` at step 0; XY motion differs after the first physics step. This supports an observation timing/adapter explanation for early divergence, subject to the material provenance caveat.

| Env | Arrival ROS-free / ROS (s) | 2 s progress ROS-free / ROS (m) | Tracking mean ROS-free / ROS (m) | Recovery events |
|---:|---:|---:|---:|---:|
| 0 | 2.20 / 2.40 | 0.494 / 0.446 | 0.022 / 0.028 | 0 / 0 |
| 1 | 2.05 / 2.00 | 0.537 / 0.562 | 0.021 / 0.023 | 0 / 0 |
| 2 | 1.95 / 2.00 | 0.574 / 0.561 | 0.025 / 0.027 | 0 / 0 |
| 3 | 2.25 / 2.45 | 0.483 / 0.438 | 0.024 / 0.028 | 0 / 0 |

Contact and torque values, per-variable first-divergence steps, seal SHA values, and the identity checks are in `p1-g3-rosfree-direct-core-v3-attempt-2-comparison.json`. This single StateGroup and seed demonstrates the new control loop, not comparative navigation quality or recovery parity. Curved crossover and further collection require a separate decision.
