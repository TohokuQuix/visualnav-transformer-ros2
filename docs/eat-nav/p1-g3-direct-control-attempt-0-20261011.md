# P1 G3 direct duplicate Path control, attempt 0

## Decision

**Fail** under the frozen direct duplicate post-Path reproducibility gate. The two direct trials completed, but arrival time differed by **0.40 s** (limit 0.10 s) and two-second common-goal progress differed by **0.11609 m** (strict limit 0.025 m). No retry was made: this is an observed post-Path physical/control divergence, not a prelaunch or logging defect. Curved candidates, Evaluator training, and full collection were not run.

This result applies to first-differential, noise block 0, one paired direct duplicate. It does not establish a population-level distribution of duplicate variation.

## Frozen execution and preservation

- Protocol: `eatnav_p1_longhorizon_g3_direct_control_v1`; [manifest](p1-g3-direct-control-v1-manifest.json), SHA `6ff0af4a6a55498e1bd7c2240e1b7d360b3563cda4c575de08bab752af5fde79`.
- Persistent attempt: `/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-direct-control-v1-attempt-0/batch-0/`. It is preserved with a valid host seal over 291 files; seal SHA `5bda703cc76c464664143067a22c6592c20fbf4e35873412eebd294431a36409`. The seal correctly labels the gate outcome `G3_batch_failed_or_inconclusive`.
- Source SHA `1ed15b909023e8248ddbc491a3d0fb59281a427f14bf4530c9f1e75b470c0740`; host runner SHA `dca9c6774aad668506be2a5d131c7c3be5c9f251a2b29f3f609e8822e156fa28`.
- Direct reference Path SHA `4ec126da6c09e7ebc02f2febba35666ef57a334cec9745301554a13aa1d0155e` for both trial branches. One bridge reset published that Path once to each direct branch and once to a command-only identical monitor follower. No curved Path was published. The monitor command was not injected into physics.
- Four-env frozen terrain, 20-step settling, visible state clone, shared four-frame robot-relative DEM, Actor checkpoint, paired noise, 50 ms control step, 40 s episode cap, and exact-stamp follower/bridge were retained. Only envs 0 and 1 received direct commands and were counted as trials; envs 2 and 3 received zero action/command after Path publication.
- Pre-Path physical state equality passed with root/joint spread zero. Shared DEM equality passed; 16 shared arrays verified. Actor pre-noise, post-noise/pre-delay, and post-delay inputs were equal (direct max absolute difference zero). Native DEM equality remained false and was recorded separately. Initial direct contact force/history max difference was 176.835 N; Critic observation max difference was 1.0. The direct Actor input gate was 0.0 against 1e-4.
- Follower readiness: three live followers, three Path subscribers, TF ready, clock active, zero rejected or pre-Path commands. External router PID 6026/start tick 306994, launcher PID 6019/start tick 306981, ZID `87abc71b81b6bdf4a3557eaec78222aa`; identity and stdout/stderr SHA unchanged before/after.

## Trace and gate

Both trial traces contain 89 contiguous rows (logical stamps 2.00–6.40 s). Every row passed clock → TF → command → bridge acceptance → physics stamp/order checks; no episode reset, termination, truncation, nonfinite value, or recovery mode occurred. Native exit was 0, owned residual process count 0, and all writers closed before sealing. Trace SHAs: slot 0 `cf88b69b320c69a4465c29591068dd0ff1b0747cc51e719d91142bd93085225a`; slot 1 `e0eb2b94e60d6cb9ed4384e88b468b7484d1ba4d3cfdbba3c3f0b66a9c7a5807`.

| Metric | Direct 0 | Direct 1 | Gate |
| --- | ---: | ---: | --- |
| Local arrival step | 48 (2.40 s after publication) | 40 (2.00 s) | spread 0.40 s > 0.10 s: **Fail** |
| Stop hold exclusive step | 89 | 81 | 40-step post-arrival hold recorded |
| Mean tracking error | 0.03125 m | 0.02577 m | spread 0.00548 m ≤ 0.02 m: Pass |
| Two-second common-goal progress | 0.44637 m | 0.56246 m | spread 0.11609 m ≥ 0.025 m: **Fail** |
| Recovery category | default; no linear/angular recovery | same | Pass |
| Arrival/censor category | arrived | arrived | Pass |

The initial follower command and Actor action matched exactly; the paired noise arrays matched for all 89 steps. At step 0, before physics, contact force/history differed by up to 176.835 N (largest at `chassis_r05_sprocket`, 0 versus 176.835 N) and Critic value differed by 0.10733. After that first physics step, tile-local XY positions differed by 0.001075 m. At step 1, Actor input max difference was 0.007604, Actor action max difference 0.025566, and follower command max difference 0.000361. By step 39, tile-local XY distance was 0.11609 m. The first difference order is therefore **contact/Critic at pre-Path → motion after step 0 → Actor and follower at step 1**. Critic value is diagnostic and is not an input to the Actor; the data do not show it causing the motion difference.

The post-Path contact/body response also differed: mean norm of the 60-body contact-force array was 326.36 versus 419.32 N; maximum single-body contact force was 1030.11 versus 2379.62 N. Mean absolute simulator applied torque was 174.01 versus 191.25. Both direct branches remained in default follower mode without recovery or termination. These are descriptive metrics for one pair, not independent evidence of a Path effect.

## Resources and interpretation

Wall time 52.56 s; GPU device-total peak 5194.75 MiB (cap 7680 MiB), GPU attribution gap peak 103.98 MiB (cap 256 MiB), 60 owned GPU samples, minimum host available RAM 88.79 GB, peak container anon+shmem 3.50 GB, peak measured persistent storage 7.95 MB (cap 1 GB). Host/container SHA checks for the referenced DEM evidence, completion marker, and traces ran before duplicate gate evaluation; independent read-only host seal and shared-array verification passed afterward.

The initial visible state, shared DEM, Actor input/action, command, Path, and paired noise were matched. The remaining pre-Path contact difference is a strong candidate for the earliest physical disturbance; the first physics step then produces motion divergence and closed-loop feedback amplifies it. The exact PhysX hidden-state mechanism is not identified from this one run. Native DEM and Critic/contact differences remain explicitly visible; neither was artificially synchronized.

**Recommended next decision:** retain this Fail and investigate whether reproducibility requires restoring or independently replaying contact/PhysX state, or use a statistical paired-run protocol with duplicate variation as the control baseline. Do not treat the current branch clone as deterministic counterfactual evidence and do not proceed to curved G3 Path comparisons under the present gate.
