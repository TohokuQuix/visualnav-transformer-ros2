# Radial inventory termination source audit

This audit precedes the termination-aware continuation. The installed host and container copies have matching SHA-256 values.

| Source | SHA-256 | Relevant lines |
|---|---|---|
| Isaac Lab `manager_based_rl_env.py` | `974977825dbbd916460b4f9bb12176ab2cde41026fadc05e1de02d8fbbc94725` | 153–241, 369–395 |
| Isaac Lab `termination_manager.py` | `cb6333f11fa6472f7d8034bae9835714e83ae3edde97894b767f1351a7eea99d` | 97–123, 129–193 |
| Onix `envs.py` | `10c67c223eed605e012fe336ddfe4a2760d0e5cd74bca1c38a0789fc668bc013` | 23–67 |
| Onix `rough_env_cfg.py` | `f357e63f7f8b95caa5b6c6ac064cf1b76cc38a797edd7c435c01f2373d79ac71` | 770–776; frozen `tilt_over` 85 degrees |

`OnixManagerBasedRLEnv.step()` runs `cfg.decimation=10` physics steps, increments `episode_length_buf` and `common_step_counter`, calls `termination_manager.compute()`, stores `reset_buf`, `reset_terminated`, and `reset_time_outs`, computes reward, then calls `recorder_manager.record_pre_reset(reset_env_ids)` and `_reset_idx(reset_env_ids)` before computing and returning the observation. The returned observation for reset envs is therefore post-reset. `TerminationManager.get_term(name)` and `active_terms` expose current per-term flags. `terminated` and `time_outs` distinguish physical and timeout conditions; `dones` is their union. `_reset_idx` zeroes `episode_length_buf` after manager/scene/event reset. No built-in episode or reset serial counter is provided by these inspected sources; the measurement records local counters beginning at the explicit initial `env.reset()` and increments them only for observed selected-env resets. Logical control step `n` ends at global physics step `10n`, with timestamp `physics_step × 0.005 s`.

The installed Onix subclass implements its own `step()`, so an observation returned by the RSL wrapper cannot be used as a termination pre-reset observation. The narrow pre-reset capture point is an instance-local wrapper around `_reset_idx`: it observes the already-computed `reset_buf` and term flags, captures root/joint/gravity and a history-preserving Actor observation before calling the original `_reset_idx`, and never changes the reset target, physics, thresholds, or dynamics. Non-selected resets remain independent and do not close selected streams. Any missing pre-reset signal, unknown term, or counter inconsistency invalidates the measurement.

The old `shard-15` is immutable and measurement-invalid; its 140-step trace cannot be promoted. New measurement code and artifacts belong to a new root. The 15 sealed old shards remain read-only.
