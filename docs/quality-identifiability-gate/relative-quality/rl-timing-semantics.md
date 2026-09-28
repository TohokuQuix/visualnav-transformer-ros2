# RL timing semantics

- Critic: V(s_t), evaluated from obs before env.step(action_t).
- Reward: raw env reward for s_t -> s_{t+1}, recorded in the same row as action_t; PPO timeout correction is not applied by collector.
- Pose/timestamp: s_{t+1} snapshot after step; terminal row snapshot is pre-reset; (global_step+1)*step_dt, identifies reward/pose endpoint; critic corresponds one control step earlier.
- Aligned 2 s window: anchor pose row i: V0=row i+1; rewards=row i+1..j inclusive; endpoint V=row j+1; j selected by timestamp at anchor+2.0 s.
- Gamma 0.99 per 0.05 s control step: 40 rewards, gamma^40 endpoint bootstrap.
- RewardManager already multiplies every term by dt=0.05 s. Raw reward decomposition was not stored.
- Terminal endpoints: 29/61,977 complete windows (train 22, validation 7, held-out 0) lack endpoint Critic. They remain in the identity audit; V-end-dependent labels are null and excluded from fair Probe comparison.
- True terminal auxiliary return uses bootstrap 0. Timeout auxiliary return follows PPO's gamma*preterminal-V reward correction because `is_finite_horizon=false`.
- Existing P10 Quality uses pre-action value rows i..j-1; it is preserved for compatibility even though trajectory pose anchors are post-action.

## Source hashes

- collector: `/tmp/eatnav-relative-collect.py` SHA-256 `022d818bbe5f48a6a38fc4bd78065c16491d5035b389eed01a9753efe0c61195`
- onix_env: `/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/envs.py` SHA-256 `10c67c223eed605e012fe336ddfe4a2760d0e5cd74bca1c38a0789fc668bc013`
- ppo: `/tmp/eatnav-relative-ppo.py` SHA-256 `4373ac1b2f9fdf14d9da57516968fc95d8f605d2967fee01dc61bf0d09423478`
- wrapper: `/tmp/eatnav-relative-wrapper.py` SHA-256 `1d27c1a7ac75d6bc62ce5376bceb23cac08c168a1343930a72073acddaf76693`
- reward_manager: `/tmp/eatnav-relative-reward-manager.py` SHA-256 `6be44a98df5aed8a0ebac2913ebea33010633e007e816697a6d99f7ccfd731db`
