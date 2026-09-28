# Goal provenance

- **heldout_complete**: 6208
- **train_complete**: 49598
- **cross_episode_sources**: 0
- **train_no_progress_threshold_m**: 0.025012025423347954
- **train_high_progress_threshold_m**: 0.3547052890062332
- **goal_source**: future realized endpoint transformed world XY to anchor base XY, capped at 0.8 m
- **command_source**: raw 3-component base velocity command [vx, vy, yaw_rate]; m/s and rad/s, not a commanded goal
- **goal_normalization**: no goal normalization; radius cap only
- **terminal_window_rule**: complete-only: all 8 future points precede same-episode end
- **progress_goal_pearson**: 0.9353153522740343
- **q_goal_pearson**: -0.33061909313049587
- **command_goal_direction_abs_deg_median**: 8.282904339007162
- **command_goal_distance_difference**: NA: velocity command [m/s] and relative goal [m] have different units
- **terminal_time_goal_pearson**: -0.0036610886373337676
- **heldout_terminal_reason_window_counts**: {'time_out': 6141, 'tilt_over': 67}
- **source_hashes**: {'manifest': '55d9e2d89631c2afa19dabfb02b2753616bc2b71907843f5570bd3a8e41e1475', 'quality_checkpoint': 'fcaf3b9397f3344a0017a52e324ad11ce879c29d9737e1b924d3800293e00987', 'baseline_checkpoint': 'c8187d5b57baf6ddfb6ab36029de7b644b8dcfa699553bb6fbe260942c7465fb'}
