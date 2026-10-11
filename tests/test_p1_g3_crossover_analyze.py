"""Offline metric separation and fixed-window exposure checks."""

import pytest

from eat_nav_p1_g3_crossover_analyze import _path_geometry, _trial


def test_curved_endpoint_progress_is_not_common_goal_progress():
    rows = []
    for step in range(40):
        x = step * 0.01
        rows.append({"control_step": step, "root_state": [x, 0.0],
                     "post_root_state": [x + 0.01, 0.0],
                     "contact_net_forces_w": [[1.0, 0.0, 0.0]],
                     "applied_torque": [2.0],
                     "follower_diagnostic": {"mode": "Track", "linear_recovery": False,
                                             "angular_recovery": False}})
    publication = {"map_xy": [[0.0, 0.0], [0.0, 1.0]],
                   "common_goal_map_xy": [1.0, 0.0], "path_role": "left_20"}
    result = _trial(rows, {"arrival_step": None, "right_censored": True},
                    publication, [0.0, 0.0], {"env_id": 0})
    assert result["common_goal_progress_2s_m"] == pytest.approx(0.4)
    assert result["local_endpoint_progress_2s_m"] < 0
    assert result["fixed_window_steps"] == 40
    assert result["contact_force_fixed_2s_mean_N"] == 1.0
    assert result["joint_torque_fixed_2s_mean_abs"] == 2.0


def test_geometry_baseline_distinguishes_straight_and_curved_paths():
    straight = _path_geometry([[0, 0], [.5, 0], [1, 0]])
    curved = _path_geometry([[0, 0], [.5, .2], [1, 0]])
    assert straight["path_max_abs_discrete_curvature_m_inv"] == 0
    assert curved["path_max_abs_discrete_curvature_m_inv"] > 0
    assert curved["path_length_m"] > straight["path_length_m"]
