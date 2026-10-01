from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from eat_nav_goal_subgoal_v2_analyze import goal_map, path_distance, polyline_distance


def test_goal_map_uses_frozen_anchor_yaw_and_origin():
    summary = {'map_path': [[10., 20.], [11., 20.]],
               'anchor': {'root_state': [10., 20., 0., 1., 0., 0., 0.]}}
    assert np.allclose(goal_map(summary, [0.4, -0.2]), [10.4, 19.8])


def test_polyline_distance_uses_segments_not_only_waypoints():
    d = polyline_distance([[0.5, 0.1], [2., 0.]], [[0., 0.], [1., 0.]])
    assert np.allclose(d, [.1, 1.])
    assert path_distance([[0., 0.], [1., 0.]], [[0., 0.], [1., 0.]]) == 0
