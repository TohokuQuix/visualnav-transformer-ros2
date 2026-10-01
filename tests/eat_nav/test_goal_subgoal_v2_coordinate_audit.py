"""Frame fixtures for the saved-rollout-only coordinate reanalysis."""
from pathlib import Path
import math
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import eat_nav_goal_subgoal_v2_coordinate_audit as audit
import eat_nav_goal_subgoal_v2_analyze as old


def test_synthetic_origin_translation_and_roundtrip():
    origin = np.array([28., -36.])
    path = np.array([[.4, -.2], [.6, -.1]])
    world = path + origin
    audit.assert_close(world - origin, path, 'world to map')
    audit.assert_close((world - origin) + origin, world, 'map to world')


def test_nonzero_anchor_yaw_and_four_goal_directions():
    yaw = math.pi / 3
    summary = {'map_path': [[.4, -.2]],
               'anchor': {'root_state': [28.4, -36.2, 0.,
                                         math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]}}
    c, s = math.cos(yaw), math.sin(yaw)
    for goal in ([.4, 0.], [-.4, 0.], [0., .4], [0., -.4]):
        expected = [.4 + c * goal[0] - s * goal[1], -.2 + s * goal[0] + c * goal[1]]
        audit.assert_close(old.goal_map(summary, goal), expected, 'robot goal to map')
        audit.assert_close(np.linalg.norm(np.asarray(expected) - summary['map_path'][0]),
                           np.linalg.norm(goal), 'goal distance')


def test_path_endpoint_goal_and_step_zero_anchor():
    origin = np.array([11.85, -35.85])
    anchor = np.array([-.02, .17])
    robot_goal = np.array([.4, 0.])
    goal = anchor + robot_goal
    path = np.array([anchor, anchor + [.2, 0.], goal])
    root_world_step_zero = anchor + origin
    audit.assert_close(root_world_step_zero - origin, path[0], 'step0 anchor')
    audit.assert_close(np.linalg.norm(path[-1] - goal), 0., 'Path endpoint to goal')


def test_pairwise_realized_distance_is_translation_invariant():
    origin = np.array([20., -20.])
    left = np.array([[.1, .2], [.3, .2], [.4, .3]])
    right = np.array([[.1, .2], [.2, .4], [.5, .4]])
    before = np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1)))
    after = np.sqrt(np.mean(np.sum(((left + origin) - (right + origin)) ** 2, axis=1)))
    audit.assert_close(before, after, 'pairwise translation invariance', atol=1e-12)
    cumulative = np.array([[.2, 0.], [.4, .1], [.6, .1]])
    increments = np.diff(np.vstack(([0., 0.], cumulative)), axis=0)
    audit.assert_close(np.cumsum(increments, axis=0), cumulative, 'increment/cumulative roundtrip')
