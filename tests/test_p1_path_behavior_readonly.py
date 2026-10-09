import math
import pytest

from scripts.eat_nav_p1_path_behavior_readonly import (
    initial_frame_xy,
    polyline_distance,
    goal_progress,
    path_y_at_forward,
)


def test_initial_frame_and_path_distance_geometry():
    assert initial_frame_xy((2.0, 3.0), (1.0, 2.0), math.pi / 2) == pytest.approx((1.0, -1.0))
    assert math.isclose(polyline_distance((0.5, 1.0), [(0, 0), (1, 0)]), 1.0)
    assert math.isclose(goal_progress((0.75, 0), (0.5, 0)), 0.5)


def test_goal_progress_preserves_lateral_penalty():
    assert goal_progress((0.75, 0), (0.5, 0.2)) < goal_progress((0.75, 0), (0.5, 0))


def test_path_y_at_forward_interpolates_nominal_lateral_target():
    assert path_y_at_forward(0.5, [(0.1, 0), (0.9, 0.4)]) == pytest.approx(0.2)
