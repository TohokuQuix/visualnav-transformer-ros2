"""Heading-aware selection in the student ROS-separated GUI demo."""

import numpy as np
import pytest

from eat_nav_interfaces.msg import ActorState22
from visualnav_transformer.eat_nav.ros2_phase2.nodes import build_model_evaluation


class History:
    def assemble(self, actor, now_ns):
        return np.zeros((4, 2, 20, 20), np.float32), np.zeros(22, np.float32), {}


class Worker:
    def predict_heading(self, dem, state):
        xy = np.zeros((14, 2), np.float32)
        yaw = np.zeros(14, np.float32)
        yaw[2] = np.pi / 2
        return xy, yaw, 'checkpoint-sha'


def evaluate(**kwargs):
    return build_model_evaluation(
        History(), ActorState22(), 100, 1, (0., 0., 0.), (0., 1.),
        mode='learned14_adopted', checkpoint='checkpoint', python='python',
        results='results', normalization='normalization', worker=Worker(),
        predict_yaw=True, **kwargs,
    )


def test_weight_two_selects_candidate_facing_global_goal():
    positional = evaluate(scoring_mode='xy_only')
    weighted = evaluate(scoring_mode='xy_heading', heading_reward_weight_m=2.0)
    assert positional.selected_index == 0
    assert weighted.selected_index == 2
    assert weighted.scores_m[2] == pytest.approx(2.0, abs=1e-6)
    assert weighted.predicted_yaw_valid


def test_heading_selection_requires_yaw_and_nonnegative_finite_weight():
    with pytest.raises(ValueError):
        build_model_evaluation(
            History(), ActorState22(), 100, 1, (0., 0., 0.), (0., 1.),
            mode='learned14_adopted', checkpoint='checkpoint', python='python',
            results='results', normalization='normalization', worker=Worker(),
            scoring_mode='xy_heading', predict_yaw=False,
        )
    for weight in (-1., float('nan')):
        with pytest.raises(ValueError):
            evaluate(scoring_mode='xy_heading', heading_reward_weight_m=weight)


def test_gui_default_is_heading_mode_with_weight_two():
    from eat_nav_phase3_gui_launch import build_parser
    args = build_parser().parse_args([])
    assert args.scoring_mode == 'xy_heading'
    assert args.heading_reward_weight_m == 2.0
