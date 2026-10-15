import numpy as np
import pytest

from visualnav_transformer.eat_nav.heading_score import score_body_goal


def test_zero_weight_matches_position_progress_and_first_tie():
    xy = np.array([[0.2, 0.0], [0.2, 0.0], [0.0, 0.1]])
    result = score_body_goal(xy, np.array([0., 1., 2.]), np.array([1., 0.]), 0.)
    expected = 1. - np.linalg.norm(np.array([1., 0.]) - xy, axis=1)
    np.testing.assert_array_equal(result['position_score_m'], expected)
    np.testing.assert_array_equal(result['total_score_m'], expected)
    assert result['index'] == 0


def test_heading_rewards_turn_toward_goal():
    xy = np.zeros((2, 2))
    result = score_body_goal(xy, np.array([0., np.pi / 2]), np.array([0., 1.]), .1)
    assert result['index'] == 1
    np.testing.assert_allclose(result['heading_score'], [0., 1.], atol=1e-12)


def test_heading_penalty_peaks_at_ninety_degrees_and_recovers_for_reverse():
    yaw = np.deg2rad([0., 45., 90., 135., 180., -90., -180.])
    result = score_body_goal(np.zeros((len(yaw), 2)), yaw,
                             np.array([1., 0.]), 2.0)
    np.testing.assert_allclose(result['heading_score'],
                               [0., np.sqrt(.5)-1., -1.,
                                np.sqrt(.5)-1., 0., -1., 0.], atol=1e-12)
    assert result['weighted_heading_score_m'][2] == pytest.approx(-2.)
    assert result['weighted_heading_score_m'][4] == pytest.approx(0.)


def test_reverse_heading_can_improve_alignment_to_goal_behind_robot():
    result = score_body_goal(np.zeros((2, 2)), np.array([0., np.pi/2]),
                             np.array([0., -1.]), 1.)
    assert result['heading_score'][1] > result['heading_score'][0]


def test_near_endpoint_heading_has_no_direction():
    result = score_body_goal(np.array([[.99, 0.], [0., 0.]]),
                             np.array([np.pi, 0.]), np.array([1., 0.]),
                             .2, goal_radius_m=.05)
    assert result['heading_score'][0] == 0.
    assert result['index'] == 0


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        score_body_goal(np.array([[np.nan, 0.]]), np.array([0.]), np.array([1., 0.]), .1)
    with pytest.raises(ValueError):
        score_body_goal(np.array([[0., 0.]]), np.array([0.]), np.array([0., 0.]), .1)


def test_goal_neighborhood_keeps_position_score_without_heading_reward():
    result = score_body_goal(np.array([[0., 0.], [0.01, 0.]]),
                             np.array([0., 1.]), np.array([0.01, 0.]), 2.0)
    np.testing.assert_array_equal(result['heading_score'], np.zeros(2))
    assert result['index'] == 1
