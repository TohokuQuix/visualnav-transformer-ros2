import math

import pytest

from eat_nav_evaluator_multihead import direct_baseline, duplicate_margin, pair_label, group_pairs, masked_bt_loss, masked_huber_loss, make_labels


def test_direct_baseline_applies_to_direct_repeats_and_candidate():
    baseline = direct_baseline([1.0, 1.0, 1.0])
    assert baseline == {"mean": 1.0, "sample_variance": 0.0, "repeat_count": 3}
    assert [value - baseline["mean"] for value in (1.0, 1.0, 1.2)] == pytest.approx([0.0, 0.0, .2])


def test_duplicate_margin_uses_only_direct_repeats_and_numeric_floor():
    assert duplicate_margin([[1., 1., 1.], [2., 2.1, 2.]], 1e-6) == pytest.approx(.1)
    assert duplicate_margin([[1., 1., 1.]], 1e-6) == 1e-6
    with pytest.raises(ValueError):
        duplicate_margin([[1., 1.]], 1e-6)


def test_pair_labels_and_conflict_use_group_only_and_tie_mask():
    assert pair_label(1.2, 1.0, .1) == "i_better"
    assert pair_label(1.0, 1.2, .1) == "j_better"
    assert pair_label(1.0, 1.1, .1) == "tie_within_duplicate_noise"
    assert pair_label(float("nan"), 1.0, .1) == "invalid_or_censored"
    rows = [
        {"state_group_id": "g1", "candidate_id": "direct_reference", "reward": 1., "goal": .5, "endpoint": .5},
        {"state_group_id": "g1", "candidate_id": "left_1", "reward": 2., "goal": .4, "endpoint": .6},
        {"state_group_id": "g2", "candidate_id": "direct_reference", "reward": 3., "goal": .2, "endpoint": .7},
    ]
    pairs = group_pairs(rows, {"reward": .01, "goal": .01, "endpoint": .01})
    assert len(pairs) == 1
    assert pairs[0]["reward_label"] == "j_better"
    assert pairs[0]["goal_label"] == "i_better"
    assert pairs[0]["reward_goal_conflict"] is True


def test_masked_bt_loss_excludes_ties_and_censored_pairs():
    pairs = [(2., 0., "i_better"), (0., 2., "j_better"), (1., 1., "tie_within_duplicate_noise"),
             (1., 0., "invalid_or_censored")]
    assert masked_bt_loss(pairs) == pytest.approx(math.log1p(math.exp(-2.)))
    assert masked_bt_loss(pairs[2:]) is None


def test_masked_huber_loss_excludes_invalid_and_uses_finite_delta():
    assert masked_huber_loss([(2., 1., True), (100., 0., False)], delta=1.) == .5
    assert masked_huber_loss([(2., 1., False)], delta=1.) is None


def test_make_labels_uses_three_direct_repeats_and_unique_candidate_pairs():
    rows = []
    for repeat in range(3):
        rows.append({"state_group_id": "g", "candidate_id": "direct_reference", "repeat_id": repeat,
                     "reward": 1., "goal": .4, "endpoint": .6, "valid": True})
    for index in range(6):
        rows.append({"state_group_id": "g", "candidate_id": f"left_{index}", "repeat_id": 0,
                     "reward": 1. + .1 * (index + 1), "goal": .4 - .01 * index,
                     "endpoint": .6 + .01 * index, "valid": True})
    result = make_labels(rows, {"reward": 1e-6, "goal": 1e-6, "endpoint": 1e-6})
    assert len(result["absolute_and_relative_labels"]) == 9
    assert len(result["candidate_pairs"]) == 21
    assert result["absolute_and_relative_labels"][0]["direct_relative_reward"] == 0.
    assert result["direct_baselines"]["g"]["reward"]["repeat_count"] == 3
    assert result["candidate_pairs"][0]["state_group_id"] == "g"
    left = next(row for row in result["absolute_and_relative_labels"] if row["candidate_id"] == "left_0")
    assert left["direct_relative_preference"]["reward"] == "candidate_better"
    assert left["direct_relative_preference"]["goal"] == "tie_within_duplicate_noise"
    assert left["direct_relative_preference"]["endpoint"] == "tie_within_duplicate_noise"
