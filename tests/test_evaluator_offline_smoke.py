import numpy as np

from visualnav_transformer.eat_nav.evaluator_offline_smoke import (
    anchor_features, path_features, group_loss, audit_records,
)
from visualnav_transformer.eat_nav.portable_input_v2 import SIM_24J, SIM_FLIPPERS
from visualnav_transformer.eat_nav.evaluator_offline_train import fit_scaler, loso_splits, pareto_set
from eat_nav_evaluator_offline_train import pair_loss
from eat_nav_evaluator_offline_analyze import distillation_adoption


def test_prepath_anchor_does_not_read_postpath_trace():
    names = list(SIM_FLIPPERS) + list(SIM_24J[0]) + list(SIM_24J[1])
    summary = {
        "anchor": {
            "dem_history": np.zeros((4, 2, 20, 20)).tolist(),
            "policy_pre_noise": [0.0] * 22,
            "policy_actor_input": [0.0] * 22,
            "flipper_previous_target": 1.0,
            "joint_position": [0.0] * len(names),
            "joint_velocity": [0.0] * len(names),
        },
        "raw_joint_effort_field_order": names,
        "goal_metadata": {"goal_xy": [1.0, 0.0],
                          "base_path": [[0.0, 0.0]] + [[i / 8, 0.0] for i in range(1, 9)]},
        "logical_t0_ns": 100,
        "trace": [{"policy_actor_input": [99.0] * 22}],
    }
    first = anchor_features(summary)
    summary["trace"] = [{"policy_actor_input": [-99.0] * 22}]
    second = anchor_features(summary)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert first["arm_applicability_mask"].tolist() == [0.0]


def test_sprocket_state_is_scaled_angular_speed_not_metres_per_second():
    names = list(SIM_FLIPPERS) + list(SIM_24J[0]) + list(SIM_24J[1])
    joint_velocity = [0.0] * len(names)
    for name, angular in (("chassis_r08_sprocket_axle", 20.0),
                          ("chassis_r01_sprocket_axle", 40.0),
                          ("chassis_l08_sprocket_axle", -10.0),
                          ("chassis_l01_sprocket_axle", -30.0)):
        joint_velocity[names.index(name)] = angular
    summary = {"anchor": {"dem_history": np.zeros((4, 2, 20, 20)).tolist(),
                          "policy_pre_noise": [0.0] * 22,
                          "policy_actor_input": [0.0] * 22,
                          "flipper_previous_target": 1.0,
                          "joint_position": [0.0] * len(names),
                          "joint_velocity": joint_velocity},
               "raw_joint_effort_field_order": names,
               "goal_metadata": {"goal_xy": [1.0, 0.0],
                                 "base_path": [[0, 0]] + [[i/8, 0] for i in range(1, 9)]},
               "logical_t0_ns": 100}
    state = anchor_features(summary)
    assert np.isclose(state["right_sprocket_angular_state_scaled"][0], 1.5)
    assert np.isclose(state["left_sprocket_angular_state_scaled"][0], -1.0)


def test_path_geometry_and_coverage_are_candidate_specific():
    dem = np.zeros((4, 2, 20, 20), dtype=float)
    dem[:, 1] = 1.0
    p1 = np.array([[i / 8, 0.0] for i in range(1, 9)])
    p2 = np.array([[i / 8, 0.1 * i / 8] for i in range(1, 9)])
    one = path_features(dem, p1)
    two = path_features(dem, p2)
    assert not np.array_equal(one["path_geometry"], two["path_geometry"])
    assert one["path_coverage"].shape == (64,)
    assert one["path_coverage"][0] == 1
    assert one["path_coverage"][-1] == 0  # 1 m endpoint lies beyond 0.95 m raster edge


def test_group_loss_is_macro_not_pair_count_weighted():
    # A group with many pairs must have the same weight as a small group.
    preds = [np.array([0.0, 0.0]), np.array([0.0])]
    target = [np.array([1.0, 1.0]), np.array([2.0])]
    assert np.isclose(group_loss(preds, target), 0.5 * (0.5 + 1.5))


def test_audit_rejects_postpath_body_variation_within_group():
    common = {"dem_history": np.zeros((4, 2, 20, 20)).tolist(),
              "body_angular_velocity": [0.0, 0.0, 0.0],
              "anchor_timestamp": [100]}
    records = [{"group": "g", "candidate": "direct_reference", "repeat": 0,
                "common": common},
               {"group": "g", "candidate": "left_1", "repeat": 0,
                "common": {**common, "body_angular_velocity": [1.0, 0.0, 0.0]}}]
    import pytest
    with pytest.raises(ValueError, match="pre-Path common field differs"):
        audit_records(records)


def test_train_scaler_balances_groups_and_never_sees_holdout():
    x = np.array([[0.0], [0.0], [10.0], [1000.0]])
    groups = ["a", "a", "b", "holdout"]
    mean, scale = fit_scaler(x[:3], groups[:3])
    assert np.isclose(mean[0], 5.0)
    assert scale[0] > 0


def test_loso_keeps_every_group_whole():
    groups = ["a", "a", "b", "b", "c"]
    folds = loso_splits(groups)
    assert len(folds) == 3
    for held, train, test in folds:
        assert all(groups[i] != held for i in train)
        assert all(groups[i] == held for i in test)


def test_pareto_set_keeps_reward_goal_tradeoff():
    scores = {"reward": [2.0, 0.0], "goal": [0.0, 2.0], "dominated": [-1.0, -1.0]}
    assert pareto_set(scores) == {"reward", "goal"}


def test_pair_loss_applies_each_frozen_head_weight_once():
    import torch
    pred = torch.tensor([[2.0, 0, 0.0, 0, 0, 0], [0.0, 0, 2.0, 0, 0, 0]])
    pairs = {"g": [{"candidate_i": "a", "candidate_j": "b",
                    "reward_label": "i_better", "goal_label": "j_better"}]}
    actual = pair_loss(pred, [0, 1], ["g", "g"], ["a", "b"], pairs,
                       weights=(0.25, 0.25))
    expected = 0.25 * torch.nn.functional.softplus(torch.tensor(-2.0)) * 2
    assert torch.allclose(actual, expected)


def test_distillation_rejected_when_observed_ranking_worsens():
    core = {"top1_recall": 0.10, "oracle_reward_regret": 0.08,
            "pairwise_accuracy": 0.50}
    student = {"top1_recall": 0.05, "oracle_reward_regret": 0.09,
               "pairwise_accuracy": 0.52}
    assert not distillation_adoption(core, student, mean_primary_delta=-0.01,
                                     fold_primary_improvements={"reward": 5,
                                                                "direct_relative_reward": 5,
                                                                "goal_reduction": 5})
