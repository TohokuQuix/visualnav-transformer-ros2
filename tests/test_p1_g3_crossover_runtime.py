"""Pre-Path approval and frozen mixed-Path contracts without Isaac runtime."""

from copy import deepcopy

import pytest

from visualnav_transformer.eat_nav.p1_g3_crossover_runtime import (
    compare_prepath, validate_crossover_assignment, validate_robot_physx_material,
    paired_effects,
)


def snapshot():
    return {"state_group": "first-differential", "seed": 2026101101,
            "terrain_mesh_sha256": "mesh", "terrain_material_sha256": "terrain-live",
            "robot_material_sha256": "robot-live",
            "envs": [{"env_id": i, "root_state_w": [float(i), 0, 0],
                      "joint_position": [0, 1], "joint_velocity": [0, 0],
                      "applied_torque": [0, 0], "actor_pre_noise": [0, 1],
                      "actor_post_noise_pre_delay": [0, 1],
                      "actor_input": [0, 1], "actor_action": [0, 0],
                      "noise": {"joint_pos": [0.1]}, "command": [0, 0, 0],
                      "shared_dem_frame_sha256": [c*64 for c in "abcd"],
                      "contact_current": [float(i)], "contact_history": [float(i)],
                      "critic_observation": [float(i)], "critic_value": 0.2,
                      "episode_length": 20, "reset_count": 0}
                     for i in range(4)]}


def test_cross_period_strict_and_diagnostic_signals_are_separate():
    first = snapshot()
    second = deepcopy(first)
    second["envs"][0]["contact_current"] = [999.0]
    second["envs"][0]["critic_observation"] = [123.0]
    result = compare_prepath(first, second)
    assert result["pass"] is True
    assert result["strict_failures"] == []
    assert {row["signal"] for row in result["diagnostic_differences"]} == {
        "contact_current", "critic_observation"}
    second["envs"][1]["root_state_w"][0] += 0.001
    result = compare_prepath(first, second)
    assert result["pass"] is False
    assert result["strict_failures"][0]["signal"] == "root_state_w"


def test_material_noise_dem_and_episode_drift_fail_before_path():
    first = snapshot()
    for mutate in (
        lambda x: x.__setitem__("robot_material_sha256", "different"),
        lambda x: x["envs"][2]["noise"].__setitem__("joint_pos", [0.2]),
        lambda x: x["envs"][0]["shared_dem_frame_sha256"].__setitem__(0, "bad"),
        lambda x: x["envs"][3].__setitem__("episode_length", 21),
    ):
        other = deepcopy(first)
        mutate(other)
        assert compare_prepath(first, other)["pass"] is False


def test_cross_period_rejects_non_numeric_or_nonfinite_strict_values():
    first = snapshot()
    for value in ("0.0", float("nan"), []):
        other = deepcopy(first)
        other["envs"][0]["root_state_w"] = [value, 0.0, 0.0]
        assert compare_prepath(first, other)["pass"] is False


def test_assignment_requires_each_seed_env_all_three_paths_and_direct_pairs():
    from visualnav_transformer.eat_nav.p1_path_crossover_v2 import build_crossover
    schedule = build_crossover({"first-differential": (1, 2, 3),
                                "second-graded": (11, 12, 13)})
    assert validate_crossover_assignment(schedule)["direct_duplicate_pairs"] == 12
    broken = deepcopy(schedule)
    broken["processes"][0]["trials"][0]["path_role"] = "left_20"
    with pytest.raises(ValueError, match="balance"):
        validate_crossover_assignment(broken)


def test_effective_robot_physx_material_requires_all_shapes_and_expected_values():
    properties = [[[0.8, 0.6, 0.0], [0.8, 0.6, 0.0]] for _ in range(4)]
    result = validate_robot_physx_material(properties)
    assert result["env_count"] == 4 and result["shape_count"] == 2
    assert len(result["canonical_sha256"]) == 64
    properties[2][1][0] = 0.7
    with pytest.raises(ValueError, match="effective robot material"):
        validate_robot_physx_material(properties)


def test_paired_effects_keep_seed_cluster_and_censor_separate():
    trials=[]
    for seed in (1,2,3):
        for env in range(4):
            for role,progress,arrival in (("direct",0.3,2.0),
                                          ("left_20",0.4,None),
                                          ("right_20",0.2,3.0)):
                trials.append({"anchor_id":"group", "batch_seed":seed, "env_id":env,
                               "path_role":role,"two_second_progress_m":progress,
                               "common_goal_progress_2s_m":progress,
                               "arrival_time_s":arrival,"right_censored":arrival is None,
                               "tracking_mean_m":0.1,"recovery_count":0,
                               "contact_force_mean_N":10.0,"applied_torque_mean_abs":20.0,
                               "contact_force_fixed_2s_mean_N":10.0,
                               "joint_torque_fixed_2s_mean_abs":20.0,
                               "path_length_m":0.75,
                               "path_max_abs_discrete_curvature_m_inv":0.0,
                               "endpoint_common_goal_distance_m":0.0})
    result=paired_effects(trials)
    assert result["independent_seed_conditions"]==3
    assert result["candidate_pairs"]["left_20"]==12
    assert result["state_groups"]["group"]["left_20"]["seed_count"]==3
    assert result["pair_rows"][0]["arrival_time_difference_s"] is None


def test_paired_effects_preserve_missing_prearrival_tracking():
    trials=[]
    for role in ("direct", "left_20", "right_20"):
        row={"anchor_id":"group", "batch_seed":1, "env_id":0,
             "path_role":role, "arrival_time_s":0.0, "right_censored":False}
        row.update({metric: 0.0 for metric in (
            "two_second_progress_m", "common_goal_progress_2s_m", "recovery_count",
            "contact_force_mean_N", "applied_torque_mean_abs",
            "contact_force_fixed_2s_mean_N", "joint_torque_fixed_2s_mean_abs",
            "path_length_m", "path_max_abs_discrete_curvature_m_inv",
            "endpoint_common_goal_distance_m")})
        row["tracking_mean_m"]=None
        trials.append(row)
    result=paired_effects(trials)
    assert result["state_groups"]["group"]["left_20"]["mean_seed_effect"]["tracking_mean_m"] is None
