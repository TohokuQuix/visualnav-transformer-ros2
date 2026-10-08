import hashlib
import json
import math

import numpy as np
import pytest

from visualnav_transformer.eat_nav.goal_bank_offline import (
    BEARINGS_DEG,
    audit_geometry,
    bezier_candidate,
    corridor_audit,
    freeze_anchors,
    goal_for_bearing,
    path_sha,
    pairwise_audit,
    pairwise_command_estimate,
    corridor_class_proxy,
    select_parameters,
)


def test_goal_bearings_have_frozen_distance_and_sign():
    assert BEARINGS_DEG == (-20, -10, 0, 10, 20)
    goals = [goal_for_bearing(b) for b in BEARINGS_DEG]
    assert all(np.linalg.norm(g) == pytest.approx(0.6) for g in goals)
    assert goals[0][1] < 0 < goals[-1][1]


def test_both_banks_keep_the_same_global_goal_and_correct_endpoint():
    goal = goal_for_bearing(20)
    a = bezier_candidate("A", goal, 0.12, "left", "common")
    b = bezier_candidate("B", goal, 0.18, "left", "goal_connected")
    normal = np.array([-goal[1], goal[0]]) / np.linalg.norm(goal)
    assert np.allclose(a["dense"][0], [0, 0])
    assert np.allclose(a["waypoints"][-1], goal)
    assert np.allclose(b["waypoints"][-1], goal + 0.18 * normal)
    assert np.allclose(a["global_goal"], b["global_goal"])
    assert a["waypoints"].shape == b["waypoints"].shape == (8, 2)
    assert np.max(np.linalg.norm(np.diff(a["dense"], axis=0), axis=1)) <= 0.02
    assert path_sha(a["waypoints"]) == path_sha(a["waypoints"].copy())


def test_common_endpoint_parameter_is_peak_lateral_amplitude():
    goal = goal_for_bearing(-10)
    row = bezier_candidate("A", goal, 0.08, "right", "common")
    normal = np.array([-goal[1], goal[0]]) / np.linalg.norm(goal)
    signed = row["dense"] @ normal
    reference = np.linspace(0, 1, len(row["dense"]))[:, None] * goal
    offset = np.sum((row["dense"] - reference) * normal, axis=1)
    assert min(offset) == pytest.approx(-0.08, abs=5e-4)
    assert signed.shape == offset.shape


def test_goal_connected_terminal_tangent_points_toward_global_goal():
    goal = goal_for_bearing(0)
    b = bezier_candidate("B", goal, 0.12, "left", "goal_connected")
    tangent = b["controls"][-1] - b["controls"][-2]
    remaining = goal - b["waypoints"][-1]
    assert np.dot(tangent, remaining) > 0
    assert tangent[0] > 0


def test_geometry_gate_catches_reverse_and_outside_without_rejecting_footprint():
    goal = goal_for_bearing(0)
    valid = bezier_candidate("A", goal, 0.04, "left", "common")
    assert audit_geometry(valid, goal, [0, 0, 0])["hard_valid"]
    outside = bezier_candidate("B", goal, 1.20, "left", "goal_connected")
    assert not audit_geometry(outside, goal, [0, 0, 0])["hard_valid"]
    assert "endpoint_outside_dem" in audit_geometry(outside, goal, [0, 0, 0])["invalid_reasons"]


def test_corridor_terrain_difference_requires_common_known_cells():
    goal = goal_for_bearing(0)
    direct = bezier_candidate("A", goal, 0.0, "direct", "common")
    left = bezier_candidate("A", goal, 0.12, "left", "common")
    height = np.zeros((20, 20))
    height[11:16, 13:18] = 0.1
    mask = np.ones((20, 20))
    report = corridor_audit(left, height, mask, {"main_track_sprocket": [[-0.33, -0.17], [0.33, 0.17]]})
    assert report["centerline_known_fraction"] > 0.95
    mask[:] = 0
    unknown = corridor_audit(left, height, mask, {"main_track_sprocket": [[-0.33, -0.17], [0.33, 0.17]]})
    assert unknown["centerline_known_fraction"] == 0
    assert unknown["known_height_profile_m"] == []


def test_anchor_freeze_uses_rank_and_rejects_missing_supply():
    rows = {
        "first": [
            {"rank": 2, "pose_id": "bad", "virtual_class": "transition", "physical_supply": False},
            {"rank": 3, "pose_id": "t", "virtual_class": "transition", "physical_supply": True},
            {"rank": 4, "pose_id": "d", "virtual_class": "differential", "physical_supply": True},
            {"rank": 5, "pose_id": "g", "virtual_class": "graded", "physical_supply": True},
            {"rank": 7, "pose_id": "f", "virtual_class": "flat-only", "physical_supply": True},
        ],
        "second": [
            {"rank": 1, "pose_id": "t2", "virtual_class": "transition", "physical_supply": True},
            {"rank": 2, "pose_id": "d2", "virtual_class": "differential", "physical_supply": True},
            {"rank": 3, "pose_id": "g2", "virtual_class": "graded", "physical_supply": True},
            {"rank": 8, "pose_id": "f2", "virtual_class": "flat-only", "physical_supply": True},
        ],
    }
    selected = freeze_anchors(rows)
    assert [x["pose_id"] for x in selected] == ["t", "d", "g", "f", "t2", "d2", "g2", "f2"]
    rows["second"][-1]["physical_supply"] = False
    with pytest.raises(ValueError, match="missing physical supply"):
        freeze_anchors(rows)


def test_parameter_selection_is_global_and_lexicographic():
    scores = [
        {"parameters": [0.04, 0.08, 0.12], "hard_valid_groups": 40, "follower_compatible": 120, "early_separation": 1.0},
        {"parameters": [0.08, 0.12, 0.16], "hard_valid_groups": 39, "follower_compatible": 200, "early_separation": 9.0},
        {"parameters": [0.04, 0.12, 0.16], "hard_valid_groups": 40, "follower_compatible": 110, "early_separation": 9.0},
    ]
    assert select_parameters(scores)["parameters"] == [0.04, 0.08, 0.12]


def test_pairwise_command_estimate_is_initial_only_and_signed():
    left = {'initial_command_estimate': {'linear_x': .2, 'angular_z': .1}}
    right = {'initial_command_estimate': {'linear_x': .2, 'angular_z': -.1}}
    same = pairwise_command_estimate(left, left)
    opposite = pairwise_command_estimate(left, right)
    assert same['initial_normalized_command_rms_estimate'] == 0
    assert opposite['initial_angular_difference_radps'] == pytest.approx(.2)
    assert opposite['initial_normalized_command_rms_estimate'] > 0
    assert opposite['time_scope'] == 'initial_command_only_not_rollout_RMS'


def test_corridor_pairwise_rms_requires_common_known_profile():
    goal=goal_for_bearing(0)
    direct=bezier_candidate('A',goal,0,'direct','common')
    left=bezier_candidate('A',goal,.12,'left','common')
    height=np.repeat(np.arange(20,dtype=float)[:,None]*.01,20,axis=1)
    mask=np.ones((20,20))
    boxes={'main_track_sprocket':[[-.33,-.17],[.33,.17]]}
    a=corridor_audit(direct,height,mask,boxes)
    b=corridor_audit(left,height,mask,boxes)
    pair=pairwise_audit(direct,left,a,b)
    assert pair['known_corridor_profile_rms_m'] is not None
    assert pair['common_known_corridor_bins'] >= 5
    unknown=corridor_audit(left,height,np.zeros((20,20)),boxes)
    assert pairwise_audit(direct,left,a,unknown)['known_corridor_profile_rms_m'] is None


def test_early_geometry_uses_common_absolute_arc_stations():
    short = {'stations_m': np.array([0., .5, 1.]), 'dense': np.array([[0., 0.], [.5, 0.], [1., 0.]]), 'arc_length_m': 1.}
    long = {'stations_m': np.array([0., 1., 2.]), 'dense': np.array([[0., 0.], [1., 0.], [2., 0.]]), 'arc_length_m': 2.}
    dem = {'corridors': {'main_track_sprocket': {}}, 'known_height_profile_m': []}
    pair = pairwise_audit(short, long, dem, dem)
    assert pair['early_0_5m_geometry_rms_m'] == pytest.approx(0.)
    assert pair['full_geometry_rms_m'] == pytest.approx(0.)
    assert pair['arc_length_difference_m'] == pytest.approx(1.)


def test_corridor_class_proxy_keeps_unknown_distinct_from_flat():
    path = bezier_candidate('A', goal_for_bearing(0), 0, 'direct', 'common')
    flat = corridor_class_proxy(path, np.zeros((20,20)), np.ones((20,20)))
    unknown = corridor_class_proxy(path, np.zeros((20,20)), np.zeros((20,20)))
    assert flat['class_proxy'] == 'flat-only'
    assert unknown['class_proxy'] == 'unknown'
