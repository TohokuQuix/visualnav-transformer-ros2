"""Offline contracts for the 48-trial direct Path repeat protocol."""

import json

import pytest

from visualnav_transformer.eat_nav.p1_g3_direct_repeat import (
    build_schedule, direct_pairs, extract_trial, repeatability_summary,
    validate_four_follower_final, validate_four_follower_readiness,
    validate_terrain_support,
)


def assignment():
    return {
        "batch_seeds": {
            "first-differential": [2026101101, 2026101102, 2026101103],
            "second-graded": [2026101111, 2026101112, 2026101113],
        },
        "trials": [
            {"anchor_id": group, "batch_index": block,
             "batch_seed": ((2026101101, 2026101102, 2026101103) if group == "first-differential"
                            else (2026101111, 2026101112, 2026101113))[block],
             "slot": slot,
             "condition": ("A", "B", "S", "T")[slot],
             "path_sha256": direct if slot == 0 else "curve"}
            for group, seeds in (("first-differential", range(3)),
                                 ("second-graded", range(3)))
            for block in seeds for slot in range(4)
            for direct in ["direct-sha"]
        ],
    }


def test_schedule_has_12_fresh_processes_and_48_direct_trials():
    schedule = build_schedule(assignment())
    assert len(schedule["processes"]) == 12
    assert [(x["anchor_id"], x["batch_index"], x["repeat_index"])
            for x in schedule["processes"][:4]] == [
                ("first-differential", 0, 0),
                ("first-differential", 0, 1),
                ("first-differential", 1, 0),
                ("first-differential", 1, 1),
            ]
    assert sum(len(x["trials"]) for x in schedule["processes"]) == 48
    assert all([row["slot"] for row in x["trials"]] == [0, 1, 2, 3]
               and {row["path_sha256"] for row in x["trials"]} == {"direct-sha"}
               and {row["condition"] for row in x["trials"]} == {"A"}
               for x in schedule["processes"])
    assert direct_pairs() == ((0, 1), (2, 3))


def test_schedule_rejects_missing_or_inconsistent_direct_reference():
    source = assignment()
    source["trials"] = source["trials"][:-1]
    with pytest.raises(ValueError, match="four slots"):
        build_schedule(source)
    source = assignment()
    source["trials"][0]["condition"] = "B"
    with pytest.raises(ValueError, match="unmodified"):
        build_schedule(source)


def test_schedule_rejects_trial_seed_drift_even_when_seed_list_is_valid():
    source = assignment()
    source["trials"][0]["batch_seed"] = 123
    with pytest.raises(ValueError, match="seed"):
        build_schedule(source)


def test_terrain_support_reports_mesh_and_material_drift():
    rows = [
        {"anchor_id": "first-differential", "batch_seed": 1,
         "mesh_sha256": "a", "material_sha256": "m"},
        {"anchor_id": "first-differential", "batch_seed": 2,
         "mesh_sha256": "b", "material_sha256": "m"},
    ]
    assert validate_terrain_support(rows)["first-differential"] == {
        "same_condition": False, "mesh_sha256": ["a", "b"],
        "material_sha256": ["m"],
    }


def test_repeatability_keeps_process_clusters_and_fresh_replays_separate():
    trials = [
        {"anchor_id": "first-differential", "batch_seed": 7,
         "repeat_index": repeat, "env_id": env, "arrival_time_s": value,
         "two_second_progress_m": value / 10,
         "tracking_mean_m": value / 100, "contact_force_mean_N": value * 2,
         "applied_torque_mean_abs": value * 3,
         "recovery_count": 0, "duplicate_gate_pass": env < 2}
        for repeat, values in enumerate(((1.0, 1.2, 1.4, 1.5),
                                         (1.1, 1.3, 1.6, 1.7)))
        for env, value in enumerate(values)
    ]
    result = repeatability_summary(trials)
    assert result["independent_condition_count"] == 1
    assert result["process_count"] == 2
    assert result["trial_count"] == 8
    assert result["within_process_pairs"] == 4
    assert result["fresh_process_matched_pairs"] == 4
    assert result["arrival_time_s"]["within_process_absolute_difference"]["median"] == pytest.approx(.15)
    assert result["arrival_time_s"]["within_process_signed_difference"]["mean"] == pytest.approx(.15)
    assert result["arrival_time_s"]["fresh_process_absolute_difference"]["max"] == pytest.approx(.2)
    assert result["arrival_time_s"]["fresh_process_signed_difference"]["mean"] == pytest.approx(.15)
    assert result["strata"]["by_env_id"]["0"]["arrival_time_s"]["count"] == 2
    assert result["strata"]["by_seed_bundle"]["first-differential:7"]["arrival_time_s"]["count"] == 8
    assert result["variation_breakdown"]["arrival_time_s"]["within_process_env_spread"]["count"] == 2
    assert result["variation_breakdown"]["arrival_time_s"]["fresh_process_mean_difference"]["count"] == 1
    assert result["duplicate_gate"]["pass_pairs"] == 2
    assert result["duplicate_gate"]["fail_pairs"] == 2
    assert json.loads(json.dumps(result)) == result


def test_censored_arrival_is_not_imputed_as_30_seconds():
    trials = [
        {"anchor_id": "first-differential", "batch_seed": 7,
         "repeat_index": 0, "env_id": env,
         "arrival_time_s": None if env == 0 else 2.0,
         "two_second_progress_m": .1, "tracking_mean_m": .01,
         "contact_force_mean_N": 1.0, "applied_torque_mean_abs": 2.0,
         "recovery_count": 0,
         "duplicate_gate_pass": False}
        for env in range(4)
    ]
    result = repeatability_summary(trials)
    assert result["arrival_time_s"]["observed_count"] == 3
    assert result["arrival_time_s"]["censored_count"] == 1
    assert result["arrival_time_s"]["mean"] == 2.0
    assert result["arrival_time_s"]["within_process_absolute_difference"]["count"] == 1


def test_terrain_fingerprint_change_splits_condition_and_fresh_match():
    rows = [{"anchor_id": "first-differential", "batch_seed": 7,
             "repeat_index": repeat, "env_id": env,
             "terrain_condition_key": "mesh-a" if repeat == 0 else "mesh-b",
             "arrival_time_s": 2.0, "two_second_progress_m": .2,
             "tracking_mean_m": .01, "contact_force_mean_N": 1.0,
             "applied_torque_mean_abs": 2.0, "recovery_count": 0,
             "duplicate_gate_pass": True}
            for repeat in range(2) for env in range(4)]
    result = repeatability_summary(rows)
    assert result["independent_condition_count"] == 2
    assert result["fresh_process_matched_pairs"] == 0


def test_extract_trial_keeps_contact_and_torque_separate_and_excludes_hold():
    rows = [{"control_step": step, "label_exposure": True,
             "root_state": [min(step, 40) * .005, 0.0, 0.0],
             "post_root_state": [min(step + 1, 40) * .005, 0.0, 0.0],
             "contact_net_forces_w": [[3.0, 4.0, 0.0]] if step < 40 else [[300., 0., 0.]],
             "applied_torque": [2.0, -2.0] if step < 40 else [200., 200.],
             "follower_diagnostic": {"linear_recovery": False,
                                     "angular_recovery": False, "mode": "default"}}
            for step in range(81)]
    result = extract_trial(
        rows, {"arrival_step": 40, "right_censored": False},
        [[0, 0], [.75, 0]], [0, 0, 0],
        {"anchor_id": "first-differential", "batch_seed": 7,
         "repeat_index": 0, "env_id": 1, "duplicate_gate_pass": True})
    assert result["arrival_time_s"] == pytest.approx(2.0)
    assert result["two_second_progress_m"] == pytest.approx(.2)
    assert result["tracking_mean_m"] == pytest.approx(0.0, abs=1e-12)
    assert result["contact_force_mean_N"] == 5.0
    assert result["applied_torque_mean_abs"] == 2.0
    assert result["exposure_steps"] == 40


def test_immediate_arrival_has_zero_prearrival_load_exposure():
    rows = [{"control_step": step,
             "root_state": [0., 0., 0.], "post_root_state": [0., 0., 0.],
             "contact_net_forces_w": [[100., 0., 0.]],
             "applied_torque": [50.],
             "follower_diagnostic": {"linear_recovery": False,
                                     "angular_recovery": False, "mode": "goal"}}
            for step in range(41)]
    result = extract_trial(rows, {"arrival_step": 0, "right_censored": False},
                           [[0, 0], [.75, 0]], [0, 0, 0], {})
    assert result["exposure_steps"] == 0
    assert result["contact_force_mean_N"] is None
    assert result["tracking_mean_m"] is None


def test_four_follower_readiness_requires_every_branch_and_zero_prepath_commands():
    ready = {"follower_alive": [True] * 4, "path_subscribers": [1] * 4,
             "tf_ready": [True] * 4, "status_counts": [2] * 4,
             "command_sequence": [0] * 4, "rejected": [0] * 4,
             "clock_publishers": 1, "use_sim_time": True,
             "protocol_errors": []}
    assert validate_four_follower_readiness(ready)["pass"]
    ready["tf_ready"][3] = False
    with pytest.raises(ValueError, match="readiness"):
        validate_four_follower_readiness(ready)


def test_four_follower_final_requires_each_branch_to_emit_every_command():
    snapshot = {"follower_alive": [True] * 4,
                "path_subscribers": [1] * 4, "tf_ready": [True] * 4,
                "status_counts": [5] * 4, "command_sequence": [40] * 4,
                "rejected": [0] * 4, "clock_publishers": 1,
                "use_sim_time": True, "protocol_errors": [],
                "clock_now_ns": 3950000000}
    assert validate_four_follower_final(snapshot, 40, 3950000000)["pass"]
    snapshot["command_sequence"][2] = 39
    with pytest.raises(ValueError, match="final"):
        validate_four_follower_final(snapshot, 40, 3950000000)
