"""Integrity tests for the frozen, offline goal sweep (when artifacts are available)."""
import collections
import json
import os
import pathlib

import numpy as np
import pytest

from visualnav_transformer.eat_nav.feasible_transfer import to_increments, to_waypoints


DOC = pathlib.Path(os.environ.get(
    "EAT_GOAL_OOD_ARTIFACT_ROOT",
    "/tmp/eatnav-vnt-goal-ood/docs/quality-identifiability-gate/candidate-coverage/goal-ood",
))
RAW = pathlib.Path(os.environ.get("EAT_GOAL_OOD_RAW_ROOT", "/tmp/eatnav-candidate-coverage/goal-ood"))


@pytest.fixture(scope="module")
def frozen():
    if not (DOC / "goal-sweep-inputs.json").exists() or not (RAW / "selected-states.npz").exists():
        pytest.skip("frozen diagnostic artifacts are not present")
    return (
        json.loads((DOC / "state-selection.json").read_text())["states"],
        json.loads((DOC / "goal-sweep-inputs.json").read_text()),
    )


def test_split_state_episode_uniqueness_and_no_heldout(frozen):
    states, sweep = frozen
    assert len(states) == 64
    assert collections.Counter(s["split"] for s in states) == {"train": 32, "val": 32}
    assert len({(s["split"], tuple(s["episode_key"])) for s in states}) == 64
    assert {s["split"] for s in states} == {"train", "val"}
    assert sweep["input_count"] == 832 and sweep["path_count"] == 6656


def test_all_goals_share_same_dem_and_paired_seed_list(frozen):
    states, sweep = frozen
    assert sweep["seeds"] == list(range(1001, 1009))
    with np.load(RAW / "selected-states.npz") as selected:
        assert selected["dem"].shape == (64, 4, 2, 20, 20)
        assert np.isfinite(selected["dem"]).all()
        for i, state in enumerate(states):
            rows = sweep["inputs"][13 * i:13 * (i + 1)]
            assert len(rows) == 13
            assert [r["goal_order"] for r in rows] == list(range(13))
            assert {r["state_id"] for r in rows} == {state["state_id"]}
            assert {r["split"] for r in rows} == {state["split"]}
            assert {r["dem_sha256"] for r in rows} == {state["dem_sha256"]}
            assert collections.Counter(r["declared_class"] for r in rows) == {
                "logged": 1, "ID": 4, "near_ood": 4, "far_ood": 4,
            }
            assert np.allclose(rows[0]["goal_xy"], selected["logged_goals"][i], atol=1e-7)


def test_inference_identity_and_goal_only_change(frozen):
    states, sweep = frozen
    # Inspect both splits; the inference loop reuses one DEM/state tensor for all goals.
    for i in (0, 31, 32, 63):
        state = states[i]
        path = RAW / "inference" / f"{state['state_id']}.npz"
        if not path.exists():
            pytest.skip("inference array not present")
        with np.load(path) as result:
            assert result["paths"].shape == (13, 8, 8, 2)
            assert result["goals"].shape == (13, 2)
            assert result["seeds"].tolist() == sweep["seeds"]
            expected = np.asarray([r["goal_xy"] for r in sweep["inputs"][13 * i:13 * (i + 1)]])
            assert np.allclose(result["goals"], expected, atol=1e-7)
            assert np.isfinite(result["paths"]).all()


def test_train_fitted_thresholds_and_unsupported_na(frozen):
    _, sweep = frozen
    thresholds = json.loads((DOC / "goal-support-thresholds.json").read_text())
    decision = json.loads((DOC / "offline-gate-decision.json").read_text())
    assert thresholds["source_split"] == "complete_only_train"
    assert thresholds["near_quantile"] == .95 and thresholds["far_quantile"] == .99
    assert thresholds["near_distance_m"] < thresholds["far_distance_m"]
    assert decision["scope"]["heldout_used"] is False
    assert decision["scope"]["simulator_rollouts_started"] is False
    assert {r["declared_class"] for r in sweep["inputs"]} == {
        "logged", "ID", "near_ood", "far_ood",
    }
    assert "unsupported" not in json.loads((DOC / "goal-ood-results.json").read_text())["macro"]


def test_increment_cumulative_conversion_and_clipped_endpoint(frozen):
    states, _ = frozen
    with np.load(RAW / "selected-states.npz") as selected:
        for i in (0, 31, 32, 63):
            action = selected["actions"][i]
            endpoint = to_waypoints(action)[-1]
            restored = endpoint * min(1., .8 / max(float(np.linalg.norm(endpoint)), 1e-20))
            assert np.allclose(to_increments(to_waypoints(action)), action, atol=1e-6)
            assert np.allclose(restored, states[i]["logged_goal"], atol=1e-6)


def test_endpoint_error_direction_and_radial_pair_against_raw_paths(frozen):
    states, sweep = frozen
    adherence = json.loads((DOC / "goal-adherence-results.json").read_text())
    directions = json.loads((DOC / "goal-direction-pairs.json").read_text())
    for i in (0, 31, 32, 63):
        with np.load(RAW / "inference" / f"{states[i]['state_id']}.npz") as result:
            path, goals = result["paths"], result["goals"]
            errors = np.linalg.norm(path[1:5, :, -1] - goals[1:5, None], axis=-1)
            assert np.isclose(np.median(errors), adherence["states"][i]["classes"]["ID"]["error"]["median"])
            expected = np.asarray([r["goal_xy"] for r in sweep["inputs"][13 * i:13 * (i + 1)]])
            assert np.allclose(goals, expected, atol=1e-7)
            pairs = {
                "near_left_vs_right": path[6, :, -1, 1] - path[5, :, -1, 1],
                "far_left_vs_right": path[10, :, -1, 1] - path[9, :, -1, 1],
                "ID_long_vs_short": np.linalg.norm(path[2, :, -1], axis=1) - np.linalg.norm(path[1, :, -1], axis=1),
            }
            for name, delta in pairs.items():
                saved = directions["states"][i]["pairs"][name]
                assert np.isclose(np.mean(delta > 0), saved["success_fraction"])
                assert np.isclose(np.median(delta), saved["endpoint_difference_median_m"])


def test_state_macro_uses_states_not_pooled_paths(frozen):
    adherence = json.loads((DOC / "goal-adherence-results.json").read_text())
    for split in ("train", "val"):
        for goal_class in ("ID", "near_ood", "far_ood"):
            subset = [s["classes"][goal_class] for s in adherence["states"] if s["split"] == split]
            saved = adherence["macro"][f"{split}:{goal_class}"]
            assert len(subset) == saved["states"] == 32
            assert np.isclose(np.mean([s["error"]["median"] for s in subset]), saved["state_macro_error_median_m"])
            assert np.isclose(np.mean([s["within_0p2m_fraction"] for s in subset]), saved["state_macro_within_0p2m"])
