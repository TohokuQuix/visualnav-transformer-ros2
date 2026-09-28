import numpy as np
import pytest

from visualnav_transformer.eat_nav.quality_gate_data import (
    assert_split_integrity,
    audit_windows,
    raw_dem_status,
    build_canonical_pair,
    select_overfit_pairs,
)


def _window(index, episode, q, x):
    action = np.zeros((8, 2), dtype=np.float32)
    action[:, 0] = x
    return dict(window_id=f"w{index}", episode_key=("s", episode, 0),
                terrain_seed=episode, terrain_type="boxes", quality=q,
                dem=np.full((4, 2, 2, 2), index, dtype=np.float32),
                goal=np.array([0.2, 0.1], dtype=np.float32), action=action,
                action_mask=np.ones(8, dtype=bool))


def test_split_audit_rejects_episode_or_terrain_seed_leakage():
    a = _window(1, 1, 0.1, 0.1)
    b = _window(2, 2, 0.9, 0.2)
    assert_split_integrity({"train": [a], "val": [b], "test": []})
    with pytest.raises(ValueError, match="terrain seed leakage"):
        assert_split_integrity({"train": [a], "val": [dict(b, terrain_seed=1)], "test": []})
    with pytest.raises(ValueError, match="episode leakage"):
        assert_split_integrity({"train": [a], "val": [dict(b, episode_key=a["episode_key"])], "test": []})


def test_canonical_pair_has_identical_dem_and_goal_but_distinct_targets():
    low = _window(1, 1, 0.1, 0.1)
    high = _window(2, 2, 0.9, 0.2)
    pair = build_canonical_pair(low, high)
    assert np.array_equal(pair[0]["dem"], pair[1]["dem"])
    assert np.array_equal(pair[0]["goal"], pair[1]["goal"])
    assert np.array_equal(pair[0]["dem"], low["dem"])
    assert not np.array_equal(pair[0]["action"], pair[1]["action"])


def test_overfit_selection_rejects_window_reuse_and_caps_episode_uses():
    windows = [_window(i, i // 2, 0.1 if i % 2 == 0 else 0.9, i * 0.01)
               for i in range(12)]
    candidates = [(0, 3, 0.2, 0.3), (0, 5, 0.3, 0.4),
                  (2, 5, 0.2, 0.3), (4, 7, 0.2, 0.3),
                  (6, 9, 0.2, 0.3), (8, 11, 0.2, 0.3)]
    chosen = select_overfit_pairs(windows, candidates, min_rms=0.1,
                                  min_direction=0.1, count=4,
                                  max_episode_uses=1)
    ids = [i for pair in chosen for i in pair[:2]]
    assert len(ids) == len(set(ids))
    episode_ids = [windows[i]["episode_key"] for i in ids]
    assert len(episode_ids) == len(set(episode_ids))


def test_audit_reports_quality_bins_terrain_and_termination():
    a = _window(1, 1, 0.1, 0.1)
    b = _window(2, 2, 0.9, 0.2)
    a["termination_reason"] = ("tilt_over",)
    b["termination_reason"] = ("time_out",)
    result = audit_windows([a, b])
    assert result["q_bin_counts"] == {"0": 1, "3": 1}
    assert result["terrain_type_counts"] == {"boxes": 2}
    assert result["termination_window_counts"] == {"tilt_over": 1, "time_out": 1}
    assert result["episode_count"] == 2


def test_sparse_raw_record_without_dem_is_expected_non_dem_step():
    assert raw_dem_status({"step": 1}) == "not_sampled"
    assert raw_dem_status({"step": 0, "dem": {"transformed": [1], "error": None}}) == "valid"
    assert raw_dem_status({"step": 0, "dem": {"transformed": None, "error": "bad"}}) == "invalid"
