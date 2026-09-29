"""Stage 0 selection must be reproducible and blind to rollout outcomes."""

import pytest

from visualnav_transformer.eat_nav.candidate_coverage_selection import (
    freeze_pilot_selection,
)


def row(terrain, episode, step, goal, view_index, **extra):
    return {
        "split": "train", "complete": True, "natural_history": True,
        "terrain_type": terrain, "episode_key": [0, view_index, episode],
        "view_index": view_index, "step": step, "local_step": step,
        "goal": [goal, 0.0], **extra,
    }


def test_selection_uses_full_prefix_and_never_outcome_fields():
    rows = [
        row("boxes", 1, 12, .5, 0, reward=999, critic_value=-1),
        row("boxes", 2, 45, .5, 1, reward=-999),
        row("rail", 1, 20, .3, 2),
        row("rail", 2, 19, .29, 3),
    ]
    result = freeze_pilot_selection(rows, terrains=["boxes", "rail", "slope"],
                                    seed=7, min_goal_m=.3, max_full_prefix_steps=40)
    assert {r["terrain_type"] for r in result["selected"]} == {"boxes", "rail"}
    assert result["selected"][0]["step"] == 12
    assert result["na_terrains"] == {"slope": "NA / no state within full-prefix budget"}
    assert result["all_prefix_steps"]["count"] == 4
    assert result["eligible_prefix_steps"]["count"] == 2
    assert "reward" not in str(result) and "critic_value" not in str(result)
    changed = [dict(r, reward=-100000, critic_value=100000) for r in rows]
    assert freeze_pilot_selection(changed, terrains=["boxes", "rail", "slope"],
                                  seed=7, min_goal_m=.3, max_full_prefix_steps=40) == result


def test_selection_rejects_partial_or_nontrain_windows():
    for change in ({"split": "val"}, {"complete": False}, {"natural_history": False}):
        item = row("boxes", 1, 20, .5, 0)
        item.update(change)
        with pytest.raises(ValueError):
            freeze_pilot_selection([item], terrains=["boxes"], seed=7,
                                   min_goal_m=.3, max_full_prefix_steps=40)


def test_selection_rejects_budget_exceeded_and_duplicate_identity():
    rows = [row("boxes", 1, 20, .5, 0), row("rail", 2, 30, .5, 1)]
    with pytest.raises(ValueError, match="budget"):
        freeze_pilot_selection(rows, terrains=["boxes", "rail"], seed=7,
                               min_goal_m=.3, max_full_prefix_steps=40,
                               approved_budget_sec=1, run_count_per_state=16,
                               cost_intercept_sec=46, cost_slope_sec=.83)
    with pytest.raises(ValueError, match="duplicate"):
        freeze_pilot_selection([rows[0], rows[0]], terrains=["boxes"], seed=7,
                               min_goal_m=.3, max_full_prefix_steps=40)


def test_full_prefix_counts_anchor_step_itself():
    result = freeze_pilot_selection([row("boxes", 1, 40, .5, 0)],
                                    terrains=["boxes"], seed=7, min_goal_m=.3,
                                    max_full_prefix_steps=40)
    assert result["selected"] == []
    assert result["all_prefix_steps"]["min"] == 41
