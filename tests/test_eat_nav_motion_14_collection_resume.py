import pytest

from visualnav_transformer.eat_nav.motion_14_collection import build_collection_schedule
from eat_nav_motion_14_collection_resume_v2 import baseline_for


def test_reference_is_same_state_group_and_posture():
    rows = build_collection_schedule()
    first = rows[0]
    second_group = rows[32]
    assert first['terrain_seed'] != second_group['terrain_seed']
    # The frozen v1 coordinator selected this wrong cross-seed reference.
    old_lookup = next(row for row in rows if
                      row['flipper_initialization_mode'] == second_group['flipper_initialization_mode']
                      and row['stage'] == 'prepath' and row['repeat_index'] == 0)
    assert old_lookup == first
    assert old_lookup != second_group
    assert baseline_for(rows, second_group) == second_group
    assert baseline_for(rows, rows[33]) == second_group
    assert baseline_for(rows, rows[34]) == second_group
    assert baseline_for(rows, rows[48]) == rows[48]
    with pytest.raises(ValueError):
        baseline_for(rows[:32], second_group)
