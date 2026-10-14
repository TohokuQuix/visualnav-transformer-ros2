import json

import pytest

from visualnav_transformer.eat_nav.motion_14_stage_a import (
    PILOT_GROUP, build_pilot_schedule, candidate_role, validate_schedule,
    geometry_audit,
)


def test_stage_a_frozen_schedule():
    rows = build_pilot_schedule()
    assert len(rows) == 32
    assert sum(row['stage'] == 'prepath' for row in rows) == 4
    assert sum(row['stage'] == 'candidate' for row in rows) == 28
    assert {row['terrain_seed'] for row in rows} == {PILOT_GROUP['terrain_seed']}
    assert {row['flipper_initialization_mode'] for row in rows} == {
        'front_folded', 'rear_folded'}
    for posture in ('front_folded', 'rear_folded'):
        subset = [row for row in rows if row['flipper_initialization_mode'] == posture]
        assert [row['repeat_index'] for row in subset[:2]] == [0, 1]
        assert {row['candidate_id'] for row in subset[2:]} == {
            candidate_role(direction, angle)
            for direction in (1, -1) for angle in (0, -30, 30, -60, 60, -90, 90)}
    validate_schedule(rows)
    bad = json.loads(json.dumps(rows))
    bad[-1]['candidate_id'] = bad[-2]['candidate_id']
    with pytest.raises(ValueError):
        validate_schedule(bad)


def test_fourteen_geometry_and_legacy_sha():
    result = geometry_audit()
    assert result['pass']
    assert len(result['candidates']) == 14
    for row in result['candidates']:
        assert row['pass']
        if abs(row['angle_deg']) <= 30:
            assert row['legacy_path_sha256'] == row['path_sha256']
