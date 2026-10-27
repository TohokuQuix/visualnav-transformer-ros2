from scripts.eat_nav_paired_long_horizon_pilot_run import (
    CANDIDATES,
    build_schedule,
    build_state_groups,
)


def test_passive_pilot_schedule_is_unique_paired_and_not_failure_induced():
    groups = build_state_groups()
    schedule = build_schedule(groups)

    assert len(groups) == 12
    assert len({group['terrain_seed'] for group in groups}) == 12
    assert len(schedule) == 60
    assert sum(row['stage'] == 'prepath' for row in schedule) == 12
    assert sum(row['stage'] == 'candidate' for row in schedule) == 48
    assert all(row['failure_induction'] is False
               for row in schedule if 'failure_induction' in row)
    for offset in range(0, len(schedule), len(CANDIDATES) + 1):
        rows = schedule[offset:offset + len(CANDIDATES) + 1]
        assert rows[0]['stage'] == 'prepath'
        assert [row['candidate_id'] for row in rows[1:]] == [
            candidate['candidate_id'] for candidate in CANDIDATES]
        assert len({row['terrain_seed'] for row in rows}) == 1