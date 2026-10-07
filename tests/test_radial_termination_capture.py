import pytest

from visualnav_transformer.eat_nav.radial_termination_capture import build_reset_events, verify_pose_coverage


def fixture(**changes):
    values = dict(selected={5: 'rail-a00', 20: 'step-a02'}, physics_step=140,
                  done_ids=[5], terminated_ids=[5], truncated_ids=[],
                  term_flags={'tilt_over': [5], 'time_out': []}, timeout_terms={'time_out'},
                  pre_states={5: {'root_state_w': [0.0] * 13}},
                  actor_values={5: [0.0] * 22}, episode_lengths={5: 14},
                  reset_counts={5: 0, 20: 0}, trace_steps={5: 140, 20: 140},
                  dem_frame_ids={5: ['frame-00', 'frame-01']},
                  contacts={5: {'bounds_valid': True, 'finite': True,
                                'buffer_full_or_truncation_suspected': False,
                                'buffer_capacity': 4096}})
    values.update(changes)
    return values


def test_one_selected_physical_termination_has_exact_pre_reset_identity():
    event, = build_reset_events(**fixture())
    assert (event['pose_id'], event['env_id'], event['termination_step']) == ('rail-a00', 5, 140)
    assert event['termination_reason'] == ['tilt_over']
    assert event['episode_id'] == 0 and event['reset_counter_before'] == 0
    assert event['reset_counter_after'] == 1
    assert event['post_reset_state'] is False
    assert event['last_valid_step'] == 140


def test_two_selected_terminations_are_independent():
    data = fixture(done_ids=[5, 20], terminated_ids=[5, 20],
                   term_flags={'tilt_over': [5, 20], 'time_out': []},
                   pre_states={5: {'root_state_w': [0.0] * 13}, 20: {'root_state_w': [1.0] * 13}},
                   actor_values={5: [0.0] * 22, 20: [1.0] * 22},
                   episode_lengths={5: 14, 20: 14},
                   dem_frame_ids={5: [], 20: []}, contacts={5: fixture()['contacts'][5],
                                                             20: fixture()['contacts'][5]})
    assert [x['env_id'] for x in build_reset_events(**data)] == [5, 20]


def test_nonselected_reset_does_not_close_selected_streams():
    data = fixture(done_ids=[7], terminated_ids=[7], term_flags={'tilt_over': [7], 'time_out': []})
    assert build_reset_events(**data) == []


def test_done_and_term_flags_must_agree():
    with pytest.raises(ValueError, match='term.*done|done.*term'):
        build_reset_events(**fixture(term_flags={'tilt_over': [], 'time_out': []}))


@pytest.mark.parametrize('change', [
    {'actor_values': {5: [0.0] * 21}},
    {'trace_steps': {5: 139}},
    {'episode_lengths': {5: 13}},
    {'contacts': {5: {'bounds_valid': False}}},
])
def test_missing_or_post_reset_evidence_is_invalid(change):
    with pytest.raises(ValueError):
        build_reset_events(**fixture(**change))


def test_timeout_and_unknown_reason_are_not_physical_termination():
    with pytest.raises(ValueError, match='timeout'):
        build_reset_events(**fixture(terminated_ids=[], truncated_ids=[5],
                                     term_flags={'tilt_over': [], 'time_out': [5]}))
    with pytest.raises(ValueError, match='unknown'):
        build_reset_events(**fixture(term_flags={'tilt_over': [], 'time_out': []}))


def test_pose_coverage_rejects_duplicate_missing_and_extra_results():
    assert verify_pose_coverage(['a','b'], [{'pose_id':'a'}, {'pose_id':'b'}]) is None
    with pytest.raises(ValueError): verify_pose_coverage(['a','b'], [{'pose_id':'a'}, {'pose_id':'a'}])
    with pytest.raises(ValueError): verify_pose_coverage(['a','b'], [{'pose_id':'a'}])
    with pytest.raises(ValueError): verify_pose_coverage(['a','b'], [{'pose_id':'a'}, {'pose_id':'b'}, {'pose_id':'c'}])


def test_selected_retermination_after_stream_close_is_nonselected_for_capture():
    assert build_reset_events(**fixture(selected={20:'step-a02'})) == []
