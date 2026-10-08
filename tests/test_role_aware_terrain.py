from visualnav_transformer.eat_nav.role_aware_terrain import classify_role, select_five


def _path(kind, onset=None):
    return {'class_proxy': kind, 'known': True, 'onset_m': onset,
            'known_fraction': 1.0}


def _row(identifier, seed, family, kinds, pair=False):
    return {'anchor_id': identifier, 'seed_label': seed, 'source_family': family,
            'physical_eligible': True, 'hard_valid': True,
            'follower_compatible': True, 'paths': [_path(k, .3 if k == 'transition' else None)
                for k in kinds],
            'pair_count': 1 if pair else 0, 'max_pair_rms_m': .02 if pair else 0,
            'selection_rank': 1}


def test_precedence_and_independent_roles():
    differential = _row('d1', 'first', 'a', ['transition'] * 5, True)
    classified = classify_role(differential)
    assert classified['primary_role'] == 'R-differential'
    assert classified['flags']['R-transition-common']
    rows = [differential,
            _row('d2', 'second', 'b', ['differential'] * 5, True),
            _row('t', 'first', 'c', ['transition'] * 5),
            _row('g', 'second', 'd', ['graded'] * 5),
            _row('f', 'first', 'e', ['flat-only'] * 5)]
    result = select_five([classify_role(x) for x in rows])
    assert result['pass']
    assert len(result['selected']) == 5


def test_missing_separate_graded_fails_without_role_reuse():
    rows = [_row('d1', 'first', 'a', ['graded'] * 5, True),
            _row('d2', 'second', 'b', ['differential'] * 5, True),
            _row('t', 'first', 'c', ['transition'] * 5),
            _row('f', 'second', 'd', ['flat-only'] * 5)]
    result = select_five([classify_role(x) for x in rows])
    assert not result['pass']
    assert result['missing_roles'] == ['R-graded-common']


def test_unknown_and_partial_grade_do_not_pass():
    row = _row('g', 'first', 'a', ['graded'] * 4 + ['unknown'])
    assert classify_role(row)['primary_role'] is None
    row['paths'][-1]['class_proxy'] = 'graded'
    row['paths'][-1]['known'] = False
    assert classify_role(row)['primary_role'] is None


def test_distinct_source_identity_gate():
    rows = [_row('d1', 'first', 'a', ['differential'] * 5, True),
            _row('d2', 'second', 'a', ['differential'] * 5, True),
            _row('t', 'first', 'a', ['transition'] * 5),
            _row('g', 'second', 'a', ['graded'] * 5),
            _row('f', 'first', 'b', ['flat-only'] * 5)]
    result = select_five([classify_role(x) for x in rows])
    assert not result['pass']
    assert result['missing_roles'] == []
