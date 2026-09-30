import pytest

from visualnav_transformer.eat_nav.evaluator_readiness_analysis import oracle_curve, readiness


def test_oracle_recall_uses_valid_bank_in_denominator():
    rows = [
        {'candidate_id': 'bank-nominal', 'kind': 'nominal', 'constraint_valid': True, 'delta_v': 0.0},
        {'candidate_id': 'nomad-seed-1001', 'kind': 'nomad', 'constraint_valid': True, 'delta_v': .10},
        {'candidate_id': 'nomad-seed-1002', 'kind': 'nomad', 'constraint_valid': True, 'delta_v': .12},
        {'candidate_id': 'bank-strong_left', 'kind': 'bank', 'constraint_valid': True, 'delta_v': .20},
        {'candidate_id': 'bank-strong_right', 'kind': 'bank', 'constraint_valid': False, 'delta_v': .40},
    ]
    result = oracle_curve(rows, (1, 2), epsilon=.01)
    assert result['feasible_union_best']['candidate_id'] == 'bank-strong_left'
    assert [x['recall'] for x in result['k']] == [False, False]
    assert result['k'][1]['regret'] == pytest.approx(.08)


def test_readiness_is_state_macro_not_candidate_count():
    states = [{'state_id': str(i), 'delta_v_range': .06 if i < 4 else .01,
               'advantage_range': .06 if i < 4 else .01, 'drawdown_range': .0,
               'nomad_distinct_ranks': 2 if i < 4 else 1} for i in range(7)]
    assert readiness(states, min_states=4)['ready']
    states[3]['delta_v_range'] = .01
    assert not readiness(states, min_states=4)['ready']
