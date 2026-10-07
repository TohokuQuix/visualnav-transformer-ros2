import numpy as np

from eat_nav_corridor_report import (paired_group_bootstrap, selection_metrics,
                                     _decision, independent_interaction_families,
                                     validate_fit_coverage, validate_scaler_parity)


def test_pareto_constant_prediction_exposes_low_precision_and_large_set():
    predicted = {'a': [1., 1.], 'b': [1., 1.], 'c': [1., 1.]}
    actual = {'a': [3., 3.], 'b': [1., 2.], 'c': [2., 1.]}
    result = selection_metrics(predicted, actual)
    assert result['pareto_recall'] == 1
    assert result['pareto_precision'] == 1 / 3
    assert result['predicted_pareto_size'] == 3
    assert result['all_candidates_tied'] is True


def test_bootstrap_resamples_group_differences_not_candidate_rows():
    rows = [{'group': 'g1', 'difference': -1.}, {'group': 'g1', 'difference': -1.},
            {'group': 'g2', 'difference': 1.}]
    result = paired_group_bootstrap(rows, 'difference', seed=123, repeats=100)
    assert result['group_count'] == 2
    assert np.isclose(result['effect'], 0.)
    assert result['interpretation'] == 'insufficient_support'


def test_architecture_signal_requires_both_primary_mae_heads():
    comparisons = ('A2-A1', 'A3-A1', 'A3-A2', 'A4-A3')
    def evidence(state):
        return {'group_count': 14, 'interpretation': state, 'ci95': [-1, -.1]}
    paired = {'E-anchor': {'reward_mae': {key: evidence('negative') for key in comparisons},
                           'goal_mae': {key: evidence('inconclusive') for key in comparisons}}}
    transition = {'transition_corridor': {'E-anchor': {
        'oracle_reward_regret': {'A3-A1': evidence('negative')},
        'reward_pair_accuracy': {'A3-A1': evidence('positive')}}}}
    status = {'group_support': {f'g{i}': {'cluster_count': 1, 'class': 'transition_corridor'}
                                for i in range(28)}}
    result = _decision(status, paired, transition, {'differential_corridor': 1,
                                                'transition_corridor': 12})
    assert result['a3_corridor_cnn_adds_heldout_signal'] == 'inconclusive'
    assert result['terrain_conditioned_candidate_selection_supported'] == 'insufficient_terrain_interaction_support'


def test_distinct_corridor_groups_sharing_one_anchor_count_as_one_family():
    support = {'left': {'class': 'transition_corridor', 'cluster_count': 5},
               'right': {'class': 'transition_corridor', 'cluster_count': 6}}
    records = [{'group': 'left', 'family': 'anchor'},
               {'group': 'right', 'family': 'anchor'}]
    assert independent_interaction_families(support, records) == {'anchor'}


def test_fit_coverage_rejects_missing_architecture_seed_combination():
    folds = [{'fold_id': 'f', 'mode': 'E-anchor'}]
    config = {'models': {'variants': ['A1', 'A2']}, 'training': {'seeds': [123]}}
    meta = [{'fold_id': 'f', 'mode': 'E-anchor', 'variant': 'A1', 'seed': 123}]
    import pytest
    with pytest.raises(ValueError, match='fit coverage'):
        validate_fit_coverage(meta, folds, config)


def test_scaler_parity_rejects_architecture_specific_preprocessing():
    base = {'mode': 'E-anchor', 'fold_id': 'f', 'input_scaler_sha256': 'same',
            'target_scaler_sha256': 'same', 'profile_scaler_sha256': 'same',
            'height_scaler': [0., 1.]}
    validate_scaler_parity([base, dict(base)])
    other = dict(base, target_scaler_sha256='different')
    import pytest
    with pytest.raises(ValueError, match='scaler parity'):
        validate_scaler_parity([base, other])
