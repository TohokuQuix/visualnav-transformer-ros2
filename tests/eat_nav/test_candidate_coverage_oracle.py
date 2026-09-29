"""Oracle best-of-K and recall use only evaluated candidate outcomes."""
import pytest
from visualnav_transformer.eat_nav.candidate_coverage_oracle import oracle_best,recall_curve,outcome_spread


def test_recall_curve_uses_pool_best_with_frozen_epsilon():
    rows=[{'candidate_id':'nomad-seed-1001','delta_v':.1},
          {'candidate_id':'nomad-seed-1002','delta_v':.4},
          {'candidate_id':'nomad-seed-1003','delta_v':.2},
          {'candidate_id':'bank-nominal','delta_v':.3}]
    assert oracle_best(rows,['nomad-seed-1001'],'delta_v') == pytest.approx(.1)
    curve=recall_curve(rows,['nomad-seed-1001','nomad-seed-1002','nomad-seed-1003'],
                       metric='delta_v',epsilon=.01,ks=[1,2,3])
    assert [r['recalled'] for r in curve] == [False,True,True]
    assert [r['best'] for r in curve] == pytest.approx([.1,.4,.4])


def test_terminal_none_values_are_explicitly_unsupported():
    rows=[{'candidate_id':'nomad-seed-1001','delta_v':None},
          {'candidate_id':'bank-nominal','delta_v':.3}]
    assert oracle_best(rows,['nomad-seed-1001'],'delta_v') is None
    curve=recall_curve(rows,['nomad-seed-1001'],metric='delta_v',epsilon=.01,ks=[1])
    assert curve[0]['recalled'] is False
    assert outcome_spread(rows,'delta_v')['valid_count']==1
