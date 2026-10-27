"""Degenerate-label and paired-candidate evaluation contracts."""
import importlib
import numpy as np
import pytest
from test_fall_stall_learning import learning_arrays


def api():
    return importlib.import_module('visualnav_transformer.eat_nav.fall_stall_metrics')


def test_tied_average_precision_and_degenerate_metrics():
    result=api().binary_metrics(np.array([1,0,1,0]),np.array([.2]*4))
    assert result['average_precision']==.5
    assert result['precision'] is None and result['recall']==0
    assert result['brier']==pytest.approx(.34)
    assert sum(b['count'] for b in result['reliability'])==4
    result=api().binary_metrics(np.array([0,0]),np.array([.1,.7]))
    assert result['average_precision'] is None and result['recall'] is None
    assert result['false_positive_rate']==.5


def test_pair_ties_unknown_selection_and_unsupported_heads():
    arrays=learning_arrays(2);prob=np.full((2,4,4),.2);progress=arrays['progress_targets'].copy()
    arrays['event_mask'][0,3,3,0]=False
    result=api().evaluate_predictions(arrays,[0,1],[11,22],prob,progress,[True,True,True,False])
    assert result['events']['turn_stall']['metrics'] is None
    assert result['events']['tip_over']['pair_ranking']['agreement']==.5
    assert result['selection']['eligible_groups']==1
    assert result['selection']['excluded_unknown_tip_groups']==1
    assert result['selection']['observed_tip_overs']==0
    assert result['progress']['10s']['mae_m']==0
    assert result['events']['tip_over']['support']['unknown']==1
    assert result['events']['tip_over']['support']['positive_base_seeds']==2


def test_average_precision_with_multiple_tie_blocks():
    # Scores .9: one positive/one negative; .1: one positive. AP=.5*.5+.5*(2/3).
    assert api().binary_metrics(np.array([1,0,1]),np.array([.9,.9,.1]))['average_precision']==pytest.approx(7/12)


def test_selection_and_pair_order_use_logits_when_sigmoid_saturates():
    arrays=learning_arrays(1);p=np.ones((1,4,4));scores=np.ones((1,4,4))*50
    scores[0,:,0]=[50,60,40,55]
    result=api().evaluate_predictions(arrays,[0],[11],p,arrays['progress_targets'],[True]*4,risk_scores=scores)
    assert result['selection']['selected_candidates'][0]['candidate_index']==2
    assert result['events']['tip_over']['pair_ranking']['agreement']==1.
