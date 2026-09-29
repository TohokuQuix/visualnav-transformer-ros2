"""Secondary Evaluator ranking is computed only after all oracle outcomes exist."""
import numpy as np
import pytest
from visualnav_transformer.eat_nav.candidate_coverage_evaluator import ranking_diagnostics


def test_ranking_diagnostics_counts_oracle_top3_and_pairwise_accuracy():
    predictions=np.array([
        [[.1],[.3],[.2]],
        [[.1],[.3],[.2]],
        [[.1],[.3],[.2]],
    ])
    result=ranking_diagnostics(predictions,[.1,.4,.2],epsilon=.01)
    assert result['top1_recall'] is True
    assert result['top3_recall'] is True
    assert result['pairwise_accuracy']==pytest.approx(1.)
    assert result['pairwise_support']==3
    assert result['ensemble_spread_mean']==pytest.approx(0,abs=1e-12)


def test_ranking_diagnostics_excludes_missing_oracle_and_no_false_top1():
    predictions=np.array([[[.4],[.2],[.3]]]*3)
    result=ranking_diagnostics(predictions,[None,.5,.1],epsilon=.01)
    assert result['top1_recall'] is False
    assert result['top3_recall'] is True
    assert result['pairwise_support']==1
