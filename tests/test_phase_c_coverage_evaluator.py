import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_score_spread_reports_candidate_and_ensemble_axes():
    from eat_nav_phase_c_coverage_evaluator import score_spread
    p=np.zeros((3,8,4));p[:,1,0]=2
    p[1]+=0.1;p[2]-=0.1
    r=score_spread(p)
    assert r['heads']['delta_v']['candidate_range']>1.9
    assert r['heads']['advantage_h']['ensemble_disagreement_median']>0
    assert not r['all_same_score']
