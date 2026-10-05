"""StateGroup-level comparison and frozen bootstrap gate."""
from eat_nav_g40_boundary_analyze import compare_rankings, stability_gate


def test_identical_rankings_pass_all_descriptive_comparisons():
    a=[0.,1.,2.,3.,4.,5.,6.]
    result=compare_rankings(a,a)
    assert result['kendall']==1.
    assert result['spearman']==1.
    assert result['top1_agreement']
    assert result['top3_overlap']==3
    assert result['direct_sign_agreement']==1.


def test_complete_reversal_detected_without_pair_independence():
    a=[0.,1.,2.,3.,4.,5.,6.]
    b=list(reversed(a))
    result=compare_rankings(a,b)
    assert result['kendall']==-1.
    assert result['pairwise_sign_agreement']==0.
    assert result['wholesale_reversal']


def test_gate_fails_if_any_frozen_condition_fails():
    good={'kendall':.9,'top1_agreement':True,'top3_overlap':3,
          'direct_sign_agreement':1.,'wholesale_reversal':False}
    assert stability_gate([good]*7)['status']=='g40_bootstrap_stable'
    bad=dict(good,top1_agreement=False)
    assert stability_gate([good]*5+[bad]*2)['status']=='g40_bootstrap_sensitive'
