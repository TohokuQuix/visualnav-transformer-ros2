import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_variance_and_effective_rank_nonnegative():
    from eat_nav_phase_c_coverage_sampling import distribution_metrics
    paths=np.array([[[1.,0.],[2.,0.]],[[1.,.2],[2.,.4]],[[1.,-.2],[2.,-.4]]])
    report=distribution_metrics(paths)
    assert report['pairwise_rms_median_m']>0
    assert 1<=report['effective_rank']<=3
    assert report['endpoint_covariance_m2'][1][1]>0


def test_coverage_requires_command_and_path_and_ood():
    from eat_nav_phase_c_coverage_sampling import coverage_decision
    spec={'coverage':{'min_fraction_two_command_clusters':.75,'min_fraction_three_path_clusters':.5,'min_terrain_count_with_two_command_clusters':2,'max_nonfinite':0,'max_train_p95_ood_fraction':.05,'shape_not_endpoint_only':True,'evaluator_input_plausibility_required':True}}
    rows=[{'command_cluster_count':2,'path_cluster_count':3,'terrain_type':t,'ood_count':0,'candidate_count':8,'nonfinite_count':0,'shape_spread_m':.1,'evaluator_input_plausible':True} for t in ['a','b','a','b']]
    assert coverage_decision(rows,spec)['pass']
    rows[0]['ood_count']=8
    assert not coverage_decision(rows,spec)['pass']


def test_instrumented_ddpm_matches_stock_sample():
    import torch
    from visualnav_transformer.eat_nav.config import EATConfig
    from visualnav_transformer.eat_nav.model import EATNav
    from eat_nav_phase_c_coverage_sampling import sampled_trace
    torch.manual_seed(3)
    model=EATNav(EATConfig(quality_enabled=False)).eval()
    dem=torch.zeros((1,4,2,20,20));goal=torch.zeros((1,2))
    with torch.inference_mode():
        expected=model.sample(dem,goal,None,seed=1001)[0].numpy()
        actual,noisy,pred,updated=sampled_trace(model,dem,goal,1001)
    assert np.allclose(actual,expected,atol=1e-6)
    assert noisy.shape==(11,8,2) and pred.shape==(10,8,2) and updated.shape==(10,8,2)
