import numpy as np
from visualnav_transformer.eat_nav.quality_gate_evaluation import episode_cluster_ci, terrain_support


def test_bootstrap_resamples_episode_clusters_not_windows():
    values=np.array([1.,1.,0.])
    episodes=['a','a','b']
    ci,samples=episode_cluster_ci(values,episodes,seed=1,repeats=100)
    assert set(np.round(samples,5)) <= {0.,round(2/3,5),1.}
    assert len(ci)==2


def test_sparse_terrain_q_bin_is_na_but_common_support_is_reported():
    distributions={'train':{'a':{0:50,1:50,2:50,3:50},'b':{0:50,1:50,2:50,3:50}},
                   'val':{'a':{0:50,1:50,2:50,3:6},'b':{0:50,1:50,2:50,3:50}},
                   'test':{'a':{0:50,1:50,2:50,3:50},'b':{0:50,1:50,2:50,3:50}}}
    result=terrain_support(distributions,min_bin_count=30)
    assert result['common_terrain_types']==['a','b']
    assert result['by_split']['val']['a']['status']=='insufficient support'
    assert result['by_split']['val']['a']['metric'] is None
    assert result['by_split']['val']['b']['status']=='supported'


def test_pair_bootstrap_uses_both_episode_ids():
    from visualnav_transformer.eat_nav.quality_gate_evaluation import pair_cluster_ci
    cross=np.array([2.,2.])
    same=np.array([1.,1.])
    episodes=[('a','b'),('c','d')]
    result=pair_cluster_ci(cross,episodes,same,episodes,seed=3,repeats=200)
    assert result['observed_mean_difference']==1.0
    assert result['ci_95'][0]==1.0
    assert result['ci_95'][1]==1.0


def test_trajectory_errors_use_valid_points_only_and_physical_xy():
    from visualnav_transformer.eat_nav.quality_gate_evaluation import trajectory_errors
    prediction=np.zeros((1,8,2),dtype=np.float32)
    target=np.zeros_like(prediction)
    prediction[0,0,0]=1.0
    prediction[0,1,0]=99.0
    mask=np.array([[True]+[False]*7])
    result=trajectory_errors(prediction,target,mask)
    assert result['mse'][0]==0.5
    assert result['rms_m'][0]==1.0
    assert result['endpoint_m'][0]==1.0
    assert result['path_length_error_m'][0]==1.0
