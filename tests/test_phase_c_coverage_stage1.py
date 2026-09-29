import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_neighborhood_filters_episode_mask_height_and_roughness():
    from eat_nav_phase_c_coverage_stage1 import matching_neighbor
    a={'episode_key':[1,2,3],'terrain_type':'boxes','goal':np.array([1.,0.]),'dem':np.stack([np.stack([np.zeros((2,2)),np.ones((2,2))])]*4)}
    b={**a,'episode_key':[1,2,4]}
    cfg={'goal_distance_m':.1,'goal_direction_deg':30,'min_mask_iou':.8,'max_dem_rmse_m':.05,'roughness_max_abs_difference_m':.05}
    assert matching_neighbor(a,b,cfg) is not None
    assert matching_neighbor(a,a,cfg) is None
    c={**b,'dem':b['dem'].copy()};c['dem'][:,0]=.1
    assert matching_neighbor(a,c,cfg) is None


def test_path_residual_has_zero_endpoint():
    from eat_nav_phase_c_coverage_stage1 import residual
    path=np.array([[.1,.2],[.5,.3],[.9,.4]])
    r=residual(path)
    assert np.allclose(r[-1],0)
    assert np.allclose(path,(np.arange(1,4)/3)[:,None]*path[-1]+r)


def test_numpy_anchor_index_serializes_as_json_integer():
    import json
    from eat_nav_phase_c_coverage_stage1 import python_index
    assert json.dumps({'anchor_view_index':python_index(np.int64(7))})=='{"anchor_view_index": 7}'
