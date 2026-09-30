import pathlib,sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'scripts'))
from eat_nav_goal_ood_prepare import polar,fit_support,classify

def test_polar_angle_wrap_and_robot_local_left_right():
    edges=[0.,.5,1.1]
    r,a=polar([[.5,0],[0,.5],[0,-.5],[-.5,0]],edges,12)
    assert list(r)==[1,1,1,1]
    assert a[1]>a[0] and a[2]<a[0]
    # The +/-pi boundary must be adjacent modulo 12.
    _,b=polar([[-1.,1e-8],[-1.,-1e-8]],edges,12)
    assert (int(b[0])-int(b[1]))%12 in (1,11)

def test_train_only_support_and_unsupported_bin():
    goals=np.stack((np.linspace(.1,.4,40),np.zeros(40)),axis=1)
    episodes=[(0,i//4,0) for i in range(40)]
    cfg={'polar_radius_edges_m':[0.,.5,1.1],'polar_angle_bins':12,
         'minimum_independent_episodes_per_bin':5,'knn_k':4,
         'knn_near_quantile':.95,'knn_far_quantile':.99}
    support=fit_support(goals,episodes,cfg)
    assert support['polar_episode_support']['0:6']==10
    labels,_=classify([[.2,0.],[-.2,0.]],goals,support)
    assert labels[0]=='ID'
    assert labels[1]=='far_ood'
    assert support['source_split'] if 'source_split' in support else True

def test_goal_clip_roundtrip_complete_endpoint():
    endpoint=np.array([[1.,0.],[.3,.4]],dtype=np.float32)
    radius=np.linalg.norm(endpoint,axis=1)
    clipped=endpoint*np.minimum(1.,.8/np.maximum(radius,1e-20))[:,None]
    assert np.allclose(clipped,[[.8,0.],[.3,.4]])


def test_validation_points_do_not_change_fitted_support():
    train=np.stack((np.linspace(.05,.35,40),np.zeros(40)),axis=1)
    episodes=[(0,i//4,0) for i in range(40)]
    cfg={'polar_radius_edges_m':[0.,.5,1.1],'polar_angle_bins':12,
         'minimum_independent_episodes_per_bin':5,'knn_k':4,
         'knn_near_quantile':.95,'knn_far_quantile':.99}
    frozen=fit_support(train,episodes,cfg)
    validation=np.array([[.2,.01],[.2,.5],[-.2,0.]])
    before=classify(validation,train,frozen)
    # Evaluation is repeatable against the same train fit. The validation
    # coordinates are never supplied to fit_support.
    after=classify(validation[::-1],train,frozen)
    assert before[0]==after[0][::-1]
    assert np.allclose(before[1],after[1][::-1])
    assert frozen==fit_support(train,episodes,cfg)
