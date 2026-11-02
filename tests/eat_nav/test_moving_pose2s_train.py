import importlib
import numpy as np
import pytest


def mod():return importlib.import_module('eat_nav_moving_pose2s_train')


def test_parent_leakage_rejected():
    m=mod();index=[dict(field='a',parent_id='a/1',seed=1,is_initial=True),dict(field='a',parent_id='a/1',seed=1,is_initial=False)]
    with pytest.raises(ValueError,match='parent'):m.verify_roles(index,{'train':[0],'validation':[1]})


def test_hierarchical_weights_do_not_count_long_parent_twice():
    m=mod();index=[dict(field='a',parent_id='p1',t0_ns=0),dict(field='a',parent_id='p2',t0_ns=0),dict(field='a',parent_id='p2',t0_ns=1)]
    weights=m.hierarchical_weights(index,[0,1,2])
    assert np.isclose(weights[0],weights[1]+weights[2])


def test_initial_variant_filters_all_roles_consistently():
    m=mod();index=[dict(field='a',parent_id='p1',is_initial=True),dict(field='a',parent_id='p1',is_initial=False),dict(field='a',parent_id='p2',is_initial=True),dict(field='a',parent_id='p2',is_initial=False)]
    r=m.select_training_roles(index,{'train':[0,1],'validation':[2,3],'audit_eval':[]},'initial')
    assert r=={'train':[0],'validation':[2]}


def test_component_hierarchy_recomputed_after_pose_mask():
    m=mod();idx=[dict(field='a',parent_id='a',state_signature='a0'),dict(field='a',parent_id='a',state_signature='a1'),dict(field='b',parent_id='b',state_signature='b0')]
    a={'pose_mask':np.array([[True],[False],[True]]),'candidate_mask':np.ones((3,1),bool),'event_mask':np.ones((3,1,4),bool)}
    w=m.component_weights(a,idx,[0,1,2])
    assert np.isclose(w['position'][0,0],.5)
    assert w['position'][1,0]==0
    assert np.isclose(w['position'][2,0],.5)
