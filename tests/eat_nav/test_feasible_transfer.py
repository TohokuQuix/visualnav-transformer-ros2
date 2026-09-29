"""Frozen candidate identity and geometry semantics for offline transfer analysis."""
import pytest
import numpy as np
from visualnav_transformer.eat_nav.feasible_transfer import (
    validate_candidate_identity, to_increments, to_waypoints,
    state_pairwise_accuracy,
)


def test_candidate_identity_requires_exact_state_generator_seed_and_run():
    schedule=[{'run_id':'pilot-004','state_id':'coverage-000','terrain_type':'boxes',
               'kind':'nomad','seed':1001,'name':None}]
    offline={'coverage-000':{'nomad-seed-1001':{'candidate_id':'nomad-seed-1001','seed':1001}}}
    rows=[{'run_id':'pilot-004','state_id':'coverage-000','terrain_type':'boxes',
           'kind':'nomad','seed':1001,'candidate_id':'nomad-seed-1001'}]
    assert validate_candidate_identity(rows,schedule,offline)==[('coverage-000','nomad-seed-1001','pilot-004')]
    with pytest.raises(ValueError,match='identity'):
        validate_candidate_identity([{**rows[0],'seed':1002}],schedule,offline)
    with pytest.raises(ValueError,match='duplicate'):
        validate_candidate_identity(rows*2,schedule,offline)


def test_increment_waypoint_roundtrip_and_first_point_origin():
    increments=np.array([[.2,.1],[.1,-.1]],dtype=np.float32)
    waypoint=to_waypoints(increments)
    assert np.allclose(waypoint,[[.2,.1],[.3,0.]])
    assert np.allclose(to_increments(waypoint),increments)


def test_pairwise_accuracy_uses_state_unit_and_excludes_ties():
    rows=[{'state_id':'a','candidate_id':'x','delta_v':1.,'score':3.},
          {'state_id':'a','candidate_id':'y','delta_v':2.,'score':2.},
          {'state_id':'a','candidate_id':'z','delta_v':2.005,'score':1.},
          {'state_id':'b','candidate_id':'x','delta_v':0.,'score':0.},
          {'state_id':'b','candidate_id':'y','delta_v':1.,'score':1.}]
    result=state_pairwise_accuracy(rows,epsilon=.01)
    assert result['a']['support']==2 and result['a']['accuracy']==0.
    assert result['b']['support']==1 and result['b']['accuracy']==1.
    assert result['state_macro_accuracy']==.5


def test_map_xy_to_base_and_command_integrator_agree_for_straight_motion():
    from visualnav_transformer.eat_nav.feasible_transfer import map_xy_to_base, integrate_commands
    map_points=np.array([[10.,20.05],[10.,20.10]])
    base=map_xy_to_base(map_points,anchor_xy=(10.,20.),yaw_rad=np.pi/2)
    assert np.allclose(base,[[.05,0.],[.10,0.]],atol=1e-6)
    command=np.array([[1.,0.,0.]]*2)
    assert np.allclose(integrate_commands(command,dt_sec=.05),[[.05,0.],[.10,0.]])


def test_cluster_recall_counts_modes_once_and_excludes_infeasible():
    from visualnav_transformer.eat_nav.feasible_transfer import cluster_labels, feasible_mode_recall
    paths=np.array([[[0.,0.],[1.,0.]],[[0.,0.],[1.01,0.]],[[0.,0.],[1.,1.]]])
    labels=cluster_labels(paths,threshold=.05)
    assert labels[0]==labels[1] and labels[2]!=labels[0]
    ids=['nomad-seed-1001','nomad-seed-1002','bank-left']
    result=feasible_mode_recall(ids,labels,[True,True,True],['nomad-seed-1001'],ks=[1])
    assert result[0]['numerator']==1 and result[0]['denominator']==2
    assert result[0]['recall']==.5
    filtered=feasible_mode_recall(ids,labels,[True,True,False],['nomad-seed-1001'],ks=[1])
    assert filtered[0]['denominator']==1 and filtered[0]['recall']==1.


def test_regret_uses_all_valid_oracle_and_tie_floor():
    from visualnav_transformer.eat_nav.feasible_transfer import oracle_regret
    cases=[{'candidate_id':'nomad-seed-1001','delta_v':1.},
           {'candidate_id':'bank-left','delta_v':1.2}]
    assert oracle_regret(cases,['nomad-seed-1001'])==pytest.approx(.2)
    assert oracle_regret(cases,['nomad-seed-1001','bank-left'])==0.


def test_command_clusters_keep_linear_and_angular_limits_separate():
    from visualnav_transformer.eat_nav.feasible_transfer import command_cluster_labels
    commands=np.array([[[.1,0.,0.]]*4,[[.12,0.,0.]]*4,[[.1,0.,.1]]*4])
    labels=command_cluster_labels(commands,linear_threshold=.03,angular_threshold=.05)
    assert labels[0]==labels[1] and labels[2]!=labels[0]
