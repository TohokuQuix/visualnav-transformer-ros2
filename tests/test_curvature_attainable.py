import math

import numpy as np
import pytest

from visualnav_transformer.eat_nav.curvature_attainable import (
    build_curve, cubic_controls, integrate_profile, piecewise_arc_line_arc,
    eight_point_curvature, solve_distinct_modes,
    solve_symmetric_target,
)
from scripts.eat_nav_curvature_attainable import (
    row_for, source_visible_path_terms, verify_anchor_identity,
)


def test_cubic_has_forward_initial_tangent_and_exact_endpoint():
    goal=np.array([.6*math.cos(math.radians(20)),.6*math.sin(math.radians(20))])
    curve=build_curve('G1',cubic_controls(goal,.12,.14,.03),goal)
    assert np.allclose(curve['dense'][0],[0,0])
    assert np.linalg.norm(curve['dense'][-1]-goal)<1e-10
    assert np.allclose(curve['initial_tangent_xy'],[1,0],atol=1e-10)
    assert curve['waypoints'].shape==(8,2)
    assert np.max(np.linalg.norm(np.diff(curve['dense'],axis=0),axis=1))<=.02


def test_constant_curvature_profile_reaches_twenty_degree_goal():
    beta=math.radians(20)
    curvature=2*math.sin(beta)/.6
    length=2*beta/curvature
    curve=integrate_profile(length,lambda t:np.full_like(t,curvature))
    goal=np.array([.6*math.cos(beta),.6*math.sin(beta)])
    assert np.linalg.norm(curve[-1]-goal)<1e-4


def test_piecewise_connection_has_continuous_join_and_straight_zero_case():
    curve=piecewise_arc_line_arc(.6,0.,0.,(.25,.5,.25))
    assert np.allclose(curve[0],[0,0])
    assert np.allclose(curve[-1],[.6,0],atol=1e-5)
    assert np.max(np.abs(curve[:,1]))<1e-9


def test_discrete_curvature_is_not_averaged_across_vertices():
    straight=np.stack([np.linspace(0,.6,9),np.zeros(9)],axis=1)
    corner=np.array([[0,0],[.1,0],[.2,0],[.2,.1],[.2,.2]],float)
    assert eight_point_curvature(straight)==pytest.approx(0.)
    assert eight_point_curvature(corner)>1.236


def test_mode_solver_returns_exact_maximum_independent_set():
    # 0 conflicts with 1; 2 is distinct from both.
    result=solve_distinct_modes(3,[(0,1)])
    assert result['optimal']
    assert result['maximum_mode_count']==2
    assert len(result['selected_indices'])==2


def test_symmetric_target_requires_direct_and_each_side():
    sides=['direct','left','right','left','right']
    passed=solve_symmetric_target(sides,[],pair_count=2,direct_index=0)
    assert passed['feasible'] and len(passed['selected_indices'])==5
    blocked=solve_symmetric_target(sides,[(1,3)],pair_count=2,direct_index=0)
    assert not blocked['feasible']


def test_numerical_endpoint_tolerance_uses_actual_path_endpoint():
    goal=np.array([.6,0.])
    curve=build_curve('G1',cubic_controls(goal,.12,.14,.02),goal)
    curve['waypoints'][-1,0]+=.0005
    row=row_for(curve,goal)
    assert np.array_equal(row['endpoint'],curve['waypoints'][-1])
    assert np.linalg.norm(row['endpoint']-goal)==pytest.approx(.0005)


def test_source_visible_path_terms_use_eight_actual_points():
    path=np.stack((np.linspace(.075,.6,8),np.zeros(8)),axis=1)
    command={'initial_command_estimate':{'carrot_xy':[.575,0.]},'command_mode':'tracking'}
    terms=source_visible_path_terms(path,command,{'lookahead_distance':.5})
    assert terms['path_point_count']==8
    assert terms['nearest_projection_segment_index']==0
    assert terms['remaining_polyline_m']==pytest.approx(.525)
    assert not terms['goal_extension_branch_expected']


def test_anchor_freeze_rejects_dem_history_change_even_when_ids_match():
    frozen=[{'anchor_id':f'a{i}','pose_id':f'p{i}',
             'dem_history_sha256':[f'd{i}-{j}' for j in range(4)],
             'source_path_shas':{'pose':f's{i}'},'settled_map_xy_yaw':[0.,0.,0.]}
            for i in range(8)]
    current=[dict(row) for row in frozen]
    assert verify_anchor_identity(current,frozen)['all_fields_match_prior_eight_anchor_freeze']
    current[2]['dem_history_sha256']=['wrong']*4
    with pytest.raises(ValueError,match='dem_history_sha256'):
        verify_anchor_identity(current,frozen)
