import math
import os
from pathlib import Path

import numpy as np
import pytest

from visualnav_transformer.eat_nav.goal075_offline import (
    goal075, contract075, initial_forward_tangent_valid,
    side_relative_to_direct, shared_symmetric_bank, audit_candidate,
    prefix_row, coverage_record, variable_count_mask,
)
from visualnav_transformer.eat_nav.curvature_attainable import generate_goal_library
from scripts.eat_nav_goal_bank_review import _read_follower_config, FOLLOWER_CONFIG
from scripts.eat_nav_goal075_review import (
    verify_inputs, freeze_contract, verify_g0_regression_archive,
    analyze_group_075,
)


def test_goal_contract_separates_path_distance_from_teacher_horizon():
    for bearing in (-20,-10,0,10,20):
        goal=goal075(bearing)
        assert np.linalg.norm(goal)==pytest.approx(.75,abs=1e-12)
        assert math.degrees(math.atan2(goal[1],goal[0]))==pytest.approx(bearing,abs=1e-12)
    contract=contract075()
    assert contract['future_path_points']==8
    assert contract['teacher_duration_s']==pytest.approx(2.)
    assert contract['nominal_time_to_endpoint_s']==pytest.approx(2.5)
    assert contract['primary_terrain_arc_end_m']==pytest.approx(.6)


def test_nonforward_g0a_tangent_is_not_admitted_even_if_endpoint_matches():
    goal=goal075(20)
    assert not initial_forward_tangent_valid(goal/np.linalg.norm(goal))
    assert initial_forward_tangent_valid(np.array([1.,0.]))


def test_input_freeze_verifies_eight_physical_anchors_and_prior_generator():
    root=os.environ.get('EATNAV_FROZEN_ARTIFACT_ROOT')
    if not root:
        pytest.skip('requires preserved EAT-Nav backup root')
    frozen=verify_inputs(Path(root))
    assert len(frozen['anchors'])==8
    assert len(frozen['old_groups'])==40
    assert frozen['provenance']['prior_curvature_decision']=='candidate_diversity_infeasible_under_frozen_follower_contract'
    assert frozen['provenance']['generator_sha_match']


def test_contract_freeze_refuses_preexisting_output_root(tmp_path):
    with pytest.raises(FileExistsError):
        freeze_contract(tmp_path)


def test_frozen_060_g0_path_sha_regression_matches_all_selected_definitions():
    root=os.environ.get('EATNAV_FROZEN_ARTIFACT_ROOT')
    if not root:pytest.skip('requires preserved EAT-Nav backup root')
    frozen=verify_inputs(Path(root))
    result=verify_g0_regression_archive(
        frozen['old_groups'],frozen['prior']['redesign060']['selected-candidate-definitions.json'])
    assert result['matched_selected_prior_paths']==560


def test_side_is_fixed_relative_to_forward_tangent_direct_path():
    stations=np.linspace(0,.75,51)
    direct={'stations_m':stations,'dense':np.stack((stations,.03*np.sin(np.pi*stations/.75)),axis=1)}
    left={'stations_m':stations,'dense':direct['dense']+np.array([0.,.015])}
    right={'stations_m':stations,'dense':direct['dense']-np.array([0.,.015])}
    assert side_relative_to_direct(left,direct,goal075(20))[0]=='left'
    assert side_relative_to_direct(right,direct,goal075(20))[0]=='right'
    assert side_relative_to_direct(direct,direct,goal075(20))[0]=='direct'


def test_shared_five_bank_requires_same_ids_across_groups():
    ids=['direct','l1','l2','r1','r2','r3']
    sides=['direct','left','left','right','right','right']
    common={'g1':set(ids),'g2':set(ids)-{'r2'}}
    assert shared_symmetric_bank(ids,sides,common,{},'direct',pair_count=2)['selected_ids']==['direct','l1','l2','r1','r3']
    common['g2'].remove('l2')
    assert not shared_symmetric_bank(ids,sides,common,{},'direct',pair_count=2)['feasible']


def test_strict_forward_tangent_excludes_nonzero_bearing_g0a_but_keeps_g2():
    goal=goal075(20)
    curves,_=generate_goal_library(goal)
    baseline=next(c for c in curves if c['family']=='G0A' and c['variant_id']=='A-direct-0.00')
    constant=next(c for c in curves if c['family']=='G2' and c['variant_id']=='constant')
    anchor={'settled_map_xy_yaw':[0.,0.,0.],
            'dem':np.stack((np.zeros((20,20)),np.ones((20,20)))),
            'proxy_bboxes':{'main_track_sprocket':[[-.15,-.12],[.15,.12]]}}
    follower=_read_follower_config(FOLLOWER_CONFIG)
    old=audit_candidate(baseline,goal,anchor,follower)
    new=audit_candidate(constant,goal,anchor,follower)
    assert 'initial_tangent_error' in old['reasons']
    assert not old['compatible']
    assert new['compatible']
    too_bent={**constant,'eight_point_max_curvature_per_m':1.3}
    assert 'eight_point_curvature_screen' in audit_candidate(too_bent,goal,anchor,follower)['reasons']


def test_group_audit_keeps_forward_direct_and_exact_mode_solver():
    root=os.environ.get('EATNAV_FROZEN_ARTIFACT_ROOT')
    if not root:pytest.skip('requires preserved EAT-Nav backup root')
    frozen=verify_inputs(Path(root))
    anchor=frozen['anchors'][0]
    goal=goal075(20)
    curves,failures=generate_goal_library(goal)
    result=analyze_group_075(anchor,20,curves,failures,_read_follower_config(FOLLOWER_CONFIG))
    assert result['direct_id']=='G2:constant'
    assert result['direct_id'] in result['compatible_ids']
    assert result['maximum_distinct']['optimal']
    assert result['generated_common_goal']==len([c for c in curves if c['family']!='G0B'])


def test_primary_corridor_clips_at_point_six_and_excludes_later_geometry():
    stations=np.round(np.linspace(0,.75,16),2)
    row={'stations_m':stations,'dense':np.stack((stations,np.where(stations>.6,.1,0.)),axis=1),
         'arc_length_m':.75}
    short=prefix_row(row,.6)
    assert short['stations_m'][-1]==pytest.approx(.6)
    assert short['dense'][-1,1]==pytest.approx(0.)
    assert len(short['dense'])<len(row['dense'])


def test_coverage_reports_all_four_proxies_and_outside_margin():
    stations=np.linspace(0,.75,16)
    row={'stations_m':stations,'dense':np.stack((stations,np.zeros_like(stations)),axis=1),
         'arc_length_m':.75}
    proxies={name:[[-.1,-.1],[.1,.1]] for name in
             ('main_track_sprocket','initial_flipper_support','articulated_reach','full_collision_union')}
    result=coverage_record(row,np.zeros((20,20)),np.ones((20,20)),proxies)
    assert set(result['proxies'])==set(proxies)
    assert result['primary_centerline_known_fraction']==pytest.approx(1.)
    assert result['full_centerline_known_fraction']==pytest.approx(1.)
    assert result['minimum_full_tile_margin_m']>0


def test_variable_count_uses_mask_without_padding_or_duplicate_ids():
    result=variable_count_mask(['d','l','r'],'d')
    assert result['candidate_count']==3
    assert result['valid_mask']==[True,True,True]
    with pytest.raises(ValueError,match='duplicate'):
        variable_count_mask(['d','l','l'],'d')
