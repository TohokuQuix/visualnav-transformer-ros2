import math

import numpy as np
import pytest

from visualnav_transformer.eat_nav.local_motion_primitives import (
    BEARINGS_DEG, constant_curvature_primitive, goal_labels,
    physical_and_label_ids,
)
from scripts.eat_nav_local_primitive_review import (
    classify_goal_use, decide, runtime_goal_dependency_audit,
    audit_primitive, terrain_support_for_anchor, local_contract,
    conditional_p1_manifest, relabel_archived_trace,
    state_group_comparison,
    freeze_review,
    build_previous_bank_group_rows, verify_inputs,
    select_p1_anchors,
    _runtime_forecast,
)


@pytest.mark.parametrize('bearing', BEARINGS_DEG)
def test_arc_reaches_bearing_with_forward_initial_tangent(bearing):
    curve = constant_curvature_primitive(.75, bearing)
    theta = math.radians(bearing)
    assert curve['analytic_curvature_per_m'] == pytest.approx(2*math.sin(theta)/.75, abs=1e-12)
    assert np.asarray(curve['waypoints']).shape == (8, 2)
    assert np.asarray(curve['waypoints'])[-1] == pytest.approx([.75*math.cos(theta), .75*math.sin(theta)], abs=1e-9)
    assert np.asarray(curve['dense'])[0] == pytest.approx([0, 0], abs=1e-12)
    assert np.asarray(curve['initial_tangent_xy']) == pytest.approx([1, 0], abs=1e-12)
    assert curve['eight_point_max_curvature_per_m'] <= 1.236
    assert curve['analytic_max_curvature_per_m'] <= 1.236
    assert curve['dense_measured_max_curvature_per_m']==pytest.approx(abs(curve['analytic_curvature_per_m']),abs=2e-4)
    assert np.diff(curve['stations_m']).min() > 0


def test_five_paths_are_unique_but_reused_at_each_anchor():
    curves = [constant_curvature_primitive(.75, bearing) for bearing in BEARINGS_DEG]
    assert len({curve['path_sha256'] for curve in curves}) == 5
    assert curves[0]['path_sha256'] != curves[-1]['path_sha256']


def test_goal_relabel_preserves_locomotion_and_changes_navigation():
    xy = [[0., 0.], [.15, .0], [.30, .05]]
    low = goal_labels(xy, [0, 0, 0], [1.0, 2.0], -20, local_endpoint=[.75, 0])
    high = goal_labels(xy, [0, 0, 0], [1.0, 2.0], 20, local_endpoint=[.75, 0])
    assert low['locomotion'] == high['locomotion']
    assert low['navigation']['goal_distance_reduction_2s'] != high['navigation']['goal_distance_reduction_2s']


def test_relabels_share_physical_identity_and_split():
    records = physical_and_label_ids('anchor-a', 10, ['train'] * 5)
    assert len({r['physical_run_id'] for r in records}) == 1
    assert len({r['label_id'] for r in records}) == 5
    with pytest.raises(ValueError, match='split'):
        physical_and_label_ids('anchor-a', 10, ['train', 'val', 'train', 'train', 'train'])


def test_unaccounted_goal_use_rejects_source_validity():
    assert classify_goal_use('diagnostic_fields', 'mission_logged_goal') == 'logging_only'
    assert classify_goal_use('outcome', 'goal_distance') == 'offline_label_only'
    assert classify_goal_use('follower_command', 'mission_logged_goal') == 'physical_execution_dependency'
    assert decide(source=True, geometry=True, follower=True, terrain=True, coverage=True, fixture=True) == 'local_motion_primitive_bank_ready_for_p1_rollout_review'
    assert decide(source=False, geometry=True, follower=True, terrain=True, coverage=True, fixture=True) == 'global_goal_relabeling_not_source_valid'


def test_frozen_runtime_source_graph_has_no_global_goal_physical_edge():
    result = runtime_goal_dependency_audit()
    assert result['relabeling_source_valid']
    assert {row['classification'] for row in result['goal_uses']} == {'logging_only', 'offline_label_only'}
    assert {'Path generation','Path publication','follower command','Actor observation/action',
            'reward and termination','post-run goal distance'} <= set(result['dataflow'])


def test_contract_keeps_local_endpoint_separate_from_common_goal():
    contract=local_contract()
    assert contract['local_endpoint_chord_distance_m']==.75
    assert contract['common_global_goal_distance_m']==.75
    assert contract['local_endpoint_is_navigation_teacher'] is False
    assert contract['primary_terrain_arc_m']==[0.,.6]
    assert contract['full_terrain_diagnostic']=='through each complete Path endpoint'


def test_review_freeze_requires_new_root_and_writes_contract_before_geometry(tmp_path):
    from pathlib import Path
    backup=Path('/home/kojima/eatnav-backups/2026-10-08-pre-shutdown')
    if not backup.exists(): pytest.skip('preserved archive unavailable')
    root=tmp_path/'review'
    freeze_review(backup,root)
    assert (root/'local-motion-primitive-contract.json').exists()
    assert not (root/'primitive-path-definitions.json').exists()
    import json
    provenance=json.loads((root/'source-provenance.json').read_text())
    assert len(provenance['new_primitive_generator_sha256'])==64
    assert len(provenance['new_review_runner_sha256'])==64
    with pytest.raises(FileExistsError):freeze_review(backup,root)


def test_straight_primitive_is_follower_compatible_and_separate_goal_is_allowed():
    curve=constant_curvature_primitive(.75,0)
    result=audit_primitive(curve, [0,0,0], {'lookahead_distance':.4,'linear_speed':.3,
        'angular_gain':1.,'angular_direction_sign':1.,'max_angular_speed':.4,
        'goal_tolerance':.1,'turn_in_place_heading_threshold':1.,'min_linear_speed_scale':.1})
    assert result['hard_valid']
    assert result['follower_compatible']
    assert result['endpoint_error_m'] < 1e-9
    assert not audit_primitive({**curve,'dense_measured_max_curvature_per_m':1.3},
                               [0,0,0],{'lookahead_distance':.4,'linear_speed':.3,
        'angular_gain':1.,'angular_direction_sign':1.,'max_angular_speed':.4,
        'goal_tolerance':.1,'turn_in_place_heading_threshold':1.,
        'min_linear_speed_scale':.1})['follower_compatible']


def test_primary_terrain_support_never_counts_late_only_difference():
    curves=[constant_curvature_primitive(.75,b) for b in (-20,0,20)]
    height=np.zeros((20,20));height[:,17:]=.2
    mask=np.ones((20,20))
    bboxes={name:[[-.10,-.05],[.10,.05]] for name in
            ('main_track_sprocket','initial_flipper_support','articulated_reach','full_collision_union')}
    report=terrain_support_for_anchor(curves,height,mask,bboxes)
    assert len(report['candidate_coverage'])==3
    assert all(p['primary']['common_known_corridor_bins']>=0 for p in report['pairs'])
    assert all(p['primary_separated'] is False for p in report['pairs'])


def test_full_terrain_diagnostic_reaches_actual_arc_endpoint():
    curve=constant_curvature_primitive(.75,20)
    assert curve['arc_length_m']>.75
    height=np.zeros((20,20));mask=np.ones((20,20))
    bboxes={name:[[-.10,-.05],[.10,.05]] for name in
            ('main_track_sprocket','initial_flipper_support','articulated_reach','full_collision_union')}
    report=terrain_support_for_anchor([curve],height,mask,bboxes)
    assert report['candidate_coverage'][curve['variant_id']]['full_diagnostic_arc_length_m']==pytest.approx(curve['arc_length_m'])
    candidate=report['candidate_coverage'][curve['variant_id']]
    assert {'primary_main_track_slope','primary_main_track_step_m',
            'primary_main_track_roughness_m','interaction_onset_primary_m'}<=set(candidate)


def test_unexecuted_p1_manifest_has_28_physical_processes():
    selected=[{'anchor_id':f'a{i}','terrain_class':c,'seed_label':'first','source_family':f's{i}'}
              for i,c in enumerate(('transition','differential','graded','flat-only'))]
    result=conditional_p1_manifest(selected,ready=True)
    assert len(result['processes'])==28
    assert sum(p['primitive_bearing_deg']==0 for p in result['processes'])==12
    assert not result['executed']
    assert conditional_p1_manifest(selected,ready=False)['processes']==[]


def test_p1_anchor_selection_uses_distinct_source_families_without_outcome():
    rows=[{'anchor_id':'a','terrain_class':'transition','source_family':'rail'},
          {'anchor_id':'b','terrain_class':'differential','source_family':'boxes'},
          {'anchor_id':'c','terrain_class':'graded','source_family':'step'},
          {'anchor_id':'d','terrain_class':'flat-only','source_family':'rail'},
          {'anchor_id':'e','terrain_class':'flat-only','source_family':'pyramid'}]
    selected=select_p1_anchors(rows)
    assert [r['anchor_id'] for r in selected]==['a','b','c','e']


def test_runtime_forecast_keeps_device_cap_and_process_tree_memory_separate():
    from pathlib import Path
    backup=Path('/home/kojima/eatnav-backups/2026-10-08-pre-shutdown')
    if not backup.exists():pytest.skip('preserved resource data unavailable')
    result=_runtime_forecast(backup)
    assert result['P1_28_processes']['point_hours']>0
    assert result['gpu_device_total_hard_cap_mib']==6656
    assert result['measured_process_tree_anon_plus_shmem_peak_gib']>result['measured_cpu_anon_peak_gib']


def test_archived_trace_relabels_share_sha_and_locomotion(tmp_path):
    from pathlib import Path
    source=Path('/home/kojima/eatnav-backups/2026-10-08-pre-shutdown/eatnav-expanded-feasibility-ros-continuation-20261005/runs/expanded-train-002-left_1')
    if not source.exists(): pytest.skip('preserved archived trace unavailable')
    result=relabel_archived_trace(source)
    assert result['verified_step_count']>=40
    assert len(result['labels'])==5
    assert len({r['physical_trace_sha256'] for r in result['labels']})==1
    assert all(r['label_id'].startswith(result['physical_run_identity']['run_id']+':goal:') for r in result['labels'])
    assert len({r['locomotion']['discounted_reward_2s'] for r in result['labels']})==1
    assert len({r['navigation']['goal_distance_reduction_2s'] for r in result['labels']})>1
    assert result['physical_invariants']['termination_or_censor_status']=='horizon_reached_no_terminal'
    assert all(r['physical_invariants_sha256']==result['physical_invariants']['sha256'] for r in result['labels'])
    assert result['frame_validation']['base_path_exact_match']
    assert result['frame_validation']['anchor_root_exact_match']
    assert result['frame_validation']['map_path_transform_max_error_m']<1e-6


def test_state_group_comparison_preserves_common_goal_and_local_endpoint_distinction():
    old={'anchor_id':'a','bearing_deg':20,'state_group_id':'a-b+20',
         'baseline_060':{'compatible_common_goal':3},
         'revised_075':{'compatible_common_goal':2}}
    group={'anchor_id':'a','terrain':{'primary_separated_pair_count':1,'late_only_pair_count':2},
           'candidates':[{'hard_valid':True,'follower_compatible':True,
                          'analytic_max_curvature_per_m':.5,
                          'follower':{'initial_command_estimate':{'angular_z':.1}}}]}
    row=state_group_comparison(old,group)
    assert row['common_global_goal_bearing_deg']==20
    assert row['local_primitive_count']==1
    assert row['old_060_compatible_count']==3
    assert row['old_075_compatible_count']==2
    assert row['new_local_endpoint_residual_m'][20]==pytest.approx(0.)
    assert row['new_local_endpoint_residual_m'][0]>0


def test_previous_bank_comparison_contains_numeric_A_B_and_G1_G3():
    from pathlib import Path
    backup=Path('/home/kojima/eatnav-backups/2026-10-08-pre-shutdown')
    if not backup.exists():pytest.skip('preserved prior bank unavailable')
    frozen=verify_inputs(backup,Path('/home/kojima/eatnav-backups'))
    result=build_previous_bank_group_rows(frozen)
    assert len(result)==40
    row=result[0]
    assert {'Bank_A_060','Bank_B_060','G1_G3_common_endpoint_060'}<=set(row)
    for key in ('Bank_A_060','Bank_B_060','G1_G3_common_endpoint_060'):
        assert {'compatible_count','max_curvature_per_m','first_yaw_command_rad_s',
                'centerline_known_fraction_mean','endpoint_to_global_goal_residual_m_max',
                'known_terrain_separated_pair_count'}<=set(row[key])
    assert max(r['Bank_B_060']['endpoint_to_global_goal_residual_m_max_all_definitions'] or 0
               for r in result)>0
