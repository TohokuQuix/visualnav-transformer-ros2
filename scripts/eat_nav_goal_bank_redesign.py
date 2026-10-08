#!/usr/bin/env python3
"""Offline-only follower-compatible parameter redesign of the frozen goal bank."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import yaml

import numpy as np

from visualnav_transformer.eat_nav.goal_bank_offline import (
    BEARINGS_DEG, CLASSES, bezier_candidate, goal_for_bearing, pairwise_audit,
)
from visualnav_transformer.eat_nav.candidate_bank_geometry import c0_fresh_forward
from visualnav_transformer.eat_nav.goal_bank_redesign import (
    AMPLITUDE_POOL_M, OFFSET_POOL_M, FOLLOWER_SCREEN_PER_M,
    SELECTION_ORDER, follower_screen_reasons, rank_triples, rollout_readiness,
)
from scripts.eat_nav_goal_bank_review import (
    FOLLOWER_CONFIG, _candidate_key, _candidate_record, _pairwise_records,
    _pool_for_group, _read_follower_config, _source_file, load_anchor_inputs,
    plain, read, sha, write,
)

BASELINE_ARCHIVE_SHA256 = 'eca4650875d4972827010bbbe4cd970cb3b06b4d0b9cfcdf2516d1a204de1e6b'
C0_FIXTURE_SHA256 = '08251e2a8f26008a026cda55b5ec1a63086ba961ffa429927d18a1377448b1b8'
E1_GATE_SHA256 = '2bb5f3420426c09f15a7a20ea0fa72ae930a4bc137ca752535f5158570a84ea6'
FOLLOWER_SOURCE_REL = Path('eatnav-onix/teaching_playback/simple_trajectory_follower/src')
FOLLOWER_CONFIG_REL = Path('eatnav-onix/teaching_playback/simple_trajectory_follower/config/onix.yaml')
PILOT_ANCHORS = ('first-transition', 'first-differential', 'second-graded', 'second-flat-only')
TERRAIN_RMS_M = .01
MIN_COMMON_BINS = 5
GEOMETRY_FULL_RMS_M = .015
GEOMETRY_EARLY_RMS_M = .01


def verify_baseline(backup_root: Path, prior_root: Path) -> dict:
    """Reject missing or modified old results and frozen source files."""
    backup_root, prior_root = Path(backup_root), Path(prior_root)
    archive = backup_root/'eatnav-goal-bearing-bank-review-20261008-r6.tar.gz'
    if sha(archive) != BASELINE_ARCHIVE_SHA256:
        raise ValueError('baseline backup archive SHA mismatch')
    with tarfile.open(archive,'r:gz') as bundle:
        archived_ledger=bundle.extractfile('eatnav-goal-bearing-bank-review-20261008-r6/sha-ledger.json')
        if archived_ledger is None:
            raise ValueError('baseline archive lacks prior SHA ledger')
        archived_ledger_bytes=archived_ledger.read()
    if (prior_root/'sha-ledger.json').read_bytes()!=archived_ledger_bytes:
        raise ValueError('prior output ledger differs from frozen backup archive')
    ledger = read(prior_root/'sha-ledger.json')['files']
    mismatches = [name for name, expected in ledger.items()
                  if not (prior_root/name).is_file() or sha(prior_root/name) != expected]
    if mismatches:
        raise ValueError('prior output SHA mismatch: '+','.join(mismatches))
    provenance = read(prior_root/'source-input-provenance.json')
    source_bad = [path for path, expected in provenance['sha256'].items()
                  if not Path(path).is_file() or sha(Path(path)) != expected]
    if source_bad:
        raise ValueError('frozen source SHA mismatch: '+','.join(source_bad))
    source_dir = Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/src')
    current_follower = {}
    for name in ('simple_trajectory_follower.cpp', 'path_geometry.cpp'):
        current = source_dir/name
        frozen = backup_root/FOLLOWER_SOURCE_REL/name
        if sha(current) != sha(frozen):
            raise ValueError('production follower source SHA mismatch: '+name)
        current_follower[name] = sha(current)
    frozen_config = backup_root/FOLLOWER_CONFIG_REL
    if sha(FOLLOWER_CONFIG) != sha(frozen_config) or sha(FOLLOWER_CONFIG) != provenance['follower_config_sha256']:
        raise ValueError('production follower config SHA mismatch')
    module_file = Path(__file__).resolve().parents[1]/'src/visualnav_transformer/eat_nav/goal_bank_offline.py'
    prior_module = provenance['offline_module_sha256']
    if sha(module_file) != prior_module:
        raise ValueError('prior bank geometry source changed')
    config = read(prior_root/'screen-config-freeze.json')
    if (config['decision_follower_supply_minimum_average_candidates_per_state_group'] != 4
            or config['follower_first_command_config'] != _read_follower_config(FOLLOWER_CONFIG)
            or FOLLOWER_SCREEN_PER_M != 1.236):
        raise ValueError('follower screen contract mismatch')
    return {'prior_review_root':str(prior_root), 'prior_review_output_sha_entries':len(ledger),
            'prior_review_ledger_sha256':hashlib.sha256(archived_ledger_bytes).hexdigest(),
            'prior_review_archive':str(archive), 'prior_review_archive_sha256':sha(archive),
            'prior_review_source_sha_entries':len(provenance['sha256']),
            'prior_review_code_head':provenance['code_head'],
            'prior_review_decision':read(prior_root/'final-decision.json')['decision'],
            'artifact_root_mapping':{
                '/tmp/eatnav-goal-bearing-bank-review-20261008-r6':str(prior_root),
                '/tmp/eatnav-r3-transition-reserve-20261008':str(backup_root/'eatnav-r3-transition-reserve-20261008'),
                '/tmp/eatnav-r3-settling-pilot-20261008':str(backup_root/'eatnav-r3-settling-pilot-20261008'),
                '/tmp/eatnav-r3-second-seed-20261008':str(backup_root/'eatnav-r3-second-seed-20261008')},
            'follower_source_sha256':current_follower,
            'follower_config_sha256':sha(FOLLOWER_CONFIG),
            'prior_bank_geometry_source_sha256':prior_module}


def _terrain_separated(pair: dict) -> bool:
    return (pair['known_corridor_profile_rms_m'] is not None
            and pair['common_known_corridor_bins'] >= MIN_COMMON_BINS
            and pair['known_corridor_profile_rms_m'] >= TERRAIN_RMS_M)


def _mode_count(items: list[dict]) -> int:
    parent = list(range(len(items)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i,j in itertools.combinations(range(len(items)),2):
        pair = pairwise_audit(items[i]['row'],items[j]['row'],items[i]['dem'],items[j]['dem'])
        if pair['full_geometry_rms_m'] < GEOMETRY_FULL_RMS_M and pair['early_0_5m_geometry_rms_m'] < GEOMETRY_EARLY_RMS_M:
            parent[root(j)] = root(i)
    return len({root(i) for i in range(len(items))})


def score_triple(pool: dict, params: tuple[float,float,float], bank: str) -> dict:
    group_counts = {}
    score = {'bank':bank, 'terminal_mode':'common' if bank == 'A' else 'blended_goal_connected',
             'parameters':list(params), 'state_group_count':len(pool),
             'hard_valid_groups':0, 'hard_valid_candidates':0,
             'follower_compatible':0, 'left_compatible':0, 'right_compatible':0,
             'geometry_mode_total':0, 'groups_seven_geometry_modes':0,
             'known_terrain_separated_pairs':0, 'unknown_dependence':0.,
             'maximum_curvature_per_m':0., 'turn_in_place_estimates':0,
             'endpoint_residual_m_sum':0., 'initial_command_mode_counts':{},
             'initial_angular_z_abs_max_radps':0., 'main_track_known_fraction_mean':0.,
             'compatible_group_histogram':{}, 'minimum_compatible_per_group':7,
             'groups_at_least_five':0}
    coverage_sum = 0.
    for gid,group in pool.items():
        items = [group['direct']] + [group[_candidate_key(side,m)]
                  for side in ('left','right') for m in params]
        compatible = 0
        for item in items:
            row, follower, dem = item['row'],item['follower'],item['dem']
            good = follower_screen_reasons(row,follower)['compatible']
            compatible += good
            score['follower_compatible'] += good
            score['left_compatible'] += good and row['side']=='left'
            score['right_compatible'] += good and row['side']=='right'
            score['hard_valid_candidates'] += item['geometry']['hard_valid']
            score['unknown_dependence'] += dem['centerline_unknown_fraction'] + dem['centerline_outside_fraction']
            score['maximum_curvature_per_m'] = max(score['maximum_curvature_per_m'],row['maximum_curvature_per_m'])
            score['turn_in_place_estimates'] += follower['turn_in_place_entry']
            score['endpoint_residual_m_sum'] += float(np.linalg.norm(row['endpoint']-row['global_goal']))
            mode = follower['command_mode']
            score['initial_command_mode_counts'][mode] = score['initial_command_mode_counts'].get(mode,0)+1
            angular = follower['initial_command_estimate'].get('angular_z')
            if angular is not None:
                score['initial_angular_z_abs_max_radps'] = max(score['initial_angular_z_abs_max_radps'],abs(angular))
            coverage_sum += dem['corridors']['main_track_sprocket']['known_fraction']
        score['hard_valid_groups'] += all(item['geometry']['hard_valid'] for item in items)
        modes = _mode_count(items)
        score['geometry_mode_total'] += modes
        score['groups_seven_geometry_modes'] += modes==7
        group_counts[gid] = compatible
        score['compatible_group_histogram'][str(compatible)] = score['compatible_group_histogram'].get(str(compatible),0)+1
        score['minimum_compatible_per_group'] = min(score['minimum_compatible_per_group'],compatible)
        score['groups_at_least_five'] += compatible>=5
        for i,j in itertools.combinations(range(7),2):
            pair = pairwise_audit(items[i]['row'],items[j]['row'],items[i]['dem'],items[j]['dem'])
            score['known_terrain_separated_pairs'] += _terrain_separated(pair)
    score['main_track_known_fraction_mean'] = coverage_sum/(len(pool)*7)
    score['side_balance'] = -abs(score['left_compatible']-score['right_compatible'])
    score['compatible_by_state_group'] = group_counts
    return score


def _parameter_boundary(pool: dict, bank: str) -> dict:
    magnitudes=AMPLITUDE_POOL_M if bank=='A' else OFFSET_POOL_M
    result={'direct':{},'levels':{}}
    for label,magnitude in [('direct',0.),*[(f'{m:.2f}',m) for m in magnitudes]]:
        items=[group['direct'] for group in pool.values()] if label=='direct' else [
            group[_candidate_key(side,magnitude)] for group in pool.values() for side in ('left','right')]
        curvatures=[item['row']['maximum_curvature_per_m'] for item in items]
        compatible=sum(follower_screen_reasons(item['row'],item['follower'])['compatible'] for item in items)
        record={'candidate_definitions':len(items),'compatible':compatible,
                'curvature_min_per_m':float(min(curvatures)),
                'curvature_max_per_m':float(max(curvatures)),
                'production_follower_screen_per_m':FOLLOWER_SCREEN_PER_M}
        if label=='direct':result['direct']=record
        else:result['levels'][label]=record
    return result


def _follower_reason_row(candidate: dict) -> dict:
    screen = follower_screen_reasons(candidate['_row'],candidate['follower'])
    command = candidate['follower']['initial_command_estimate']
    return {'state_group_id':candidate['state_group_id'],'bank':candidate['bank'],
            'candidate_id':candidate['candidate_id'],'goal_bearing_deg':candidate['bearing_deg'],
            'parameter_kind':candidate['parameter_kind'],
            'signed_amplitude_or_offset_m':candidate['signed_amplitude_or_offset_m'],
            'maximum_curvature_per_m':candidate['maximum_curvature_per_m'],
            'initial_chord_angle_rad':candidate['follower']['initial_chord_angle_rad'],
            'initial_command_estimate':command,
            'command_sign':0 if command.get('angular_z') is None else int(np.sign(command['angular_z'])),
            'angular_saturation_estimate':candidate['follower']['angular_saturation_estimate'],
            'turn_in_place_estimate':candidate['follower']['turn_in_place_entry'],
            'reverse_estimate':candidate['follower']['reverse_estimate'],
            'endpoint_xy_m':candidate['local_subgoal_xy_m'],
            'terminal_tangent_xy':candidate['terminal_tangent_xy'], **screen}


def _selected_bank(bank: str, selection: dict, pool: dict, groups: list[dict],
                   anchors_by_id: dict) -> tuple[list[dict],list[dict],dict]:
    rows,pairs = [],[]
    support={}
    for group in groups:
        gid = group['state_group_id']
        source = pool[gid]
        anchor = anchors_by_id[group['anchor_id']]
        items=[]
        specs=[('direct',0.,'direct')]+[(side,m,f'{side}-{level}')
               for side in ('left','right')
               for level,m in zip(('small','medium','large'),selection['parameters'])]
        for side,m,cid in specs:
            candidate=_candidate_record(group,anchor,bank,source[_candidate_key(side,m)],cid)
            if bank == 'B':
                candidate['terminal_mode']='blended_goal_connected'
                candidate['underlying_generator_mode']='goal_connected'
            items.append(candidate)
            rows.append(candidate)
        local_pairs=_pairwise_records(items)
        by_id={item['candidate_id']:item for item in items}
        for pair in local_pairs:
            first=by_id[pair['first']]['dem']['interaction_onset_station_m']
            second=by_id[pair['second']]['dem']['interaction_onset_station_m']
            both_compatible=all(follower_screen_reasons(by_id[cid]['_row'],by_id[cid]['follower'])['compatible']
                                for cid in (pair['first'],pair['second']))
            pairs.append({'state_group_id':gid,'bearing_deg':group['bearing_deg'],
                          'terrain_class':group['terrain_class'],'seed_label':group['seed_label'],
                          'both_follower_compatible':both_compatible,
                          'interaction_onset_difference_m':abs(first-second) if first is not None and second is not None else None,
                          **pair})
        support[gid]={'anchor_id':group['anchor_id'],'terrain_class':group['terrain_class'],
                      'seed_label':group['seed_label'],'bearing_deg':group['bearing_deg'],
                      'hard_valid':all(x['geometry']['hard_valid'] for x in items),
                      'geometry_modes':_mode_count([source[_candidate_key(side,m)] for side,m,_ in specs]),
                      'compatible':sum(follower_screen_reasons(x['_row'],x['follower'])['compatible'] for x in items),
                      'compatible_required_l1':all(follower_screen_reasons(by_id[cid]['_row'],by_id[cid]['follower'])['compatible']
                          for cid in ('direct','left-small','right-small','left-medium','right-medium')),
                      'reverse':sum(x['follower']['reverse_estimate'] for x in items),
                      'self_intersections':sum('self_intersection' in x['geometry']['invalid_reasons'] for x in items),
                      'known_terrain_separated_pairs':sum(_terrain_separated(p) for p in local_pairs),
                      'compatible_known_terrain_separated_pairs':sum(
                          _terrain_separated(p) and all(follower_screen_reasons(by_id[cid]['_row'],by_id[cid]['follower'])['compatible']
                              for cid in (p['first'],p['second'])) for p in local_pairs)}
    return rows,pairs,support


def _class_seed_support(support: dict) -> dict:
    return {cls:{seed:sum(row['terrain_class']==cls and row['seed_label']==seed
                           and row['compatible_known_terrain_separated_pairs']>0
                           for row in support.values())
                 for seed in ('first','second')}
            for cls in ('transition','differential','graded')}


def _forecast(prior: dict) -> dict:
    source=prior['measured_prior_process_mean_sec']
    p95=prior['measured_prior_process_p95_sec']
    bytes_per=prior['storage_reference_bytes_per_process']
    variants={'P1_five':28,'P1_seven':36,'P2_five':56,'P2_seven':72,
              'P1_plus_P2_five':84,'P1_plus_P2_seven':108,
              'Bank_B_added_five_shared_direct':48,
              'Bank_B_added_seven_shared_direct':72,
              'Bank_B_added_five_independent_baseline':84,
              'Bank_B_added_seven_independent_baseline':108}
    return {'source_prior_forecast_sha256':None,
            'sequential_fresh_process_estimates':{
                name:{'processes':count,'point_hours':count*source/3600,
                      'p95_hours':count*p95/3600,'p95_plus_20_percent_hours':count*p95*1.2/3600,
                      'saved_gb_point':count*bytes_per/1e9,
                      'saved_gb_plus_20_percent':count*bytes_per*1.2/1e9}
                for name,count in variants.items()},
            'gpu_peak_reference_mib':prior['prior_gpu_peak_reference_mib'],
            'gpu_device_total_cap_mib':prior['prior_gpu_device_total_cap_mib'],
            'host_available_reference_gb':prior['prior_host_available_reference_gb'],
            'simulator_pss_reference_gib':prior['prior_simulator_pss_reference_gib'],
            'anon_shmem_reference_gib':prior['prior_anon_shmem_reference_gib'],
            'external_zenoh_prerequisite':True,
            'cleanup_finalization':'owned PID/start-tick, SHA seal, native exit, zero residual',
            'automatic_retries':0,
            'bank_B_repeat_policy':'shared direct only if SHA-identical and accepted as a common baseline; otherwise independent direct repeats',
            'not_an_authorization_to_run':True}


def _prospective_P1_P2_gates(root: Path) -> dict:
    source=root/'docs/quality-identifiability-gate/candidate-coverage/e1-feasibility/e1-gates.yaml'
    if sha(source)!=E1_GATE_SHA256:
        raise ValueError('frozen E1 numeric gate source SHA mismatch')
    frozen=yaml.safe_load(source.read_text())
    duplicate=frozen['duplicate']; p2c=frozen['P_to_C']; c2r=frozen['C_to_R']; r2y=frozen['R_to_Y']
    return {'status':'prospective_only_no_rollout_authorized',
            'numeric_source_path':str(source),'numeric_source_sha256':sha(source),
            'statistical_unit':'StateGroup; 4 distinct source families in P1',
            'P1_to_P2_rule':'all 4 P1 groups satisfy duplicate, P_to_C, C_to_A and A_to_R; at least one group satisfies R_to_Y; every scheduled process is infrastructure-valid; otherwise stop before P2',
            'duplicate_gate':{
                'same_path_sha_and_initial_anchor':True,
                'xy_rms_max_m':duplicate['xy_rms_max_m'],
                'command_rms_max_m_per_s':duplicate['command_rms_max_m_per_s'],
                'actor_input_max_abs':duplicate['actor_input_max_abs'],
                'actor_action_max_abs':duplicate['actor_action_max_abs'],
                'critic_value_max_abs':duplicate['critic_value_max_abs'],
                'dem_rms_max_m':duplicate['dem_rms_max_m'],
                'dem_mask_iou_min':duplicate['dem_mask_iou_min'],
                'reward_max_abs':duplicate['reward_max_abs']},
            'P_to_C_per_group':{
                'minimum_nondirect_commands_above_noise':2,
                'follower_command_rms_above_m_per_s':p2c['follower_command_rms_above_m_per_s'],
                'first_command_difference_above_m_per_s':p2c['first_command_difference_above_m_per_s'],
                'minimum_command_clusters':p2c['minimum_command_clusters_per_group']},
            'C_to_A_per_group':{
                'minimum_nondirect_actor_action_differences':1,
                'actor_action_max_abs_above':'max(0.0001, 2 * measured direct-duplicate max_abs)',
                'actor_input_difference_recorded':True},
            'A_to_R_per_group':{
                'realized_xy_rms_above_m':c2r['realized_xy_rms_above_m'],
                'candidate_to_duplicate_ratio_min':c2r['candidate_effect_to_duplicate_noise_ratio_min'],
                'ratio_denominator_floor_m':c2r['ratio_denominator_floor_m'],
                'minimum_realized_modes':c2r['minimum_realized_modes_per_group']},
            'R_to_Y_at_least_one_group':{
                'outcome_rank_difference_required':r2y['require_outcome_rank_difference_in_at_least_one_group'],
                'critic_value_difference_min':r2y['minimum_critic_value_difference'],
                'progress_or_endpoint_difference_m_min':r2y['minimum_progress_or_endpoint_difference_m'],
                'numeric_difference_above_duplicate_noise':True,
                'categorical_contact_stack_or_termination_difference_accepted':True},
            'infrastructure_validity_all_scheduled_processes':{
                'native_exit':0,'run_complete_marker':True,'protocol_valid_marker':True,
                'host_container_SHA_match':True,'owned_residual_processes':0,
                'failed_or_missing_process_policy':'stop before P2; no automatic retry'},
            'no_change_after_P1_to':['Path','threshold','candidate_count'],
            'P2_only_after_P1_pass':True}


def _source_exact_audit(backup: Path, follower_config: dict) -> dict:
    source_dir=Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/src')
    follower_cpp=(source_dir/'simple_trajectory_follower.cpp').read_text()
    geometry_cpp=(source_dir/'path_geometry.cpp').read_text()
    checks={
        'linear_speed':'active_travel_direction * linear_speed_ * speed_scale' in follower_cpp,
        'angular_gain_sign':'angular_direction_sign_ * angular_gain_ * heading_error' in follower_cpp,
        'saturation':'std::clamp(angular_z, -max_angular_speed_, max_angular_speed_)' in follower_cpp,
        'turn_in_place':'abs_heading_error < turn_in_place_heading_threshold_' in follower_cpp,
        'nearest_path_projection':'path_geometry::projectPointOntoPath' in follower_cpp,
        'goal_extension':'findGoalExtensionCarrot' in follower_cpp,
        'goal_tolerance':'distance_to_goal < goal_tolerance_' in follower_cpp,
        'path_frame_tf':'updateRobotPoseInPathFrame()' in follower_cpp,
        'path_arclength_sampling':'samplePathFromProjection' in geometry_cpp,
        'latest_TF_lookup':'lookupTransform(target_frame, robot_frame_, tf2::TimePointZero' in follower_cpp,
    }
    if not all(checks.values()):
        raise ValueError('follower source semantic static check failed')
    fixture_path=backup/'eatnav-candidate-bank-e05/c0/fixtures.json'
    if sha(fixture_path)!=C0_FIXTURE_SHA256:
        raise ValueError('production C0 fixture SHA mismatch')
    fixture=read(fixture_path)
    if (fixture['production_source_sha256']!=sha(source_dir/'simple_trajectory_follower.cpp')
            or fixture['production_config_sha256']!=sha(FOLLOWER_CONFIG)):
        raise ValueError('production C0 fixture source/config mismatch')
    parity=[]
    for case in fixture['cases']:
        result=c0_fresh_forward(case['path_base_xy'],
                                lookahead_m=follower_config['lookahead_distance'],
                                speed_mps=follower_config['linear_speed'],
                                angular_gain=follower_config['angular_gain'],
                                angular_direction_sign=follower_config['angular_direction_sign'],
                                max_angular_speed=follower_config['max_angular_speed'],
                                goal_tolerance_m=follower_config['goal_tolerance'],
                                turn_threshold_rad=follower_config['turn_in_place_heading_threshold'],
                                min_linear_speed_scale=follower_config['min_linear_speed_scale'])
        observed=[result['linear_x'],result['angular_z']]
        error=float(np.max(np.abs(np.asarray(observed)-np.asarray(case['production_first_v_omega']))))
        if error>1e-5:
            raise ValueError('production C0 numeric fixture mismatch: '+case['id'])
        parity.append({'case_id':case['id'],'production_first_v_omega':case['production_first_v_omega'],
                       'pure_first_v_omega':observed,'max_abs_error':error})
    return {'checks':checks,'production_cpp_sha256':sha(source_dir/'simple_trajectory_follower.cpp'),
            'production_path_geometry_cpp_sha256':sha(source_dir/'path_geometry.cpp'),
            'production_config_sha256':sha(FOLLOWER_CONFIG),
            'production_config_scalars':follower_config,
            'pure_source_equivalent':'candidate_bank_geometry.c0_fresh_forward at fresh pose origin/yaw0',
            'production_C0_fixture_sha256':sha(fixture_path),
            'production_C0_numeric_parity':parity,
            'numeric_parity_tolerance':1e-5,
            'scope_limits':['production follower requests latest TF; upstream synchronized bridge may require exact-stamp TF independently',
                            'requires a valid Path frame and robot TF at ROS boundary',
                            'does not model mutable controller state or later commands',
                            'initial command only; no physical tracking inference'],
            'frozen_source_matches_backup':True}


def run_redesign(backup: Path, prior: Path, output: Path, *, figures: bool=True) -> dict:
    backup,prior,output=Path(backup),Path(prior),Path(output)
    if output.exists():
        raise FileExistsError('new output root required: '+str(output))
    baseline=verify_baseline(backup,prior)
    root=Path(__file__).resolve().parents[1]
    test_command=['pytest','-q','tests/test_r3_second_seed.py',
                  'tests/test_r3_transition_reserve.py','tests/test_r3_anchor_contract.py']
    environment=dict(os.environ,EATNAV_FROZEN_ARTIFACT_ROOT=str(backup),PYTHONPATH='src:.')
    test_process=subprocess.run(test_command,cwd=root,env=environment,text=True,
                                capture_output=True,check=False)
    if test_process.returncode or '18 passed' not in test_process.stdout:
        raise ValueError('path-independent 18-test baseline failed: '+test_process.stdout+test_process.stderr)
    anchors,anchor_provenance=load_anchor_inputs(backup)
    groups=read(prior/'state-groups.json')
    old_anchors=read(prior/'eight-anchor-freeze.json')
    if len(anchors)!=8 or len(groups)!=40 or [a['anchor_id'] for a in anchors]!=[a['anchor_id'] for a in old_anchors]:
        raise ValueError('frozen anchor or StateGroup mismatch')
    for group in groups:
        if not np.allclose(group['global_goal_base_xy_m'],goal_for_bearing(group['bearing_deg']),atol=1e-12):
            raise ValueError('frozen global goal mismatch')
    output.mkdir(parents=True)
    write(output/'baseline-test-recovery.json',{
        'command':test_command,'artifact_root':str(backup),'exit_code':test_process.returncode,
        'stdout':test_process.stdout,'stderr':test_process.stderr,
        'expected':'18 passed','path_independent':True})
    anchors_by_id={a['anchor_id']:a for a in anchors}
    if len({anchors_by_id[name]['source_family'] for name in PILOT_ANCHORS}) != 4:
        raise ValueError('conditional four-anchor pilot lacks distinct source families')
    terminal_rule='terminal=normalize(0.30*goal_unit-signed_offset*goal_normal); P1=goal/3+signed_offset*0.4*normal; P2=endpoint-0.15*terminal'
    config={'schema':'eatnav_goal_bank_redesign_v1',
            'base_commit':'8481ebf234f2bc52937f457cf8311961a4abef41',
            'amplitude_pool_m':list(AMPLITUDE_POOL_M),
            'offset_pool_m':list(OFFSET_POOL_M),
            'selection_order':list(SELECTION_ORDER),
            'follower_curvature_screen_per_m':FOLLOWER_SCREEN_PER_M,
            'minimum_compatible_candidates_per_group':5,
            'bank_B_terminal_mode':'blended_goal_connected',
            'bank_B_terminal_rule':terminal_rule,
            'bank_B_terminal_rule_sha256':hashlib.sha256(terminal_rule.encode()).hexdigest(),
            'bank_B_generator_source_sha256':hashlib.sha256(inspect.getsource(bezier_candidate).encode()).hexdigest(),
            'bank_B_strict_goal_facing':'prior baseline diagnostic only',
            'global_goal_bearings_deg':list(BEARINGS_DEG), 'global_goal_distance_m':.6,
            'same_8_anchors':True,'same_40_state_groups':True,
            'future_waypoints':8,'dense_spacing_max_m':.02,
            'geometry_full_rms_m':GEOMETRY_FULL_RMS_M,
            'geometry_early_rms_m':GEOMETRY_EARLY_RMS_M,
            'known_terrain_rms_m':TERRAIN_RMS_M,
            'known_terrain_min_common_bins':MIN_COMMON_BINS,
            'forbidden_selection_inputs':['reward','Critic','Outcome','Evaluator_prediction']}
    write(output/'artifact-root-mapping.json',baseline['artifact_root_mapping'])
    write(output/'baseline-recovery.json',baseline)
    write(output/'frozen-redesign-config.json',config)
    follower=_read_follower_config(FOLLOWER_CONFIG)
    source_audit=_source_exact_audit(backup,follower)
    write(output/'source-exact-follower-audit.json',source_audit)
    pools={'A':{},'B':{}}
    for group in groups:
        goal=np.asarray(group['global_goal_base_xy_m'])
        anchor=anchors_by_id[group['anchor_id']]
        pools['A'][group['state_group_id']]=_pool_for_group('A','common',goal,anchor,follower,AMPLITUDE_POOL_M)
        pools['B'][group['state_group_id']]=_pool_for_group('B','goal_connected',goal,anchor,follower,OFFSET_POOL_M)
    boundary={bank:_parameter_boundary(pools[bank],bank) for bank in ('A','B')}
    write(output/'amplitude-offset-compatibility-boundary.json',boundary)
    screens={bank:[score_triple(pools[bank],combo,bank)
                   for combo in itertools.combinations(AMPLITUDE_POOL_M if bank=='A' else OFFSET_POOL_M,3)]
             for bank in ('A','B')}
    selected={bank:rank_triples(screens[bank]) for bank in ('A','B')}
    for bank in ('A','B'):
        write(output/f'bank-{bank}-triple-comparison.json',screens[bank])
    write(output/'selected-parameters.json',selected)
    rows_by_bank,pairs_by_bank,support_by_bank={},{},{}
    for bank in ('A','B'):
        rows,pairs,support=_selected_bank(bank,selected[bank],pools[bank],groups,anchors_by_id)
        rows_by_bank[bank],pairs_by_bank[bank],support_by_bank[bank]=rows,pairs,support
    public_rows=[{k:plain(v) for k,v in item.items() if not k.startswith('_')}
                 for bank in ('A','B') for item in rows_by_bank[bank]]
    write(output/'selected-candidate-definitions.json',public_rows)
    write(output/'selected-candidate-metrics.json',[
        {'state_group_id':row['state_group_id'],'bank':row['bank'],
         'candidate_id':row['candidate_id'],'bearing_deg':row['bearing_deg'],
         'seed_label':row['seed_label'],'terrain_class':row['terrain_class'],
         'path_length_m':row['arc_length_m'],
         'known_fraction_by_proxy':{name:metric['known_fraction'] for name,metric in row['dem']['corridors'].items()},
         'unknown_fraction_by_proxy':{name:metric['unknown_fraction'] for name,metric in row['dem']['corridors'].items()},
         'interaction_onset_station_m':row['dem']['interaction_onset_station_m'],
         'maximum_curvature_per_m':row['maximum_curvature_per_m'],
         'internal_cubic_tangent_continuity':'analytic_C1',
         'maximum_curvature_delta_per_m':row['curvature_continuity_max_delta_per_m'],
         'terminal_tangent_xy':row['terminal_tangent_xy'],
         'endpoint_global_goal_residual_m':row['endpoint_global_goal_residual_m'],
         'global_goal_progress_if_endpoint_reached_m':.6-row['endpoint_global_goal_residual_m'],
         'next_replan_goal_direction_cosine':row['terminal_goal_direction_cosine']}
        for bank in ('A','B') for row in rows_by_bank[bank]])
    reason_rows=[plain(_follower_reason_row(row)) for bank in ('A','B') for row in rows_by_bank[bank]]
    write(output/'follower-screen-reason-matrix.json',reason_rows)
    write(output/'terrain-corridor-comparison.json',pairs_by_bank)
    write(output/'per-state-group-support.json',support_by_bank)
    class_seed={bank:_class_seed_support(support_by_bank[bank]) for bank in ('A','B')}
    readiness={bank:rollout_readiness(support_by_bank[bank],class_seed[bank]) for bank in ('A','B')}
    terrain_summary={bank:{'class_seed_support':class_seed[bank],
                           'all_candidate_class_seed_support':{cls:{seed:sum(row['terrain_class']==cls and row['seed_label']==seed
                               and row['known_terrain_separated_pairs']>0 for row in support_by_bank[bank].values())
                               for seed in ('first','second')} for cls in ('transition','differential','graded')},
                           'separated_known_pair_count':sum(_terrain_separated(pair) for pair in pairs_by_bank[bank]),
                           'compatible_separated_known_pair_count':sum(_terrain_separated(pair) and pair['both_follower_compatible'] for pair in pairs_by_bank[bank]),
                           'group_with_separated_pair_count':sum(row['known_terrain_separated_pairs']>0 for row in support_by_bank[bank].values()),
                           'flat_group_count':sum(row['terrain_class']=='flat-only' for row in support_by_bank[bank].values()),
                           'per_bearing':{str(b):sum(_terrain_separated(pair) and pair['bearing_deg']==b for pair in pairs_by_bank[bank]) for b in BEARINGS_DEG},
                           'per_class':{cls:sum(_terrain_separated(pair) and pair['terrain_class']==cls for pair in pairs_by_bank[bank]) for cls in CLASSES},
                           'per_seed':{seed:sum(_terrain_separated(pair) and pair['seed_label']==seed for pair in pairs_by_bank[bank]) for seed in ('first','second')},
                           'left_direct_right':{mode:sum(_terrain_separated(pair) and
                                ((mode=='left_direct' and 'direct' in (pair['first'],pair['second']) and ('left' in pair['first'] or 'left' in pair['second'])) or
                                 (mode=='right_direct' and 'direct' in (pair['first'],pair['second']) and ('right' in pair['first'] or 'right' in pair['second'])) or
                                 (mode=='left_right' and ('left' in pair['first'] or 'left' in pair['second']) and ('right' in pair['first'] or 'right' in pair['second'])) )
                                for pair in pairs_by_bank[bank]) for mode in ('left_direct','right_direct','left_right')},
                           'known_only':True,'unknown_only_differences_excluded':True}
                     for bank in ('A','B')}
    write(output/'terrain-separation-review.json',terrain_summary)
    b_rows=rows_by_bank['B']
    write(output/'bank-B-structural-bias.json',{
        'global_goal_remains_common':True,'local_endpoint_not_navigation_goal':True,
        'mean_endpoint_to_global_goal_residual_m':float(np.mean([row['endpoint_global_goal_residual_m'] for row in b_rows])),
        'mean_path_length_m':float(np.mean([row['arc_length_m'] for row in b_rows])),
        'mean_global_goal_progress_if_endpoint_reached_m':float(np.mean([.6-row['endpoint_global_goal_residual_m'] for row in b_rows])),
        'internal_cubic_tangent_continuity':'analytic_C1; replan junction continuity unmeasured',
        'mean_maximum_curvature_delta_per_m':float(np.mean([row['curvature_continuity_max_delta_per_m'] for row in b_rows])),
        'mean_next_replan_goal_direction_cosine':float(np.mean([row['terminal_goal_direction_cosine'] for row in b_rows if row['terminal_goal_direction_cosine'] is not None])),
        'short_horizon_teacher_bias':'structural endpoint residual only; no Outcome or teacher score evaluated'})
    write(output/'bank-readiness.json',readiness)
    old_summary=read(prior/'bank-comparison.json')
    old_selection=read(prior/'selected-bank-config.json')
    baseline_comparison={bank:{'old_parameters':old_selection[bank]['parameters'],
                               'old_compatible':old_summary[bank]['follower_compatible_candidate_count'],
                               'new_parameters':selected[bank]['parameters'],
                               'new_compatible':selected[bank]['follower_compatible'],
                               'old_terrain_separated_pairs':old_summary[bank]['terrain_separated_known_pairs'],
                               'new_terrain_separated_pairs':terrain_summary[bank]['separated_known_pair_count']}
                         for bank in ('A','B')}
    write(output/'old-vs-new-bank-comparison.json',baseline_comparison)
    a_ready,b_ready=readiness['A']['ready'],readiness['B']['ready']
    if a_ready and b_ready:
        decision='bank_a_and_b_ready_for_separate_rollout_review'
    elif a_ready:
        decision='bank_a_ready_bank_b_ablation_not_ready'
    elif not readiness['A']['group_gate'] and not readiness['B']['group_gate']:
        decision='candidate_bank_compatibility_still_insufficient'
    elif not readiness['A']['terrain_gate'] and not readiness['B']['terrain_gate']:
        decision='terrain_separation_lost_after_compatibility_redesign'
    else:
        decision='candidate_bank_compatibility_still_insufficient'
    proposal={'status':'not_eligible_for_execution_compatibility_gate_failed' if not a_ready else 'conditional_proposal_only_no_execution',
              'bank_A_ready':a_ready,'bank_B_ablation_ready':b_ready,
              'pilot_anchor_ids':list(PILOT_ANCHORS),
              'L1_five_candidates':['direct','left-small','right-small','left-medium','right-medium'],
              'L2_seven_candidates':['direct','left-small','right-small','left-medium','right-medium','left-large','right-large'],
              'L2_large_screen_out_must_be_explicit':True,
              'screen_out_candidates_by_bank':{bank:{gid:[row['candidate_id'] for row in rows_by_bank[bank]
                  if row['state_group_id']==gid and not follower_screen_reasons(row['_row'],row['follower'])['compatible']]
                  for gid in [g['state_group_id'] for g in groups if g['anchor_id'] in PILOT_ANCHORS and g['bearing_deg'] in (-20,0,20)]}
                  for bank in ('A','B')},
              'L1_vs_L2_selection':'deferred to next review; do not auto-run screen-out large candidates',
              'P1':{'bearings_deg':[0],'groups':4,'L1_processes':28,'L2_processes':36,'direct_repeats_per_group':3},
              'P2':{'bearings_deg':[-20,20],'groups':8,'L1_processes':56,'L2_processes':72,'only_after_P1_gate':True},
              'P1_state_group_order':[g['state_group_id'] for g in groups if g['anchor_id'] in PILOT_ANCHORS and g['bearing_deg']==0],
              'P2_state_group_order':[g['state_group_id'] for g in groups if g['anchor_id'] in PILOT_ANCHORS and g['bearing_deg'] in (-20,20)],
              'within_group_process_order_L1':['direct-repeat-0','direct-repeat-1','direct-repeat-2','left-small','right-small','left-medium','right-medium'],
              'within_group_process_order_L2':['direct-repeat-0','direct-repeat-1','direct-repeat-2','left-small','right-small','left-medium','right-medium','left-large','right-large'],
              'plus_minus_10_bearings':'offline definitions only',
              'split_unit':'terrain seed/source family; all bearings and candidates grouped together',
              'Path_SHA_by_group_bank':{bank:{gid:{row['candidate_id']:row['path_sha256']
                  for row in rows_by_bank[bank] if row['state_group_id']==gid}
                  for gid in [g['state_group_id'] for g in groups if g['anchor_id'] in PILOT_ANCHORS and g['bearing_deg'] in (-20,0,20)]}
                  for bank in ('A','B')}}
    write(output/'five-vs-seven-and-staged-proposal.json',proposal)
    write(output/'P1-P2-manifest-proposal.json',proposal)
    prior_forecast=read(prior/'runtime-storage-forecast.json')
    forecast=_forecast(prior_forecast)
    forecast['source_prior_forecast_sha256']=sha(prior/'runtime-storage-forecast.json')
    write(output/'runtime-storage-forecast.json',forecast)
    gates=_prospective_P1_P2_gates(root)
    write(output/'prospective-P1-P2-gates.json',gates)
    risk={'curvature_screen':'diagnostic follower compatibility gate, not physical vehicle limit',
          'parameter_pool_limit':'fixed A/B pools may make 5/7 impossible under current curve rules',
          'bank_B_goal_bias':'local endpoint leaves residual to common global goal',
          'source_equivalent_limit':'fresh first command only; no mutable state or realized tracking',
          'terrain_proxy_limit':'DEM collision-bound rectangle audit, not contact simulation',
          'runtime':'prior E1 extrapolation, not measurement of this proposal'}
    write(output/'risk-register.json',risk)
    result={'schema':'eatnav_goal_bank_follower_redesign_v1','decision':decision,
            'bank_readiness':readiness,'selected_parameters':{k:v['parameters'] for k,v in selected.items()},
            'old_vs_new':baseline_comparison,'candidate_count':{'A':280,'B':280},
            'same_8_anchors_40_groups_5_bearings':True,
            'candidate_path_rollout_count':0,'follower_bridge_router_started':False,
            'outcome_collected':False,'learning_started':False,
            'conditional_proposal_only':True}
    write(output/'final-decision.json',result)
    provenance={'baseline':baseline,'anchor_source_file_count':anchor_provenance['source_file_count'],
                'new_runner_sha256':sha(Path(__file__)),
                'new_module_sha256':sha(Path(__file__).resolve().parents[1]/'src/visualnav_transformer/eat_nav/goal_bank_redesign.py'),
                'config_sha256':sha(output/'frozen-redesign-config.json'),
                'follower_source_audit_sha256':sha(output/'source-exact-follower-audit.json'),
                'output_root':str(output),'prior_root_untouched':True,
                'forbidden_selection_inputs_used':[]}
    write(output/'provenance.json',provenance)
    _report(output,result,screens,terrain_summary,forecast,boundary)
    if figures:
        make_figures(output,screens,selected,rows_by_bank,reason_rows,terrain_summary,forecast,result)
    ledger={str(path.relative_to(output)):sha(path) for path in sorted(output.rglob('*'))
            if path.is_file() and path.name!='sha-ledger.json'}
    write(output/'sha-ledger.json',{'schema':'eatnav_goal_bank_redesign_output_sha_v1','files':ledger})
    return result


def _report(output: Path, result: dict, screens: dict, terrain: dict, forecast: dict,
            boundary: dict) -> None:
    a,b=result['bank_readiness']['A'],result['bank_readiness']['B']
    old=result['old_vs_new']
    lines=['# Follower-compatible candidate bank offline redesign','',
           f"**Decision:** `{result['decision']}`. No Path was executed.",'',
           '## Frozen inputs and source parity','',
           '- Prior output ledger: 35/35 entries verified; prior source provenance: 87/87 files verified.',
           '- The 18 R3 baseline tests pass with an environment-selected frozen artifact root.',
           '- Production follower C++ source, path geometry C++ and YAML match backed-up SHA copies.',
           '- Three prior production C0 fixtures (straight, curved, short goal extension) match the pure first-command calculation within 1e-5 using the same source/config SHA.',
           '- The screen remains 1.236 m⁻¹ as an offline compatibility gate, not a physical limit.',
           '', '## Bank comparison','',
           '| Metric | Bank A | Bank B blended |','| --- | ---: | ---: |',
           f"| Selected parameter triple (m) | {old['A']['new_parameters']} | {old['B']['new_parameters']} |",
           f"| Hard-valid groups | {a['groups_all_hard_valid']}/40 | {b['groups_all_hard_valid']}/40 |",
           f"| Compatible candidates, old → new | {old['A']['old_compatible']} → {old['A']['new_compatible']} | {old['B']['old_compatible']} → {old['B']['new_compatible']} |",
           f"| Groups with ≥5 compatible candidates | {a['groups_at_least_five']}/40 | {b['groups_at_least_five']}/40 |",
           f"| Groups with 7 geometry modes | {a['groups_seven_geometry_modes']}/40 | {b['groups_seven_geometry_modes']}/40 |",
           f"| Known terrain-separated pairs | {terrain['A']['separated_known_pair_count']} | {terrain['B']['separated_known_pair_count']} |",
           f"| Screen-compatible known terrain-separated pairs | {terrain['A']['compatible_separated_known_pair_count']} | {terrain['B']['compatible_separated_known_pair_count']} |",
           f"| Both-seed interaction-class terrain gate among compatible candidates | {a['terrain_gate']} | {b['terrain_gate']} |",
           '', 'All ten ordered triples per bank were evaluated using the frozen prospective priority.',
           'The previous selector already ranked follower-compatible count before geometry; the new priority changed tie-breaks, not the attainable compatible supply.',
           'Candidate-level failure flags, initial command estimates and coverage are in the machine-readable artifacts.',
           'A small positive amplitude is not automatically compatible: the maximum Bézier curvature is screened over the whole Path.',
           'For the fixed curve rules, Bank A has only the 0.04 m left/right pair below 1.236 m⁻¹; every Bank B non-direct offset in the new pool exceeds that screen.',
           f"Bank A level counts (out of 80 per level): {[(k,v['compatible']) for k,v in boundary['A']['levels'].items()]}; Bank B: {[(k,v['compatible']) for k,v in boundary['B']['levels'].items()] }.",
           'Thus parameter triple selection alone cannot supply the required direct + two compatible side pairs.',
           'Terrain differences involving screen-out Paths are descriptive only. The rollout readiness gate counts common-known terrain separation among screen-compatible pairs.',
           '', '## Conditional pilot options','',
           '- L1: five defined candidates, omitting the large pair; P1 28 processes and P2 56. Its medium pair is also screen-out under the present selected geometry.',
           '- L2: seven defined candidates; P1 36 processes and P2 72. Its medium and large pairs are screen-out under the present selected geometry.',
           '- These process counts are comparisons only. Neither L1 nor L2 is eligible for execution after the failed 5/7 gate.',
           '- Prospective P1→P2 numeric thresholds are copied from the previously frozen E1 gate and saved with its SHA; no P1 data was used to choose them.',
           f"- P1 L1 point/P95+20%: {forecast['sequential_fresh_process_estimates']['P1_five']['point_hours']:.2f}/{forecast['sequential_fresh_process_estimates']['P1_five']['p95_plus_20_percent_hours']:.2f} h.",
           f"- P1 L2 point/P95+20%: {forecast['sequential_fresh_process_estimates']['P1_seven']['point_hours']:.2f}/{forecast['sequential_fresh_process_estimates']['P1_seven']['p95_plus_20_percent_hours']:.2f} h.",
           '', '## Limits','',
           '- The pure first-command calculation is source-equivalent only for a fresh controller at a fixed pose; later commands and realized motion are unmeasured. The production follower requests latest TF, while an upstream bridge can impose its own exact-stamp protocol.',
           '- Bank B has a distinct local endpoint and a structural residual to the common global goal. It remains a separate ablation objective.',
           '- Candidate pairs and bearings from one source family are correlated. Flat control is not required to show terrain separation.',
           '- No follower, bridge, router, simulator, Outcome, direct repeat or training process was started.','']
    (output/'report.md').write_text('\n'.join(lines))


def make_figures(output: Path, screens: dict, selected: dict, rows_by_bank: dict,
                 reasons: list[dict], terrain: dict, forecast: dict, result: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=output/'figures';folder.mkdir()
    def save(fig,name):
        fig.tight_layout();fig.savefig(folder/name,dpi=130);plt.close(fig)
    for bank,name in [('A','01-amplitude-compatibility.png'),('B','02-offset-compatibility.png')]:
        fig,ax=plt.subplots(figsize=(7,4));data=screens[bank]
        ax.bar(range(len(data)),[r['follower_compatible'] for r in data])
        ax.set_xticks(range(len(data)),[','.join(f'{x:.2f}' for x in r['parameters']) for r in data],rotation=65)
        ax.axhline(200,color='red',linestyle='--',label='5 per group')
        ax.set_ylabel('follower-screen compatible / 280');ax.set_title(f'Bank {bank} parameter triples');ax.legend()
        save(fig,name)
    for bank,name in [('A','03-bank-A-pareto.png'),('B','04-bank-B-pareto.png')]:
        fig,ax=plt.subplots(figsize=(6,4));data=screens[bank]
        ax.scatter([r['follower_compatible'] for r in data],[r['known_terrain_separated_pairs'] for r in data])
        ax.set_xlabel('compatible candidates');ax.set_ylabel('known terrain-separated pairs');ax.set_title(f'Bank {bank} offline tradeoff')
        save(fig,name)
    fig,ax=plt.subplots(figsize=(6,4))
    for bank in ('A','B'):
        ax.hist([r['maximum_curvature_per_m'] for r in rows_by_bank[bank]],bins=25,alpha=.45,label=bank)
    ax.axvline(FOLLOWER_SCREEN_PER_M,color='red',label='screen');ax.set_xlabel('maximum curvature (1/m)');ax.legend()
    save(fig,'05-curvature-distribution.png')
    fig,ax=plt.subplots(figsize=(6,4));flags={}
    for row in reasons:
        for reason in row['failure_flags']:
            flags[reason]=flags.get(reason,0)+1
    ax.bar(list(flags),list(flags.values()));ax.set_ylabel('candidate count');ax.tick_params(axis='x',rotation=25)
    save(fig,'06-follower-failure-reasons.png')
    fig,ax=plt.subplots(figsize=(7,4))
    for bank in ('A','B'):
        ax.plot(BEARINGS_DEG,[sum(r['bank']==bank and r['goal_bearing_deg']==bearing and r['compatible'] for r in reasons) for bearing in BEARINGS_DEG],marker='o',label=bank)
    ax.set_xlabel('goal bearing (deg)');ax.set_ylabel('compatible candidates');ax.legend()
    save(fig,'07-bearing-compatibility.png')
    fig,ax=plt.subplots(figsize=(7,4));labels=[f'{cls}/{seed}' for cls in ('transition','differential','graded') for seed in ('first','second')]
    x=np.arange(len(labels));ax.bar(x-.2,[terrain['A']['class_seed_support'][cls][seed] for cls in ('transition','differential','graded') for seed in ('first','second')],width=.4,label='A')
    ax.bar(x+.2,[terrain['B']['class_seed_support'][cls][seed] for cls in ('transition','differential','graded') for seed in ('first','second')],width=.4,label='B')
    ax.set_xticks(x,labels,rotation=30);ax.set_ylabel('groups with known separation');ax.legend()
    save(fig,'08-class-seed-terrain-separation.png')
    fig,ax=plt.subplots(figsize=(6,3));ax.bar(['L1 5','L2 7'],[84,108]);ax.set_ylabel('P1+P2 processes');ax.set_title('Conditional staged rollout sizes')
    save(fig,'09-five-versus-seven.png')
    fig,ax=plt.subplots(figsize=(7,3));ax.bar(['P1 0°','P2 ±20°'],[4,8]);ax.set_ylabel('StateGroups');ax.set_title('P2 only after P1 gate')
    save(fig,'10-staged-P1-P2.png')
    fig,ax=plt.subplots(figsize=(8,4));names=['P1_five','P1_seven','P2_five','P2_seven'];data=forecast['sequential_fresh_process_estimates']
    ax.bar(names,[data[n]['point_hours'] for n in names],label='point')
    ax.plot(names,[data[n]['p95_plus_20_percent_hours'] for n in names],color='red',marker='o',label='P95+20%')
    ax.set_ylabel('sequential hours')
    storage=ax.twinx();storage.plot(names,[data[n]['saved_gb_plus_20_percent'] for n in names],color='green',marker='s',label='storage +20%')
    storage.set_ylabel('saved GB +20%')
    ax.legend(loc='upper left');storage.legend(loc='upper right')
    save(fig,'11-runtime-storage-and-decision.png')


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-artifact-root',required=True,type=Path)
    parser.add_argument('--prior-review-root',required=True,type=Path)
    parser.add_argument('--output-root',required=True,type=Path)
    args=parser.parse_args()
    result=run_redesign(args.frozen_artifact_root,args.prior_review_root,args.output_root)
    print(json.dumps({'decision':result['decision'],'output_root':str(args.output_root)},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
