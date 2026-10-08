#!/usr/bin/env python3
"""Offline attainable-set audit of curvature-constrained goal candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.eat_nav_goal_bank_review import (FOLLOWER_CONFIG, _read_follower_config,
    load_anchor_inputs, plain, read, sha, write)
from scripts.eat_nav_goal_bank_redesign import (_forecast, _prospective_P1_P2_gates,
    _source_exact_audit, BASELINE_ARCHIVE_SHA256)
from visualnav_transformer.eat_nav.curvature_attainable import (
    CURVATURE_LIMIT_PER_M, ENDPOINT_TOLERANCE_M, eight_point_curvature, generate_goal_library,
    solve_distinct_modes, solve_symmetric_target)
from visualnav_transformer.eat_nav.candidate_bank_geometry import arc_resample
from visualnav_transformer.eat_nav.goal_bank_offline import (
    BEARINGS_DEG, audit_geometry, corridor_audit, corridor_class_proxy,
    follower_diagnostic, goal_for_bearing, pairwise_audit, path_sha)

REDESIGN_ARCHIVE_SHA = '36a6b011510a633c16b0bc9ba35b8c7d941e87739dd68cb9b758308b8e619c59'
BASE_COMMIT = '87b42aa836c213ce78a5c853c64878b14a0e43d1'
BASE_DECISION_SOURCES = (
    'src/visualnav_transformer/eat_nav/candidate_bank_geometry.py',
    'src/visualnav_transformer/eat_nav/goal_bank_offline.py',
    'src/visualnav_transformer/eat_nav/goal_bank_redesign.py',
    'src/visualnav_transformer/eat_nav/radial_interaction.py',
    'scripts/eat_nav_goal_bank_review.py',
    'scripts/eat_nav_goal_bank_redesign.py',
)
MODE = {'full_rms_min_m':.01, 'swept_exclusive_min_m2':.01,
        'signed_lateral_max_difference_min_m':.012,
        'curvature_profile_rms_min_per_m':.12,
        'distinct_path_sha_required':True}
CONFIG = {'schema':'eatnav_curvature_attainable_v1',
          'goal_distance_m':.60,'bearings_deg':list(BEARINGS_DEG),
          'future_path_xy_points':8,'curvature_screen_per_m':CURVATURE_LIMIT_PER_M,
          'endpoint_tolerance_m':ENDPOINT_TOLERANCE_M,
          'dense_spacing_max_m':.02,'initial_tangent_max_error_rad':1e-6,
          'known_corridor_rms_min_m':.01,'known_corridor_common_bins_min':5,
          'mode_separation':MODE,
          'mode_rule':'all five criteria; identical or near C0 alone does not invalidate Path',
          'fixed_bank_B_policy':'separate local-endpoint ablation; never combine with common-goal family',
          'forbidden_inputs':['Outcome','reward','Critic','learned evaluator'],
          'no_rollout':True}


def verify_redesign(backup:Path, prior:Path) -> dict:
    archive=backup/'eatnav-goal-bank-redesign-20261008-r4.tar.gz'
    if sha(archive)!=REDESIGN_ARCHIVE_SHA:
        raise ValueError('redesign archive SHA mismatch')
    with tarfile.open(archive,'r:gz') as bundle:
        ledger_bytes=bundle.extractfile('eatnav-goal-bank-redesign-20261008-r4/sha-ledger.json').read()
    if (prior/'sha-ledger.json').read_bytes()!=ledger_bytes:
        raise ValueError('redesign ledger differs from archive')
    ledger=read(prior/'sha-ledger.json')['files']
    bad=[name for name,digest in ledger.items() if not (prior/name).is_file() or sha(prior/name)!=digest]
    if bad:
        raise ValueError('redesign artifact SHA mismatch: '+','.join(bad))
    return {'archive':str(archive),'archive_sha256':sha(archive),
            'ledger_sha256':hashlib.sha256(ledger_bytes).hexdigest(),
            'verified_files':len(ledger),'root':str(prior)}


def verify_baseline_portable(backup:Path, prior:Path) -> dict:
    """Resolve archived roots through the supplied backup, not historical /tmp."""
    archive=backup/'eatnav-goal-bearing-bank-review-20261008-r6.tar.gz'
    if sha(archive)!=BASELINE_ARCHIVE_SHA256:
        raise ValueError('prior review archive SHA mismatch')
    with tarfile.open(archive,'r:gz') as bundle:
        archived=bundle.extractfile('eatnav-goal-bearing-bank-review-20261008-r6/sha-ledger.json').read()
    if (prior/'sha-ledger.json').read_bytes()!=archived:
        raise ValueError('prior review ledger differs from backup')
    ledger=read(prior/'sha-ledger.json')['files']
    bad=[name for name,digest in ledger.items() if not (prior/name).is_file() or sha(prior/name)!=digest]
    if bad:
        raise ValueError('prior review artifact SHA mismatch: '+','.join(bad))
    provenance=read(prior/'source-input-provenance.json')
    saved_backup=Path('/home/kojima/eatnav-backups/2026-10-08-pre-shutdown')
    source_map={}
    root=Path(__file__).resolve().parents[1]
    for path,digest in provenance['sha256'].items():
        old=Path(path)
        if old.is_relative_to(saved_backup):
            resolved=backup/old.relative_to(saved_backup)
        elif old.name=='e1-gates.yaml':
            resolved=root/'docs/quality-identifiability-gate/candidate-coverage/e1-feasibility/e1-gates.yaml'
        else:
            resolved=old
        if not resolved.is_file() or sha(resolved)!=digest:
            raise ValueError('mapped source SHA mismatch: '+path+' -> '+str(resolved))
        source_map[path]=str(resolved)
    frozen_source=backup/'eatnav-onix/teaching_playback/simple_trajectory_follower/src'
    live_source=Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/src')
    for name in ('simple_trajectory_follower.cpp','path_geometry.cpp'):
        if sha(frozen_source/name)!=sha(live_source/name):
            raise ValueError('follower source does not match frozen backup: '+name)
    if sha(FOLLOWER_CONFIG)!=sha(backup/'eatnav-onix/teaching_playback/simple_trajectory_follower/config/onix.yaml'):
        raise ValueError('production follower config does not match backup')
    module=root/'src/visualnav_transformer/eat_nav/goal_bank_offline.py'
    if sha(module)!=provenance['offline_module_sha256']:
        raise ValueError('prior goal geometry module SHA mismatch')
    return {'archive':str(archive),'archive_sha256':sha(archive),
            'ledger_sha256':hashlib.sha256(archived).hexdigest(),
            'verified_files':len(ledger),'mapped_source_sha_count':len(source_map),
            'mapped_source_paths':source_map,'root':str(prior)}


def verify_g0_regression(groups:list[dict],libraries:dict,redesign:Path) -> dict:
    prior=read(redesign/'selected-candidate-definitions.json')
    matched=0
    failures=[]
    for item in prior:
        group=next(g for g in groups if g['state_group_id']==item['state_group_id'])
        curves=libraries[group['bearing_deg']][0]
        family='G0A' if item['bank']=='A' else 'G0B'
        side=item['side']
        magnitude=abs(float(item['signed_amplitude_or_offset_m']))
        variant=f"{item['bank']}-{side}-{magnitude:.2f}"
        curve=next((c for c in curves if c['family']==family and c['variant_id']==variant),None)
        if curve is None or curve['path_sha256']!=item['path_sha256']:
            failures.append(item['state_group_id']+':'+item['bank']+':'+variant)
        else:matched+=1
    if failures:
        raise ValueError('G0 prior Path SHA regression: '+','.join(failures[:10]))
    return {'matched_selected_prior_paths':matched,'prior_count':len(prior),
            'selected_candidate_definitions_sha256':sha(redesign/'selected-candidate-definitions.json'),
            'G0A_B_exact_path_sha':True}


def verify_base_sources(root:Path) -> dict:
    result={}
    for relative in BASE_DECISION_SOURCES:
        expected=subprocess.check_output(['git','show',f'{BASE_COMMIT}:{relative}'],cwd=root)
        actual=(root/relative).read_bytes()
        if actual!=expected:
            raise ValueError('decision dependency differs from base commit: '+relative)
        result[relative]={'sha256':hashlib.sha256(actual).hexdigest(),
                          'verified_against_base_commit':BASE_COMMIT}
    return result


def verify_anchor_identity(anchors:list[dict],frozen:list[dict]) -> dict:
    if len(anchors)!=8 or len(frozen)!=8:
        raise ValueError('eight anchor identity required')
    tested=[]
    for anchor,old in zip(anchors,frozen):
        differences=[key for key,expected in old.items()
                     if key not in anchor or plain(anchor[key])!=expected]
        if differences:
            raise ValueError(f"frozen anchor mismatch {old.get('anchor_id')}: {differences}")
        tested.append({'anchor_id':old['anchor_id'],'pose_id':old['pose_id'],
                       'dem_history_sha256':old['dem_history_sha256'],
                       'source_path_shas':old['source_path_shas']})
    return {'all_fields_match_prior_eight_anchor_freeze':True,
            'anchor_count':8,'verified':tested}


def row_for(curve:dict,goal:np.ndarray) -> dict:
    bank='B' if curve['family']=='G0B' else 'A'
    return {'bank':bank,'dense':curve['dense'],'stations_m':curve['stations_m'],
            'waypoints':curve['waypoints'],'endpoint':np.asarray(curve['waypoints'])[-1],
            'global_goal':goal,'arc_length_m':curve['arc_length_m'],
            'maximum_curvature_per_m':curve['analytic_or_dense_max_curvature_per_m']}


def signed_side(curve:dict,goal:np.ndarray) -> tuple[str,float]:
    normal=np.array([-goal[1],goal[0]])/np.linalg.norm(goal)
    dense=np.asarray(curve['dense'])
    chord=dense@normal
    weighted=float(np.mean(chord))
    return ('left' if weighted>1e-6 else 'right' if weighted< -1e-6 else 'direct',weighted)


def source_visible_path_terms(path:np.ndarray,command:dict,follower:dict) -> dict:
    """Record C0's path-only terms; command authority remains c0_fresh_forward."""
    path=np.asarray(path,dtype=float)
    segments=np.diff(path,axis=0)
    lengths=np.linalg.norm(segments,axis=1)
    best=None
    for index,length in enumerate(lengths):
        if length<1e-6:continue
        ratio=float(np.clip(-path[index]@segments[index]/length**2,0,1))
        projection=path[index]+ratio*segments[index]
        remaining=(1-ratio)*length+float(lengths[index+1:].sum())
        key=(float(np.linalg.norm(projection)),-remaining)
        if best is None or key<best[0]:best=(key,index,projection,remaining)
    if best is None:
        return {'valid':False,'reason':'degenerate_Path'}
    _,index,projection,remaining=best
    return {'valid':True,'path_point_count':len(path),
            'first_path_point_xy_m':path[0].tolist(),
            'last_path_point_xy_m':path[-1].tolist(),
            'first_chord_heading_rad':float(math.atan2(path[0,1],path[0,0])),
            'path_polyline_length_m':float(lengths.sum()),
            'nearest_projection_segment_index':index,
            'nearest_projection_xy_m':projection.tolist(),
            'nearest_projection_distance_m':float(np.linalg.norm(projection)),
            'remaining_polyline_m':float(remaining),
            'lookahead_distance_m':float(follower['lookahead_distance']),
            'goal_extension_branch_expected':remaining<=float(follower['lookahead_distance']),
            'carrot_xy_m':command['initial_command_estimate'].get('carrot_xy'),
            'command_mode':command['command_mode']}


def separation(first:dict,second:dict,first_dem:dict,second_dem:dict,
               goal:np.ndarray) -> dict:
    pair=pairwise_audit(first,second,first_dem,second_dem)
    length=min(first['arc_length_m'],second['arc_length_m'])
    stations=np.linspace(0,length,101)
    def sample(row):
        return np.stack([np.interp(stations,row['stations_m'],row['dense'][:,axis])
                         for axis in range(2)],axis=1)
    a,b=sample(first),sample(second)
    normal=np.array([-goal[1],goal[0]])/np.linalg.norm(goal)
    lateral=float(np.max(np.abs((a-b)@normal)))
    def profile(row):
        return np.interp(stations,row['stations_m'],row['signed_curvature_profile_per_m'])
    curvature_rms=float(np.sqrt(np.mean((profile(first)-profile(second))**2)))
    distinct=(first['path_sha256']!=second['path_sha256'] and
              pair['full_geometry_rms_m']>=MODE['full_rms_min_m'] and
              pair['pairwise_exclusive_area_m2']>=MODE['swept_exclusive_min_m2'] and
              lateral>=MODE['signed_lateral_max_difference_min_m'] and
              curvature_rms>=MODE['curvature_profile_rms_min_per_m'])
    terrain=(pair['common_known_corridor_bins']>=CONFIG['known_corridor_common_bins_min']
             and pair['known_corridor_profile_rms_m'] is not None
             and pair['known_corridor_profile_rms_m']>=CONFIG['known_corridor_rms_min_m'])
    return {'distinct':bool(distinct),'terrain_separated':bool(terrain),
            'lateral_max_difference_m':lateral,'curvature_profile_rms_per_m':curvature_rms,
            **pair}


def evaluate_curve(curve:dict,goal:np.ndarray,anchor:dict,follower:dict) -> dict:
    row=row_for(curve,goal)
    geometry=audit_geometry(row,goal,anchor['settled_map_xy_yaw'])
    endpoint_goal_error=float(np.linalg.norm(np.asarray(curve['waypoints'])[-1]-goal))
    tangent_error=float(math.atan2(abs(curve['initial_tangent_xy'][1]),
                                  curve['initial_tangent_xy'][0]))
    reasons=list(geometry['invalid_reasons'])
    if curve['endpoint_error_m']>ENDPOINT_TOLERANCE_M:
        reasons.append('endpoint_numerical_residual')
    if curve['family']!='G0B' and endpoint_goal_error>ENDPOINT_TOLERANCE_M:
        reasons.append('common_global_goal_residual')
    if curve['family'] in ('G1','G2','G3') and tangent_error>CONFIG['initial_tangent_max_error_rad']:
        reasons.append('initial_tangent_error')
    if curve['analytic_or_dense_max_curvature_per_m']>CURVATURE_LIMIT_PER_M+1e-9:
        reasons.append('analytic_curvature_screen')
    if curve['eight_point_max_curvature_per_m']>CURVATURE_LIMIT_PER_M+1e-9:
        reasons.append('eight_point_curvature_screen')
    command=follower_diagnostic(row,follower)
    if command['command_mode']!='tracking':
        reasons.append('first_command_'+command['command_mode'])
    if command['reverse_estimate']:
        reasons.append('reverse_estimate')
    side,lateral=signed_side(curve,goal)
    result={'family':curve['family'],'variant_id':curve['variant_id'],
            'path_sha256':path_sha(curve['waypoints']),'side':side,
            'signed_mean_lateral_m':lateral,
            'analytic_curve_compatible':geometry['hard_valid'] and
                 curve['endpoint_error_m']<=ENDPOINT_TOLERANCE_M and
                 (curve['family']=='G0B' or endpoint_goal_error<=ENDPOINT_TOLERANCE_M) and
                 curve['analytic_or_dense_max_curvature_per_m']<=CURVATURE_LIMIT_PER_M+1e-9,
            'eight_point_compatible':curve['eight_point_max_curvature_per_m']<=CURVATURE_LIMIT_PER_M+1e-9,
            'first_command_compatible':command['command_mode']=='tracking' and not command['reverse_estimate'],
            'combined_compatible':not reasons,
            'hard_valid':geometry['hard_valid'] and endpoint_goal_error<=ENDPOINT_TOLERANCE_M,
            'invalid_reasons':reasons,'analytic_or_dense_max_curvature_per_m':curve['analytic_or_dense_max_curvature_per_m'],
            'eight_point_max_curvature_per_m':curve['eight_point_max_curvature_per_m'],
            'endpoint_goal_error_m':endpoint_goal_error,
            'endpoint_numerical_error_m':curve['endpoint_error_m'],
            'initial_tangent_error_rad':tangent_error,
            'arc_length_m':curve['arc_length_m'],
            'first_command':command,
            'source_visible_path_terms':source_visible_path_terms(curve['waypoints'],command,follower)}
    if result['combined_compatible']:
        dem=corridor_audit(row,anchor['dem'][0],anchor['dem'][1],anchor['proxy_bboxes'])
        if (dem['centerline_known_fraction']==0 or
                dem['corridors']['main_track_sprocket']['known_fraction']==0):
            result['combined_compatible']=False
            result['invalid_reasons'].append('unknown_only_corridor')
        else:
            result['terrain']={'centerline_known_fraction':dem['centerline_known_fraction'],
                 'centerline_unknown_fraction':dem['centerline_unknown_fraction'],
                 'main_track_known_fraction':dem['corridors']['main_track_sprocket']['known_fraction'],
                 'interaction_onset_station_m':dem['interaction_onset_station_m'],
                 'path_aligned_class_proxy':corridor_class_proxy(row,anchor['dem'][0],anchor['dem'][1])['class_proxy']}
        return result,dem
    return result,None


def analyze_group(group:dict,anchor:dict,curves:list[dict],failures:list[dict],
                  follower:dict) -> tuple[dict,list[dict],list[dict]]:
    goal=np.asarray(group['global_goal_base_xy_m'])
    records=[]; compatible=[]
    for curve in curves:
        record,dem=evaluate_curve(curve,goal,anchor,follower)
        records.append(record)
        if record['combined_compatible'] and curve['family']!='G0B':
            compatible.append((curve,record,dem))
    conflicts=[]; terrain_pairs=[]; pair_rows=[];mode_failures=Counter()
    for i,j in itertools.combinations(range(len(compatible)),2):
        a,ar,ad=compatible[i];b,br,bd=compatible[j]
        metric=separation(a,b,ad,bd,goal)
        if not metric['distinct']:
            conflicts.append((i,j))
            if a['path_sha256']==b['path_sha256']:mode_failures['identical_path_sha']+=1
            if metric['full_geometry_rms_m']<MODE['full_rms_min_m']:mode_failures['centerline_rms']+=1
            if metric['pairwise_exclusive_area_m2']<MODE['swept_exclusive_min_m2']:mode_failures['swept_exclusive_area']+=1
            if metric['lateral_max_difference_m']<MODE['signed_lateral_max_difference_min_m']:mode_failures['signed_lateral']+=1
            if metric['curvature_profile_rms_per_m']<MODE['curvature_profile_rms_min_per_m']:mode_failures['curvature_profile']+=1
        elif metric['terrain_separated']:
            terrain_pairs.append((i,j))
        if metric['distinct'] or metric['terrain_separated']:
            pair_rows.append({'first':ar['variant_id'],'second':br['variant_id'],
                              'first_family':ar['family'],'second_family':br['family'],
                              **metric})
    maximum=solve_distinct_modes(len(compatible),conflicts)
    direct=next((i for i,(_,r,_) in enumerate(compatible)
                 if r['family']=='G0A' and r['variant_id']=='A-direct-0.00'),None)
    sides=[r['side'] for _,r,_ in compatible]
    symmetric={str(1+2*n):solve_symmetric_target(sides,conflicts,pair_count=n,
                              direct_index=direct) if direct is not None else {'feasible':False}
               for n in (1,2,3)}
    record={'state_group_id':group['state_group_id'],'anchor_id':anchor['anchor_id'],
            'seed_label':anchor['seed_label'],'source_family':anchor['source_family'],
            'terrain_class':anchor['terrain_class'],'bearing_deg':group['bearing_deg'],
            'generated':len(curves),'construction_failures':len(failures),
            'hard_valid':sum(r['hard_valid'] for r in records),
            'analytic_compatible':sum(r['analytic_curve_compatible'] for r in records),
            'eight_point_compatible':sum(r['eight_point_compatible'] for r in records),
            'first_command_compatible':sum(r['first_command_compatible'] for r in records),
            'compatible_common_goal':len(compatible),
            'compatible_B_ablation':sum(r['combined_compatible'] and r['family']=='G0B' for r in records),
            'compatible_by_family':dict(Counter(r['family'] for _,r,_ in compatible)),
            'rejection_reasons':dict(Counter(reason for r in records for reason in r['invalid_reasons'])),
            'mode_conflict_reasons':dict(mode_failures),
            'maximum_distinct_finite_library':maximum,
            'symmetric_direct_plus_pairs':symmetric,
            'distinct_terrain_separated_pair_count':len(terrain_pairs),
            'terrain_pair_by_family':dict(Counter('+'.join(sorted((compatible[i][1]['family'],compatible[j][1]['family'])))
                              for i,j in terrain_pairs)),
            'compatible_path_shas':[r['path_sha256'] for _,r,_ in compatible],
            'compatible_ids':[r['family']+':'+r['variant_id'] for _,r,_ in compatible]}
    return record,records,pair_rows


def alternative_attainable(curves:list[dict],goal:np.ndarray,anchor:dict,
                           follower:dict,screen_per_m:float) -> dict:
    """Diagnostic-only full mode recheck for one anchor and one isolated change."""
    valid=[]
    for curve in curves:
        if curve['family']=='G0B':continue
        row=row_for(curve,goal)
        geometry=audit_geometry(row,goal,anchor['settled_map_xy_yaw'])
        if not geometry['hard_valid'] or curve['endpoint_error_m']>ENDPOINT_TOLERANCE_M:
            continue
        if np.linalg.norm(curve['waypoints'][-1]-goal)>ENDPOINT_TOLERANCE_M:
            continue
        if max(curve['analytic_or_dense_max_curvature_per_m'],
               curve['eight_point_max_curvature_per_m'])>screen_per_m:
            continue
        command=follower_diagnostic(row,follower)
        if command['command_mode']!='tracking' or command['reverse_estimate']:
            continue
        dem=corridor_audit(row,anchor['dem'][0],anchor['dem'][1],
                           {'main_track_sprocket':anchor['proxy_bboxes']['main_track_sprocket']})
        if (dem['centerline_known_fraction']==0 or
                dem['corridors']['main_track_sprocket']['known_fraction']==0):
            continue
        valid.append((curve,dem,signed_side(curve,goal)[0]))
    conflicts=[]
    for i,j in itertools.combinations(range(len(valid)),2):
        if not separation(valid[i][0],valid[j][0],valid[i][1],valid[j][1],goal)['distinct']:
            conflicts.append((i,j))
    maximum=solve_distinct_modes(len(valid),conflicts)
    direct=next((i for i,(curve,_,_) in enumerate(valid)
                 if curve['family']=='G0A' and curve['variant_id']=='A-direct-0.00'),None)
    sides=[side for _,_,side in valid]
    targets={str(1+2*n):solve_symmetric_target(sides,conflicts,pair_count=n,
               direct_index=direct)['feasible'] if direct is not None else False for n in (1,2,3)}
    return {'anchor_id':anchor['anchor_id'],'compatible_supply':len(valid),
            'side_supply':dict(Counter(sides)),
            'maximum_distinct_finite_library':maximum['maximum_mode_count'],
            'maximum_solver_optimal':maximum['optimal'],
            'symmetric_direct_plus_pairs':targets,
            'not_a_frozen_readiness_result':True}


def sensitivity_diagnostic(groups:list[dict],counts:list[dict],libraries:dict,
                           all_records:list[dict],anchors_by:dict,follower:dict) -> dict:
    """Isolated mathematical perturbations; never reuse as a frozen experiment."""
    bearing_cases={}
    for bearing in BEARINGS_DEG:
        group=next(g for g in groups if g['bearing_deg']==bearing)
        records=[r for r in all_records if r['state_group_id']==group['state_group_id']]
        common=[r for r in records if r['family']!='G0B' and r['hard_valid'] and r['first_command_compatible']]
        by_screen={}
        for screen in (1.236,1.5,1.8,2.,2.7,3.):
            usable=[r for r in common if max(r['analytic_or_dense_max_curvature_per_m'],
                                             r['eight_point_max_curvature_per_m'])<=screen]
            by_screen[str(screen)]={side:len({r['path_sha256'] for r in usable if r['side']==side})
                                   for side in ('direct','left','right')}
        required={}
        for side in ('left','right'):
            values=sorted({r['path_sha256']:max(r['analytic_or_dense_max_curvature_per_m'],
                                               r['eight_point_max_curvature_per_m'])
                           for r in common if r['side']==side}.values())
            required[side]=values[1] if len(values)>=2 else None
        curves=libraries[bearing][0]
        point_count={}
        for n in (8,12,16):
            rescued=0;total=0
            for curve,r in zip(curves,records):
                if r['family']=='G0B' or not r['hard_valid'] or not r['first_command_compatible']:
                    continue
                if curve['analytic_or_dense_max_curvature_per_m']>CURVATURE_LIMIT_PER_M:continue
                total+=1
                path=arc_resample(curve['dense'],n)
                if eight_point_curvature(np.vstack(([0.,0.],path)))<=CURVATURE_LIMIT_PER_M:
                    rescued+=1
            point_count[str(n)]={'discrete_screen_pass':rescued,'analytic_screen_pass':total,
                                 'not_source_parity_validated':n!=8}
        bearing_cases[str(bearing)]={'unique_side_supply_by_curvature_screen':by_screen,
             'minimum_screen_for_two_per_side_ignoring_mode_separation':required,
             'point_count_discrete_only':point_count,
             'mode_conflict_reasons':next(c['mode_conflict_reasons'] for c in counts if c['state_group_id']==group['state_group_id'])}
    anchor=anchors_by[groups[0]['anchor_id']]
    alternative_screens={}
    for screen in (1.5,1.8,2.):
        alternative_screens[str(screen)]={str(b):alternative_attainable(
            libraries[b][0],np.asarray(goal_for_bearing(b)),anchor,follower,screen)
            for b in BEARINGS_DEG}
    distances={}
    for distance in (.75,.9):
        distances[str(distance)]={}
        for bearing in BEARINGS_DEG:
            goal=np.asarray(goal_for_bearing(bearing))*distance/.6
            curves,_=generate_goal_library(goal)
            usable=[c for c in curves if c['family']!='G0B'
                    and c['analytic_or_dense_max_curvature_per_m']<=CURVATURE_LIMIT_PER_M
                    and c['eight_point_max_curvature_per_m']<=CURVATURE_LIMIT_PER_M]
            sides=Counter(signed_side(c,goal)[0] for c in usable)
            distances[str(distance)][str(bearing)]={
                'screen_compatible_curve_supply_without_DEM_or_C0':dict(sides),
                'two_per_side_supply_only':sides['left']>=2 and sides['right']>=2,
                'representative_anchor_mode_recheck':alternative_attainable(
                    curves,goal,anchor,follower,CURVATURE_LIMIT_PER_M),
                'not_a_frozen_readiness_result':True}
    return {'status':'diagnostic_only_no_adoption',
      'frozen':{'goal_distance_m':.6,'bearings_deg':list(BEARINGS_DEG),
                'curvature_screen_per_m':CURVATURE_LIMIT_PER_M,'point_count':8},
      'finite_library_observation':{'minimum_modes':min(r['maximum_distinct_finite_library']['maximum_mode_count'] for r in counts),
          'five_symmetric_groups':sum(r['symmetric_direct_plus_pairs']['5']['feasible'] for r in counts)},
      'frozen_bearing_analysis':bearing_cases,
      'changed_curvature_screen_one_anchor_mode_recheck':alternative_screens,
      'goal_distance_isolated_geometry_only':distances,
      'bearing_isolated_scope':{'bearing_0_only_groups':8,'bearing_0_five_symmetric_groups':sum(r['bearing_deg']==0 and r['symmetric_direct_plus_pairs']['5']['feasible'] for r in counts),
                                'remove_20_deg_groups_is_not_a_pass_of_frozen_gate':True},
      'interpretation':'The representative first-transition anchor reaches symmetric five at a 1.8/m mathematical curvature screen or 0.75m goal distance, when each is changed alone and the same finite-library mode solver is rerun. The former exceeds the frozen production compatibility screen; the latter requires a changed experiment and verification across all 40 physical StateGroups. No altered setting is adopted.',
      'no_threshold_or_path_count_changed':True}


def _plot(output:Path,counts:list[dict],examples:dict) -> None:
    figdir=output/'figures';figdir.mkdir(exist_ok=True)
    palette={'G0A':'#555555','G0B':'#bb8822','G1':'#2972b2','G2':'#20a078','G3':'#ae4b80'}
    figures=[('01-G0-G3-example-curves','curves'),('02-curvature-profiles','curvature'),
             ('03-compatibility-regions','regions'),('04-attainable-by-bearing','counts'),
             ('05-compatible-terrain-separation','terrain'),('06-eight-point-effects','discretization'),
             ('07-five-seven-gates','gates'),('08-decision-flow','flow')]
    for name,kind in figures:
        fig,ax=plt.subplots(figsize=(7.5,4.6))
        if kind in ('curves','curvature','regions','discretization'):
            sample=examples['all_curves'] if kind in ('regions','discretization') else examples['curves']
            for curve,record in sample:
                if kind=='curves':
                    ax.plot(curve['dense'][:,0],curve['dense'][:,1],label=curve['family']+':'+curve['variant_id'],
                            color=palette[curve['family']])
                    ax.scatter(curve['waypoints'][:,0],curve['waypoints'][:,1],s=10,color=palette[curve['family']])
                elif kind=='curvature':
                    ax.plot(curve['stations_m'],curve['signed_curvature_profile_per_m'],
                            label=curve['family']+':'+curve['variant_id'],color=palette[curve['family']])
                elif kind=='regions':
                    ax.scatter(curve['arc_length_m'],curve['analytic_or_dense_max_curvature_per_m'],
                               marker='o' if record['combined_compatible'] else 'x',color=palette[curve['family']],s=13)
                else:
                    ax.scatter(curve['analytic_or_dense_max_curvature_per_m'],curve['eight_point_max_curvature_per_m'],
                               marker='o' if record['combined_compatible'] else 'x',color=palette[curve['family']],s=13)
            if kind in ('curvature','regions','discretization'):
                ax.axhline(CURVATURE_LIMIT_PER_M,color='red',linestyle='--',label='1.236 screen')
            if kind=='discretization':ax.axvline(CURVATURE_LIMIT_PER_M,color='red',linestyle='--')
            labels={'curves':('robot-forward x (m)','robot-left y (m)'),
                    'curvature':('arc length (m)','signed curvature (1/m)'),
                    'regions':('Path arc length (m)','maximum dense curvature (1/m)'),
                    'discretization':('maximum dense curvature (1/m)','eight-point curvature (1/m)')}
            ax.set_xlabel(labels[kind][0]);ax.set_ylabel(labels[kind][1])
        elif kind=='counts':
            for bearing in BEARINGS_DEG:
                subset=[r['maximum_distinct_finite_library']['maximum_mode_count'] for r in counts if r['bearing_deg']==bearing]
                ax.scatter([bearing]*len(subset),subset)
            ax.set_xticks(BEARINGS_DEG)
            ax.set_xlabel('goal bearing (degrees)');ax.set_ylabel('distinct finite-library modes')
        elif kind=='terrain':
            labels=[r['terrain_class'] for r in counts]
            ax.bar(range(len(counts)),[r['distinct_terrain_separated_pair_count'] for r in counts])
            ax.set_xticks(range(len(counts)),labels,rotation=90,fontsize=5)
            ax.set_xlabel('StateGroup terrain class');ax.set_ylabel('compatible known-terrain pairs')
        elif kind=='gates':
            ax.bar(['3','5','7'],[sum(r['symmetric_direct_plus_pairs'][n]['feasible'] for r in counts) for n in ('3','5','7')])
            ax.axhline(40,color='red',linestyle='--')
            ax.set_xlabel('direct + symmetric candidate target');ax.set_ylabel('qualifying StateGroups / 40')
        else:
            ax.text(.04,.75,'source SHA + C0 parity → G0–G3 geometry → follower screen',transform=ax.transAxes)
            ax.text(.04,.5,'→ distinct finite-library modes → 3 / 5 / 7 gates',transform=ax.transAxes)
            ax.text(.04,.25,'→ offline decision; no rollout',transform=ax.transAxes)
            ax.set_axis_off()
        ax.set_title(name.replace('-',' '));ax.grid(alpha=.2)
        if kind in ('curves','curvature'):ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=140);plt.close(fig)


def run(backup:Path,prior:Path,redesign:Path,output:Path) -> dict:
    if output.exists():raise FileExistsError('new output root required')
    provenance=verify_baseline_portable(backup,prior)
    provenance['redesign']=verify_redesign(backup,redesign)
    anchors,anchor_sources=load_anchor_inputs(backup)
    groups=read(prior/'state-groups.json')
    old_anchors=read(prior/'eight-anchor-freeze.json')
    if len(anchors)!=8 or len(groups)!=40:
        raise ValueError('anchor or group freeze mismatch')
    anchor_identity=verify_anchor_identity(anchors,old_anchors)
    for group in groups:
        if not np.allclose(group['global_goal_base_xy_m'],goal_for_bearing(group['bearing_deg']),atol=1e-12):
            raise ValueError('goal freeze mismatch')
    source_exact=_source_exact_audit(backup,_read_follower_config(FOLLOWER_CONFIG))
    root=Path(__file__).resolve().parents[1]
    gates=_prospective_P1_P2_gates(root)
    provenance.update({'anchor_source':anchor_sources,
         'frozen_anchor_identity':anchor_identity,
         'base_decision_source_sha256':verify_base_sources(root),
         'selected_anchors':[
        {'anchor_id':a['anchor_id'],'pose_id':a['pose_id'],'seed_label':a['seed_label'],
         'source_family':a['source_family'],'dem_history_sha256':a['dem_history_sha256'],
         'source_path_shas':a['source_path_shas']} for a in anchors],
         'goal_group_source':str(prior/'state-groups.json'),
         'goal_group_sha256':sha(prior/'state-groups.json'),
         'redesign_candidate_sha256':sha(redesign/'selected-candidate-definitions.json'),
         'prior_candidate_sha256':sha(prior/'candidate-definitions.json'),
         'prior_compatibility_report_sha256':sha(redesign/'report.md'),
         'frozen_e1_gates_sha256':gates['numeric_source_sha256']})
    provenance['attainable_generator_source_sha256']=sha(root/'src/visualnav_transformer/eat_nav/curvature_attainable.py')
    provenance['attainable_runner_source_sha256']=sha(Path(__file__))
    output.mkdir(parents=True)
    write(output/'source-provenance.json',provenance)
    write(output/'attainable-set-config.json',CONFIG)
    write(output/'generator-family-contracts.json',{
        'G0A':'exact prior common-endpoint cubic baseline','G0B':'exact prior local-endpoint ablation; excluded from common bank',
        'G1':'fixed forward start tangent cubic, deterministic control grid',
        'G2':'integrated constant/ramp/symmetric/sign-changing curvature profiles; bounded least-squares endpoint solve',
        'G3':'arc-line-arc and biarc with continuous pose/tangent, bounded two-curvature solve',
        'G0A_amplitudes_m':[0.,.04,.08,.12,.16,.20],
        'G0B_offsets_m':[0.,.04,.06,.08,.10,.12],
        'G1_forward_handles_m':[.08,.12,.16,.20],
        'G1_terminal_handles_m':[.08,.14,.20],
        'G1_terminal_lateral_m':[-.08,-.06,-.04,-.02,0.,.02,.04,.06,.08],
        'G2_profiles':['constant','linear_ramp','symmetric_ramp','sign_change'],
        'G2_signchange_lengths_m':[.60,.66,.72,.78],
        'G2_signchange_a_per_m':[-.8,-.4,0.,.4,.8],
        'G3_lengths_m':[.60,.612,.62,.68,.74,.80],
        'G3_segment_fractions':[[.5,0.,.5],[.495,.01,.495],[.25,.5,.25],[.2,.5,.3],[.3,.5,.2]],
        'G3_curvature_initial_seeds':[[0.,0.],[.7,-.7],[-.7,.7]],
        'solver_endpoint_tolerance_m':ENDPOINT_TOLERANCE_M,
        'endpoint_projection_allowed':False,'selection_uses_Outcome':False})
    write(output/'follower-source-parity.json',source_exact)
    anchor_by={a['anchor_id']:a for a in anchors}
    by_bearing={}
    for bearing in BEARINGS_DEG:
        by_bearing[bearing]=generate_goal_library(goal_for_bearing(bearing))
    g0_parity=verify_g0_regression(groups,by_bearing,redesign)
    write(output/'g0-path-regression.json',g0_parity)
    counts=[];all_records=[];all_pairs=[];examples={'curves':[]}
    for index,group in enumerate(groups):
        curves,failures=by_bearing[group['bearing_deg']]
        record,candidates,pairs=analyze_group(group,anchor_by[group['anchor_id']],curves,failures,
                                              _read_follower_config(FOLLOWER_CONFIG))
        counts.append(record)
        all_records.extend([{'state_group_id':record['state_group_id'],**candidate} for candidate in candidates])
        all_pairs.extend([{'state_group_id':record['state_group_id'],**pair} for pair in pairs])
        if index==0:
            examples['all_curves']=list(zip(curves,candidates))
        if group['bearing_deg']==0 and group['anchor_id']==anchors[0]['anchor_id']:
            for family in ('G0A','G0B','G1','G2','G3'):
                options=[(curve,r) for curve,r in zip(curves,candidates)
                         if curve['family']==family and r['combined_compatible']]
                if options:
                    examples['curves'].append(max(options,key=lambda pair:abs(pair[1]['signed_mean_lateral_m'])))
        print(f"{index+1}/40 {record['state_group_id']} compatible={record['compatible_common_goal']} distinct={record['maximum_distinct_finite_library']['maximum_mode_count']}",flush=True)
    # Dense per-candidate/per-pair diagnostics remain outside Git.
    write(output/'curve-library-records.json',all_records)
    write(output/'compatible-distinct-pairs.json',all_pairs)
    family_stats={family:{'generated_per_bearing':{str(b):sum(c['family']==family for c in by_bearing[b][0]) for b in BEARINGS_DEG},
                          'construction_failures_per_bearing':{str(b):sum(c['family']==family for c in by_bearing[b][1]) for b in BEARINGS_DEG},
                          'compatible_across_groups':sum(r['combined_compatible'] and r['family']==family for r in all_records)}
                  for family in ('G0A','G0B','G1','G2','G3')}
    write(output/'curve-library-summary.json',{'families':family_stats,
         'construction_failures':{str(b):by_bearing[b][1] for b in BEARINGS_DEG},
         'rejection_reason_counts':dict(Counter(reason for row in all_records for reason in row['invalid_reasons'])),
         'curve_records_file':'curve-library-records.json','pair_file':'compatible-distinct-pairs.json'})
    values=[r['maximum_distinct_finite_library']['maximum_mode_count'] for r in counts]
    target_counts={n:sum(r['symmetric_direct_plus_pairs'][str(n)]['feasible'] for r in counts) for n in (3,5,7)}
    all_optimal=all(r['maximum_distinct_finite_library']['optimal'] for r in counts)
    breakdown={key:{str(value):[r['maximum_distinct_finite_library']['maximum_mode_count'] for r in counts if r[key]==value]
                    for value in sorted({r[key] for r in counts})}
               for key in ('bearing_deg','terrain_class','seed_label','source_family')}
    mode_report={'groups':counts,'finite_library_optimal_all_groups':all_optimal,
       'unconstrained_distinct_min_median_max':[min(values),float(np.median(values)),max(values)],
       'unconstrained_groups_at_least':{str(n):sum(x>=n for x in values) for n in (3,5,7)},
       'symmetric_direct_plus_pairs_groups':target_counts,'breakdown':breakdown,
       'mode_criteria':MODE,'finite_library_only_not_continuous_attainability_proof':True}
    write(output/'candidate-attainable-counts.json',mode_report)
    terrain_support={cls:{seed:sum(r['terrain_class']==cls and r['seed_label']==seed and r['distinct_terrain_separated_pair_count']>0 for r in counts)
                          for seed in ('first','second')}
                     for cls in ('transition','differential','graded')}
    write(output/'terrain-separation-compatible-only.json',{
        'class_seed_support':terrain_support,
        'flat_control_terrain_separation_not_required':True,
        'groups':[{k:r[k] for k in ('state_group_id','terrain_class','seed_label','bearing_deg','distinct_terrain_separated_pair_count','terrain_pair_by_family')} for r in counts],
        'common_known_only':True,'claims_available':['geometrically_distinct','follower_screen_compatible','known_terrain_separated'],
        'claims_unavailable':['realized_differently','Outcome_different']})
    terrain_gate=all(value>0 for cls in terrain_support.values() for value in cls.values())
    gates_report={str(n):{'all_40_symmetric':target_counts[n]==40,
                     'terrain_both_seeds':terrain_gate,
                     'ready':target_counts[n]==40 and terrain_gate and all_optimal}
                  for n in (3,5,7)}
    if gates_report['7']['ready']:decision='seven_candidate_curvature_constrained_bank_ready_for_pilot_review'
    elif gates_report['5']['ready']:decision='five_candidate_curvature_constrained_bank_ready_for_pilot_review'
    elif max(values)<=3 and all_optimal:decision='three_mode_only_requires_scope_review'
    else:decision='candidate_diversity_infeasible_under_frozen_follower_contract'
    if not all_optimal:decision='offline_geometry_or_source_parity_invalid'
    # Diagnostic perturbations remain entirely outside the frozen experiment.
    sensitivity=sensitivity_diagnostic(groups,counts,by_bearing,all_records,anchor_by,
                                       _read_follower_config(FOLLOWER_CONFIG))
    write(output/'sensitivity-diagnostic.json',sensitivity)
    forecast_source=read(redesign/'runtime-storage-forecast.json')
    forecast=_forecast(read(prior/'runtime-storage-forecast.json'))
    write(output/'conditional-rollout-proposal.json',{
        'status':'conditional_only_not_authorized',
        'bank_supported':gates_report['5']['ready'],
        'P1_five_processes':28,'P2_five_processes':56,
        'P1_seven_processes':36,'P2_seven_processes':72,
        'four_source_families':['first-transition','first-differential','second-graded','second-flat-only'],
        'duplicate_direct_repeats_per_group':3,
        'frozen_e1_protocol':gates,
        'prior_forecast_sha256':sha(redesign/'runtime-storage-forecast.json'),
        'source_prior_forecast_sha256':sha(prior/'runtime-storage-forecast.json'),
        'forecast_20_percent_margin':forecast,
        'prior_redesign_forecast':forecast_source,
        'P2_requires_P1_gate':True,'no_rollout_executed':True})
    write(output/'final-decision.json',{'decision':decision,'gates':gates_report,
         'groups':40,'finite_library_optimal_all_groups':all_optimal,
         'follower_bridge_router_simulator_started':False,'Outcome_collected':False,
         'bank_B_separate_ablation':True,
         'limited_to_finite_library_and_offline_first_command':True})
    _plot(output,counts,examples)
    by_bearing={b:{'unconstrained_distinct':values_by_bearing,
                   'symmetric_3':sum(r['bearing_deg']==b and r['symmetric_direct_plus_pairs']['3']['feasible'] for r in counts),
                   'symmetric_5':sum(r['bearing_deg']==b and r['symmetric_direct_plus_pairs']['5']['feasible'] for r in counts),
                   'symmetric_7':sum(r['bearing_deg']==b and r['symmetric_direct_plus_pairs']['7']['feasible'] for r in counts)}
                for b in BEARINGS_DEG
                for values_by_bearing in [[r['maximum_distinct_finite_library']['maximum_mode_count'] for r in counts if r['bearing_deg']==b]]}
    report=f'''# Curvature-constrained attainable-set offline review

## Decision

`{decision}`. Neither the five- nor seven-candidate bank passes the frozen offline gate. This does not authorize a three-candidate rollout.

## Frozen inputs and parity

Eight SHA-verified physical anchors (two seeds, four DEM frames each), 40 anchor/bearing StateGroups, prior Bank A/B definitions and ledgers, production C++/YAML, and the E1 gates were checked before generation. The 560 selected prior G0 Paths reproduced their Path SHA exactly. All three production C0 fixtures pass the 1e-5 numeric tolerance. The fixed contract remains 0.60 m goal distance, bearings -20/-10/0/+10/+20 degrees, eight future Path XY points, and 1.236 m^-1 offline curvature screen.

## G0–G3 geometry and attainable modes

Compatible common-goal curves by family across 40 groups: {family_stats}. Each curve was checked separately for dense/analytic curvature, eight-point curvature, fresh C0 first command, endpoint/tangent, finite geometry, reverse, and known DEM corridor. Bank B is a separate local-endpoint ablation and is never mixed into common-goal mode counts.

The exact finite-library independent-set solver reached optimal status in all groups: {all_optimal}. Unconstrained distinct modes have min/median/max {min(values)}/{np.median(values):g}/{max(values)}. By bearing (eight physical anchors each): {by_bearing}. Although 16/40 groups have five unconstrained distinct modes, none has direct plus two mutually distinct left/right pairs. Symmetric 3/5/7 target groups: {target_counts[3]}/{target_counts[5]}/{target_counts[7]} of 40. The five gate is {gates_report['5']['ready']}; the seven gate is {gates_report['7']['ready']}.

At nonzero bearings, the counterbearing side supplies only one candidate under the frozen screen in this library. The second opposite-side Path needs approximately 1.778 m^-1 curvature even before distinct-mode checks. At bearing 0, both side pools exist but pairwise centerline/swept/lateral/curvature-profile separation prevents two left/right pairs. Testing 12 or 16 Path points does not add analytic-compatible supply for this library; production first-command parity for those representations was not established. As an isolated sensitivity check on the first-transition anchor only, a 1.8 m^-1 mathematical curvature screen or a 0.75 m goal distance reaches symmetric five in all five bearings after rerunning the distinct-mode solver and known-DEM/initial-command checks. The former is outside the production follower screen; the latter is a new goal contract. Neither establishes readiness across 40 groups or changes the frozen result.

## Terrain and scope

Compatible common-known terrain-separated group support by class and seed: {terrain_support}. Transition and graded groups have no separated pair under the current known-corridor metric. Flat-control groups do not require terrain separation. The available claims are geometric distinctness, the offline follower screen, and known-terrain separation. Realized tracking and Outcome differences are unmeasured. The finite deterministic library does not prove global impossibility over all continuous paths, and 1.236 m^-1 is a compatibility screen rather than a physical vehicle limit.

The conditional P1/P2 proposal retains the frozen E1 SHA and measured runtime/resource reference with 20% margin. No follower, bridge, router, simulator, rollout, training, or robot control ran.
'''
    (output/'review-report.md').write_text(report)
    files={p.relative_to(output).as_posix():{'sha256':sha(p),'size_bytes':p.stat().st_size}
           for p in output.rglob('*') if p.is_file() and p.name!='artifact-sha-ledger.json'}
    write(output/'artifact-sha-ledger.json',{'files':files})
    return read(output/'final-decision.json')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-artifact-root',required=True,type=Path)
    parser.add_argument('--prior-review-root',required=True,type=Path)
    parser.add_argument('--prior-redesign-root',required=True,type=Path)
    parser.add_argument('--output-root',required=True,type=Path)
    args=parser.parse_args()
    result=run(args.frozen_artifact_root,args.prior_review_root,args.prior_redesign_root,args.output_root)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
