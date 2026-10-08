#!/usr/bin/env python3
"""Read-only R3 inputs -> offline goal-bearing candidate-bank design review."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from visualnav_transformer.eat_nav.goal_bank_offline import (
    AMPLITUDE_POOL_M, BEARINGS_DEG, CLASSES, OFFSET_POOL_M,
    audit_geometry, bezier_candidate, corridor_audit, corridor_class_proxy, follower_diagnostic,
    freeze_anchors, goal_for_bearing, pairwise_audit, pairwise_command_estimate, path_sha,
    select_parameters,
)


ROOT_NAME = 'eatnav-goal-bearing-bank-review-20261008'
FOLLOWER_CONFIG = Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml')
E1_FORECAST = Path('eatnav-expanded-feasibility-ros-continuation-20261005/runtime-storage-forecast.json')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [plain(v) for v in value]
    return value


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.tmp')
    pending.write_text(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False)+'\n')
    pending.replace(path)


def _ledger_paths(ledger: Path) -> dict[str,str]:
    data = read(ledger)
    return {Path(row['path']).relative_to(Path(row['path']).parents[2]).as_posix(): row['sha256']
            for row in data['files']}


def _source_file(provenance: dict, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    provenance['sha256'][str(path)] = sha(path)


def _read_follower_config(path: Path) -> dict:
    # Parse scalar values only; the production YAML is part of provenance.
    keys = ('lookahead_distance','linear_speed','angular_gain','angular_direction_sign',
            'max_angular_speed','goal_tolerance','turn_in_place_heading_threshold',
            'min_linear_speed_scale')
    values = {}
    for line in path.read_text().splitlines():
        if ':' not in line:
            continue
        key, value = line.strip().split(':',1)
        if key in keys:
            values[key] = float(value.strip())
    if set(values) != set(keys):
        raise ValueError('incomplete production follower scalar config')
    return values


def _verify_shard(shard: Path, pose_id: str, files: list[Path], provenance: dict) -> None:
    seal_path = shard/'seal.json'
    seal = read(seal_path)
    if (seal.get('native_exit_code') != 0 or seal.get('owned_residual') is not False
            or pose_id not in seal.get('pose_ids', []) or
            seal.get('artifact_gate', {}).get('physics_steps') != 200):
        raise ValueError('unsealed or incomplete source shard: '+str(shard))
    _source_file(provenance, seal_path)
    ledger_path = shard/'sha-ledger.json'
    _source_file(provenance, ledger_path)
    ledger = read(ledger_path)
    expected = {Path(row['path']).relative_to(Path('/tmp')/shard.parent.name/shard.name).as_posix(): row['sha256']
                for row in ledger['files']}
    for path in files:
        relative = path.relative_to(shard).as_posix()
        _source_file(provenance, path)
        if expected.get(relative) != provenance['sha256'][str(path)]:
            raise ValueError('source SHA ledger mismatch: '+str(path))


def _find_scan_pose(backup: Path, seed: str, pose_id: str, radius: float, provenance: dict) -> dict:
    folder = (backup/'eatnav-anchor-placement-20261008/scan-v3' if seed == 'first'
              else backup/'eatnav-r3-second-seed-20261008/scan-v1')
    file = folder/f'radius-{radius:.2f}.json'
    _source_file(provenance,file)
    ledger = (backup/'eatnav-anchor-placement-20261008/analysis-v9/sha-ledger.json' if seed == 'first'
              else backup/'eatnav-r3-second-seed-20261008/scan-sha-ledger.json')
    _verify_prior_file(file, ledger, provenance)
    poses = [row for row in read(file)['candidates'] if row['pose_id'] == pose_id]
    if len(poses) != 1:
        raise ValueError('scan pose identity not unique: '+pose_id)
    return poses[0]


def _verify_prior_file(path: Path, ledger_path: Path, provenance: dict) -> None:
    _source_file(provenance, ledger_path)
    data=read(ledger_path)
    entries=data['files'] if isinstance(data,dict) and 'files' in data else data
    expected={row['path']:row['sha256'] for row in entries} if isinstance(entries,list) else entries
    root=ledger_path.parent
    if ledger_path.name in ('sha-ledger.json',) and root.name.startswith('analysis-v'):
        root=root.parent
    elif ledger_path.name in ('scan-sha-ledger.json','final-output-sha-ledger.json'):
        root=ledger_path.parent
    relative=path.relative_to(root).as_posix()
    if expected.get(relative)!=sha(path):
        raise ValueError('frozen source ledger mismatch: '+str(path))
    provenance.setdefault('verified_prior_ledgers',{})[str(path)]=str(ledger_path)


def load_anchor_inputs(backup: Path) -> tuple[list[dict],dict]:
    """Freeze prior ranked supply, prove all DEM/Actor/trace bytes are sealed."""
    backup = Path(backup)
    if not backup.is_dir():
        raise FileNotFoundError(backup)
    provenance = {'backup_root': str(backup), 'sha256': {},
                  'prior_roots_read_only': True, 'outcome_fields_used_for_selection': []}
    first_root = backup/'eatnav-r3-settling-pilot-20261008'
    reserve_root = backup/'eatnav-r3-transition-reserve-20261008'
    second_root = backup/'eatnav-r3-second-seed-20261008'
    first_result = first_root/'analysis/per-pose-results.json'
    reserve_result = reserve_root/'analysis/reserve-per-pose.json'
    second_result = second_root/'analysis-v2/per-pose-results.json'
    first_manifest = first_root/'primary-manifest.json'
    reserve_manifest = reserve_root/'reserve-manifest.json'
    second_manifest = second_root/'selection/pilot-manifest.json'
    for file in (first_result,reserve_result,second_result,first_manifest,reserve_manifest,second_manifest):
        _source_file(provenance,file)
        _verify_prior_file(file, (first_root if file.is_relative_to(first_root) else
                                  reserve_root if file.is_relative_to(reserve_root) else second_root)
                           /'final-output-sha-ledger.json', provenance)
    first_meta = {r['pose_id']:r for r in read(first_manifest)['poses']}
    reserve_meta = {r['pose_id']:r for r in read(reserve_manifest)['selected']}
    second_meta = {r['pose_id']:r for r in read(second_manifest)['poses']}
    first_rows = []
    for row in read(first_result)['poses']:
        meta = first_meta[row['pose_id']]
        first_rows.append({**row,'rank':meta['rank'],'virtual_class':meta['virtual_class'],
                           'source_root':str(first_root),'metadata':meta})
    for row in read(reserve_result):
        meta = reserve_meta[row['pose_id']]
        first_rows.append({**row,'rank':meta['selection_rank_in_r3'],
                           'virtual_class':meta['classification'],
                           'source_root':str(reserve_root),'metadata':meta})
    second_rows = []
    for row in read(second_result):
        meta = second_meta[row['pose_id']]
        second_rows.append({**row,'rank':meta['rank'],'virtual_class':meta['virtual_class'],
                            'source_root':str(second_root),'metadata':meta})
    chosen = freeze_anchors({'first':first_rows,'second':second_rows})
    anchors = []
    for row in chosen:
        source_root = Path(row['source_root'])
        shard = source_root/row['shard_id']
        pose = shard/'run/poses'/f"{row['pose_id']}.json"
        descriptor = read(pose)
        dem_paths = [shard/'run'/relative for relative in descriptor['dem_paths']]
        actor_path = shard/'run'/descriptor['settled_actor_path']
        trace_path = shard/'run'/descriptor['trace_path']
        _verify_shard(shard,row['pose_id'],[pose,*dem_paths,actor_path,trace_path],provenance)
        if descriptor['physics_step_count'] != 200 or len(dem_paths) != 4:
            raise ValueError('incomplete source pose: '+row['pose_id'])
        frames = [read(path) for path in dem_paths]
        indices = [f['frame_index'] for f in frames]
        if indices != [0,1,2,3]:
            raise ValueError('DEM history identity/order mismatch')
        for frame in frames:
            data = np.asarray(frame['transformed'],dtype=float)
            if (frame['shape'] != [2,20,20] or data.shape != (2,20,20) or
                    frame['axis'] != 'robot-relative-grid-xy' or frame['frame_id'] != 'map' or
                    frame['resolution_m'] != .1 or not frame['finite'] or not np.isfinite(data).all()):
                raise ValueError('DEM frame contract mismatch: '+str(frame))
        if frames[-1]['raw_sha256'] != row['dem_frame_sha256']:
            raise ValueError('settled DEM SHA identity mismatch')
        actor = read(actor_path)
        if actor['shape'] != [22] or actor['finite'] is not True or not np.isfinite(actor['values']).all():
            raise ValueError('incomplete Actor body state')
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        if len(trace)!=200 or trace[-1]['physics_step']!=200:
            raise ValueError('incomplete 200-step trace')
        state = trace[-1]['state']
        if trace[-1]['post_reset_state'] or trace[-1]['episode_id']!=0:
            raise ValueError('source trace crossed episode reset')
        map_pose = [float(state['root_state_w'][0]),float(state['root_state_w'][1]),
                    float(state['euler_yaw_signed_rad'])]
        scan = _find_scan_pose(backup,row['seed_label'],row['pose_id'],row['radius_m'],provenance)
        if scan['dem_sha256'] != row['metadata'].get('virtual_dem_sha256', row['metadata'].get('dem_sha256')):
            raise ValueError('virtual scan DEM SHA mismatch')
        proxy = scan['footprint_diagnostic']
        boxes = {name:proxy[name]['local_bbox_xy_m'] for name in
                 ('main_track_sprocket','initial_flipper_support','articulated_reach','full_collision_union')}
        anchors.append({
            'anchor_id': f"{row['seed_label']}-{row['terrain_class']}",
            'pose_id':row['pose_id'],'seed_label':row['seed_label'],
            'terrain_class':row['terrain_class'],'terrain':row['metadata']['terrain'],
            'terrain_seed':row['metadata']['terrain_seed'],'source_family':row['source_family'],
            'selection_rank':row['rank'],'radius_m':row['radius_m'],
            'requested_xy_yaw':row['requested_xy_yaw'], 'settled_map_xy_yaw':map_pose,
            'dem_history_sha256':[f['raw_sha256'] for f in frames],
            'dem_file_sha256':[sha(path) for path in dem_paths],
            'dem':np.asarray(frames[-1]['transformed'],dtype=float),
            'dem_history_frames':frames,'body_state_actor':actor['values'],
            'actor_config_identity':actor['config_identity'],'proxy_bboxes':boxes,
            'source_shard':str(shard),'settled_class':row['settled_class'],
            'known_onset':row['known_onset'],
            'physical_supply':row['physical_supply'],
            'source_path_shas':{str(p):provenance['sha256'][str(p)] for p in [pose,*dem_paths,actor_path,trace_path]},
        })
    provenance['source_file_count'] = len(provenance['sha256'])
    provenance['source_family_count'] = len({a['source_family'] for a in anchors})
    return anchors, provenance


def _goal_cell(goal: np.ndarray, height: np.ndarray, mask: np.ndarray) -> dict:
    col = int(np.rint(goal[0]/.1+9.5))
    row = int(np.rint(goal[1]/.1+9.5))
    inside = 0 <= col < 20 and 0 <= row < 20 and bool(np.all(np.abs(goal) <= 1.0))
    return {'row':row if inside else None,'column':col if inside else None,
            'inside':inside, 'known': bool(mask[row,col]>.5) if inside else False,
            'height_m':float(height[row,col]) if inside and mask[row,col]>.5 else None}


def _map_goal(goal: np.ndarray, pose: list[float]) -> tuple[list[float],float]:
    c,s=math.cos(pose[2]),math.sin(pose[2])
    rotation=np.array([[c,-s],[s,c]])
    mapped=rotation@goal+np.asarray(pose[:2])
    restored=rotation.T@(mapped-np.asarray(pose[:2]))
    return mapped.tolist(),float(np.max(np.abs(restored-goal)))


def _candidate_key(side: str, magnitude: float) -> str:
    return 'direct' if side == 'direct' else f'{side}_{magnitude:.2f}'


def _pool_for_group(bank: str, mode: str, goal: np.ndarray, anchor: dict, follower: dict,
                    magnitudes: tuple[float,...] | None = None) -> dict:
    if magnitudes is None:
        magnitudes = AMPLITUDE_POOL_M if bank == 'A' else OFFSET_POOL_M
    specs=[('direct',0.0)]+[(side,m) for side in ('left','right') for m in magnitudes]
    result={}
    for side,magnitude in specs:
        row=bezier_candidate(bank,goal,magnitude,side,mode)
        geometry=audit_geometry(row,goal,anchor['settled_map_xy_yaw'])
        # Parameter screening uses the actual known DEM and main-track proxy.
        # The selected bank receives the complete four-proxy corridor audit.
        dem=corridor_audit(row,anchor['dem'][0],anchor['dem'][1],
                           {'main_track_sprocket':anchor['proxy_bboxes']['main_track_sprocket']})
        result[_candidate_key(side,magnitude)]={
            'row':row,'geometry':geometry,'follower':follower_diagnostic(row,follower),'dem':dem}
    return result


def _score_combo(pool: dict, magnitudes: tuple[float,float,float], bank: str, mode: str) -> dict:
    score={'bank':bank,'terminal_mode':mode,'parameters':list(magnitudes),
           'hard_valid_groups':0,'follower_compatible':0,
           'early_separation':0.0,'full_separation':0.0,
           'known_terrain_separation':0.0,'known_terrain_pair_count':0,
           'unknown_dependence':0.0,'curvature_penalty':0.0,'side_balance':0,
           'hard_valid_candidates':0}
    left_count=right_count=0
    for group in pool.values():
        chosen=[group['direct']]+[group[_candidate_key(side,m)] for side in ('left','right') for m in magnitudes]
        valid=[c['geometry']['hard_valid'] for c in chosen]
        score['hard_valid_candidates']+=sum(valid)
        score['hard_valid_groups']+=int(all(valid))
        score['follower_compatible']+=sum(c['follower']['follower_compatible_screen'] for c in chosen)
        left_count+=sum(group[_candidate_key('left',m)]['follower']['follower_compatible_screen'] for m in magnitudes)
        right_count+=sum(group[_candidate_key('right',m)]['follower']['follower_compatible_screen'] for m in magnitudes)
        score['unknown_dependence']+=sum(c['dem']['centerline_unknown_fraction']+c['dem']['centerline_outside_fraction'] for c in chosen)
        score['curvature_penalty']+=sum(c['row']['maximum_curvature_per_m']
                                        if c['row']['maximum_curvature_per_m'] is not None else 1e6
                                        for c in chosen)
        direct=chosen[0]
        for other in chosen[1:]:
            pair=pairwise_audit(direct['row'],other['row'],direct['dem'],other['dem'])
            score['early_separation']+=pair['early_0_5m_geometry_rms_m']
            score['full_separation']+=pair['full_geometry_rms_m']
            if pair['known_corridor_profile_rms_m'] is not None and pair['common_known_corridor_bins']>=5:
                score['known_terrain_separation']+=pair['known_corridor_profile_rms_m']
                score['known_terrain_pair_count']+=1
    score['side_balance']=-abs(left_count-right_count)
    return score


def _pairwise_records(items: list[dict]) -> list[dict]:
    records=[]
    for first,second in itertools.combinations(items,2):
        pair=pairwise_audit(first['_row'],second['_row'],first['_dem'],second['_dem'])
        commands=pairwise_command_estimate(first['follower'],second['follower'])
        records.append({'first':first['candidate_id'],'second':second['candidate_id'],
                        **pair,**commands})
    return records


def _candidate_record(group: dict, anchor: dict, bank: str, item: dict,
                      candidate_id: str) -> dict:
    row=item['row']
    dem=corridor_audit(row,anchor['dem'][0],anchor['dem'][1],anchor['proxy_bboxes'])
    class_proxy=corridor_class_proxy(row,anchor['dem'][0],anchor['dem'][1])
    endpoint=np.asarray(row['endpoint'])
    global_goal=np.asarray(row['global_goal'])
    remaining=global_goal-endpoint
    terminal=np.asarray(row['terminal_tangent_xy'])
    replan_cos=float(np.dot(terminal,remaining)/np.linalg.norm(remaining)) if np.linalg.norm(remaining)>1e-12 else None
    data={'state_group_id':group['state_group_id'],'anchor_id':anchor['anchor_id'],
          'seed_label':anchor['seed_label'],'source_family':anchor['source_family'],
          'terrain':anchor['terrain'],'terrain_class':anchor['terrain_class'],
          'bearing_deg':group['bearing_deg'],'bank':bank,'candidate_id':candidate_id,
          'side':row['side'],'signed_amplitude_or_offset_m':row['magnitude_m']*(1 if row['side']=='left' else -1 if row['side']=='right' else 0),
          'parameter_kind':'peak_lateral_amplitude_m' if bank=='A' else 'endpoint_offset_m',
          'terminal_mode':row['terminal_mode'],'frame':'robot-local-base',
          'global_goal_xy_m':global_goal,'local_subgoal_xy_m':endpoint,
          'controls_xy_m':row['controls'],'waypoints_xy_m':row['waypoints'],
          'dense_centerline_xy_m':row['dense'],'dense_stations_m':row['stations_m'],
          'path_sha256':item['geometry']['path_sha256'],'arc_length_m':row['arc_length_m'],
          'initial_tangent_xy':row['initial_tangent_xy'],'terminal_tangent_xy':terminal,
          'maximum_curvature_per_m':row['maximum_curvature_per_m'],
          'curvature_continuity_max_delta_per_m':row['curvature_continuity_max_delta_per_m'],
          'geometry':item['geometry'],'follower':item['follower'],'dem':dem,
          'candidate_corridor_class_proxy':class_proxy,
          'candidate_corridor_class_matches_anchor':(class_proxy['class_proxy']==anchor['terrain_class']
                                                      if class_proxy['class_proxy']!='unknown' else None),
          'endpoint_global_goal_residual_m':float(np.linalg.norm(remaining)),
          'terminal_goal_direction_cosine':replan_cos,
          'primary_navigation_goal':'global_goal_only',
          '_row':row,'_dem':dem}
    return data


def _public_candidate(item: dict) -> dict:
    return {k:plain(v) for k,v in item.items() if not k.startswith('_')}


def _summarize_bank(rows: list[dict], pairs: list[dict], group_count: int) -> dict:
    valid=sum(r['geometry']['hard_valid'] for r in rows)
    compatible=sum(r['follower']['follower_compatible_screen'] for r in rows)
    turn=sum(r['follower']['turn_in_place_entry'] for r in rows)
    reverse=sum(r['follower']['reverse_estimate'] for r in rows)
    terrain_sep=sum(p['known_corridor_profile_rms_m'] is not None and p['common_known_corridor_bins']>=5
                    and p['known_corridor_profile_rms_m']>=.01 for p in pairs)
    directional={}
    for key,condition in {
        'direct_vs_left':lambda p: 'direct' in (p['first'],p['second']) and (p['first'].startswith('left') or p['second'].startswith('left')),
        'direct_vs_right':lambda p: 'direct' in (p['first'],p['second']) and (p['first'].startswith('right') or p['second'].startswith('right')),
        'left_vs_right':lambda p: (p['first'].startswith('left') and p['second'].startswith('right')) or (p['first'].startswith('right') and p['second'].startswith('left')),
    }.items():
        selected=[p for p in pairs if condition(p)]
        known=[p['known_corridor_profile_rms_m'] for p in selected
               if p['known_corridor_profile_rms_m'] is not None and p['common_known_corridor_bins']>=5]
        directional[key]={'pair_count':len(selected),'common_known_pair_count':len(known),
                          'mean_known_corridor_rms_m':float(np.mean(known)) if known else None,
                          'known_separated_pair_count':sum(v>=.01 for v in known)}
    group_data={}
    for r in rows:
        group_data.setdefault(r['state_group_id'],[]).append(r)
    group_pairs={}
    for p in pairs:
        group_pairs.setdefault(p['state_group_id'],[]).append(p)
    group_support={}
    for gid, group in group_data.items():
        separated=sum(p['known_corridor_profile_rms_m'] is not None
                      and p['common_known_corridor_bins']>=5
                      and p['known_corridor_profile_rms_m']>=.01
                      for p in group_pairs.get(gid,[]))
        group_support[gid]={'follower_compatible':sum(r['follower']['follower_compatible_screen'] for r in group),
                            'non_direct_follower_compatible':sum(r['candidate_id']!='direct' and r['follower']['follower_compatible_screen'] for r in group),
                            'known_terrain_separated_pairs':separated,
                            'hard_valid':all(r['geometry']['hard_valid'] for r in group)}
    valid_groups=sum(all(r['geometry']['hard_valid'] for r in group) for group in group_data.values())
    mode_groups=sum(all(p['full_geometry_rms_m']>=.015 or p['early_0_5m_geometry_rms_m']>=.01
                        for p in group_pairs.get(gid,[])) for gid in group_data)
    bearing_support={str(b): {'groups':sum(r['bearing_deg']==b and r['candidate_id']=='direct' for r in rows),
                              'hard_valid':sum(r['bearing_deg']==b and r['geometry']['hard_valid'] for r in rows),
                              'follower_compatible':sum(r['bearing_deg']==b and r['follower']['follower_compatible_screen'] for r in rows)}
                     for b in BEARINGS_DEG}
    class_support={c:{'groups':sum(r['terrain_class']==c and r['candidate_id']=='direct' for r in rows),
                      'hard_valid':sum(r['terrain_class']==c and r['geometry']['hard_valid'] for r in rows),
                      'follower_compatible':sum(r['terrain_class']==c and r['follower']['follower_compatible_screen'] for r in rows)}
                   for c in CLASSES}
    return {'candidate_count':len(rows),'hard_valid_candidate_count':valid,
            'hard_valid_state_groups':valid_groups,'state_groups':group_count,
            'seven_geometry_mode_supply_groups':mode_groups,
            'geometry_mode_definition':'each pair full RMS >=0.015 m or early <=0.5m RMS >=0.01 m; descriptive only',
            'terrain_separated_known_pairs':terrain_sep,
            'terrain_separated_groups':sum(s['known_terrain_separated_pairs']>0 for s in group_support.values()),
            'candidate_class_proxy_known':sum(r['candidate_corridor_class_proxy']['class_proxy']!='unknown' for r in rows),
            'candidate_class_proxy_matches_anchor':sum(r['candidate_corridor_class_matches_anchor'] is True for r in rows),
            'groups_meeting_follower_minimum':sum(s['follower_compatible']>=4 for s in group_support.values()),
            'minimum_follower_compatible_per_group':min(s['follower_compatible'] for s in group_support.values()),
            'group_support':group_support,
            'terrain_pair_definition':'common-known main-track corridor profile bins >=5 and RMS >=0.01 m; unknown-only difference excluded',
            'left_direct_right_terrain_difference':directional,
            'follower_compatible_candidate_count':compatible,
            'maximum_curvature_per_m':max(r['maximum_curvature_per_m'] for r in rows),
            'turn_in_place_candidate_count':turn,'reverse_candidate_count':reverse,
            'centerline_known_fraction_mean':float(np.mean([r['dem']['centerline_known_fraction'] for r in rows])),
            'centerline_unknown_or_outside_mean':float(np.mean([r['dem']['centerline_unknown_fraction']+r['dem']['centerline_outside_fraction'] for r in rows])),
            'main_track_known_fraction_mean':float(np.mean([r['dem']['corridors']['main_track_sprocket']['known_fraction'] for r in rows])),
            'proxy_known_coverage':{name:{'mean':float(np.mean([r['dem']['corridors'][name]['known_fraction'] for r in rows])),
                                            'minimum':float(np.min([r['dem']['corridors'][name]['known_fraction'] for r in rows]))}
                                    for name in ('main_track_sprocket','initial_flipper_support','articulated_reach','full_collision_union')},
            'left_follower_compatible':sum(r['side']=='left' and r['follower']['follower_compatible_screen'] for r in rows),
            'right_follower_compatible':sum(r['side']=='right' and r['follower']['follower_compatible_screen'] for r in rows),
            'mean_local_endpoint_global_goal_residual_m':float(np.mean([r['endpoint_global_goal_residual_m'] for r in rows])),
            'mean_terminal_goal_direction_cosine':float(np.mean([r['terminal_goal_direction_cosine'] for r in rows if r['terminal_goal_direction_cosine'] is not None])) if any(r['terminal_goal_direction_cosine'] is not None for r in rows) else None,
            'bearing_support':bearing_support,'class_support':class_support,
            'seed_support':{seed:{'groups':sum(r['seed_label']==seed and r['candidate_id']=='direct' for r in rows),
                                  'hard_valid':sum(r['seed_label']==seed and r['geometry']['hard_valid'] for r in rows)}
                            for seed in ('first','second')},
            'source_family_support':{family:{'groups':sum(r['source_family']==family and r['candidate_id']=='direct' for r in rows),
                                             'hard_valid':sum(r['source_family']==family and r['geometry']['hard_valid'] for r in rows),
                                             'follower_compatible':sum(r['source_family']==family and r['follower']['follower_compatible_screen'] for r in rows)}
                                     for family in sorted({r['source_family'] for r in rows})}}


def _forecast(backup: Path, provenance: dict) -> dict:
    source=backup/E1_FORECAST
    _source_file(provenance,source)
    measured=read(source)
    design_report=Path(__file__).resolve().parents[1]/'docs/quality-identifiability-gate/evaluator-design-review/evaluator-design-review.md'
    _source_file(provenance,design_report)
    per_run=measured['measured_process_mean_sec']
    p95=measured['measured_process_p95_sec']
    # The later 140-process design review reported 1.19 GB for 140 sealed
    # runs, a more conservative storage reference than the earlier 50-run
    # host-only forecast.  This is an estimate, not a new measurement.
    saved_per_run=1.19e9/140
    variants={'4_anchors_1_bearing':36,'4_anchors_2_bearings':72,
              '4_anchors_3_bearings':108,'8_anchors_1_bearing':72,
              'staged_0_then_plus_minus_20':108,
              'two_banks_shared_three_direct_repeats':180,
              'two_banks_independent_three_direct_repeats_each':216}
    estimates={}
    for name,count in variants.items():
        estimates[name]={'candidate_processes':count,
                         'point_hours':count*per_run/3600,
                         'p95_hours':count*p95/3600,
                         'plus_20_percent_hours':count*p95*1.2/3600,
                         'saved_gb_point':count*saved_per_run/1e9,
                         'saved_gb_plus_20_percent':count*saved_per_run*1.2/1e9}
    return {'source':str(source),'source_sha256':sha(source),
            'expanded_design_review_source':str(design_report),
            'expanded_design_review_source_sha256':sha(design_report),
            'measured_prior_process_mean_sec':per_run,
            'measured_prior_process_p95_sec':p95,
            'prior_gpu_peak_reference_mib':5960,
            'prior_simulator_pss_reference_gib':5.38,
            'prior_anon_shmem_reference_gib':3.72,
            'prior_host_available_reference_gb':87.7,
            'prior_gpu_device_total_cap_mib':6656,
            'storage_reference_bytes_per_process':saved_per_run,
            'forecast_scope':'sequential fresh-process; includes prior startup/finalization overhead in per-run measurement; excludes new retries',
            'external_zenoh_prerequisite':True,
            'cleanup_and_finalization':'owned PID/start-tick, sealed marker, native exit, zero residual, host/container SHA per process',
            'retry_policy':'no automatic retry; infrastructure-invalid and measurement-invalid kept separate; new attempt requires review',
            'alternatives':estimates,
            'one_bank_108_process_plus_anchor_replays':108,
            'two_bank_direct_path_identity':'direct Path SHA is identical across A/B for every StateGroup',
            'two_bank_repeat_policy':'180 processes if three sealed direct repeats are shared; 216 if each bank requires its own three-repeat baseline',
            'anchor_only_replay_count':'extra if required; not hidden in candidate-process counts',
            'minimum_goal_bearing_comparison':'4 anchors x 2 bearings (-20,+20) = 72 processes; four classes and both side bearings',
            'staged_review':'phase 1 four anchors x 0 deg = 36; phase 2 same anchors x +/-20 deg = 72; stop between phases'}


def _decision(a: dict, b: dict) -> tuple[str,list[str]]:
    banks={'A':a,'B':b}
    ready={name:(row['hard_valid_state_groups']==row['state_groups']
                 and row['seven_geometry_mode_supply_groups']==row['state_groups']
                 and row['terrain_separated_known_pairs']>0
                 and row['groups_meeting_follower_minimum']==row['state_groups'])
           for name,row in banks.items()}
    reasons=[f"Bank {name}: {row['hard_valid_state_groups']}/{row['state_groups']} hard-valid groups; "
             f"{row['seven_geometry_mode_supply_groups']}/{row['state_groups']} geometry-mode groups; "
             f"{row['groups_meeting_follower_minimum']}/{row['state_groups']} groups have >=4 follower-screen candidates; "
             f"{row['terrain_separated_known_pairs']} common-known terrain pairs"
             for name,row in banks.items()]
    if ready['A'] and not ready['B']:
        return 'candidate_bank_common_endpoint_ready_for_rollout_review',reasons
    if ready['B'] and not ready['A']:
        return 'candidate_bank_local_subgoal_ready_for_rollout_review',reasons
    if ready['A'] and ready['B']:
        return 'candidate_bank_both_variants_require_small_comparison',reasons
    if all(row['hard_valid_state_groups']<row['state_groups'] for row in banks.values()):
        return 'candidate_bank_requires_parameter_redesign',reasons
    if all(row['seven_geometry_mode_supply_groups']<row['state_groups'] for row in banks.values()):
        return 'candidate_bank_goal_bearing_support_incomplete',reasons
    if all(row['terrain_separated_known_pairs']==0 for row in banks.values()):
        return 'candidate_bank_insufficient_terrain_separation',reasons
    return 'candidate_bank_follower_compatibility_insufficient',reasons


def _pilot_proposal(groups: list[dict], rows: list[dict], bank: str) -> dict:
    chosen={'first-transition','first-differential','second-graded','second-flat-only'}
    selected=[g for g in groups if g['anchor_id'] in chosen and g['bearing_deg'] in (-20,0,20)]
    if len(selected)!=12:
        raise ValueError('pilot StateGroup count mismatch')
    paths={}
    for row in rows:
        if row['bank']==bank and row['state_group_id'] in {g['state_group_id'] for g in selected}:
            paths.setdefault(row['state_group_id'],{})[row['candidate_id']]=row['path_sha256']
    if any(len(paths.get(g['state_group_id'],{}))!=7 for g in selected):
        raise ValueError('pilot bank incomplete')
    return {'status':'proposal_only_no_rollout_authorized','bank':bank,
            'selection_rule':'one physical-supply anchor per class; 2 first + 2 second seed; four distinct source families; fixed before Outcome',
            'pilot_anchor_ids':sorted(chosen),'state_groups':selected,
            'path_sha256_by_state_group':paths,
            'direct_repeats_per_state_group':3,'non_direct_once_per_state_group':6,
            'candidate_processes_per_state_group':9,'state_group_count':12,
            'candidate_process_count':108,
            'anchor_only_replay_processes':'separate if required',
            'plus_minus_10_degree_goals':'defined offline; not in initial pilot',
            'four_class_2_bearing_reduced_option_processes':72,
            'pilot_not_final_collection':True}


def _report(summary: dict, selected: dict, mode_comparison: dict, unique_paths: int,
            provenance: dict, forecast: dict,
            decision: str, reasons: list[str]) -> str:
    a,b=summary['A'],summary['B']
    lines=['# Goal-bearing / candidate-bank offline design review','',
           f'**Decision:** `{decision}`. This is an offline Path-design decision only; no candidate was run.',
           '', '## Frozen inputs', '',
           f"- 8 anchors: 2 seeds × 4 DEM classes, selected by prior R3 rank. Independent source families: {provenance['source_family_count']}.",
           f"- {provenance['source_file_count']} source files SHA-checked, including all selected four-frame DEM histories, Actor body states, traces and seals.",
           '- Global goals are 0.60 m away at −20°, −10°, 0°, +10° and +20° in robot-local base frame.',
           '- Within a StateGroup, anchor, four DEM frames, body state and global goal are fixed; only Path changes.',
           f'- The 560 anchor-specific definitions contain {unique_paths} unique Path hashes because a bearing/candidate geometry is deliberately reused across anchors; definitions are not independent trajectories.',
           '', '## Global parameter selection', '',
           f"- Bank A peak lateral amplitudes: {selected['A']['parameters']} m.",
           f"- Bank B endpoint offsets: {selected['B']['parameters']} m; terminal control: `{selected['B']['terminal_mode']}`.",
           '- The supplied pools and lexicographic outcome-blind priorities were used across all 40 StateGroups.',
           f"- Bank B terminal alternatives: strict goal-facing max curvature {mode_comparison['strict_goal_facing']['maximum_curvature_per_m']:.2f} m⁻¹ / initial turn estimates {mode_comparison['strict_goal_facing']['turn_in_place_estimates']}; blended goal-connected max curvature {mode_comparison['goal_connected']['maximum_curvature_per_m']:.2f} m⁻¹ / initial turn estimates {mode_comparison['goal_connected']['turn_in_place_estimates']}.",
           '- Strict goal-facing wins the frozen geometry-priority ordering despite its higher curvature. This is a design concern for review, not evidence of tracking.',
           '', '## Bank comparison', '',
           '| Metric | A common endpoint | B local subgoal |',
           '| --- | ---: | ---: |',
           f"| Hard-valid candidates | {a['hard_valid_candidate_count']}/280 | {b['hard_valid_candidate_count']}/280 |",
           f"| Seven hard-valid StateGroups | {a['hard_valid_state_groups']}/40 | {b['hard_valid_state_groups']}/40 |",
           f"| Seven distinct geometry modes | {a['seven_geometry_mode_supply_groups']}/40 | {b['seven_geometry_mode_supply_groups']}/40 |",
           f"| Follower-compatible screen | {a['follower_compatible_candidate_count']}/280 | {b['follower_compatible_candidate_count']}/280 |",
           f"| StateGroups with ≥4 follower-screen candidates | {a['groups_meeting_follower_minimum']}/40 | {b['groups_meeting_follower_minimum']}/40 |",
           f"| Common-known terrain-separated pairs | {a['terrain_separated_known_pairs']} | {b['terrain_separated_known_pairs']} |",
           f"| StateGroups with ≥1 known-terrain-separated pair | {a['terrain_separated_groups']}/40 | {b['terrain_separated_groups']}/40 |",
           f"| Path-aligned class proxy matching source class | {a['candidate_class_proxy_matches_anchor']}/{a['candidate_class_proxy_known']} known | {b['candidate_class_proxy_matches_anchor']}/{b['candidate_class_proxy_known']} known |",
           f"| Max curvature (m⁻¹) | {a['maximum_curvature_per_m']:.2f} | {b['maximum_curvature_per_m']:.2f} |",
           f"| Initial turn-in-place estimates | {a['turn_in_place_candidate_count']} | {b['turn_in_place_candidate_count']} |",
           f"| Reverse estimates | {a['reverse_candidate_count']} | {b['reverse_candidate_count']} |",
           f"| Mean local-endpoint→global-goal residual (m) | {a['mean_local_endpoint_global_goal_residual_m']:.3f} | {b['mean_local_endpoint_global_goal_residual_m']:.3f} |",
           '', 'The 1.236 m⁻¹ criterion is a production-follower command screen, not a physical vehicle limit. The first-command estimate does not establish realized tracking.',
           'Known-terrain separation uses only common known DEM samples; unknown-only differences are excluded.',
           'Path-aligned three-band terrain class is a diagnostic proxy using the prior thresholds; it is not the frozen axis-aligned anchor classifier.',
           'Known coverage for all four collision-bound proxy families and per-StateGroup separation are in bank-comparison.json.',
           '', '## Pilot and forecast', '',
           '- Proposed 4-anchor / 3-bearing pilot: 12 StateGroups; 3 direct repeats plus 6 non-direct Paths = 108 fresh processes for one bank.',
           '- Across both banks the direct Path is identical within each StateGroup: 180 processes with three shared direct repeats, or 216 with three independent repeats per bank, before anchor-only replay.',
           f"- One-bank 108-process reference: {forecast['alternatives']['4_anchors_3_bearings']['point_hours']:.2f} h point, {forecast['alternatives']['4_anchors_3_bearings']['plus_20_percent_hours']:.2f} h P95+20%, {forecast['alternatives']['4_anchors_3_bearings']['saved_gb_plus_20_percent']:.2f} GB saved with 20% margin.",
           '- Minimum bearing comparison with all four classes is 4 anchors × two bearings (−20°, +20°), 72 processes; the staged 0° then ±20° design is 36 + 72.',
           '- External Zenoh, per-process SHA seal, cleanup and resource checks require separate rollout approval.',
           '', '## Interpretation and limits', '',
           '- Bank B retains a common global navigation goal. Its shifted endpoint leaves a structural global-goal residual; a short-horizon teacher may penalize this before replan. No Outcome exists here to quantify that bias.',
           '- Geometry, swept area, known terrain profile and initial follower command are four distinct offline claims. Realized trajectory and Outcome remain unmeasured.',
           '- Bearing-derived groups and candidate pairs are correlated. Split by terrain seed/source family; no 40-independent-StateGroup claim.',
           '- Representative footprint metrics use frozen USD collision-bound corner rectangles, not a contact-response simulation.',
           '', '## Decision reasons', '']
    lines += [f'- {reason}' for reason in reasons]
    lines += ['', 'Raw candidate definitions and 12 figures are in the output root; selected small artifacts are copied into Git for review. No follower, bridge, router, simulator, Outcome or training run was started.', '']
    return '\n'.join(lines)


def run_review(backup: Path, output: Path, *, figures: bool=True) -> dict:
    backup=Path(backup)
    output=Path(output)
    if output.exists():
        raise FileExistsError('new output root required: '+str(output))
    output.mkdir(parents=True)
    anchors,provenance=load_anchor_inputs(backup)
    follower_path=FOLLOWER_CONFIG
    historical_follower=backup/'eatnav-onix/teaching_playback/simple_trajectory_follower/config/onix.yaml'
    if sha(follower_path)!=sha(historical_follower):
        raise ValueError('production follower config differs from frozen prior source')
    _source_file(provenance,follower_path)
    _source_file(provenance,historical_follower)
    follower=_read_follower_config(follower_path)
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=Path(__file__).resolve().parents[1]).strip()
        branch=subprocess.check_output(['git','branch','--show-current'],text=True,cwd=Path(__file__).resolve().parents[1]).strip()
        status=subprocess.check_output(['git','status','--porcelain'],text=True,cwd=Path(__file__).resolve().parents[1]).strip().splitlines()
    except (OSError,subprocess.CalledProcessError) as exc:
        raise ValueError('Git provenance unavailable') from exc
    provenance.update({'source_file_count':len(provenance['sha256']),
                       'code_head':head,'branch':branch,
                       'worktree':str(Path(__file__).resolve().parents[1]),
                       'worktree_status_at_run':status,
                       'script_sha256':sha(Path(__file__)),
                       'offline_module_sha256':sha(Path(__file__).resolve().parents[1]/'src/visualnav_transformer/eat_nav/goal_bank_offline.py'),
                       'follower_config_sha256':sha(follower_path)})
    frozen=[]
    for anchor in anchors:
        frozen.append({k:plain(v) for k,v in anchor.items() if k not in
                       ('dem','dem_history_frames','proxy_bboxes')})
    write(output/'source-input-provenance.json',provenance)
    write(output/'eight-anchor-freeze.json',frozen)
    config={'schema':'eatnav_goal_bank_offline_v1','base_commit':'ecadcd3fa9f7cc57ecab092777f387d14d8d98d9',
            'bearings_deg':list(BEARINGS_DEG),'goal_distance_m':.6,'path_future_point_count':8,
            'path_start':'implicit robot origin; eight future poses exclude origin',
            'amplitude_pool_m':list(AMPLITUDE_POOL_M),'offset_pool_m':list(OFFSET_POOL_M),
            'bank_A_terminal_mode':'common','bank_B_terminal_modes_compared':['goal_connected','strict_goal_facing'],
            'dem_resolution_m':.1,'dem_shape':[2,20,20],
            'geometry_mode_rms_thresholds_m':{'full':.015,'early_0_5m':.01},
            'known_terrain_separation_rms_m':.01,'known_terrain_min_common_bins':5,
            'terrain_selection_metric':'known main-track corridor profile RMS at common known stations',
            'decision_follower_supply_minimum_average_candidates_per_state_group':4,
            'follower_first_command_config':follower,
            'follower_config_sha256':sha(follower_path),
            'selection_order':['hard_valid_groups','follower_compatible','early_separation',
                               'full_separation','known_terrain_separation','unknown_dependence',
                               'curvature_penalty','side_balance','parameter_simplicity'],
            'forbidden_selection_inputs':['reward','Critic','Outcome','Evaluator_prediction']}
    write(output/'screen-config-freeze.json',config)
    groups=[]
    for anchor in anchors:
        for bearing in BEARINGS_DEG:
            goal=goal_for_bearing(bearing)
            map_goal,inverse_error=_map_goal(goal,anchor['settled_map_xy_yaw'])
            cell=_goal_cell(goal,anchor['dem'][0],anchor['dem'][1])
            group={'state_group_id':f"{anchor['anchor_id']}-b{bearing:+03d}",
                   'anchor_id':anchor['anchor_id'],'seed_label':anchor['seed_label'],
                   'source_family':anchor['source_family'],'terrain':anchor['terrain'],
                   'terrain_class':anchor['terrain_class'],'bearing_deg':bearing,
                   'global_goal_base_xy_m':goal,'global_goal_map_xy_m':map_goal,
                   'map_inverse_error_m':inverse_error,'global_goal_cell':cell,
                   'global_goal_id':f"{anchor['anchor_id']}-goal-{bearing:+03d}",
                   'anchor_dem_history_sha256':anchor['dem_history_sha256'],
                   'anchor_body_state_actor_sha256':hashlib.sha256(json.dumps(anchor['body_state_actor'],separators=(',',':')).encode()).hexdigest(),
                   'state_contract':'anchor+four DEM frames+body state+global goal fixed; candidate Path only varies'}
            group['global_goal_sha256']=hashlib.sha256(json.dumps(plain(goal),separators=(',',':')).encode()).hexdigest()
            groups.append(group)
    if len(groups)!=40 or any(g['map_inverse_error_m']>1e-9 or not g['global_goal_cell']['inside'] for g in groups):
        raise ValueError('goal-bearing StateGroup contract failed')
    write(output/'state-groups.json',groups)
    write(output/'global-goal-contract.json',{'distance_m':.6,'bearings_deg':list(BEARINGS_DEG),
          'navigation_goal':'common global goal within each StateGroup',
          'map_transform':'settled root XY/yaw; exact inverse checked',
          'state_groups':40,'within_group_only_Path_varies':True})
    write(output/'local-subgoal-contract.json',{'bank':'B','local_endpoint':'G+s*normal(G)',
          'navigation_goal':'same global goal G for all candidates',
          'primary_teacher_terms':['global-goal distance reduction','endpoint-to-global-goal error',
                                   'global-goal encounter','direct-relative global-goal improvement'],
          'local_subgoal_arrival':'Path tracking diagnostic only; never primary navigation success'})
    anchors_by_id={a['anchor_id']:a for a in anchors}
    pools={('A','common'):{},('B','goal_connected'):{},('B','strict_goal_facing'):{}}
    for group in groups:
        anchor=anchors_by_id[group['anchor_id']]
        goal=np.asarray(group['global_goal_base_xy_m'])
        for (bank,mode),collection in pools.items():
            collection[group['state_group_id']]=_pool_for_group(bank,mode,goal,anchor,follower)
    screens=[]
    for (bank,mode),pool in pools.items():
        magnitudes=AMPLITUDE_POOL_M if bank=='A' else OFFSET_POOL_M
        for combo in itertools.combinations(magnitudes,3):
            screens.append(_score_combo(pool,combo,bank,mode))
    selection={'A':select_parameters([s for s in screens if s['bank']=='A']),
               'B':select_parameters([s for s in screens if s['bank']=='B'])}
    mode_comparison={}
    for mode in ('goal_connected','strict_goal_facing'):
        mode_choice=select_parameters([s for s in screens if s['bank']=='B' and s['terminal_mode']==mode])
        selected_rows=[]
        for group in pools[('B',mode)].values():
            selected_rows += [group['direct']]+[
                group[_candidate_key(side,m)] for side in ('left','right')
                for m in mode_choice['parameters']]
        mode_comparison[mode]={'parameters':mode_choice['parameters'],
                               'hard_valid_groups':mode_choice['hard_valid_groups'],
                               'follower_compatible':mode_choice['follower_compatible'],
                               'early_separation':mode_choice['early_separation'],
                               'known_terrain_separation':mode_choice['known_terrain_separation'],
                               'maximum_curvature_per_m':max(r['row']['maximum_curvature_per_m'] for r in selected_rows),
                               'turn_in_place_estimates':sum(r['follower']['turn_in_place_entry'] for r in selected_rows),
                               'reverse_estimates':sum(r['follower']['reverse_estimate'] for r in selected_rows)}
    write(output/'parameter-screen-results.json',screens)
    write(output/'selected-bank-config.json',selection)
    write(output/'bank-B-terminal-mode-comparison.json',mode_comparison)
    rows_by_bank={'A':[],'B':[]}
    pairs_by_bank={'A':[],'B':[]}
    for bank in ('A','B'):
        choice=selection[bank]
        pool=pools[(bank,choice['terminal_mode'])]
        for group in groups:
            anchor=anchors_by_id[group['anchor_id']]
            source=pool[group['state_group_id']]
            items=[]
            specs=[('direct',0.0,'direct')]+[
                (side,m,f'{side}-{name}') for side in ('left','right')
                for name,m in zip(('small','medium','large'),choice['parameters'])]
            for side,magnitude,candidate_id in specs:
                item=_candidate_record(group,anchor,bank,source[_candidate_key(side,magnitude)],candidate_id)
                items.append(item)
                rows_by_bank[bank].append(item)
            for pair in _pairwise_records(items):
                pairs_by_bank[bank].append({'state_group_id':group['state_group_id'],
                                            'bearing_deg':group['bearing_deg'],
                                            'terrain_class':group['terrain_class'],**pair})
    if any(len(rows_by_bank[b])!=280 for b in ('A','B')):
        raise ValueError('expected 280 definitions per bank')
    candidates=[_public_candidate(r) for bank in ('A','B') for r in rows_by_bank[bank]]
    unique_path_count=len({r['path_sha256'] for r in candidates})
    write(output/'candidate-definitions.json',candidates)
    write(output/'geometry-validity.json',[{'state_group_id':r['state_group_id'],'bank':r['bank'],
          'candidate_id':r['candidate_id'],**r['geometry']} for r in candidates])
    write(output/'follower-diagnostics.json',[{'state_group_id':r['state_group_id'],'bank':r['bank'],
          'candidate_id':r['candidate_id'],**r['follower']} for r in candidates])
    write(output/'terrain-corridor-metrics.json',[{'state_group_id':r['state_group_id'],'bank':r['bank'],
          'candidate_id':r['candidate_id'],'dem':r['dem']} for r in candidates])
    write(output/'pairwise-corridor-comparison.json',pairs_by_bank)
    bank_summary={bank:_summarize_bank(rows_by_bank[bank],pairs_by_bank[bank],40) for bank in ('A','B')}
    write(output/'bank-comparison.json',bank_summary)
    bearing_report={}
    for bearing in BEARINGS_DEG:
        by_bank={}
        for bank in ('A','B'):
            subset=[r for r in rows_by_bank[bank] if r['bearing_deg']==bearing]
            pair_subset=[p for p in pairs_by_bank[bank] if p['bearing_deg']==bearing]
            by_bank[bank]={**bank_summary[bank]['bearing_support'][str(bearing)],
                           'known_interaction_direct_groups':sum(r['candidate_id']=='direct' and r['dem']['interaction_onset_station_m'] is not None for r in subset),
                           'known_terrain_separated_pairs':sum(p['known_corridor_profile_rms_m'] is not None and p['common_known_corridor_bins']>=5 and p['known_corridor_profile_rms_m']>=.01 for p in pair_subset),
                           'mean_swept_exclusive_area_m2':float(np.mean([p['pairwise_exclusive_area_m2'] for p in pair_subset])),
                           'maximum_curvature_per_m':max(r['maximum_curvature_per_m'] for r in subset),
                           'left_follower_compatible':sum(r['side']=='left' and r['follower']['follower_compatible_screen'] for r in subset),
                           'right_follower_compatible':sum(r['side']=='right' and r['follower']['follower_compatible_screen'] for r in subset),
                           'local_subgoal_endpoint_inside':sum(np.all(np.abs(r['local_subgoal_xy_m'])<=1.0) for r in subset),
                           'local_subgoal_endpoint_known':sum(_goal_cell(np.asarray(r['local_subgoal_xy_m']),anchors_by_id[r['anchor_id']]['dem'][0],anchors_by_id[r['anchor_id']]['dem'][1])['known'] for r in subset)}
        bearing_report[str(bearing)]={'goal_cells_known':sum(g['bearing_deg']==bearing and g['global_goal_cell']['known'] for g in groups),
                                      'goal_cells_inside':sum(g['bearing_deg']==bearing and g['global_goal_cell']['inside'] for g in groups),
                                      'bank_support':by_bank,
                                      'terrain_class_retention':'frozen source-anchor class is fixed; path-aligned proxy is a separate diagnostic',
                                      'candidate_corridor_class_proxy_by_bank':{bank:{cls:sum(r['bearing_deg']==bearing and r['candidate_corridor_class_proxy']['class_proxy']==cls for r in rows_by_bank[bank]) for cls in (*CLASSES,'unknown')} for bank in ('A','B')}}
    write(output/'goal-bearing-comparison.json',bearing_report)
    source_grouping={}
    for anchor in anchors:
        source_grouping.setdefault(anchor['source_family'],[]).append(anchor['anchor_id'])
    split={'unit':'terrain seed/source family, not bearing, candidate pair or Path',
           'same_split_for':['five goal bearings','seven candidates','direct repeats',
                             'Bank A/B derivations','future Outcome'],
           'source_family_to_anchor_ids':source_grouping,
           'independent_source_family_count':len(source_grouping),
           'anchor_count':8,'state_group_count':40,
           'no_split_assignments_from_outcomes':True}
    write(output/'split-contract.json',split)
    decision,reasons=_decision(bank_summary['A'],bank_summary['B'])
    if decision=='candidate_bank_common_endpoint_ready_for_rollout_review':
        pilot_bank='A'
    elif decision=='candidate_bank_local_subgoal_ready_for_rollout_review':
        pilot_bank='B'
    else:
        pilot_bank='A' if bank_summary['A']['follower_compatible_candidate_count']>=bank_summary['B']['follower_compatible_candidate_count'] else 'B'
    pilot=_pilot_proposal(groups,[*rows_by_bank['A'],*rows_by_bank['B']],pilot_bank)
    pilot['provisional_when_decision_not_ready']=decision not in (
        'candidate_bank_common_endpoint_ready_for_rollout_review',
        'candidate_bank_local_subgoal_ready_for_rollout_review')
    write(output/'pilot-manifest-proposal.json',pilot)
    forecast=_forecast(backup,provenance)
    provenance['source_file_count']=len(provenance['sha256'])
    write(output/'runtime-storage-forecast.json',forecast)
    write(output/'prospective-rollout-gates.json',{
        'status':'prospective_only_no_rollout_authorized',
        'evaluation_within_bearing_only':True,
        'metrics':['P_to_C_command_mode','C_to_A_Actor_input_action_difference',
                   'A_to_R_realized_trajectory_mode','R_to_Y_2s_outcome_difference',
                   'direct_repeat_reproducibility','candidate_to_duplicate_ratio',
                   'global_goal_distance_reduction','discounted_reward_2s',
                   'direct_relative_reward','cross_track','Path_progress','termination',
                   'tilt','contact','effort','infrastructure_invalid_separation',
                   'coordinate_corrected_map_frame_geometry'],
        'no_cross_bearing_candidate_preference_comparison':True})
    risk={'source_family_dependence':'8 anchors represent only 6 distinct source families',
          'bank_B_teacher_bias':'local endpoint leaves nonzero global-goal residual on a short 2 s horizon',
          'follower_diagnostic_limit':'first-command arithmetic and curvature screen are not realized response',
          'DEM_unknown':'unknown differences excluded from terrain separation',
          'proxy_limit':'oriented collision-bound rectangles are descriptive, not contact response',
          'goal_bearing_goal_cell':'known/unknown status recorded per anchor and bearing',
          'long_runtime':'108 sequential fresh processes extrapolate beyond former 4 h cap; separate approval/cap required',
          'no_outcome':'candidate preference and vehicle feasibility remain unmeasured'}
    risk['bank_B_strict_terminal_priority']='the frozen priority selects strict goal-facing geometry despite much higher curvature than blended goal-connected; follower compatibility is not improved'
    write(output/'risk-register.json',risk)
    path_collection_sha=hashlib.sha256(''.join(r['path_sha256'] for r in candidates).encode()).hexdigest()
    final={'schema':'eatnav_goal_bank_review_decision_v1','decision':decision,'reasons':reasons,
           'base_commit':'ecadcd3fa9f7cc57ecab092777f387d14d8d98d9',
           'branch':branch,'anchor_count':8,'independent_source_family_count':len(source_grouping),
           'state_group_count':40,'candidate_counts':{'A':280,'B':280},
           'unique_path_sha256_count':unique_path_count,
           'selected_parameters':selection,'bank_summary':bank_summary,
           'pilot_bank_provisional':pilot_bank,'pilot_candidate_processes_one_bank':108,
           'candidate_paths_generated_offline':True,'candidate_paths_executed':False,
           'follower_bridge_router_started':False,'outcome_collected':False,
           'path_collection_sha256':path_collection_sha}
    write(output/'final-decision.json',final)
    (output/'report.md').write_text(_report(bank_summary,selection,mode_comparison,unique_path_count,
                                           provenance,forecast,decision,reasons))
    write(output/'source-input-provenance.json',provenance)
    if figures:
        make_figures(output,anchors,groups,rows_by_bank,pairs_by_bank,bank_summary,pilot,forecast,decision)
    ledger={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='sha-ledger.json'}
    write(output/'sha-ledger.json',{'schema':'eatnav_goal_bank_output_sha_v1','files':ledger})
    return final


def make_figures(output: Path, anchors: list[dict], groups: list[dict],
                 rows_by_bank: dict, pairs_by_bank: dict, bank_summary: dict,
                 pilot: dict, forecast: dict, decision: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    folder=output/'figures'
    folder.mkdir()

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder/name,dpi=150)
        plt.close(fig)

    # 01: each source DEM and all five local goals.
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for ax,anchor in zip(axes.flat,anchors):
        height=anchor['dem'][0]
        mask=anchor['dem'][1]>.5
        ax.imshow(np.ma.masked_where(~mask,height),origin='lower',extent=(-1,1,-1,1),cmap='terrain')
        ax.imshow(np.ma.masked_where(mask,np.ones((20,20))),origin='lower',extent=(-1,1,-1,1),cmap='Greys',alpha=.35)
        ax.scatter(0,0,c='black',s=15)
        for bearing in BEARINGS_DEG:
            goal=goal_for_bearing(bearing)
            ax.plot([0,goal[0]],[0,goal[1]],linewidth=.8,alpha=.6)
            ax.scatter(goal[0],goal[1],s=14)
        ax.set_title(anchor['anchor_id']+' / '+anchor['terrain'],fontsize=8)
        ax.set_xlim(-.1,.9);ax.set_ylim(-.6,.6);ax.set_aspect('equal')
    save(fig,'01-anchors-five-goal-bearings.png')

    # 02: seven Paths from the same representative StateGroup for each bank.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    sample='first-transition-b+00'
    for ax,bank in zip(axes,('A','B')):
        for row in rows_by_bank[bank]:
            if row['state_group_id']!=sample:
                continue
            points=np.asarray(row['dense_centerline_xy_m'])
            ax.plot(points[:,0],points[:,1],label=row['candidate_id'])
            ax.scatter(*np.asarray(row['local_subgoal_xy_m']),s=8)
        ax.scatter(.6,0,marker='*',s=90,c='black',label='global goal')
        ax.set_title('Bank '+bank+' / same global goal')
        ax.set_aspect('equal');ax.set_xlim(0,.75);ax.set_ylim(-.3,.3)
        ax.legend(fontsize=6,ncol=2)
    save(fig,'02-bank-A-B-paths.png')

    # 03: local endpoint is never the navigation task goal.
    fig,ax=plt.subplots(figsize=(6,4))
    ax.scatter(.6,0,marker='*',s=140,c='black',label='global task goal G')
    ax.scatter(0,0,s=45,c='gray',label='robot')
    for offset in OFFSET_POOL_M:
        ax.scatter(.6,offset,s=30,c='tab:blue')
        ax.scatter(.6,-offset,s=30,c='tab:orange')
        ax.annotate('',xy=(.6,0),xytext=(.6,offset),arrowprops={'arrowstyle':'->','alpha':.45})
    ax.set_xlim(-.05,.8);ax.set_ylim(-.32,.32);ax.set_aspect('equal');ax.legend()
    ax.set_title('Bank B: shifted Path endpoint, common navigation goal')
    save(fig,'03-local-subgoal-global-goal.png')

    # 04: bearing changes the task; Paths within one bearing vary independently.
    fig,ax=plt.subplots(figsize=(6,4))
    for bearing in BEARINGS_DEG:
        gid=f'first-transition-b{bearing:+03d}'
        for row in rows_by_bank['A']:
            if row['state_group_id']==gid and row['candidate_id']=='direct':
                points=np.asarray(row['dense_centerline_xy_m'])
                ax.plot(points[:,0],points[:,1],label=f'{bearing:+d}°')
    ax.set_xlim(0,.65);ax.set_ylim(-.3,.3);ax.set_aspect('equal');ax.legend()
    ax.set_title('Five bearing-specific direct Paths')
    save(fig,'04-bearing-direct-path-geometry.png')

    # 05: common-known profile, with blank intervals where source DEM is unknown.
    fig,ax=plt.subplots(figsize=(7,4))
    for row in rows_by_bank['A']:
        if row['state_group_id']=='first-transition-b+00' and row['candidate_id'] in ('direct','left-large','right-large'):
            profile=np.asarray(row['dem']['known_height_profile_m'])
            if len(profile): ax.plot(profile[:,0],profile[:,1],label=row['candidate_id'])
    ax.set_xlabel('Path arclength (m)');ax.set_ylabel('Robot-relative height (m)')
    ax.set_title('Known-only centerline DEM profile');ax.legend()
    save(fig,'05-terrain-corridor-profile.png')

    # 06: area and terrain separation are deliberately separate axes.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for bank,color in (('A','tab:blue'),('B','tab:orange')):
        pairs=pairs_by_bank[bank]
        axes[0].hist([p['pairwise_exclusive_area_m2'] for p in pairs],bins=16,alpha=.5,label=bank,color=color)
        axes[1].hist([p['known_corridor_profile_rms_m'] for p in pairs if p['known_corridor_profile_rms_m'] is not None],bins=16,alpha=.5,label=bank,color=color)
    axes[0].set_xlabel('Pairwise exclusive swept area (m²)')
    axes[1].set_xlabel('Common-known main-track corridor RMS (m)')
    for ax in axes: ax.legend();ax.set_ylabel('Pair count')
    save(fig,'06-swept-area-versus-known-terrain.png')

    # 07: follower command screen is not a vehicle turn limit.
    fig,ax=plt.subplots(figsize=(7,4))
    for bank,marker in (('A','o'),('B','x')):
        data=rows_by_bank[bank]
        ax.scatter([r['maximum_curvature_per_m'] for r in data],
                   [abs(r['follower']['initial_command_estimate'].get('angular_z') or 0) for r in data],
                   s=12,alpha=.45,label=bank,marker=marker)
    ax.axvline(1.236,color='black',linestyle='--',label='follower command screen')
    ax.set_xlabel('Maximum Path curvature (m⁻¹)');ax.set_ylabel('|initial estimated ω| (rad/s)')
    ax.set_title('Offline curvature and first-command diagnostic');ax.legend()
    save(fig,'07-curvature-follower-screen.png')

    # 08: class and seed support without counting derived Paths as independent.
    fig,ax=plt.subplots(figsize=(7,4))
    x=np.arange(len(CLASSES));width=.3
    for i,seed in enumerate(('first','second')):
        counts=[sum(a['seed_label']==seed and a['terrain_class']==cls for a in anchors) for cls in CLASSES]
        ax.bar(x+(i-.5)*width,counts,width,label=seed)
    ax.set_xticks(x,CLASSES);ax.set_ylabel('Frozen anchors');ax.set_ylim(0,1.3)
    ax.set_title('Eight anchors; six independent source families');ax.legend()
    save(fig,'08-class-seed-support.png')

    # 09: exact proposed process matrix, repeated direct distinguished.
    fig,ax=plt.subplots(figsize=(7,3.5))
    labels=['first-transition','first-differential','second-graded','second-flat-only']
    matrix=np.full((4,3),9)
    ax.imshow(matrix,vmin=0,vmax=10,cmap='Blues')
    ax.set_xticks(range(3),['−20°','0°','+20°']);ax.set_yticks(range(4),labels)
    for i in range(4):
        for j in range(3): ax.text(j,i,'3 direct + 6',ha='center',va='center',fontsize=8)
    ax.set_title('Pilot proposal: 12 StateGroups × 9 = 108 processes')
    save(fig,'09-pilot-state-group-matrix.png')

    # 10: forecast from measured prior per-process costs.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    names=['4×1','4×2','4×3','8×1']
    keys=['4_anchors_1_bearing','4_anchors_2_bearings','4_anchors_3_bearings','8_anchors_1_bearing']
    times=[forecast['alternatives'][k]['plus_20_percent_hours'] for k in keys]
    stores=[forecast['alternatives'][k]['saved_gb_plus_20_percent'] for k in keys]
    axes[0].bar(names,times);axes[0].set_ylabel('P95 + 20% hours')
    axes[1].bar(names,stores);axes[1].set_ylabel('Saved GB + 20%')
    for ax in axes: ax.set_xlabel('Anchors × bearings; 9 processes per group')
    save(fig,'10-runtime-storage-process-options.png')

    # 11: prospective measurement stages; none run here.
    fig,ax=plt.subplots(figsize=(10,2.8));ax.axis('off')
    labels=['Path P','follower command C','Actor A','realized motion R','Outcome Y']
    for i,label in enumerate(labels):
        ax.text((i+.5)/5,.55,label,ha='center',va='center',bbox={'boxstyle':'round','fc':'#dbeafe','ec':'#2563eb'})
        if i<4: ax.annotate('',xy=((i+1.42)/5,.55),xytext=((i+.72)/5,.55),arrowprops={'arrowstyle':'->'})
    ax.text(.5,.12,'Prospective only — this review observes P geometry and estimates initial C',ha='center')
    save(fig,'11-prospective-P-C-A-R-Y-gate.png')

    # 12: decision summarizes the two banks and the rollout authorization boundary.
    fig,ax=plt.subplots(figsize=(8,4));ax.axis('off')
    a,b=bank_summary['A'],bank_summary['B']
    text=(f"Decision: {decision}\n\n"
          f"A: {a['hard_valid_candidate_count']}/280 hard-valid, {a['follower_compatible_candidate_count']}/280 follower-screen, {a['terrain_separated_known_pairs']} terrain pairs\n"
          f"B: {b['hard_valid_candidate_count']}/280 hard-valid, {b['follower_compatible_candidate_count']}/280 follower-screen, {b['terrain_separated_known_pairs']} terrain pairs\n\n"
          'Offline bank definition only → separate review before any candidate rollout')
    ax.text(.04,.85,text,va='top',fontsize=11,bbox={'boxstyle':'round','fc':'#f3f4f6','ec':'#374151'})
    save(fig,'12-decision-flow.png')


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup-root',type=Path,default=Path('/home/kojima/eatnav-backups/2026-10-08-pre-shutdown'))
    parser.add_argument('--output-root',type=Path,default=Path('/tmp')/ROOT_NAME)
    args=parser.parse_args()
    result=run_review(args.backup_root,args.output_root)
    print(json.dumps({'decision':result['decision'],'output_root':str(args.output_root),
                      'candidate_counts':result['candidate_counts']},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
