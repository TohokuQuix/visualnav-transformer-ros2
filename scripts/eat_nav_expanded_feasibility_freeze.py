"""Freeze 50-process schedule, train-only E0 diagnostics and resource forecast."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

from eat_nav_expanded_feasibility_preflight import CONFIG,forecast_runtime,sha,write_once


def main():
    cfg=yaml.safe_load(CONFIG.read_text());root=Path(cfg['output_root'])
    freeze=json.loads((root/'input-freeze.json').read_text())
    if (sha(root/'selected-states.json')!=freeze['selected_states_sha256'] or
        sha(root/'candidate-paths.jsonl.gz')!=freeze['candidate_bank_sha256'] or
        not json.loads((root/'offline-e0-audit.json').read_text())['all_hard_valid']):
        raise ValueError('X3 input/E0 SHA or hard gate differs')
    old=Path(cfg['source_e1_root']);addendum=Path(cfg['source_coordinate_addendum'])
    states=json.loads((root/'selected-states.json').read_text())['train']
    with gzip.open(root/'candidate-paths.jsonl.gz','rt') as stream:
        bank=[json.loads(line) for line in stream]
    if len(states)!=5 or len(bank)!=35:raise ValueError('five StateGroups/seven paths each required')
    rows=[]
    for i,state in enumerate(states):
        paths={p['candidate_id']:p for p in bank if p['state_id']==state['state_id']}
        folder=root/'inputs'/f'state-{i:03d}'
        if list(paths)!=cfg['candidate_ids']:raise ValueError('candidate order differs')
        for role,path_id,repeat in ([('anchor_only','direct_reference',0)]
            +[('nominal_duplicate','direct_reference',j) for j in range(3)]
            +[('candidate_outcome',p,0) for p in cfg['candidate_ids'][1:]]):
            process_id=(f"{state['state_id']}-anchor_only-00" if role=='anchor_only' else
                f"{state['state_id']}-nominal_duplicate-{repeat:02d}" if role=='nominal_duplicate' else
                f"{state['state_id']}-{path_id}")
            rows.append({'order':len(rows),'process_id':process_id,'state_group_id':state['state_id'],
                'state_index':i,'terrain':state['terrain'],'terrain_seed':state['terrain_seed'],
                'episode_key':state['episode_key'],'window_id':state['source_window_id'],
                'role':role,'path_id':path_id,'seed':cfg['run_seed'],'repeat_id':repeat,
                'oracle_denominator':role=='candidate_outcome' or (role=='nominal_duplicate' and repeat==0),
                'robot_local_waypoint_sha256':__import__('hashlib').sha256(
                    np.asarray(paths[path_id]['waypoints_xy_m'],dtype=np.float32).tobytes()).hexdigest(),
                'base_path_file_sha256':sha(folder/f'{path_id}-base-path.json'),
                'map_path_file_sha256':'pending_anchor_only',
                'input_sha256':freeze['per_state'][i]})
    if (len(rows)!=50 or sum(x['role']=='anchor_only' for x in rows)!=5 or
        sum(x['role']=='nominal_duplicate' for x in rows)!=15 or
        sum(x['oracle_denominator'] for x in rows)!=35):
        raise ValueError('50/5/15/35 schedule accounting differs')
    write_once(root/'run-manifest.json',{'schema_version':'eatnav_expanded_feasibility_50_v1',
        'rows':rows,'state_groups':5,'fresh_processes':50,'unique_outcomes':35,
        'anchor_only':5,'direct_repeats':15,'non_direct':30,
        'existing_E1_groups_reused_not_rerun':2,'old_E1_protocol_valid_run_count':18})
    old_runs=[]
    for folder in sorted((old/'runs').iterdir()):
        path=folder/'run-complete.json'
        if path.is_file():old_runs.append(json.loads(path.read_text())['elapsed_sec'])
    if len(old_runs)!=18:raise ValueError('old E1 runtime support differs')
    anchor_durations=[json.loads(p.read_text())['process_elapsed_sec'] for p in
        sorted(Path('/tmp/eatnav-g0e-formal-anchor-20261003/runs').glob('anchor-*/final-result.json'))]
    if len(anchor_durations)!=2:raise ValueError('old formal anchor runtime support differs')
    old_wall=json.loads((old/'final-decision-reviewed.json').read_text())['wall_clock_sec']
    baseline_overhead=max(0.,old_wall-sum(old_runs))
    point=45*float(np.mean(old_runs))+5*float(np.mean(anchor_durations))+baseline_overhead*50/18
    p95=45*float(np.quantile(old_runs,.95))+5*max(anchor_durations)+baseline_overhead*50/18
    with_margin=p95*cfg['resources']['forecast_margin']
    prior_bytes=json.loads((old/'postrun-verification.json').read_text())['combined_storage_bytes']
    storage_prediction=int(prior_bytes*50/18*cfg['resources']['forecast_margin'])
    forecast={'measured_18_process_wall_sec':old_wall,'measured_18_process_elapsed_sum_sec':sum(old_runs),
        'measured_process_mean_sec':float(np.mean(old_runs)),'measured_process_p95_sec':float(np.quantile(old_runs,.95)),
        'measured_anchor_sec':anchor_durations,'old_wall_overhead_sec':baseline_overhead,
        'new_processes':50,'new_anchor':5,'new_candidate_processes':45,
        'point_forecast_sec':point,'p95_forecast_sec':p95,'p95_plus_20pct_sec':with_margin,
        'wall_cap_sec':cfg['resources']['wall_clock_cap_sec'],'old_combined_storage_bytes':prior_bytes,
        'storage_forecast_with_margin_bytes':storage_prediction,
        'storage_cap_bytes':cfg['resources']['storage_cap_bytes'],
        'status':'pass' if with_margin<=cfg['resources']['wall_clock_cap_sec'] and storage_prediction<=cfg['resources']['storage_cap_bytes'] else 'stop'}
    write_once(root/'runtime-storage-forecast.json',forecast)
    if forecast['status']!='pass':raise ValueError('X8 runtime/storage forecast stop before rollout')
    # This diagnostic is train-only and never enters eligibility or candidate choice.
    derived=Path('/tmp/eatnav-candidate-coverage/goal-ood/train-derived.npz')
    if sha(derived)!='3c1ec88a0a94e62f02c116642cbbd7c2ef2c48b0b414e49521fc8ebe544955ec':
        raise ValueError('train-only derived array SHA differs')
    with np.load(derived) as data:
        goals=np.asarray(data['goals']);actions=np.asarray(data['actions'])
    target=np.asarray(cfg['goal_robot_local_xy_m'])
    neighbors=actions[np.linalg.norm(goals-target,axis=1)<=.05].cumsum(axis=1)
    if len(neighbors)<10:raise ValueError('train-only goal neighborhood insufficient')
    nn=[]
    for row in bank:
        path=np.asarray(row['waypoints_xy_m'])
        distance=np.sqrt(np.mean(np.sum((neighbors-path)**2,axis=-1),axis=-1))
        nn.append({'state_id':row['state_id'],'candidate_id':row['candidate_id'],
            'nearest_train_8point_RMS_m':float(distance.min()),'goal_neighbor_count':len(neighbors)})
    write_once(root/'train-only-path-nearest-neighbor.json',{'rows':nn,'source_sha256':sha(derived),
        'goal_neighbor_radius_m':.05,'selection_or_gate_used':False})
    prior=json.loads((root/'source-provenance.json').read_text())
    source_paths={
        'candidate_bank_config':CONFIG,'old_e1_gates':old/'e1-gates.yaml',
        'old_e1_safety':old/'e1-safety-stops.yaml','old_manifest':old/'continuation-manifest.json',
        'coordinate_frame_spec':addendum/'coordinate-frame-spec.json',
        'follower_binary':Path('/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon/exact-follower-build/simple_trajectory_follower_node'),
        'follower_source':Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/src/simple_trajectory_follower.cpp'),
        'follower_params':Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml'),
        'bridge_source':Path('/tmp/eatnav-vnt-e1-feasibility/scripts/eat_nav_command_delivery_bridge.py'),
        'exact_stamp_source':Path('/tmp/eatnav-vnt-e1-feasibility/src/visualnav_transformer/eat_nav/exact_stamp_protocol.py'),
        'process_ownership_source':Path('/tmp/eatnav-vnt-e1-feasibility/scripts/eat_nav_e1_process_ownership.py'),
        'router_binary':Path('/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd'),
        'router_config':Path('/opt/ros/jazzy/share/rmw_zenoh_cpp/config/DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5'),
        'router_host_client_config':Path('/opt/ros/jazzy/share/rmw_zenoh_cpp/config/DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5'),
    }
    values={key:{'path':str(path),'sha256':sha(path)} for key,path in source_paths.items()}
    prior['expanded_source_sha256']=values
    prior['resource_forecast_sha256']=sha(root/'runtime-storage-forecast.json')
    prior['run_manifest_sha256']=sha(root/'run-manifest.json')
    prior['original_worktree_clean_at_creation']=True
    prior['branch']='exp/eatnav-expanded-feasibility'
    prior['starting_HEAD']='ca6e112886469ce8af24f08b7654bc2d57ad7388'
    prior['candidate_bank_config_sha256']=sha(CONFIG)
    prior['old_source_commits']={'E1':json.loads((old/'source-freeze.json').read_text())['HEAD'],
        'artifact_recovery':'b8ce3e4eb911cba9e1f2a2dd07d0b65c680bd464'}
    # X1 provenance is refined before any new outcome; preserve the original copy.
    write_once(root/'source-provenance-x0-full.json',prior)
    write_once(root/'preflight-freeze.json',{'status':'X0_X1_X3_X8_FROZEN_NO_ROLLOUT',
        'config_sha256':sha(CONFIG),'selected_states_sha256':sha(root/'selected-states.json'),
        'candidate_bank_sha256':sha(root/'candidate-paths.jsonl.gz'),
        'e0_audit_sha256':sha(root/'offline-e0-audit.json'),
        'manifest_sha256':sha(root/'run-manifest.json'),
        'runtime_forecast_sha256':sha(root/'runtime-storage-forecast.json'),
        'source_provenance_full_sha256':sha(root/'source-provenance-x0-full.json'),
        'old_E1_decision_sha256':sha(old/'final-gate-decision.json'),
        'old_coordinate_addendum_sha256':sha(addendum/'original-vs-corrected-decision.json'),
        'new_candidate_outcomes_seen':False})
    print('X8_FORECAST_PASS',round(point),round(p95),round(with_margin),storage_prediction,flush=True)


if __name__=='__main__':main()
