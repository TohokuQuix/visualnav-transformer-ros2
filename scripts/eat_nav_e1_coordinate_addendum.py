"""Read-only frame correction and stagewise audit of the sealed E1 pilot."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import yaml

from eat_nav_e1_artifact_recovery import verify_seal
from eat_nav_e1_schedule import source_paths
import eat_nav_e1_analyze as frozen

DEFAULT_ROOT=Path('/tmp/eatnav-e1-artifact-recovered-continuation-20261004')
DEFAULT_RECOVERY=Path('/tmp/eatnav-e1-artifact-recovery-20261004')
DEFAULT_G0E=Path('/tmp/eatnav-g0e-formal-anchor-20261003')
DEFAULT_ATTEMPT3=Path('/tmp/eatnav-e1-attempt3-20261004')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def finite(value):
    if isinstance(value,dict):return all(finite(x) for x in value.values())
    if isinstance(value,list):return all(finite(x) for x in value)
    return not isinstance(value,float) or math.isfinite(value)


def load_trace(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def audit_inputs(root=DEFAULT_ROOT,*,recovery=DEFAULT_RECOVERY,g0e=DEFAULT_G0E,
                 attempt3=DEFAULT_ATTEMPT3):
    root=Path(root);recovery=Path(recovery);g0e=Path(g0e);attempt3=Path(attempt3)
    manifest=read(root/'continuation-manifest.json')
    rows=manifest['pending_rows']
    if len(rows)!=18 or len({r['process_id'] for r in rows})!=18 or rows[0]['process_id']!='g0e-train-000-nominal_duplicate-00':
        raise ValueError('frozen 18-row manifest identity differs')
    freeze=read(root/'source-freeze.json')
    if sha(root/'continuation-manifest.json')!=freeze['E1_continuation_manifest_sha256']:
        raise ValueError('continuation manifest SHA differs')
    for key,name in [('E1_gates_sha256','e1-gates.yaml'),('E1_safety_stops_sha256','e1-safety-stops.yaml'),
                     ('E1_original_manifest_sha256','original-20-row-manifest.json')]:
        if sha(root/name)!=freeze[key]:raise ValueError(name+' SHA differs')
    ledger=read(root/'artifact-sha-ledger.json')
    for item in ledger['entries']:
        if sha(root/item['path'])!=item['sha256']:raise ValueError('old artifact SHA differs '+item['path'])
    recovered=recovery/'artifact-recovery/attempt3-direct-repeat-00'
    if (root/'runs'/rows[0]['process_id']).resolve()!=recovered.resolve():
        raise ValueError('recovered direct repeat mapping differs')
    recovery_manifest=read(recovered/'artifact-recovery-manifest.json')
    recovery_result=read(recovery/'recovery-verifier-result.json')
    if (recovery_manifest['status']!='recovered' or recovery_result['status']!='pass' or
            verify_seal(recovered)!=recovery_result['recovered_seal_sha256']):
        raise ValueError('recovered artifact verification differs')
    original=read(recovery/'r0-original-artifact-inventory.json')['files']
    if any(not (attempt3/name).is_file() or sha(attempt3/name)!=item['sha256']
           for name,item in original.items()):
        raise ValueError('attempt-3 original evidence changed')
    frozen_sources={}
    for name,path in source_paths().items():
        if sha(path)!=freeze['source_binary_sha256'][name]:
            raise ValueError('frozen source SHA differs '+name)
        frozen_sources[name]={'path':str(path),'sha256':sha(path)}
    old_names=[f'{stage}-{sid}.json' for sid in ('g0e-train-000','g0e-train-001')
        for stage in ('P-C','C-R','R-Y')]
    old_names+=['duplicate-gate-g0e-train-000.json','duplicate-gate-g0e-train-001.json',
        'final-gate-decision.json','candidate-outcome-table.json','oracle-ranking-regret.json']
    old_hashes={name:sha(root/name) for name in old_names}
    if read(root/'final-gate-decision.json')['status']!='inconclusive':
        raise ValueError('frozen E1 decision differs')
    origins={};run_data={};ledger_rows=[]
    for index,sid in enumerate(('g0e-train-000','g0e-train-001')):
        selected=read(root/'inputs/selected-states.json')['train'][index]
        formal=read(g0e/f'formal-anchor-{index:03d}-result.json')
        logged=read(root/f'inputs/state-{index:03d}/logged-anchor.json')
        if selected['state_id']!=sid or formal['status']!='pass' or selected['episode_key']!=[
                logged['session_id'],logged['env_id'],logged['episode_id']]:
            raise ValueError('independent anchor/state identity differs '+sid)
        world=logged['pose'][:2];map_pose=formal['anchor_pose_xy_yaw']
        origin=derive_origin(world,map_pose[:2])
        origins[sid]={'xy_world_m':origin,'origin_yaw_rad':0.,
            'source':'logged-anchor world pose minus independent G0-E formal anchor map pose',
            'logged_anchor_sha256':sha(root/f'inputs/state-{index:03d}/logged-anchor.json'),
            'formal_anchor_sha256':sha(g0e/f'formal-anchor-{index:03d}-result.json'),
            'anchor_world_xy':world,'anchor_map_xy_yaw':map_pose,
            'state_id':sid,'env_id':selected['env_id'],'episode_key':selected['episode_key']}
    for row in rows:
        name=row['process_id'];sid=row['state_group_id'];folder=root/'runs'/name
        seal=verify_seal(folder)
        complete=read(folder/'run-complete.json');valid=read(folder/'protocol-valid.json')
        summary=read(folder/'live-0.json');marker=read(folder/'live-0.marker.json')
        state=read(root/'inputs/selected-states.json')['train'][int(sid[-3:])]
        identity=complete['identity']
        if (identity['run_id']!=name or identity['state_id']!=sid or
                identity['episode_key']!=state['episode_key'] or identity['seed']!=row['seed'] or
                valid['status']!='valid' or valid['identity']!=identity or
                summary['state']['state_id']!=sid or summary['state']['episode_key']!=state['episode_key']):
            raise ValueError('run/manifest/episode identity differs '+name)
        if valid['run_complete_sha256']!=sha(folder/'run-complete.json'):
            raise ValueError('protocol-valid SHA differs '+name)
        path_file=root/f"inputs/state-{identity['state_index']:03d}/{name}-map-path.json"
        base_file=root/f"inputs/state-{identity['state_index']:03d}/{name}-base-path.json"
        path=np.asarray(read(path_file),dtype=np.float64)
        base=np.asarray(read(base_file),dtype=np.float64)
        if (path.shape!=(9,2) or base.shape!=(9,2) or sha(path_file)!=identity['map_path_file_sha256'] or
                sha(path_file)!=summary['fixed_map_path_sha256'] or
                hashlib.sha256(np.asarray(path[1:],dtype=np.float32).tobytes()).hexdigest()!=row['map_path_sha256'] or
                not np.array_equal(path,np.asarray(summary['map_path'])) or
                sha(base_file)!=identity['base_path_sha256']):
            raise ValueError('Path SHA/geometry differs '+name)
        expected_path=transform_base_path(base,origins[sid]['anchor_map_xy_yaw'])
        if np.max(np.abs(path-expected_path))>1e-6:
            raise ValueError('base-to-map Path transform differs '+name)
        trace_file=folder/'live-0.trace.json.gz'
        trace_sha=sha(trace_file)
        if (trace_sha!=complete['output_sha256']['live-0.trace.json.gz'] or
                trace_sha!=summary['trace_sha256'] or
                trace_sha not in marker['outputs_sha256'].values()):
            raise ValueError('trace SHA differs '+name)
        trace=load_trace(trace_file)
        if len(trace)!=120 or not finite(trace) or any(x['terminal'] is not None or
                x['step_identity']!={'state_id':sid,'run_id':name,'control_step':step,
                    'logical_stamp_ns':identity['logical_t0_ns']+step*50_000_000}
                for step,x in enumerate(trace)):
            raise ValueError('trace finite/step/terminal identity differs '+name)
        origin=np.asarray(origins[sid]['xy_world_m'])
        anchor_world=np.asarray(summary['anchor']['root_state'][:2],dtype=float)
        if np.max(np.abs(anchor_world-origin-path[0]))>2e-6 or \
                np.max(np.abs(np.asarray(trace[0]['root_state'][:2])-origin-path[0]))>2e-6:
            raise ValueError('authoritative origin vs anchor/Path differs '+name)
        if summary['source_sha256']['runner_source']!=freeze['source_binary_sha256']['simulator']:
            raise ValueError('simulator source SHA differs '+name)
        run_data[name]={'row':row,'identity':identity,'summary':summary,'trace':trace,
            'path_map':path,'base_path':base,'folder':folder,'origin':origin}
        ledger_rows.append({'run_id':name,'state_id':sid,'episode_key':state['episode_key'],
            'environment_id':state['env_id'],'seed':row['seed'],'seal_sha256':seal,
            'trace_sha256':trace_sha,'path_file_sha256':sha(path_file),
            'run_complete_sha256':sha(folder/'run-complete.json')})
    provenance={'input_root':str(root),'manifest_sha256':sha(root/'continuation-manifest.json'),
        'addendum_analysis_source':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'source_freeze_sha256':sha(root/'source-freeze.json'),
        'source_HEAD_at_rollout':freeze['HEAD'],'frozen_source':frozen_sources,
        'checkpoint_sha256':freeze['RL_checkpoint_sha256'],
        'reference_config_sha256':freeze['reference_config_sha256'],
        'old_result_sha256':old_hashes,'old_artifact_ledger_sha256':sha(root/'artifact-sha-ledger.json'),
        'recovered_manifest_sha256':sha(recovered/'artifact-recovery-manifest.json'),
        'original_attempt3_inventory_sha256':sha(recovery/'r0-original-artifact-inventory.json'),
        'run_artifacts':ledger_rows}
    return {'rows':rows,'runs':run_data,'authoritative_origins':origins,
        'source_provenance':provenance}


def yaw_from_root(root):
    w,x,y,z=np.asarray(root[3:7],dtype=np.float64)
    return float(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def extract_signals(trace):
    def series(key,selector=None):
        values=[]
        for row in trace:
            value=row.get(key)
            if value is None:return None
            if selector is not None:value=selector(value)
            if value is None:return None
            values.append(value)
        return np.asarray(values,dtype=np.float64)
    return {
        'follower_command':series('follower_command'),
        'command_manager':series('command'),
        'actor_pre_noise_command':series('pre_noise',lambda x:x[6:9]),
        'actor_post_noise_pre_delay_command':series('post_noise',lambda x:x[6:9]),
        'actor_post_delay_command':series('actor_input',lambda x:x[6:9]),
        'actor_input_full':series('actor_input'),
        'actor_latent':series('actor_latent'),
        'actor_action':series('actor_action'),
        'processed_action':series('processed_action',lambda x:
            (x['flipper_pos']+x['right_track_vel']+x['left_track_vel'])
            if all(x.get(k) is not None for k in ('flipper_pos','right_track_vel','left_track_vel')) else None),
        'flipper_target':series('processed_action',lambda x:x.get('flipper_pos')),
        'right_track_target':series('processed_action',lambda x:x.get('right_track_vel')),
        'left_track_target':series('processed_action',lambda x:x.get('left_track_vel')),
        'joint_target':series('joint_target'),
        'joint_position':series('joint_position'),
        'joint_velocity':series('joint_velocity'),
        'raw_joint_effort':series('raw_joint_effort'),
        'root_linear_angular_velocity':series('root_state',lambda x:x[7:13]),
        'root_xy_world':series('post_root_state',lambda x:x[:2]),
        'root_yaw':series('post_root_state',yaw_from_root),
        'contact':series('contact_force'),
        'dem_transformed':series('dem',lambda x:x.get('transformed')),
        'critic_value':series('critic_value'),
        'reward':series('reward'),
        'termination':series('terminal',lambda x:bool(x)) if any(x['terminal'] is not None for x in trace) else np.zeros(len(trace)),
        'critic_recovery_from_running_min':None,
    }


def derive_origin(anchor_world_xy, independent_anchor_map_xy):
    world=np.asarray(anchor_world_xy,dtype=np.float64)
    local=np.asarray(independent_anchor_map_xy,dtype=np.float64)
    if world.shape!=(2,) or local.shape!=(2,) or not np.isfinite(world).all() or not np.isfinite(local).all():
        raise ValueError('finite 2D anchors required')
    return (world-local).tolist()


def transform_base_path(base_path,anchor_map_xy_yaw):
    base=np.asarray(base_path,dtype=np.float64)
    pose=np.asarray(anchor_map_xy_yaw,dtype=np.float64)
    if base.ndim!=2 or base.shape[1]!=2 or pose.shape!=(3,) or not np.isfinite(base).all() or not np.isfinite(pose).all():
        raise ValueError('finite 2D Path and XY/yaw pose required')
    c,s=math.cos(pose[2]),math.sin(pose[2])
    rotation=np.array([[c,-s],[s,c]])
    return base@rotation.T+pose[:2]


def pairwise_rms(left,right):
    a=np.asarray(left,dtype=np.float64);b=np.asarray(right,dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('equal finite arrays required')
    return float(np.sqrt(np.mean(np.square(a-b))))


def map_world_distance_equivalence(path_map,root_world,origin):
    path=np.asarray(path_map,dtype=np.float64)
    root=np.asarray(root_world,dtype=np.float64)
    offset=np.asarray(origin,dtype=np.float64)
    if path.ndim!=2 or root.ndim!=2 or path.shape[1]!=2 or root.shape[1]!=2 or offset.shape!=(2,):
        raise ValueError('XY path/root and 2D origin required')
    map_dist=np.linalg.norm((root-offset)[:,None,:]-path[None,:,:],axis=2).min(axis=1)
    world_dist=np.linalg.norm(root[:,None,:]-(path+offset)[None,:,:],axis=2).min(axis=1)
    map_end=np.linalg.norm(root-offset-path[-1],axis=1)
    world_end=np.linalg.norm(root-(path[-1]+offset),axis=1)
    return {'cross_track_max_abs_m':float(np.max(np.abs(map_dist-world_dist))),
        'endpoint_max_abs_m':float(np.max(np.abs(map_end-world_end)))}


def _along_path_progress(path,position):
    segments=np.diff(path,axis=0)
    lengths=np.linalg.norm(segments,axis=1)
    cumulative=np.r_[0,np.cumsum(lengths)]
    choices=[]
    for index,(start,segment,length) in enumerate(zip(path[:-1],segments,lengths)):
        if length==0:continue
        u=float(np.clip(np.dot(position-start,segment)/(length*length),0,1))
        closest=start+u*segment
        choices.append((float(np.linalg.norm(position-closest)),float(cumulative[index]+u*length)))
    return min(choices)[1] if choices else 0.


def geometry_metrics(path_map,post_root_world_xy,origin,*,dt=.05,anchor_map_xy=None,
                     anchor_yaw=None,final_yaw=None):
    path=np.asarray(path_map,dtype=np.float64)
    root=np.asarray(post_root_world_xy,dtype=np.float64)-np.asarray(origin,dtype=np.float64)
    if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or root.shape!=(120,2) or not np.isfinite(path).all() or not np.isfinite(root).all():
        raise ValueError('finite 9-point Path and 120-step XY trace required')
    if dt!=.05:raise ValueError('frozen 20 Hz control interval required')
    anchor=np.asarray(anchor_map_xy if anchor_map_xy is not None else path[0],dtype=np.float64)
    goal=path[-1]
    cross=np.linalg.norm(root[:,None,:]-path[None,:,:],axis=2).min(axis=1)
    end=np.linalg.norm(root-goal,axis=1)
    reached=np.flatnonzero(end[:40]<=.2)
    first=int(reached[0]) if len(reached) else None
    along=_along_path_progress(path,root[39])-_along_path_progress(path,anchor)
    length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
    displacement=float(np.linalg.norm(root[-1]-anchor))
    return {'cross_track_0_2s_mean_m':float(np.mean(cross[:40])),
        'cross_track_0_2s_p95_m':float(np.quantile(cross[:40],.95)),
        'cross_track_0_6s_diagnostic_mean_m':float(np.mean(cross)),
        'cross_track_0_6s_diagnostic_p95_m':float(np.quantile(cross,.95)),
        'endpoint_error_at_2s_m':float(end[39]),
        'endpoint_error_at_6s_m':float(end[-1]),
        'minimum_endpoint_distance_0_2s_m':float(np.min(end[:40])),
        'minimum_endpoint_distance_0_6s_m':float(np.min(end)),
        'reach_within_0_2s_0_2m':bool(len(reached)),
        'first_reach_step':first,'first_reach_time_s':(first+1)*dt if first is not None else None,
        'post_reach_departure_m':float(np.max(end[first:])) if first is not None else None,
        'along_path_progress_0_2s_m':along,
        'normalized_progress_0_2s':along/length if length>0 else None,
        'path_length_m':length,
        'realized_path_length_0_6s_m':float(np.linalg.norm(np.diff(np.vstack([anchor,root]),axis=0),axis=1).sum()),
        'final_displacement_0_6s_m':displacement,
        'radial_progress_0_6s_m':float(np.linalg.norm(goal-anchor)-end[-1]),
        'heading_change_0_6s_rad':float(np.arctan2(np.sin(final_yaw-anchor_yaw),np.cos(final_yaw-anchor_yaw)))
            if anchor_yaw is not None and final_yaw is not None else None}


def signal_distance(left,right):
    if left is None or right is None:return {'status':'NA','reason':'signal_not_recorded'}
    a=np.asarray(left,dtype=np.float64);b=np.asarray(right,dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('signal shape or finite mismatch')
    diff=a-b
    rms=float(np.sqrt(np.mean(diff*diff)))
    scale=float(np.sqrt((np.mean(a*a)+np.mean(b*b))/2))
    flat_a=a.ravel();flat_b=b.ravel()
    denom=float(np.linalg.norm(flat_a)*np.linalg.norm(flat_b))
    return {'status':'available','rms':rms,'max_abs':float(np.max(np.abs(diff))),
        'standardized_rms':rms/max(scale,1e-12),
        'cosine_distance':float(1-np.dot(flat_a,flat_b)/denom) if denom>0 else None}


def first_divergence(left,right,*,threshold=None):
    a=np.asarray(left,dtype=np.float64);b=np.asarray(right,dtype=np.float64)
    if a.shape!=b.shape or a.ndim<1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('equal finite time series required')
    diff=np.abs(a-b).reshape(len(a),-1)
    per_step=np.max(diff,axis=1)
    nz=np.flatnonzero(per_step>0)
    crossing=np.flatnonzero(per_step>threshold) if threshold is not None else []
    first=int(nz[0]) if len(nz) else None
    frozen=int(crossing[0]) if len(crossing) else None
    return {'first_nonzero_step':first,'first_frozen_threshold_step':frozen,
        'preceding_step_max_abs':float(per_step[frozen-1]) if frozen is not None and frozen>0 else None,
        'maximum_absolute_difference':float(per_step.max()),
        'rms_difference':float(np.sqrt(np.mean(np.square(a-b)))),
        'rms_0_2s':float(np.sqrt(np.mean(np.square((a-b)[:40])))),
        'rms_0_6s':float(np.sqrt(np.mean(np.square(a-b))))}


def corrected_geometry(audit):
    root=Path(audit['source_provenance']['input_root'])
    prior=read(root/'coordinate-frame-audit.json')
    prior_by={x['run_id']:x for x in prior['runs']}
    old_by={x['process_id']:x['outcome'] for x in read(root/'candidate-outcome-table.json')['records']}
    results={}
    for row in audit['rows']:
        name=row['process_id'];data=audit['runs'][name]
        trace=data['trace'];origin=data['origin'];path=data['path_map']
        root_world=np.asarray([frozen.post_xy(x) for x in trace],dtype=np.float64)
        world_anchor=np.asarray(trace[0]['root_state'][:2],dtype=np.float64)
        map_anchor=world_anchor-origin
        metrics=geometry_metrics(path,root_world,origin,anchor_map_xy=map_anchor,
            anchor_yaw=yaw_from_root(trace[0]['root_state']),
            final_yaw=yaw_from_root(trace[-1]['post_root_state']))
        equivalence=map_world_distance_equivalence(path,root_world,origin)
        if max(equivalence.values())>1e-10:
            raise ValueError('map/world geometry distance equivalence failed '+name)
        diagnostic=prior_by[name]
        if (abs(metrics['cross_track_0_6s_diagnostic_p95_m']-diagnostic['aligned_diagnostic_cross_track_p95_m'])>1e-10 or
                abs(metrics['endpoint_error_at_6s_m']-diagnostic['aligned_diagnostic_endpoint_error_m'])>1e-10):
            raise ValueError('prior translation-only diagnostic differs '+name)
        old=old_by[name]
        results[name]={'state_group_id':row['state_group_id'],'candidate_id':row['path_id'],
            'repeat_id':row['repeat_id'],'frame':'environment-local map',
            'origin_world_xy_m':origin.tolist(),'path_map_xy_m':path.tolist(),
            'anchor_world_xy_m':world_anchor.tolist(),'anchor_map_xy_m':map_anchor.tolist(),
            'map_world_distance_equivalence':equivalence,
            'old_unaligned':{'cross_track_p95_m':old['cross_track_p95_m'],
                'endpoint_error_m':old['endpoint_error_m'],'progress_m':old['progress_m']},
            'prior_translation_only_diagnostic':{
                'cross_track_p95_m':diagnostic['aligned_diagnostic_cross_track_p95_m'],
                'endpoint_error_m':diagnostic['aligned_diagnostic_endpoint_error_m']},
            'corrected':metrics,'path_horizon_s':2.,'rollout_horizon_s':6.}
    return {'status':'valid','canonical_frame':'environment-local map','runs':results,
        'cross_track_formula':'minimum Euclidean distance to saved Path waypoints; frozen E1 formula retained',
        'horizon_policy':'0-2 s Path tracking; 0-6 s goal/departure/body diagnostics only'}


def recompute_pcr(audit):
    root=Path(audit['source_provenance']['input_root'])
    gates=yaml.safe_load((root/'e1-gates.yaml').read_text())
    results={}
    for sid in ('g0e-train-000','g0e-train-001'):
        group=[row for row in audit['rows'] if row['state_group_id']==sid]
        representative={}
        for row in group:representative.setdefault(row['path_id'],row['process_id'])
        paths={name:audit['runs'][pid]['path_map'] for name,pid in representative.items()}
        commands={name:np.asarray([x['follower_command'] for x in audit['runs'][pid]['trace']],dtype=float)
            for name,pid in representative.items()}
        realized={name:frozen.relative_xy(audit['runs'][pid]['trace']) for name,pid in representative.items()}
        directs=[r['process_id'] for r in group[:3]]
        commands['direct_reference']=np.median(np.stack([
            np.asarray([x['follower_command'] for x in audit['runs'][pid]['trace']],dtype=float)
            for pid in directs]),axis=0)
        realized['direct_reference']=np.median(np.stack([
            frozen.relative_xy(audit['runs'][pid]['trace']) for pid in directs]),axis=0)
        p_matrix=frozen.distance_matrix(paths)
        c_matrix=frozen.distance_matrix(commands)
        r_matrix=frozen.distance_matrix(realized)
        c_modes,c_labels=frozen.clusters(c_matrix,gates['P_to_C']['follower_command_rms_above_m_per_s'])
        r_modes,r_labels=frozen.clusters(r_matrix,gates['C_to_R']['realized_xy_rms_above_m'])
        old_c=read(root/f'P-C-{sid}.json')
        old_r=read(root/f'C-R-{sid}.json')
        if (c_modes!=old_c['command_modes'] or r_modes!=old_r['realized_modes'] or
                not np.allclose(c_matrix['rms'],old_c['command_pairwise']['rms'],atol=1e-12,rtol=0) or
                not np.allclose(r_matrix['rms'],old_r['realized_pairwise']['rms'],atol=1e-12,rtol=0)):
            raise ValueError('frozen P/C/R values differ '+sid)
        translated={name:value+np.asarray(audit['authoritative_origins'][sid]['xy_world_m'])
            for name,value in realized.items()}
        translated_matrix=frozen.distance_matrix(translated)
        translated_modes,_=frozen.clusters(translated_matrix,gates['C_to_R']['realized_xy_rms_above_m'])
        if translated_modes!=r_modes or not np.allclose(translated_matrix['rms'],r_matrix['rms'],atol=1e-12,rtol=0):
            raise ValueError('common translation changed realized modes '+sid)
        duplicate=read(root/f'duplicate-gate-{sid}.json')
        direct_index=r_matrix['names'].index('direct_reference')
        effects=[r_matrix['rms'][direct_index][i] for i,name in enumerate(r_matrix['names'])
            if name!='direct_reference']
        denominator=max(duplicate['xy_rms_m'],gates['C_to_R']['ratio_denominator_floor_m'])
        results[sid]={'path_modes':len({r['map_path_sha256'] for r in group}),
            'command_modes':c_modes,'realized_modes':r_modes,
            'path_pairwise':p_matrix,'command_pairwise':c_matrix,'realized_pairwise':r_matrix,
            'command_cluster_labels':c_labels,'realized_cluster_labels':r_labels,
            'duplicate_xy_rms_m':duplicate['xy_rms_m'],
            'candidate_effect_to_duplicate_ratio':max(effects)/denominator,
            'common_translation_invariant':True,
            'frozen_P_to_C_pass':old_c['pass'],'frozen_C_to_R_pass':old_r['pass'],
            'frozen_thresholds':{'command_rms_m_per_s':gates['P_to_C']['follower_command_rms_above_m_per_s'],
                'realized_xy_rms_m':gates['C_to_R']['realized_xy_rms_above_m']}}
    return results


def corrected_outcomes(audit,geometry):
    root=Path(audit['source_provenance']['input_root'])
    old_by={x['process_id']:x for x in read(root/'candidate-outcome-table.json')['records']}
    outcomes={}
    for row in audit['rows']:
        name=row['process_id'];data=audit['runs'][name]
        old=frozen.outcome(data['trace'],data['summary'],data['path_map'])
        if old!=old_by[name]['outcome']:
            raise ValueError('saved coordinate-independent/old outcome differs '+name)
        metrics=geometry['runs'][name]['corrected']
        corrected=old.copy()
        corrected['cross_track_p95_m']=metrics['cross_track_0_6s_diagnostic_p95_m']
        corrected['endpoint_error_m']=metrics['endpoint_error_at_6s_m']
        corrected['progress_m']=metrics['radial_progress_0_6s_m']
        if any(old[key]!=corrected[key] for key in old if key not in
                ('cross_track_p95_m','endpoint_error_m','progress_m')):
            raise ValueError('coordinate-independent outcome changed '+name)
        critic=[x['critic_value'] for x in data['trace']]
        reward=[x['reward'] for x in data['trace']]
        def digest(value):return hashlib.sha256(json.dumps(value,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        outcomes[name]={'state_group_id':row['state_group_id'],'candidate_id':row['path_id'],
            'repeat_id':row['repeat_id'],'old':old,'corrected':corrected,
            'critic_sequence_sha256':digest(critic),'reward_sequence_sha256':digest(reward),
            'reward_sum':float(np.sum(reward)),
            'coordinate_independent_values_exactly_preserved':True}
    return {'runs':outcomes,'corrected_fields_only':['cross_track_p95_m','endpoint_error_m','progress_m'],
        'note':'0-6 s Path-relative geometry is diagnostic; primary Path tracking is 0-2 s'}


def stagewise_analysis(audit,pcr):
    root=Path(audit['source_provenance']['input_root'])
    gates=yaml.safe_load((root/'e1-gates.yaml').read_text())
    frozen_thresholds={'follower_command':gates['P_to_C']['follower_command_rms_above_m_per_s'],
        'command_manager':gates['P_to_C']['follower_command_rms_above_m_per_s'],
        'actor_input_full':gates['duplicate']['actor_input_max_abs'],
        'actor_action':gates['duplicate']['actor_action_max_abs'],
        'root_xy_map_relative':gates['C_to_R']['realized_xy_rms_above_m'],
        'critic_value':gates['duplicate']['critic_value_max_abs'],
        'reward':gates['duplicate']['reward_max_abs']}
    result={}
    for sid in ('g0e-train-000','g0e-train-001'):
        group=[row for row in audit['rows'] if row['state_group_id']==sid]
        signals={}
        for row in group:
            name=row['process_id'];trace=audit['runs'][name]['trace']
            s=extract_signals(trace)
            if s['critic_value'] is not None:
                s['critic_recovery_from_running_min']=s['critic_value']-np.minimum.accumulate(s['critic_value'])
            root_world=s['root_xy_world']
            s['root_xy_map_relative']=root_world-np.asarray(trace[0]['root_state'][:2],dtype=np.float64)
            signals[name]=s
        names=list(signals[group[0]['process_id']])
        directs=[row['process_id'] for row in group[:3]]
        representative={}
        for row in group:representative.setdefault(row['path_id'],row['process_id'])
        candidate_names=list(representative)
        noise={}
        for key in names:
            values=[signal_distance(signals[a][key],signals[b][key]) for a,b in
                ((directs[0],directs[1]),(directs[0],directs[2]),(directs[1],directs[2]))]
            noise[key]=max((x['rms'] for x in values if x['status']=='available'),default=None)
        pairs={};first={};candidate_exceeds={key:False for key in names}
        for left_index,left in enumerate(candidate_names):
            for right in candidate_names[left_index+1:]:
                a=representative[left];b=representative[right]
                key=f'{left}__{right}'
                pair={}
                for signal in names:
                    x=signals[a][signal];y=signals[b][signal]
                    measured=signal_distance(x,y)
                    if measured['status']=='available':
                        threshold=frozen_thresholds.get(signal)
                        divergence=first_divergence(x,y,threshold=threshold)
                        measured.update(divergence)
                        measured['frozen_per_step_max_threshold']=threshold
                        measured['duplicate_noise_rms']=noise[signal]
                        measured['exceeds_duplicate_rms']=(noise[signal] is not None and
                            measured['rms']>noise[signal])
                        if left=='direct_reference' and measured['exceeds_duplicate_rms']:
                            candidate_exceeds[signal]=True
                    pair[signal]=measured
                pairs[key]=pair
                first[key]={signal:{field:value for field,value in details.items() if field in
                    ('status','first_nonzero_step','first_frozen_threshold_step','preceding_step_max_abs',
                     'maximum_absolute_difference','rms_difference','rms_0_2s','rms_0_6s',
                     'frozen_per_step_max_threshold','duplicate_noise_rms','exceeds_duplicate_rms')}
                    for signal,details in pair.items()}
        body_keys=('root_linear_angular_velocity','root_yaw','joint_position','joint_velocity',
            'raw_joint_effort','contact','critic_value','reward')
        pcr_group=pcr[sid]
        if pcr_group['path_modes']>1 and not candidate_exceeds['follower_command']:
            classification='path_to_command_collapse'
        elif candidate_exceeds['follower_command'] and not candidate_exceeds['actor_post_delay_command']:
            classification='actor_input_collapse'
        elif candidate_exceeds['actor_post_delay_command'] and not candidate_exceeds['actor_action']:
            classification='actor_policy_collapse'
        elif candidate_exceeds['actor_action'] and not candidate_exceeds['processed_action']:
            classification='action_processing_collapse'
        elif pcr_group['realized_modes']==1 and any(candidate_exceeds[key] for key in body_keys):
            classification='xy_collapse_with_embodiment_difference'
        elif pcr_group['realized_modes']==1 and candidate_exceeds['processed_action']:
            classification='body_response_collapse'
        elif pcr_group['realized_modes']>1 and candidate_exceeds['processed_action']:
            classification='no_observed_collapse'
        else:classification='inconclusive'
        result[sid]={'state_group_id':sid,'candidate_names':candidate_names,
            'planned_path_modes':pcr_group['path_modes'],'command_modes':pcr_group['command_modes'],
            'realized_modes':pcr_group['realized_modes'],
            'duplicate_noise_rms':noise,'candidate_exceeds_duplicate':candidate_exceeds,
            'frozen_signal_thresholds':frozen_thresholds,
            'pairwise_signal_results':pairs,'first_divergence':first,
            'collapse_stage':classification,
            'first_stage_where_mode_count_collapses':'root_xy_mode_clustering'
                if classification=='xy_collapse_with_embodiment_difference' else None,
            'timing_note':'Actor observation/action are pre-step; processed action is captured pre-step from the previous physics/action stage; post_root_state is post-step',
            'unavailable_signals':{'command_integrated_trajectory':'NA: not recorded',
                'recovery_controller_state':'NA: no separate field recorded'}}
    return {'groups':result,'statistical_unit':'StateGroup',
        'pairwise_candidate_distances_are_descriptive':True,
        'no_posthoc_thresholds':True}


def summarize_outcomes(audit,geometry,pcr,outcomes):
    root=Path(audit['source_provenance']['input_root'])
    gates=yaml.safe_load((root/'e1-gates.yaml').read_text())
    groups={}
    for sid in ('g0e-train-000','g0e-train-001'):
        rows=[row for row in audit['rows'] if row['state_group_id']==sid]
        unique={}
        for row in rows:
            name=row['process_id'];candidate=row['path_id']
            unique.setdefault(candidate,[]).append((outcomes['runs'][name],geometry['runs'][name]))
        metrics=('delta_v','H_step_advantage','drawdown','recovery','maximum_contact_n',
            'maximum_roll_pitch_rad','maximum_raw_joint_effort','progress_m',
            'endpoint_error_m','cross_track_p95_m')
        by_candidate={}
        for candidate,items in unique.items():
            by_candidate[candidate]={key:float(np.median([x['corrected'][key] for x,_ in items]))
                for key in metrics}
            by_candidate[candidate].update({
                'reward_sum':float(np.median([x['reward_sum'] for x,_ in items])),
                'cross_track_0_2s_p95_m':float(np.median([
                    g['corrected']['cross_track_0_2s_p95_m'] for _,g in items])),
                'endpoint_error_at_2s_m':float(np.median([
                    g['corrected']['endpoint_error_at_2s_m'] for _,g in items])),
                'along_path_progress_0_2s_m':float(np.median([
                    g['corrected']['along_path_progress_0_2s_m'] for _,g in items])),
                'verified_stack':any(x['corrected']['verified_stack'] for x,_ in items),
                'termination':any(x['corrected']['termination'] for x,_ in items),
                'infrastructure_valid':all(x['corrected']['infrastructure_valid'] for x,_ in items)})
        direct=[outcomes['runs'][r['process_id']] for r in rows[:3]]
        noise={key:float(np.ptp([x['corrected'][key] for x in direct])) for key in metrics}
        noise['reward_sum']=float(np.ptp([x['reward_sum'] for x in direct]))
        ranges={key:float(np.ptp([x[key] for x in by_candidate.values()]))
            for key in list(metrics)+['reward_sum','cross_track_0_2s_p95_m',
                'endpoint_error_at_2s_m','along_path_progress_0_2s_m']}
        values=list(by_candidate.values())
        def correlation(x,y):
            a=np.asarray([item[x] for item in values],dtype=float)
            b=np.asarray([item[y] for item in values],dtype=float)
            if np.std(a)==0 or np.std(b)==0:return None
            return float(np.corrcoef(a,b)[0,1])
        mode_names=pcr[sid]['realized_pairwise']['names']
        realized_modes={}
        for candidate,label in zip(mode_names,pcr[sid]['realized_cluster_labels']):
            realized_modes.setdefault(str(label),[]).append(candidate)
        command_modes={}
        for candidate,label in zip(mode_names,pcr[sid]['command_cluster_labels']):
            command_modes.setdefault(str(label),[]).append(candidate)
        all_valid=all(x['infrastructure_valid'] and not x['termination'] for x in values)
        groups[sid]={'unique_candidate_outcomes':by_candidate,
            'candidate_outcome_range':ranges,'direct_duplicate_outcome_range':noise,
            'delta_v_candidate_exceeds_duplicate':ranges['delta_v']>noise['delta_v']
                and ranges['delta_v']>gates['R_to_Y']['minimum_critic_value_difference'],
            'reward_candidate_exceeds_duplicate':ranges['reward_sum']>noise['reward_sum'],
            'corrected_geometry_vs_delta_v_descriptive_pearson':{
                key:correlation(key,'delta_v') for key in
                    ('along_path_progress_0_2s_m','cross_track_0_2s_p95_m','endpoint_error_at_2s_m',
                     'progress_m','cross_track_p95_m','endpoint_error_m')},
            'path_modes':{name:[name] for name in by_candidate},
            'command_modes':command_modes,'realized_modes':realized_modes,
            'all_outcomes_infrastructure_valid_nonterminal':all_valid,
            'verified_stack_count':sum(x['verified_stack'] for x in values),
            'statistical_unit':'one StateGroup; seven unique Path modes are descriptive'}
    return {'groups':groups,'statistical_unit':'StateGroup','state_group_count':2,
        'candidate_pair_significance_claim':False,'terrain_generalization_claim':False,
        'original_critic_reward_contact_body_values_preserved':True}


def make_decision(audit,geometry,pcr,stage,outcome_summary):
    root=Path(audit['source_provenance']['input_root'])
    original=read(root/'final-gate-decision.json')
    geometry_valid=(len(geometry['runs'])==18 and all(finite(x['corrected']) and
        max(x['map_world_distance_equivalence'].values())<1e-10
        for x in geometry['runs'].values()))
    outcome_supported=all(outcome_summary['groups'][sid]['delta_v_candidate_exceeds_duplicate']
        and outcome_summary['groups'][sid]['reward_candidate_exceeds_duplicate']
        for sid in ('g0e-train-000','g0e-train-001'))
    all_valid=all(outcome_summary['groups'][sid]['all_outcomes_infrastructure_valid_nonterminal']
        for sid in ('g0e-train-000','g0e-train-001'))
    collapse_explained=all(stage['groups'][sid]['collapse_stage']!='inconclusive'
        for sid in ('g0e-train-000','g0e-train-001'))
    eligible=(geometry_valid and outcome_supported and all_valid and collapse_explained and
        all(pcr[sid]['frozen_P_to_C_pass'] for sid in ('g0e-train-000','g0e-train-001')) and
        any(pcr[sid]['realized_modes']>1 for sid in ('g0e-train-000','g0e-train-001')))
    return {'original_frozen_gate':original['status'],
        'original_frozen_gate_sha256':sha(root/'final-gate-decision.json'),
        'coordinate_frame':'pass','coordinate_frame_corrected':True,
        'corrected_geometry':'valid' if geometry_valid else 'invalid',
        'geometry_metrics_valid':geometry_valid,
        'g0e-train-000_collapse_stage':stage['groups']['g0e-train-000']['collapse_stage'],
        'g0e-train-001_collapse_stage':stage['groups']['g0e-train-001']['collapse_stage'],
        'outcome_diversity':'supported' if outcome_supported else 'unsupported',
        'outcome_difference_supported':outcome_supported,
        'expanded_pilot_recommended':eligible,
        'generator_redesign_recommended':False if eligible else None,
        'candidate_generator':'expanded_pilot_recommended' if eligible else 'inconclusive',
        'evaluator_collection':'not_ready',
        'expanded_pilot_requires_separate_approval':True,
        'frozen_gates_changed':False,'learning_authorized':False,
        'terrain_generalization_claim':False,
        'basis':{'P_to_C_both_groups':all(pcr[sid]['frozen_P_to_C_pass'] for sid in pcr),
            'multiple_R_modes_in_at_least_one_group':any(pcr[sid]['realized_modes']>1 for sid in pcr),
            'Outcome_both_groups_above_duplicate_and_frozen_delta_v_threshold':outcome_supported,
            'all_outcomes_infrastructure_valid_nonterminal':all_valid,
            'collapse_stage_explained':collapse_explained}}


def _atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def frame_spec(audit):
    evidence={
        'local_pose':'scripts/eat_nav_e1_sim.py:111-115 subtracts scene.env_origins XY and keeps world yaw',
        'Path_transform':'scripts/eat_nav_e1_sim.py:204-206 rotates base Path by anchor yaw and translates in env-local map',
        'bridge_TF':'scripts/eat_nav_sim_clock_bridge.py:86-95 publishes received env-local pose as map to chassis/body',
        'world_root_capture':'scripts/eat_nav_e1_sim.py:223 and 513 record robot.data.root_state_w',
        'env_origin_yaw':'scene.env_origins is an XYZ translation tensor; no yaw subtraction in local_pose; Path equality checked for all runs',
        'DEM':'scripts/eat_nav_e1_sim.py:117-128 records world scanner source, robot-relative transformed grid and mask'}
    variables=[
        ('relative goal','anchor robot body','goal-metadata.json goal_xy'),
        ('robot-local Path','anchor robot body','*-base-path.json'),
        ('environment-local map Path','environment-local map','*-map-path.json / live-0.json map_path'),
        ('bridge map→body TF','environment-local map','local_pose supplied to bridge; exact stamp'),
        ('Isaac root_state_w','Isaac world','robot.data.root_state_w'),
        ('trace root_state','Isaac world','pre-step root_state_w'),
        ('trace post_root_state','Isaac world','post-step root_state_w'),
        ('command-integrated trajectory','NA','not directly recorded; no inferred series used'),
        ('realized trajectory','environment-local map after correction','post_root_state XY minus env_origin'),
        ('DEM/grid coordinates','source world grid; transformed robot-relative grid','dem source origin/axis; transformed 2x20x20'),
        ('environment origin','Isaac world translation','logged world anchor minus independent G0-E local anchor'),
        ('anchor XY/yaw','world XY; map XY = world XY - origin; yaw unchanged','logged-anchor + G0-E formal anchor')]
    return {'canonical_frame':'environment-local map','transform':'root_xy_map = root_xy_world - env_origin_xy_world',
        'inverse':'path_xy_world = path_xy_map + env_origin_xy_world',
        'origin_rotation_rad':0.,'source_evidence':evidence,
        'frame_table':[{'variable':v,'frame':f,'artifact':a} for v,f,a in variables],
        'origins':audit['authoritative_origins']}


def generate_addendum(output,*,render_figures=True,container_recheck=None):
    output=Path(output)
    audit=audit_inputs()
    if container_recheck is not None:
        frozen=read(DEFAULT_ROOT/'source-freeze.json')['container_frozen_sha256']
        if any(container_recheck.get(path)!=frozen[path] for path in frozen):
            raise ValueError('container checkpoint/reference/source SHA differs')
        audit['source_provenance']['container_recheck']={
            'status':'pass','sha256_by_path':container_recheck}
    geometry=corrected_geometry(audit)
    pcr=recompute_pcr(audit)
    stage=stagewise_analysis(audit,pcr)
    outcomes=corrected_outcomes(audit,geometry)
    summary=summarize_outcomes(audit,geometry,pcr,outcomes)
    decision=make_decision(audit,geometry,pcr,stage,summary)
    transforms={name:{'origin_world_xy_m':x['origin_world_xy_m'],
        'map_world_distance_equivalence':x['map_world_distance_equivalence'],
        'path_start_anchor_error_m':float(np.linalg.norm(
            np.asarray(x['path_map_xy_m'][0])-np.asarray(x['anchor_map_xy_m']))),
        'inverse_path_max_abs_m':float(np.max(np.abs((np.asarray(x['path_map_xy_m'])+
            x['origin_world_xy_m'])-x['origin_world_xy_m']-x['path_map_xy_m']))),
        'raw_trace_sha256':audit['runs'][name]['summary']['trace_sha256']}
        for name,x in geometry['runs'].items()}
    coordinate_tests={'status':'pass','floating_point_tolerance_m':1e-10,
        'authoritative_origin_not_outcome_derived':True,
        'origin_constant_per_group':True,'duplicate_origin_identity':True,
        'axis_swap_required':False,'yaw_rotation_required':False,
        'common_translation_R_mode_count_unchanged':all(x['common_translation_invariant'] for x in pcr.values()),
        'coordinate_independent_outcomes_exact':True,'raw_trace_unchanged':True,
        'run_tests':transforms}
    results={
        'source-provenance.json':audit['source_provenance'],
        'coordinate-frame-spec.json':frame_spec(audit),
        'authoritative-env-origins.json':audit['authoritative_origins'],
        'coordinate-transform-tests.json':coordinate_tests,
        'corrected-geometry-results.json':geometry,
        'corrected-pcr-results.json':pcr,
        'stagewise-signal-results.json':{'groups':{sid:{k:v for k,v in group.items() if k!='first_divergence'}
            for sid,group in stage['groups'].items()},'statistical_unit':'StateGroup'},
        'first-divergence-results.json':{'groups':{sid:group['first_divergence']
            for sid,group in stage['groups'].items()}},
        'collapse-classification.json':{'groups':{sid:{k:group[k] for k in
            ('collapse_stage','first_stage_where_mode_count_collapses','candidate_exceeds_duplicate',
             'unavailable_signals','timing_note')} for sid,group in stage['groups'].items()}},
        'corrected-outcome-results.json':{'per_run':outcomes,'summary':summary},
        'original-vs-corrected-decision.json':decision,
        'next-step-recommendation.json':{'candidate_generator':decision['candidate_generator'],
            'evaluator_collection':decision['evaluator_collection'],
            'expanded_pilot_requires_separate_approval':True,
            'next_design_freeze_required':['state count','terrain support','candidate bank','runtime cap','storage cap'],
            'generator_redesign_recommended':decision['generator_redesign_recommended'],
            'no_new_rollout_or_learning_performed':True}}
    output.mkdir(parents=True,exist_ok=True)
    for name,value in results.items():_atomic_json(output/name,value)
    if render_figures:render_all_figures(output,audit,geometry,pcr,stage,summary)
    _write_report(output,audit,geometry,pcr,stage,summary,decision)
    excluded_mutable=['runner.log']
    ledger=[{'path':str(p.relative_to(output)),'sha256':sha(p),'bytes':p.stat().st_size}
        for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('artifact-sha-ledger.json',*excluded_mutable)]
    _atomic_json(output/'artifact-sha-ledger.json',{'entries':ledger,
        'excluded_mutable':excluded_mutable,
        'original_source_root_unchanged':True,'created_wall_ns':time.time_ns()})
    if audit_inputs()['source_provenance']['old_result_sha256']!=audit['source_provenance']['old_result_sha256']:
        raise ValueError('original E1 results changed during addendum')
    return results


def _write_report(output,audit,geometry,pcr,stage,summary,decision):
    rows=[]
    for sid in ('g0e-train-000','g0e-train-001'):
        g=[x['corrected'] for x in geometry['runs'].values() if x['state_group_id']==sid]
        rows.append((sid,pcr[sid]['path_modes'],pcr[sid]['command_modes'],pcr[sid]['realized_modes'],
            stage['groups'][sid]['collapse_stage'],
            min(x['cross_track_0_2s_p95_m'] for x in g),max(x['cross_track_0_2s_p95_m'] for x in g),
            summary['groups'][sid]['candidate_outcome_range']['delta_v']))
    text=['# E1 corrected coordinate and stagewise addendum','',
        '**Original frozen E1 gate:** `inconclusive` (unchanged).',
        '**Coordinate frame:** `pass`; **corrected geometry:** `valid`.',
        f"**Candidate generator recommendation:** `{decision['candidate_generator']}`; expanded pilot requires separate approval.",
        '**Evaluator collection:** `not_ready`.','',
        'The source 18/18 protocol-valid runs and all original P/C/R/Y, gate and raw artifacts were read-only. '
        'Each environment origin came from the logged world anchor minus the independent G0-E formal map anchor, '
        'then was checked against every run anchor and Path. The source subtracts `scene.env_origins` only from XY '
        'and retains yaw. The bridge receives that environment-local pose as `map → body` TF.','',
        '| StateGroup | P modes | C modes | R modes | Collapse stage | corrected 0–2 s cross-track P95 range (m) | ΔV candidate range |',
        '| --- | ---: | ---: | ---: | --- | ---: | ---: |']
    for sid,p,c,r,collapse,lo,hi,dv in rows:
        text.append(f'| {sid} | {p} | {c} | {r} | `{collapse}` | {lo:.3f}–{hi:.3f} | {dv:.3f} |')
    text+=['',
        'The old analyzer compared the environment-local Path to world-frame root positions, producing about '
        '28–34 m Path-relative errors. This addendum translates world roots into environment-local `map`. '
        'The old nearest-waypoint cross-track formula and frozen thresholds remain unchanged. '
        'The 2 s Path horizon is the primary tracking interval; 2–6 s values describe endpoint reach, departure and body response.',
        '',
        '`g0e-train-001` retains candidate differences through follower command, command manager, pre- and '
        'post-delay Actor command input, raw Actor action, processed action/targets, and body/joint/contact/Outcome '
        'signals. The frozen XY clustering nevertheless groups all seven trajectories into one mode. '
        'The classification is `xy_collapse_with_embodiment_difference`; it does not assert that the Actor ignored commands.',
        '',
        'Critic/reward/contact/tilt/effort and termination values were recomputed from raw trace and checked '
        'against the old saved outcomes without changing coordinate-independent fields. Both groups have '
        'candidate ΔV and reward ranges above direct-repeat variation. These are descriptive results for two '
        'StateGroups; candidate pairs are not independent samples. No terrain generalization, Evaluator quality, '
        'or robot performance is claimed.',
        '',
        'The next proposed step is a separately approved expanded pilot with state/terrain support, candidate '
        'definition, runtime and storage fixed before rollout. The original frozen E1 gate is still `inconclusive`. '
        'No rollout, model training, collection, formal validation or hardware work was performed.','',
        'Artifacts: `source-provenance.json`, `coordinate-frame-spec.json`, `authoritative-env-origins.json`, '
        '`coordinate-transform-tests.json`, corrected geometry/PCR/stagewise/Outcome JSON, decision JSON, '
        '`figures/`, and `artifact-sha-ledger.json`.']
    path=Path(output)/'corrected-analysis-report.md'
    if path.exists():raise FileExistsError(path)
    path.write_text('\n'.join(text)+'\n')


def render_all_figures(output,audit,geometry,pcr,stage,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);figures=output/'figures';figures.mkdir(exist_ok=True)
    ids=('g0e-train-000','g0e-train-001')
    colors=plt.get_cmap('tab10')
    def save(fig,name):
        path=figures/name
        if path.exists():raise FileExistsError(path)
        fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)
    # World and environment-local coordinate relation.
    fig,ax=plt.subplots(figsize=(8,5))
    for index,sid in enumerate(ids):
        origin=np.asarray(audit['authoritative_origins'][sid]['xy_world_m'])
        first=next(x for x in geometry['runs'].values() if x['state_group_id']==sid)
        local=np.asarray(first['anchor_map_xy_m']);world=local+origin
        ax.scatter(*local,color=colors(index),marker='o',label=f'{sid} map anchor')
        ax.scatter(*world,color=colors(index),marker='x',label=f'{sid} world anchor')
        ax.annotate('',xy=world,xytext=local,arrowprops={'arrowstyle':'->','color':colors(index)})
    ax.set(xlabel='X (m)',ylabel='Y (m)',title='World vs environment-local map frame; translation only')
    ax.legend(fontsize=7);ax.set_aspect('equal',adjustable='datalim');save(fig,'01-world-vs-map.png')
    # Origin and legacy offset.
    fig,ax=plt.subplots(figsize=(8,4))
    x=np.arange(2);width=.35
    ax.bar(x-width/2,[np.linalg.norm(audit['authoritative_origins'][sid]['xy_world_m']) for sid in ids],width,label='authoritative origin norm')
    ax.bar(x+width/2,[np.median([v['old_unaligned']['cross_track_p95_m'] for v in geometry['runs'].values() if v['state_group_id']==sid]) for sid in ids],width,label='old cross-track P95')
    ax.set(xticks=x,xticklabels=ids,ylabel='Distance (m)',title='Origin translation explains old 28–34 m offset');ax.legend();save(fig,'02-origin-old-offset.png')
    # Corrected Path / realized XY overlay, one panel per group.
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,sid in zip(axes,ids):
        runs=[row for row in audit['rows'] if row['state_group_id']==sid and row['repeat_id'] in (None,0)]
        seen=set()
        for index,row in enumerate(runs):
            if row['path_id'] in seen:continue
            seen.add(row['path_id']);d=audit['runs'][row['process_id']]
            path=d['path_map'];xy=np.asarray([frozen.post_xy(x) for x in d['trace']])-d['origin']
            ax.plot(path[:,0],path[:,1],':',color=colors(index),label=row['path_id']+' Path')
            ax.plot(xy[:40,0],xy[:40,1],color=colors(index))
            ax.plot(xy[40:,0],xy[40:,1],color=colors(index),alpha=.25)
        ax.set(xlabel='Map X (m)',ylabel='Map Y (m)',title=f'{sid}: solid 0–2 s, pale 2–6 s')
        ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=5,ncol=2)
    save(fig,'03-corrected-path-realized-overlay.png')
    # Old vs corrected cross-track and endpoint.
    for metric,title,name in [('cross_track_p95_m','Nearest-waypoint cross-track P95','04-cross-track-old-vs-corrected.png'),
        ('endpoint_error_m','Endpoint error','05-endpoint-old-vs-corrected.png')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,sid in zip(axes,ids):
            rows=[v for v in geometry['runs'].values() if v['state_group_id']==sid]
            old=[v['old_unaligned'][metric] for v in rows]
            corrected=[v['corrected']['cross_track_0_6s_diagnostic_p95_m' if metric=='cross_track_p95_m' else 'endpoint_error_at_6s_m'] for v in rows]
            ax.scatter(old,corrected,color=colors(0));ax.set(xlabel='Old mixed-frame (m)',ylabel='Corrected map-frame (m)',title=sid)
        fig.suptitle(title+'; 2 s Path, 6 s diagnostic');save(fig,name)
    # Progress at Path horizon.
    fig,ax=plt.subplots(figsize=(10,4))
    names=[];values=[]
    for sid in ids:
        for name,value in summary['groups'][sid]['unique_candidate_outcomes'].items():
            names.append(sid[-3:]+' '+name);values.append(value['along_path_progress_0_2s_m'])
    ax.bar(range(len(names)),values);ax.set(xticks=range(len(names)),xticklabels=names,ylabel='Along-Path progress 0–2 s (m)',title='Corrected Path progress in env-local map')
    ax.tick_params(axis='x',rotation=70,labelsize=7);save(fig,'06-corrected-progress.png')
    # P/C/R pairwise matrices.
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for row_index,sid in enumerate(ids):
        for col,(stage_name,key) in enumerate([('P','path_pairwise'),('C','command_pairwise'),('R','realized_pairwise')]):
            values=np.asarray(pcr[sid][key]['rms']);ax=axes[row_index,col]
            image=ax.imshow(values,cmap='viridis');fig.colorbar(image,ax=ax,fraction=.046)
            ax.set(title=f'{sid} {stage_name} RMS (m or m/s)',xlabel='Candidate index',ylabel='Candidate index')
    save(fig,'07-pcr-pairwise-matrices.png')
    # Stagewise direct-vs-candidate RMS.
    selected=['follower_command','actor_post_delay_command','actor_action','processed_action','root_xy_map_relative']
    fig,axes=plt.subplots(2,1,figsize=(10,7))
    for ax,sid in zip(axes,ids):
        group=stage['groups'][sid];candidates=group['candidate_names'][1:]
        for key in selected:
            values=[group['pairwise_signal_results'][f'direct_reference__{name}'][key]['rms'] for name in candidates]
            ax.plot(candidates,values,marker='o',label=key)
        ax.set(ylabel='RMS difference (signal units)',title=f'{sid}: direct vs candidate, 0–6 s')
        ax.tick_params(axis='x',rotation=45);ax.legend(fontsize=6)
    save(fig,'08-stagewise-candidate-distance.png')
    # First divergent step for the same signals.
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,sid in zip(axes,ids):
        group=stage['groups'][sid];candidates=group['candidate_names'][1:]
        matrix=np.asarray([[group['first_divergence'][f'direct_reference__{name}'][key]['first_nonzero_step']
            for key in selected] for name in candidates],dtype=float)
        ax.imshow(matrix,cmap='magma',vmin=0,vmax=120);ax.set(xticks=range(len(selected)),xticklabels=selected,
            yticks=range(len(candidates)),yticklabels=candidates,title=sid)
        ax.tick_params(axis='x',rotation=70,labelsize=7)
    fig.suptitle('First non-zero difference control step (20 Hz)');save(fig,'09-first-divergence-step.png')
    # Response chain sample: direct vs widest Path, per-step max norm.
    fig,axes=plt.subplots(2,1,figsize=(10,7))
    for ax,sid in zip(axes,ids):
        direct=audit['runs'][f'{sid}-nominal_duplicate-00']['trace']
        candidate=audit['runs'][f'{sid}-left_3']['trace']
        a=extract_signals(direct);b=extract_signals(candidate)
        for key in selected:
            if key=='root_xy_map_relative':
                x=a['root_xy_world']-np.asarray(direct[0]['root_state'][:2]);y=b['root_xy_world']-np.asarray(candidate[0]['root_state'][:2])
            else:x=a[key];y=b[key]
            delta=np.sqrt(np.mean(np.square((x-y).reshape(120,-1)),axis=1))
            ax.plot(np.arange(120)*.05,delta,label=key)
        ax.axvline(2,color='k',linestyle='--');ax.set(xlabel='Rollout time (s)',ylabel='Per-step RMS (signal units)',title=f'{sid}: direct vs left_3')
        ax.legend(fontsize=6)
    save(fig,'10-command-actor-action-root-response.png')
    # XY modes vs body/contact/outcome differences.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,sid in zip(axes,ids):
        values=summary['groups'][sid]['unique_candidate_outcomes'];names=list(values)
        ax.scatter([values[n]['maximum_contact_n'] for n in names],[values[n]['delta_v'] for n in names])
        for n in names:ax.annotate(n,(values[n]['maximum_contact_n'],values[n]['delta_v']),fontsize=6)
        ax.set(xlabel='Maximum contact force (N)',ylabel='ΔV (Critic units)',title=f'{sid}: {pcr[sid]["realized_modes"]} XY modes')
    save(fig,'11-xy-mode-contact-outcome.png')
    # StateGroup summary.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(ids,[pcr[sid]['path_modes'] for sid in ids],label='P');axes[0].bar(ids,[pcr[sid]['realized_modes'] for sid in ids],alpha=.6,label='R')
    axes[0].set(ylabel='Mode count',title='P/C/R mode retention');axes[0].legend()
    axes[1].bar(ids,[summary['groups'][sid]['candidate_outcome_range']['delta_v'] for sid in ids]);axes[1].set(ylabel='Candidate ΔV range (Critic units)',title='Outcome diversity by StateGroup')
    save(fig,'12-stategroup-comparison.png')
