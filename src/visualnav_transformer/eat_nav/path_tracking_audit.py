"""Frozen 112-run identity and alignment checks; no simulator or model imports."""
from __future__ import annotations
import numpy as np

LIVE_KINDS=frozenset(('nominal','nomad','bank'))


def identity_key(entry):
    return (entry['run_id'],entry['state_id'],entry['kind'],entry['candidate_id'],
            entry.get('repeat'),entry.get('seed'))


def select_live_entries(schedule):
    live=[];excluded=[]
    for entry in schedule:
        kind=entry['kind']
        if kind in LIVE_KINDS:live.append(entry)
        elif kind=='anchor':excluded.append(entry)
        else:raise ValueError('unexpected live kind '+str(kind))
    if len(live)!=112 or len(excluded)!=7 or len({identity_key(e) for e in live})!=112:
        raise ValueError('frozen 112 live and seven anchor identities required')
    return live,excluded


def validate_trace_alignment(trace,logical_t0_ns):
    if len(trace)!=120:raise ValueError('complete 120-step trace required')
    max_command=0.;max_root=0.;stamp_mismatch=0
    for step,row in enumerate(trace):
        stamp=logical_t0_ns+step*50_000_000
        if any(row[name]!=stamp for name in ('logical_sim_time_ns','clock_stamp_ns','tf_stamp_ns','follower_command_stamp_ns')) or row['path_stamp_ns']!=logical_t0_ns:
            stamp_mismatch+=1
            raise ValueError(f'stamp mismatch step {step}')
        if row['terminal'] is not None:raise ValueError(f'early terminal step {step}')
        command=np.asarray(row['command'],dtype=float);follower=np.asarray(row['follower_command'],dtype=float)
        if command.shape!=(3,) or follower.shape!=(3,) or not np.isfinite(command).all() or not np.isfinite(follower).all():
            raise ValueError(f'nonfinite or invalid command step {step}')
        max_command=max(max_command,float(np.max(np.abs(command-follower))))
        events=row['bridge_command_events']
        if len(events)!=8 or {event['branch'] for event in events}!=set(range(8)) or any(
                not event['accepted'] or event['stamp_ns']!=stamp or event['tf_stamp_ns']!=stamp
                for event in events):
            raise ValueError(f'bridge command event mismatch step {step}')
        if any(row['bridge_rejected_count']):raise ValueError(f'rejected bridge command step {step}')
        if step<119:
            a=np.asarray(row['post_root_state'],dtype=float)
            b=np.asarray(trace[step+1]['root_state'],dtype=float)
            if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f'root shape/nonfinite step {step}')
            max_root=max(max_root,float(np.max(np.abs(a-b))))
            if max_root>1e-6:raise ValueError(f'root step alignment mismatch step {step}')
    if max_command>1e-6:raise ValueError('recorded command differs from follower')
    return {'step_count':120,'control_period_sec':.05,'stamp_mismatch_count':stamp_mismatch,
            'root_step_max_abs':max_root,'command_max_abs':max_command,
            'command_relation':'trace[k].command and Actor use pre-step root[k]; trace[k].post_root_state is root[k+1]'}


def _sha(path):
    import hashlib
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def _yaw_from_root(root):
    import math
    w,x,y,z=np.asarray(root,dtype=float)[3:7]
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def audit_run(entry,folder,state_dir,ledger_row):
    import gzip,json,pathlib
    from .path_tracking_geometry import map_from_base,base_from_map,to_increments,to_waypoints
    folder=pathlib.Path(folder);state_dir=pathlib.Path(state_dir)
    if entry['kind'] not in LIVE_KINDS:raise ValueError('non-live run in audit')
    marker_path=folder/'run-complete.json';marker=json.loads(marker_path.read_text())
    if marker['identity']!=entry or ledger_row['run_id']!=entry['run_id'] or _sha(marker_path)!=ledger_row['marker_sha256']:
        raise ValueError('run identity or complete marker SHA mismatch')
    for name,digest in marker['output_sha256'].items():
        if _sha(folder/name)!=digest:raise ValueError('output SHA mismatch '+name)
    trace_path=folder/'live-0.trace.json.gz'
    if _sha(trace_path)!=ledger_row['trace_sha256']:raise ValueError('trace SHA mismatch')
    summary=json.loads((folder/'live-0.json').read_text())
    with gzip.open(trace_path,'rt') as stream:trace=json.load(stream)
    alignment=validate_trace_alignment(trace,entry['logical_t0_ns'])
    if summary['state']['state_id']!=entry['state_id'] or summary['state']['episode_key']!=entry['episode_key']:
        raise ValueError('summary state or episode mismatch')
    base_file=state_dir/f"{entry['candidate_id']}-base-path.json"
    map_file=state_dir/f"{entry['candidate_id']}-map-path.json"
    if _sha(base_file)!=entry['base_path_sha256'] or _sha(base_file)!=summary['source_sha256']['base_path']:
        raise ValueError('base Path SHA mismatch')
    if _sha(map_file)!=summary['fixed_map_path_sha256'] or _sha(map_file)!=summary['source_sha256']['fixed_map_path']:
        raise ValueError('map Path SHA mismatch')
    base=np.asarray(json.loads(base_file.read_text()),dtype=float)
    mapped=np.asarray(json.loads(map_file.read_text()),dtype=float)
    if base.shape!=mapped.shape!=(9,2) or not np.isfinite(base).all() or not np.isfinite(mapped).all():
        raise ValueError('Path point count or nonfinite Path')
    if not np.allclose(base[0],0.,atol=1e-10) or not np.array_equal(mapped,np.asarray(summary['map_path'])):
        raise ValueError('Path origin or summary map Path mismatch')
    anchor=np.asarray(summary['post_follower_anchor']['root_state'],dtype=float)
    first=np.asarray(trace[0]['root_state'],dtype=float)
    if anchor.shape!=(13,) or not np.isfinite(anchor).all() or not np.isfinite(first).all():
        raise ValueError('anchor root nonfinite or shape')
    origin=anchor[:2]-mapped[0]
    local_anchor=anchor[:2]-origin
    yaw=_yaw_from_root(anchor)
    roundtrip=float(np.max(np.abs(map_from_base(base,local_anchor,yaw)-mapped)))
    back=float(np.max(np.abs(base_from_map(mapped,local_anchor,yaw)-base)))
    increments=to_increments(base[1:])
    increment_error=float(np.max(np.abs(to_waypoints(increments)-base[1:])))
    root_error=float(np.max(np.abs(anchor-first)))
    if max(roundtrip,back,increment_error,root_error)>1e-6:
        raise ValueError('coordinate reconstruction mismatch')
    for step,row in enumerate(trace):
        for name in ('root_state','post_root_state','actor_input','actor_latent','actor_action',
                     'critic_observation','command','follower_command','contact_force'):
            if not np.isfinite(np.asarray(row[name],dtype=float)).all():
                raise ValueError(f'nonfinite {name} step {step}')
        if not np.isfinite([row['critic_value'],row['reward']]).all():
            raise ValueError(f'nonfinite value or reward step {step}')
    identity={'run_id':entry['run_id'],'state_id':entry['state_id'],'terrain':entry['terrain_type'],
        'episode_id':entry['episode_key'],'generator':entry['kind'],
        'candidate_id':entry['candidate_id'],'diffusion_seed':entry['seed'],
        'base_path_sha256':_sha(base_file),'map_path_sha256':_sha(map_file),
        'trace_sha256':_sha(trace_path),'complete_marker_sha256':_sha(marker_path),
        'anchor_pose_world':anchor[:7].tolist(),'path_point_count':len(base),
        'future_point_count':len(base)-1,'path_sampling_interval_sec':.25,
        'path_horizon_sec':2.,'run_complete':True,'included':True,
        'inclusion_reason':'complete frozen live rollout'}
    coordinate={'run_id':entry['run_id'],'env_origin_world_xy':origin.tolist(),
        'anchor_map_local_xy':local_anchor.tolist(),'anchor_yaw_rad':yaw,
        'anchor_root_first_step_max_abs':root_error,
        'map_base_roundtrip_max_abs_m':max(roundtrip,back),
        'increment_roundtrip_max_abs_m':increment_error,
        'path_start_local_xy':mapped[0].tolist(),'path_endpoint_local_xy':mapped[-1].tolist(),
        'path_start_world_xy':(mapped[0]+origin).tolist(),
        'path_endpoint_world_xy':(mapped[-1]+origin).tolist(),
        'path_length_m':float(np.linalg.norm(np.diff(mapped,axis=0),axis=1).sum()),
        **alignment}
    return {'identity':identity,'coordinate':coordinate,'trace':trace,'summary':summary,
            'base_path':base,'map_path':mapped,'env_origin':origin}


def validate_source_hashes(hashes):
    """Require global scientific inputs to match; allow frozen per-state/Path inputs."""
    invariant=('rl_checkpoint','reference_config','manifest','bridge_client_source',
               'capture_source','clock_protocol_source','state')
    for name in invariant:
        if len(hashes.get(name,()))!=1:raise ValueError(name+' source SHA drift')
    if len(hashes.get('runner_source',())) not in (1,2):
        raise ValueError('unexpected runner source protocol count')
    return True


def recovery_from_log_lines(lines,logical_t0_ns,command_count):
    import json
    events={}
    for line in lines:
        if not line.startswith('TF_RACE_JSON '):continue
        event=json.loads(line[len('TF_RACE_JSON '):])
        if event.get('phase')!='command_pre_publish':continue
        stamp=event.get('command_stamp_ns')
        if stamp in events:raise ValueError('duplicate follower command stamp in recovery log')
        events[stamp]=event
    expected={logical_t0_ns+i*50_000_000 for i in range(command_count)}
    if set(events)!=expected:raise ValueError('recovery log stamp mismatch')
    return {'linear_recovery_fired':any(x.get('linear_recovery',False) for x in events.values()),
            'angular_recovery_fired':any(x.get('angular_recovery',False) for x in events.values()),
            'command_events':len(events)}
