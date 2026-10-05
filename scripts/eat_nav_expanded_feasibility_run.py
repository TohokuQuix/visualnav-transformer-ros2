"""Frozen, external-router expanded E1 schedule. One process per manifest row."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

import eat_nav_e1_run_one as replay
import eat_nav_e1_schedule as schedule
from eat_nav_e1_analyze import duplicate_gate, load_trace
from eat_nav_e1_artifact_recovery import seal_tree, verify_seal
from eat_nav_g0e_gpu_probe import Nvml, atomic as atomic_status
from eat_nav_g0e_gpu_run import container_identity


ROOT=Path('/tmp/eatnav-expanded-feasibility-config-restored-20261005')
REPO=Path(__file__).resolve().parents[1]
EXPECTED_SIM_SHA='0e8ea4639f0496ab45b4b833ab1bc0e8c6320d3419194e42ff2b8e84e0454e2e'
EXPECTED_MANIFEST_SHA='b65868f9b3b6b7433be946f8385bb574d15574e2962f3aecd0b85f73231f016a'
EXPECTED_INPUT_FREEZE_SHA='58f2ff2138647cd17cb2d2e7d390f27eb4e28413af28ba25f2a1b93ab192154d'
EXPECTED_CONFIG_SHA=(
    '1df33506ae7b11cee668b898cb908b5ec835c000037b94b749393825c47d51b0',
    '191e6642cdd2f85a8656c06e5d894163b4a99d2339562f72a5861b248eeefd2c',
    '251ded83b0b9f16dcd546bdeceb014ecb6c6133dee8a871af940100a537bd796',
    'f2d29e2cd8057357451d10d03d91f95d91388b14c43dc3144ea1180867fbeea0',
    '726f7c67daa974b93feb2a74a6bf7a9a937adbe0f1478cf1266877995cc3119b',
)
E1_ANCHOR_CONFIG=Path('/tmp/eatnav-e1-artifact-recovered-continuation-20261004/inputs/state-000/config-120.yaml')
E1_ANCHOR_CONFIG_SHA='d89e7853cc72ce0e0090623f6a4e300637cd6f35efa2cd7058ddf3d1720be605'
WALL_CAP_SEC=14400
STORAGE_CAP_BYTES=1_000_000_000


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def disk_bytes(root=ROOT):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def router_identity(root=ROOT,*,proc_root=Path('/proc')):
    expected=read(Path(root)/'external-router-provenance.json')['router_identity']
    pid=int(expected['pid'])
    proc=Path(proc_root)/str(pid)
    stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
    result={'pid':pid,'start_ticks':int(stat[19]),
        'binary':str((proc/'exe').resolve())}
    if result!=expected or result['binary']!='/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd':
        raise RuntimeError('external router PID/start tick/binary changed')
    return result


def frozen_rows(root=ROOT):
    if sha(root/'run-manifest.json')!=EXPECTED_MANIFEST_SHA:
        raise ValueError('50-process manifest SHA differs')
    if sha(root/'input-freeze.json')!=EXPECTED_INPUT_FREEZE_SHA:
        raise ValueError('restored input freeze SHA differs')
    if sha(root/'selected-states.json')!='6b91c1c7e602db042228106f968f4d6ee4dd7f30583c4b6a78fd744bdc3065b9':
        raise ValueError('selected-state SHA differs')
    if sha(root/'candidate-paths.jsonl.gz')!='e04c146c4b2c46217915a50815245cb6c56fc10103d8ae79c1550ffe815c3514':
        raise ValueError('candidate bank SHA differs')
    rows=read(root/'run-manifest.json')['rows']
    states=read(root/'selected-states.json')['train']
    if len(rows)!=50 or len(states)!=5 or [x['order'] for x in rows]!=list(range(50)):
        raise ValueError('frozen 50-row order differs')
    for idx,row in enumerate(rows):
        if row['state_index']!=idx//10 or row['state_group_id']!=states[idx//10]['state_id']:
            raise ValueError('state index/order differs')
        if row['seed']!=123 or row['episode_key']!=states[idx//10]['episode_key']:
            raise ValueError('state seed/episode differs')
        config=root/'inputs'/f'state-{idx//10:03d}'/'config-120.yaml'
        if sha(config)!=EXPECTED_CONFIG_SHA[idx//10] or row['input_sha256']['config-120.yaml']!=sha(config):
            raise ValueError('restored config SHA differs')
        path=root/'inputs'/f"state-{row['state_index']:03d}"/f"{row['path_id']}-base-path.json"
        if sha(path)!=row['base_path_file_sha256']:
            raise ValueError('frozen base Path SHA differs '+row['process_id'])
    return rows,states


def validate_anchor_limit_contract(root,state_count,reference_limits):
    """Reject missing or changed E1 numeric limits before launching an anchor."""
    for index in range(state_count):
        path=Path(root)/'inputs'/f'state-{index:03d}'/'config-120.yaml'
        current=yaml.safe_load(path.read_text()).get('anchor_limits',{})
        missing=sorted(set(reference_limits)-set(current))
        different=sorted(key for key in reference_limits if key in current
                         and current[key]!=reference_limits[key])
        extra=sorted(set(current)-set(reference_limits))
        if missing or different or extra:
            raise ValueError(f'anchor limits state-{index:03d}: missing={missing}; '
                             f'different={different}; extra={extra}')


def original_e1_anchor_limits():
    if sha(E1_ANCHOR_CONFIG)!=E1_ANCHOR_CONFIG_SHA:
        raise ValueError('original E1 anchor config SHA differs')
    return yaml.safe_load(E1_ANCHOR_CONFIG.read_text())['anchor_limits']


def execution_entry(row,state,*,map_digest):
    if row['state_group_id']!=state['state_id'] or row['episode_key']!=state['episode_key']:
        raise ValueError('entry/state identity differs')
    index=row['state_index']
    return {'order':row['order'],'run_id':row['process_id'],
        'kind':'anchor' if row['role']=='anchor_only' else
            'duplicate' if row['role']=='nominal_duplicate' else 'candidate',
        'state_id':row['state_group_id'],'state_index':index,
        'candidate_id':row['process_id'],'path_id':row['path_id'],
        'seed':row['seed'],'anchor_step':state['step'],
        'episode_key':row['episode_key'],
        'logical_t0_ns':2_300_000_000_000+index*100_000_000_000,
        'local_path_sha256':row['robot_local_waypoint_sha256'],
        'map_path_sha256':map_digest,
        'base_path_sha256':row['base_path_file_sha256']}


def _yaw_from_wxyz(quaternion):
    w,x,y,z=map(float,quaternion)
    if not np.isfinite([w,x,y,z]).all():raise ValueError('anchor quaternion nonfinite')
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def _map_path(base,xy,yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    x,y=map(float,xy)
    return [[float(x+c*px-s*py),float(y+s*px+c*py)] for px,py in base]


def freeze_map_paths(root,state_index,rows,anchor_map,anchor,*,logged_goal,amplitudes):
    """Create all seven fixed map Paths only after an accepted anchor."""
    root=Path(root);folder=root/'inputs'/f'state-{state_index:03d}'
    q=np.asarray(anchor['root_state'][3:7],dtype=float)
    anchor_map=np.asarray(anchor_map,dtype=float)
    if anchor_map.shape!=(9,2) or not np.isfinite(anchor_map).all():
        raise ValueError('anchor map Path shape/finite')
    yaw=_yaw_from_wxyz(q);xy=anchor_map[0]
    paths={};metadata={};identity=[]
    for row in rows:
        pid=row['process_id'];path_id=row['path_id']
        base_file=folder/f'{path_id}-base-path.json'
        if sha(base_file)!=row['base_path_file_sha256']:
            raise ValueError('base Path file SHA differs')
        base=np.asarray(read(base_file),dtype=np.float64)
        if base.shape!=(9,2) or not np.isfinite(base).all():
            raise ValueError('base Path shape/finite')
        local_sha=hashlib.sha256(np.asarray(base[1:],dtype=np.float32).tobytes()).hexdigest()
        if local_sha!=row['robot_local_waypoint_sha256']:
            raise ValueError('robot-local waypoint SHA differs')
        mapped=np.asarray(_map_path(base,xy,yaw))
        if path_id=='direct_reference':
            if float(np.max(np.abs(mapped-anchor_map)))>1e-6:
                raise ValueError('anchor direct Path/map transform differs')
            mapped=anchor_map
        back=(mapped-xy)@np.array([[math.cos(yaw),-math.sin(yaw)],
                                    [math.sin(yaw),math.cos(yaw)]])
        inverse_error=float(np.max(np.abs(back-base)))
        if inverse_error>1e-6:raise ValueError('base↔map inverse transform error')
        values=mapped.tolist()
        digest=hashlib.sha256(np.asarray(mapped[1:],dtype=np.float32).tobytes()).hexdigest()
        paths[pid]=values
        metadata[pid]={'state_id':row['state_group_id'],'run_id':pid,
            'candidate_id':path_id,'amplitude_m':amplitudes[path_id],
            'base_path':base.tolist(),'base_path_sha256':sha(base_file),
            'goal_xy':base[-1].tolist(),'mission_logged_goal':list(logged_goal),
            'robot_local_8point_sha256':local_sha,'map_path_8point_sha256':digest}
        identity.append({'process_id':pid,'path_id':path_id,'base_file_sha256':sha(base_file),
            'map_waypoint_sha256':digest,'endpoint_map_xy':values[-1],
            'inverse_max_abs_m':inverse_error})
    if any((folder/f'{pid}-map-path.json').exists() or
           (folder/f'{pid}-goal-metadata.json').exists() for pid in paths):
        raise FileExistsError('map Path/goal metadata already frozen')
    for pid,values in paths.items():
        once(folder/f'{pid}-map-path.json',values)
        once(folder/f'{pid}-goal-metadata.json',metadata[pid])
    result={'status':'pass','state_index':state_index,'anchor_xy_map':xy.tolist(),
        'anchor_yaw_rad':yaw,'path_count':len(paths),'paths':identity,
        'file_sha256':{pid:{'map_path':sha(folder/f'{pid}-map-path.json'),
            'goal_metadata':sha(folder/f'{pid}-goal-metadata.json')} for pid in paths}}
    once(root/f'map-path-freeze-state-{state_index:03d}.json',result)
    return result


def candidate_amplitudes(root=ROOT):
    result={}
    with gzip.open(root/'candidate-paths.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            result.setdefault(row['state_id'],{})[row['candidate_id']]=row['amplitude_m']
    return result


def docker(*args):
    return subprocess.run(['docker',*map(str,args)],capture_output=True,text=True,check=True)


def configure_replay(root=ROOT):
    root=Path(root)
    replay.ROOT=Path(root)
    replay.INPUT=Path(root)/'inputs'
    replay.REPO=REPO
    replay.SIM=root/'eat_nav_expanded_sim_v2.py'
    replay.REMOTE_CODE=str(Path(root)/'eat_nav_expanded_sim_v2.py')
    replay.REMOTE_ROOT=str(Path(root)/'runs')
    replay.REMOTE_INPUT_ROOT=Path(root)
    replay.ENV=('PYTHONPATH='+str(root)+':/tmp/eatnav-candidate-coverage/evaluator-readiness-code/src:'
        '/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:'
        '/tmp/eatnav-extended/isaac-source')
    schedule.ROOT=Path(root)

    def frozen(entry):
        rows,states=frozen_rows(root)
        row=rows[entry['order']]
        if row['process_id']!=entry['run_id'] or states[row['state_index']]['state_id']!=entry['state_id']:
            raise ValueError('frozen execution identity differs')
        if sha(root/'eat_nav_expanded_sim_v2.py')!=EXPECTED_SIM_SHA:
            raise ValueError('adapted simulator SHA differs')
        folder=Path(root)/'inputs'/f"state-{row['state_index']:03d}"
        expected={'prefix-actions.npz':row['input_sha256']['prefix-actions.npz'],
            'suffix-actions-120.npz':row['input_sha256']['suffix-actions-120.npz'],
            'logged-anchor.json':row['input_sha256']['logged-anchor.json'],
            'config-120.yaml':row['input_sha256']['config-120.yaml'],
            f"{row['process_id']}-base-path.json":row['base_path_file_sha256']}
        if entry['kind']!='anchor':
            expected[f"{row['process_id']}-map-path.json"]=entry['map_path_file_sha256']
            expected[f"{row['process_id']}-goal-metadata.json"]=entry['goal_metadata_sha256']
        for name,digest in expected.items():
            if sha(folder/name)!=digest:raise ValueError('run input SHA differs '+name)
        if sha(replay.PROD)!='616097e44d199b8d33f6534aedd299f2d20127ed8e100c18777f9ca7a2da389d':
            raise ValueError('follower binary differs')
        if sha(replay.PARAM)!='ddedff40d13e20416a43309669ad19e7f3a06e2df8cc8c9d3248369e14d40c90':
            raise ValueError('follower config differs')
        router_identity(root)
        return folder
    replay.frozen=frozen


def setup_container(root=ROOT):
    root=Path(root)
    sim=root/'eat_nav_expanded_sim_v2.py'
    if sha(sim)!=EXPECTED_SIM_SHA:raise ValueError('adapted simulator SHA differs')
    remote=str(root)
    docker('exec','isaac-lab-onix','mkdir','-p',remote)
    source_files={'eat_nav_expanded_sim_v2.py':sim,
        'eat_nav_e1_anchor_gate.py':REPO/'scripts/eat_nav_e1_anchor_gate.py',
        'eat_nav_g0e_formal_gate.py':REPO/'scripts/eat_nav_g0e_formal_gate.py',
        'eat_nav_goal_subgoal_dem_gate.py':REPO/'scripts/eat_nav_goal_subgoal_dem_gate.py'}
    for name,path in source_files.items():
        docker('cp',str(path),f'isaac-lab-onix:{remote}/{name}')
        if docker('exec','isaac-lab-onix','sha256sum',f'{remote}/{name}').stdout.split()[0]!=sha(path):
            raise ValueError('container source SHA differs '+name)


def copy_inputs_to_container(root=ROOT):
    root=Path(root);remote=str(root)
    docker('cp',str(root/'inputs'),f'isaac-lab-onix:{remote}/inputs')
    for local in (root/'inputs').rglob('*'):
        if local.is_file():
            target=remote+'/'+str(local.relative_to(root))
            if docker('exec','isaac-lab-onix','sha256sum',target).stdout.split()[0]!=sha(local):
                raise ValueError('container input SHA differs '+str(local))


def copy_state_paths_to_container(root,state_index,rows):
    folder=Path(root)/'inputs'/f'state-{state_index:03d}'
    for row in rows:
        for suffix in ('map-path.json','goal-metadata.json'):
            path=folder/f"{row['process_id']}-{suffix}"
            docker('cp',str(path),f'isaac-lab-onix:{path}')
            if docker('exec','isaac-lab-onix','sha256sum',str(path)).stdout.split()[0]!=sha(path):
                raise ValueError('container generated Path/goal SHA differs '+str(path))


def _map_digest(root,row):
    if row['role']=='anchor_only':return 'pending_anchor_only'
    path=Path(root)/'inputs'/f"state-{row['state_index']:03d}"/f"{row['process_id']}-map-path.json"
    values=np.asarray(read(path),dtype=np.float32)
    return hashlib.sha256(values[1:].tobytes()).hexdigest()


def _entry(root,row,state):
    result=execution_entry(row,state,map_digest=_map_digest(root,row))
    if row['role']!='anchor_only':
        folder=Path(root)/'inputs'/f"state-{row['state_index']:03d}"
        result['map_path_file_sha256']=sha(folder/f"{row['process_id']}-map-path.json")
        result['goal_metadata_sha256']=sha(folder/f"{row['process_id']}-goal-metadata.json")
    return result


def _router_check(root=ROOT):
    current=router_identity(root)
    provenance=read(Path(root)/'external-router-provenance.json')
    for name,source in provenance['router_log_sources'].items():
        content=Path(source).read_text(errors='replace').lower()
        if any(term in content for term in ('warning','warn','error','refused','bind failed','mismatch','panic')):
            raise RuntimeError('external router '+name+' log warning')
    return current


def _completed_prefix(root,rows):
    completed=[]
    for row in rows:
        marker=Path(root)/'run-stage-markers'/f"{row['process_id']}.json"
        if marker.exists():
            run=Path(root)/'runs'/row['process_id']
            if read(marker)['status']!='complete' or not (run/'run-complete.json').is_file():
                raise ValueError('incomplete saved marker')
            if row['role']!='anchor_only' and read(run/'protocol-valid.json')['status']!='valid':
                raise ValueError('saved candidate protocol invalid')
            verify_seal(run)
            completed.append(row['process_id'])
        elif (Path(root)/'runs'/row['process_id']).exists():
            raise RuntimeError('partial run exists; automatic retry forbidden')
        elif any((Path(root)/'run-stage-markers'/f"{x['process_id']}.json").exists()
                 for x in rows[row['order']+1:]):
            raise ValueError('completed marker gap breaks frozen order')
    return completed


def _remaining_forecast(root,rows,completed,start):
    elapsed=time.time()-start
    if elapsed>=WALL_CAP_SEC:raise TimeoutError('four-hour pilot cap')
    if disk_bytes(root)>=STORAGE_CAP_BYTES:raise RuntimeError('one-GB storage cap')
    durations=[read(Path(root)/'runs'/name/'run-complete.json')['elapsed_sec'] for name in completed]
    if durations:
        per_run=max(read(Path(root)/'runtime-storage-forecast.json')['measured_process_p95_sec'],
                    float(np.mean(durations)))
        projected=elapsed+(len(rows)-len(completed))*per_run*1.2
    else:
        forecast=read(Path(root)/'runtime-storage-forecast.json')
        projected=elapsed+forecast['p95_plus_20pct_sec']
    if projected>WALL_CAP_SEC:
        once(Path(root)/'forecast-stop.json',{'elapsed_sec':elapsed,'completed':len(completed),
            'projected_total_sec':projected,'reason':'4h_cap_projection'})
        raise TimeoutError('pilot projected over four hours')
    return {'elapsed_sec':elapsed,'projected_total_sec':projected,
        'storage_bytes':disk_bytes(root),'completed':len(completed)}


def _anchor_complete(root,row,entry,group,amplitudes,logged_goal):
    root=Path(root);folder=root/'runs'/row['process_id']
    remote=f'{root}/runs/{row["process_id"]}'
    for name in ('anchor-gate.json','timing-progress.json','timing-events.jsonl'):
        if not (folder/name).exists():docker('cp',f'isaac-lab-onix:{remote}/{name}',str(folder/name))
    gate=read(folder/'anchor-gate.json')
    summary=read(folder/'anchor-0.json')
    if gate['status']!='pass' or summary['anchor_gate']['status']!='pass':
        raise ValueError('numeric/DEM anchor gate failed')
    if summary['state']['state_id']!=row['state_group_id']:
        raise ValueError('anchor state identity differs')
    remote_map=root/'inputs'/f"state-{row['state_index']:03d}"/f"{row['process_id']}-map-path.json"
    temp=folder/'anchor-direct-map-path.json'
    docker('cp',f'isaac-lab-onix:{remote_map}',str(temp))
    if read(temp)!=summary['map_path'] or sha(temp)!=summary['fixed_map_path_sha256']:
        raise ValueError('anchor generated map Path SHA differs')
    result=freeze_map_paths(root,row['state_index'],group,summary['map_path'],
        summary['anchor'],logged_goal=logged_goal,amplitudes=amplitudes)
    anchor_local=root/'inputs'/f"state-{row['state_index']:03d}"/f"{row['process_id']}-map-path.json"
    if sha(anchor_local)!=sha(temp):raise ValueError('frozen anchor map Path file SHA differs')
    copy_state_paths_to_container(root,row['state_index'],group)
    once(root/f"anchor-gate-{row['state_group_id']}.json",{
        'status':'pass','run_id':row['process_id'],'anchor_result_sha256':sha(folder/'anchor-0.json'),
        'anchor_gate_sha256':sha(folder/'anchor-gate.json'),
        'map_path_freeze_sha256':sha(root/f'map-path-freeze-state-{row["state_index"]:03d}.json'),
        'numeric_errors':summary['anchor_error_vs_logged'],
        'dem_history_gate':gate['dem_history_gate'],
        'map_paths':result['path_count']})


def _noise_floor(root,rows):
    traces=[load_trace(Path(root)/'runs'/row['process_id']/'live-0.trace.json.gz') for row in rows]
    keys=('follower_command','command','pre_noise','post_noise','actor_input','actor_latent',
        'actor_action','root_state','joint_position','joint_velocity','raw_joint_effort',
        'raw_joint_effort_grouped','contact_force','critic_value','reward')
    result={}
    for key in keys:
        values=[]
        for a,b in ((0,1),(0,2),(1,2)):
            try:
                left=np.asarray([r[key] for r in traces[a]],dtype=float)
                right=np.asarray([r[key] for r in traces[b]],dtype=float)
                delta=left-right
                if not np.isfinite(delta).all():raise ValueError('nonfinite '+key)
                values.append(float(np.sqrt(np.mean(delta*delta))))
            except (KeyError,TypeError):continue
        result[key]={'pairwise_rms':values,'maximum_rms':max(values) if values else None}
    return {'status':'frozen_before_lateral','state_group_id':rows[0]['state_group_id'],
        'trace_sha256':[sha(Path(root)/'runs'/r['process_id']/'live-0.trace.json.gz') for r in rows],
        'signals':result}


def run(root=ROOT):
    root=Path(root);rows,states=frozen_rows(root)
    validate_anchor_limit_contract(root,len(states),original_e1_anchor_limits())
    if sha(root/'eat_nav_expanded_sim_v2.py')!=EXPECTED_SIM_SHA:
        raise ValueError('simulator adapter SHA differs')
    if read(root/'x2-readiness-gate.json')['status']!='pass':raise ValueError('X2 pending')
    _router_check(root)
    configure_replay(root)
    if not (root/'router-launch-provenance.json').exists():
        identity=router_identity(root)
        once(root/'router-launch-provenance.json',{'pid':identity['pid'],
            'start_ticks':identity['start_ticks'],'ownership':'external',
            'external_router_provenance_sha256':sha(root/'external-router-provenance.json')})
    completed=_completed_prefix(root,rows)
    if not completed and (root/'pilot-started.marker.json').exists():
        raise RuntimeError('pilot clock exists but no completed run; automatic retry forbidden')
    if not (root/'container-setup.marker.json').exists():
        setup_container(root)
        copy_inputs_to_container(root)
        once(root/'container-setup.marker.json',{'status':'pass',
            'simulator_sha256':sha(root/'eat_nav_expanded_sim_v2.py'),
            'input_file_count':len(list((root/'inputs').rglob('*'))),
            'router_pid':router_identity(root)['pid']})
    else:setup_container(root)
    container=container_identity();nvml=Nvml()
    gates=yaml.safe_load((root/'e1-gates.yaml').read_text())
    amps=candidate_amplitudes(root)
    for row in rows:
        pid=row['process_id'];idx=row['state_index'];sid=row['state_group_id']
        if pid in completed:continue
        _router_check(root)
        if (root/'pilot-started.marker.json').exists():
            start=read(root/'pilot-started.marker.json')['start_wall_ns']/1e9
            forecast=_remaining_forecast(root,rows,completed,start)
        else:forecast={'elapsed_sec':0,'projected_total_sec':read(root/'runtime-storage-forecast.json')['p95_plus_20pct_sec']}
        group=rows[idx*10:(idx+1)*10]
        if row['role']=='candidate_outcome' and read(root/f'duplicate-gate-{sid}.json')['status']!='pass':
            raise RuntimeError('lateral candidate before duplicate pass')
        if row['role']!='anchor_only' and not (root/f'map-path-freeze-state-{idx:03d}.json').is_file():
            raise RuntimeError('map Path not frozen after anchor')
        atomic_status(root/'status.json',{'stage':'PRELAUNCH_BASELINE','run_id':pid,
            'completed_new_processes':len(completed),'forecast':forecast})
        baseline=schedule.prelaunch_baseline(row,container,nvml)
        if baseline['status']!='pass':raise RuntimeError('prelaunch baseline '+repr(baseline))
        if row['role']=='anchor_only' and not (root/'pilot-started.marker.json').exists():
            identity=router_identity(root)
            once(root/'pilot-started.marker.json',{'start_wall_ns':time.time_ns(),
                'wall_cap_sec':WALL_CAP_SEC,'storage_cap_bytes':STORAGE_CAP_BYTES,
                'frozen_manifest_sha256':sha(root/'run-manifest.json'),
                'router_pid':identity['pid'],'router_start_ticks':identity['start_ticks']})
        _router_check(root)
        entry=_entry(root,row,states[idx])
        atomic_status(root/'status.json',{'stage':'RUNNING','run_id':pid,
            'completed_new_processes':len(completed),'elapsed_sec':time.time()-
                read(root/'pilot-started.marker.json')['start_wall_ns']/1e9})
        if row['role']=='anchor_only':
            safety=yaml.safe_load((root/'e1-safety-stops.yaml').read_text())
            anon_shmem_cap=safety['infrastructure_stop']['cgroup_anon_plus_shmem_over_bytes']
            replay.run_one(entry,pid,timeout_sec=600,timing_telemetry=True,
                timing_monitor_script=REPO/'scripts/eat_nav_goal_subgoal_timing_monitor.py',
                timing_monitor_extra_args=('--anon-shmem-cap-bytes',str(anon_shmem_cap)))
            _anchor_complete(root,row,entry,group,amps[sid],states[idx]['logged_goal'])
        else:
            replay.run_one(entry,pid,timeout_sec=600,timing_telemetry=True,
                timing_monitor_script=REPO/'scripts/eat_nav_e1_monitor.py',
                timing_monitor_config=root/'e1-safety-stops.yaml',
                timing_monitor_extra_args=('--sim-source',replay.REMOTE_CODE),
                monitor_cleanup_protocol=True)
            if read(root/'runs'/pid/'protocol-valid.json')['status']!='valid':
                raise RuntimeError('candidate protocol invalid')
        runfolder=root/'runs'/pid
        seal_tree(runfolder);verify_seal(runfolder)
        once(root/'run-stage-markers'/f'{pid}.json',{'status':'complete',
            'run_complete_sha256':sha(runfolder/'run-complete.json'),
            'completed_new_processes':len(completed)+1})
        completed.append(pid)
        if row==group[3]:
            result=duplicate_gate(group[1:4],root,gates)
            once(root/f'duplicate-gate-{sid}.json',result)
            if result['status']!='pass':raise RuntimeError('duplicate gate failed '+repr(result['reason']))
            once(root/f'body-noise-floor-{sid}.json',_noise_floor(root,group[1:4]))
        if row==group[-1]:
            once(root/f'group-complete-{sid}.json',{'status':'complete',
                'run_marker_sha256':[sha(root/'runs'/r['process_id']/'run-complete.json') for r in group]})
        print('EXPANDED_RUN_COMPLETE',len(completed),'/50',pid,flush=True)
    atomic_status(root/'status.json',{'stage':'PILOT_50_COMPLETE',
        'completed_new_processes':50,'elapsed_sec':time.time()-
            read(root/'pilot-started.marker.json')['start_wall_ns']/1e9,
        'analysis_pending':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    try:run(args.root)
    except BaseException as exc:
        root=args.root
        atomic_status(root/'status.json',{'stage':'PILOT_STOPPED','reason':repr(exc),
            'completed_new_processes':len(list((root/'run-stage-markers').glob('*.json')))
                if (root/'run-stage-markers').exists() else 0,
            'automatic_retry':False})
        print('EXPANDED_PILOT_STOP',repr(exc),flush=True)
        raise
