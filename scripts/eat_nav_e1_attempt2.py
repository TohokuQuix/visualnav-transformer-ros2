"""Static formal E1 attempt-2 audit and guarded continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import yaml

import eat_nav_e1_run_one as replay
import eat_nav_e1_schedule as schedule
from eat_nav_e1_process_ownership import ownership_spec
from eat_nav_e1_recovery_continue import configure_output_root, check_router, install_transport_guard

ROOT=Path('/tmp/eatnav-e1-attempt2-20261004')
PRIOR=Path('/tmp/eatnav-e1-zenoh-recovery-20261004')
LABEL='attempt-2'
HISTORICAL=[Path('/tmp/eatnav-e1-feasibility-20261003'),PRIOR,
            Path('/tmp/eatnav-e1-cleanup-fixed-20261004'),
            Path('/tmp/eatnav-e1-lifecycle-compatible-20261004')]
FILE_FLAGS={'--manifest','--prefix','--other-env-actions','--logged-anchor','--state',
            '--base-path','--fixed-map-path','--recorded-frames','--selected-dem',
            '--config','--reference-config','--rl-checkpoint','--goal-metadata'}

def sha(path):return replay.sha(path)
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)

def configure():
    configure_output_root(ROOT)
    replay.REMOTE_CODE=str(ROOT/'eat_nav_e1_sim.py')
    schedule.SESSION='eatnav-e1-'+LABEL.replace('-','')

def copy_inputs():
    if ROOT.exists():
        if not (ROOT/'attempt-2-evidence').is_dir():raise FileExistsError('formal output root already exists')
    else:ROOT.mkdir()
    for name in ('continuation-manifest.json','original-20-row-manifest.json',
                 'e1-gates.yaml','e1-safety-stops.yaml','map-path-identities.jsonl.gz',
                 'completed-pending-identity.json','input-freeze.json'):
        if not (ROOT/name).exists():shutil.copy2(PRIOR/name,ROOT/name)
        if sha(PRIOR/name)!=sha(ROOT/name):raise ValueError('input copy SHA mismatch '+name)
    if not (ROOT/'inputs').exists():shutil.copytree(PRIOR/'inputs',ROOT/'inputs')
    for p in (PRIOR/'inputs').rglob('*'):
        if p.is_file() and sha(p)!=sha(ROOT/p.relative_to(PRIOR)):
            raise ValueError('input copy SHA mismatch '+str(p))

def cli_file_paths(cmd):
    if cmd[:6]!=['docker','exec','-e',replay.ENV,'isaac-lab-onix','/isaac-sim/python.sh']:
        raise ValueError('unexpected Docker CLI prefix')
    found={'simulator':cmd[6]}
    for flag in FILE_FLAGS:
        if cmd.count(flag)!=1:raise ValueError('missing/duplicate CLI path flag '+flag)
        found[flag]=cmd[cmd.index(flag)+1]
    return found

def audit_identity(row,entry,cmd):
    state=read(ROOT/'inputs/selected-states.json')['train'][0]
    config=yaml.safe_load((ROOT/'inputs/state-000/config-120.yaml').read_text())
    goal=read(ROOT/'inputs/state-000'/f"{row['process_id']}-goal-metadata.json")
    frames=read(ROOT/'inputs/state-000/recorded-frames.json')
    if (row['order']!=1 or row['process_id']!='g0e-train-000-nominal_duplicate-00'
            or row['state_group_id']!='g0e-train-000' or row['role']!='nominal_duplicate'
            or row['path_id']!='direct_reference' or row['seed']!=123 or row['repeat_id']!=0):
        raise ValueError('formal first row identity differs')
    if (state['state_id']!=row['state_group_id'] or state['episode_key']!=[0,22,0]
            or state['env_id']!=22 or state['terrain']!='pyramid_stairs_inv'
            or state['step']!=30 or state['source_episode']!=0):
        raise ValueError('state/episode/environment/terrain identity differs')
    source=config['source_state']
    if (config['scope']!='e1_live_candidate' or config['control_steps']!=120
            or source['anchor_step']!=30 or source['env_index']!=22
            or source['episode_id']!=0 or source['state_index']!=0):
        raise ValueError('E1 config scope or source identity differs')
    if (entry['logical_t0_ns']!=2_300_000_000_000 or entry['seed']!=123
            or cmd[cmd.index('--logical-t0-ns')+1]!=str(entry['logical_t0_ns'])
            or cmd[cmd.index('--control-steps')+1]!='120'):
        raise ValueError('logical start, seed, or control length differs')
    with np.load(ROOT/'inputs/state-000/prefix-actions.npz') as data:
        if data['actions'].shape!=(31,100,6) or data['selected_env_commands'].shape!=(31,3):
            raise ValueError('prefix action shape differs')
        if not all(np.isfinite(data[k]).all() for k in data.files):raise ValueError('prefix nonfinite')
    with np.load(ROOT/'inputs/state-000/suffix-actions-120.npz') as data:
        if data['actions'].shape!=(120,100,6) or not np.isfinite(data['actions']).all():
            raise ValueError('suffix action shape/finite differs')
    with np.load(ROOT/'inputs/g0e-selected-states.npz') as data:
        if data['dem'].shape!=(2,4,2,20,20) or not np.isfinite(data['dem'][0]).all():
            raise ValueError('selected DEM shape/finite differs')
    expected=state['dem_history_identity']
    actual=[[[f['session_id'],f['env_id'],f['episode_id']],f['step']] for f in frames]
    if len(frames)!=4 or actual!=expected or any(f['terrain_type']!=state['terrain']
            or f['terrain_seed']!=state['terrain_seed'] for f in frames):
        raise ValueError('DEM 4 frame identity differs')
    if len({f['timestamp_ns'] for f in frames})!=4 or any(
            f['dem']['axis']!='row=+world_y,col=+world_x' or f['dem']['resolution_m']!=0.1
            for f in frames):raise ValueError('DEM timestamp/axis/resolution differs')
    if (goal['run_id']!=row['process_id'] or goal['state_id']!=row['state_group_id']
            or goal['candidate_id']!=row['path_id']
            or goal['robot_local_8point_sha256']!=row['path_sha256']
            or goal['map_path_8point_sha256']!=row['map_path_sha256']):
        raise ValueError('goal or Path identity differs')
    for flag,key in (('--base-path','path_sha256'),('--fixed-map-path','map_path_sha256')):
        path=Path(cmd[cmd.index(flag)+1]); points=np.asarray(read(path),dtype=np.float32)
        if points.shape!=(9,2) or hashlib.sha256(points[1:].tobytes()).hexdigest()!=row[key]:
            raise ValueError('Path coordinates SHA differs '+flag)
    if (cmd[cmd.index('--base-path')+1]==cmd[cmd.index('--fixed-map-path')+1]
            or cmd[cmd.index('--output')+1].startswith(str(ROOT/'inputs'))
            or not cmd[cmd.index('--output')+1].startswith(str(ROOT/'runs'))):
        raise ValueError('input/output or base/map Path root confused')
    return {'state_id':state['state_id'],'episode_key':state['episode_key'],
            'env_id':state['env_id'],'terrain':state['terrain'],
            'dem_frame_identity':actual,'scope':config['scope'],
            'prefix_actions':[31,100,6],'suffix_actions':[120,100,6],
            'candidate_id':goal['candidate_id'],'seed':row['seed']}

def audit_cli(cmd,freeze):
    files=cli_file_paths(cmd)
    source=ROOT/'inputs'
    provenance=read(schedule.G0E/'formal-anchor-provenance.json')['container_input_sha256']
    expected_by_flag={'simulator':freeze['source_binary_sha256']['simulator'],
        '--manifest':provenance['/tmp/eatnav-extended/output/manifest.json'],
        '--reference-config':freeze['reference_config_sha256'],
        '--rl-checkpoint':freeze['RL_checkpoint_sha256']}
    table=[]
    for flag,remote in files.items():
        if remote.startswith(str(source)):
            host=Path(remote)
            if not host.is_file() or not os.access(host,os.R_OK):raise FileNotFoundError(host)
            expected=sha(host)
        elif flag=='simulator':host=schedule.REPO/'scripts/eat_nav_e1_sim.py';expected=expected_by_flag[flag]
        else:host=None;expected=expected_by_flag[flag]
        if host is not None and sha(host)!=expected:raise ValueError('host SHA differs '+str(host))
        test=subprocess.run(['docker','exec',schedule.DOCKER,'test','-f',remote],capture_output=True)
        readable=subprocess.run(['docker','exec',schedule.DOCKER,'test','-r',remote],capture_output=True)
        if test.returncode or readable.returncode:raise FileNotFoundError('container file missing/unreadable '+remote)
        actual=subprocess.check_output(['docker','exec',schedule.DOCKER,'sha256sum',remote],text=True).split()[0]
        if actual!=expected:raise ValueError('container SHA differs '+remote)
        table.append({'argument':flag,'host_path':str(host) if host else None,
                      'container_path':remote,'sha256':actual,'regular_readable':True})
    if len(table)!=len(FILE_FLAGS)+1:raise ValueError('CLI file argument enumeration incomplete')
    return table

def static_preflight():
    configure()
    copy_inputs()
    freeze=schedule.freeze_sources()
    if freeze['dirty']:raise ValueError('formal worktree dirty')
    schedule.setup_container()
    rows=read(ROOT/'continuation-manifest.json')['pending_rows']
    entry=schedule.execution_entry(rows[0])
    cmd=replay.simulator_command(entry,str(ROOT/'runs'/entry['run_id']),timing_telemetry=True)
    table=audit_cli(cmd,freeze)
    identity=audit_identity(rows[0],entry,cmd)
    if (ROOT/'runs'/entry['run_id']).exists():raise FileExistsError('attempt-2 output already exists')
    record={'status':'pass','command':cmd,'file_table':table,'identity':identity,
            'process_registry_roles':ownership_spec(),
            'router_ownership':'external',
            'source_freeze_sha256':sha(ROOT/'source-freeze.json'),
            'historical_roots':[{'root':str(root),'source_freeze_sha256':sha(root/'source-freeze.json')
                if (root/'source-freeze.json').is_file() else None} for root in HISTORICAL]}
    atomic(ROOT/f'{LABEL}-static-preflight.json',record)
    return record

def continue_runs():
    configure()
    if read(ROOT/f'{LABEL}-static-preflight.json')['status']!='pass':raise ValueError('static preflight')
    if read(ROOT/'router-readiness.json')['gate']['status']!='pass':raise ValueError('router readiness')
    check_router(ROOT)
    marker=ROOT/f'{LABEL}-continuation-started.marker.json'
    if marker.exists():raise FileExistsError(LABEL+' already started')
    started=time.time_ns()
    atomic(marker,{'start_wall_ns':started,
        'pending_valid_processes':18,'historical_attempts_excluded':[0,1,2]
            if LABEL=='attempt-3' else [0,1]})
    atomic(ROOT/'continuation-started.marker.json',{'start_wall_ns':started,
        'pending_processes':18,LABEL.replace('-','_')+'_marker_sha256':sha(marker)})
    install_transport_guard(ROOT)
    try:
        schedule.run()
        atomic(ROOT/'final-decision.json',{'decision':('ownership_protocol_recovered_e1_complete'
            if LABEL=='attempt-3' else 'cleanup_protocol_recovered_e1_complete'),
            'counts':schedule.process_counts(ROOT),'learning_started':False})
    except BaseException as exc:
        counts=schedule.process_counts(ROOT)
        prefix='ownership_protocol_recovered_e1' if LABEL=='attempt-3' else 'cleanup_protocol_recovered_e1'
        decision=(LABEL.replace('-','_')+'_measurement_invalid_stop' if counts['protocol_valid_process_count']==0 else
                  prefix+'_duplicate_gate_fail' if 'duplicate gate' in str(exc) else
                  prefix+'_runtime_stop')
        atomic(ROOT/'final-decision.json',{'decision':decision,'reason':repr(exc),
            'traceback':traceback.format_exc(),'counts':counts,'learning_started':False})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','continue'])
    args=parser.parse_args()
    if args.stage=='preflight':print(json.dumps(static_preflight(),sort_keys=True),flush=True)
    else:continue_runs()
