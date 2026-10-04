"""Read-only frozen-input and external-router gates for expanded E1 continuation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

from eat_nav_e1_coordinate_addendum import audit_inputs as audit_old_e1
from eat_nav_expanded_feasibility_preflight import CONFIG,sha,write_once

STOPPED=Path('/tmp/eatnav-expanded-feasibility-20261004')
RESUME=Path('/tmp/eatnav-expanded-feasibility-resume-20261005')
REPO=Path(__file__).resolve().parents[1]
EXPECTED={'selected-states.json':'6b91c1c7e602db042228106f968f4d6ee4dd7f30583c4b6a78fd744bdc3065b9',
    'candidate-paths.jsonl.gz':'e04c146c4b2c46217915a50815245cb6c56fc10103d8ae79c1550ffe815c3514',
    'run-manifest.json':'acf3444cbb1eb3312b8af95ec575ef916afc932bdac6d8e3a420d0a533c473de',
    'config':'fd282cbb3105088c02e03fb314dc05ab1d35b472cda003cf3587d7e75196a03d'}
EXPECTED_STATES=[('boxes',36,20,645838458),('hf_pyramid_slope',54,65,2480648597),
    ('k_rail_field',6,15,2373776937),('k_rail_step_field',73,75,964068328),
    ('pyramid_stairs',47,55,3361987328)]
ROUTER_BINARY=Path('/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd')
ROUTER_CONFIG=Path('/opt/ros/jazzy/share/rmw_zenoh_cpp/config/DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5')
ROUTER_CLIENT=Path('/opt/ros/jazzy/share/rmw_zenoh_cpp/config/DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5')


def audit_frozen_inputs(root=STOPPED):
    root=Path(root)
    status=json.loads((root/'status.json').read_text())
    if status['stage']!='X17_STOPPED_ROUTER_UNAVAILABLE' or status['new_processes_started']!=0:
        raise ValueError('old stopped state differs')
    if (root/'runs').exists() and list((root/'runs').iterdir()):
        raise ValueError('unexpected prior attempt in stopped root')
    if sha(CONFIG)!=EXPECTED['config']:
        raise ValueError('candidate bank config SHA differs')
    for name,digest in EXPECTED.items():
        if name!='config' and sha(root/name)!=digest:
            raise ValueError('frozen '+name+' SHA differs')
    ledger=json.loads((root/'artifact-sha-ledger.json').read_text())
    if len(ledger['entries'])!=80 or any(sha(root/item['path'])!=item['sha256'] for item in ledger['entries']):
        raise ValueError('stopped artifact ledger differs')
    states=json.loads((root/'selected-states.json').read_text())['train']
    actual=[(s['terrain'],s['env_id'],s['step'],s['terrain_seed']) for s in states]
    if actual!=EXPECTED_STATES or any(s['episode_key'][2]!=0 or s['prefix_steps']>80 for s in states):
        raise ValueError('selected StateGroup identity differs')
    manifest=json.loads((root/'run-manifest.json').read_text())
    rows=manifest['rows']
    if len(rows)!=50 or [x['order'] for x in rows]!=list(range(50)):
        raise ValueError('frozen run schedule differs')
    for i,s in enumerate(states):
        group=rows[i*10:i*10+10]
        if [x['role'] for x in group]!=['anchor_only']+['nominal_duplicate']*3+['candidate_outcome']*6:
            raise ValueError('role order differs')
        if [x['path_id'] for x in group]!=['direct_reference']*4+[
            'left_1','left_2','left_3','right_1','right_2','right_3']:
            raise ValueError('candidate order differs')
        if any(x['state_group_id']!=s['state_id'] or x['seed']!=123 or x['episode_key']!=s['episode_key']
               for x in group):raise ValueError('state/seed/episode schedule differs')
    freeze=json.loads((root/'input-freeze.json').read_text())
    if (freeze['selected_states_sha256']!=sha(root/'selected-states.json') or
        freeze['candidate_bank_sha256']!=sha(root/'candidate-paths.jsonl.gz')):
        raise ValueError('state/bank input freeze differs')
    for i in range(5):
        folder=root/'inputs'/f'state-{i:03d}'
        for name,digest in freeze['per_state'][i].items():
            if sha(folder/name)!=digest:raise ValueError('per-state input SHA differs '+name)
    forecast=json.loads((root/'runtime-storage-forecast.json').read_text())
    if (forecast['status']!='pass' or forecast['p95_plus_20pct_sec']>14400 or
        forecast['storage_forecast_with_margin_bytes']>1_000_000_000):
        raise ValueError('frozen runtime/storage forecast differs')
    old=audit_old_e1()
    if len(old['runs'])!=18:raise ValueError('old E1 run count differs')
    addendum=Path('/tmp/eatnav-e1-coordinate-addendum-r2-20261004')
    if (sha(addendum/'original-vs-corrected-decision.json')!=
        json.loads((root/'preflight-freeze.json').read_text())['old_coordinate_addendum_sha256']):
        raise ValueError('coordinate addendum SHA differs')
    return {'stopped_stage':status['stage'],'old_attempt_count':0,'run_manifest_count':len(rows),
        'new_state_count':len(states),'new_candidate_count':5*7,
        'old_protocol_valid_runs':len(old['runs']),'selected_state_identity':actual,
        'frozen_sha256':{k:sha(root/k) for k in EXPECTED if k!='config'},
        'config_sha256':sha(CONFIG),'old_stop_status_sha256':sha(root/'status.json'),
        'old_stop_ledger_sha256':sha(root/'artifact-sha-ledger.json'),
        'forecast':forecast,'existing_candidate_attempts_counted':0}


def router_process_identity(pid,*,proc_root=Path('/proc')):
    proc=Path(proc_root)/str(pid)
    if not proc.is_dir():raise ValueError('external router process absent')
    executable=(proc/'exe').resolve()
    if executable!=ROUTER_BINARY:raise ValueError('external router binary differs')
    stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
    start_ticks=int(stat[19])
    command=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if not command or start_ticks<=0:raise ValueError('router process command/start tick missing')
    started_unix_sec=None
    try:
        boot=int(next(line.split()[1] for line in (Path(proc_root)/'stat').read_text().splitlines()
            if line.startswith('btime ')))
        started_unix_sec=boot+start_ticks/os.sysconf('SC_CLK_TCK')
    except (FileNotFoundError,StopIteration,ValueError):pass
    return {'pid':int(pid),'start_ticks':start_ticks,'start_unix_sec':started_unix_sec,
        'binary':str(executable),'command':command}


def require_router_log(path):
    if path is None or not Path(path).is_file():raise ValueError('router log evidence unavailable')
    content=Path(path).read_text(errors='replace')
    lowered=content.lower()
    if not content.strip():raise ValueError('router log evidence empty')
    if any(term in lowered for term in ('bind failed','address already in use','connection refused',
        'unable to connect','config mismatch','panic')):
        raise ValueError('router log warning present')
    return {'status':'pass','path':str(path),'sha256':sha(path),'bytes':len(content.encode())}


def current_router_pid():
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if (proc/'exe').resolve()==ROUTER_BINARY:found.append(int(proc.name))
        except OSError:continue
    if len(found)!=1:raise ValueError('expected exactly one external Zenoh router process')
    return found[0]


def reachable_host():
    with socket.create_connection(('127.0.0.1',7447),timeout=2):pass


def reachable_container():
    code='import socket; s=socket.create_connection(("127.0.0.1",7447),2); s.close()'
    subprocess.run(['docker','exec','isaac-lab-onix','/usr/bin/python3','-c',code],
        capture_output=True,text=True,check=True)


def check_router(*,log_path):
    before=router_process_identity(current_router_pid())
    reachable_host();reachable_container()
    log=require_router_log(log_path)
    after=router_process_identity(before['pid'])
    if before!=after:raise ValueError('router PID/start tick changed during readiness')
    actual={'router_binary':sha(ROUTER_BINARY),'router_config':sha(ROUTER_CONFIG),
        'host_client_config':sha(ROUTER_CLIENT)}
    expected={'router_binary':'0a7f481c11a139254178ac0b81effe475d358a0fa818e32a1a8fd33abcae098b',
        'router_config':'68d6b941f1cd19e48e830b6516d993cebf43cd4147cf5c800f81ee3ff0c94b76',
        'host_client_config':'33cece505c521dfaf4c882f406e76ec0f7b14efe65b948ac9cdd984c8e6f1483'}
    if actual!=expected:raise ValueError('router binary/config SHA differs')
    return {'status':'pass',**before,'endpoint':'127.0.0.1:7447','host_connect':True,
        'container_connect':True,'router_log':log,'source_sha256':actual,
        'externally_owned':True,'runner_start_stop_or_kill':False}


def prepare():
    audit=audit_frozen_inputs()
    if RESUME.exists():raise FileExistsError('new resume root already exists')
    RESUME.mkdir()
    frozen_names=('selected-states.json','selected-states.npz','candidate-paths.jsonl.gz',
        'run-manifest.json','runtime-storage-forecast.json','offline-e0-audit.json',
        'train-only-path-nearest-neighbor.json','input-freeze.json')
    for name in frozen_names:
        shutil.copy2(STOPPED/name,RESUME/name)
        if sha(STOPPED/name)!=sha(RESUME/name):raise ValueError('frozen copy SHA differs '+name)
    shutil.copytree(STOPPED/'inputs',RESUME/'inputs')
    for source in (STOPPED/'inputs').rglob('*'):
        if source.is_file() and sha(source)!=sha(RESUME/'inputs'/source.relative_to(STOPPED/'inputs')):
            raise ValueError('frozen input copy SHA differs')
    write_once(RESUME/'x0-frozen-input-audit.json',audit)
    write_once(RESUME/'old-stop-preservation.json',{'source_root':str(STOPPED),
        'old_stop_status_sha256':sha(STOPPED/'status.json'),
        'old_stop_ledger_sha256':sha(STOPPED/'artifact-sha-ledger.json'),
        'old_attempts_excluded':0,'resume_counts_as_candidate_retry':False})
    write_once(RESUME/'status.json',{'stage':'X0_FROZEN_REAUDITED',
        'new_anchor_processes':0,'new_candidate_processes':0,'router_owned_by_runner':False})
    print('X0_REAUDIT_PASS',RESUME,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','router'])
    p.add_argument('--router-log',type=Path);args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:
        result=check_router(log_path=args.router_log)
        write_once(RESUME/'external-router-provenance.json',result)
        print('X1_ROUTER_PASS',result['pid'],result['start_ticks'],flush=True)
