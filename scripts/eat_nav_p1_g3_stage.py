"""Freeze and execute the one authorized P1 long-horizon G3 four-batch pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from eat_nav_p1_g1_lifecycle import verify_seal

BASE=Path('/home/kojima/eatnav-experiments/p1-long-horizon-v1')
G2=BASE/'g2-attempt-0'
G3=BASE/'g3-attempt-0'
REMOTE_BASE='/workspace/isaaclab_onix/.eatnav-p1-long-horizon-v1/g3-attempt-0'
GROUPS=('first-differential','second-graded')
SLOTS=('direct_duplicate_0','direct_duplicate_1','minus_20_deg','plus_20_deg')
BEARINGS=(0,0,-20,20)
G1_SEAL='c5712eb6bc816f49222f01218bb593f116a38a00f8ce07eb79bd5b637e420609'
G2_SEAL='73ec7b6cac44dfdb717a8f8a8dd2a4cce4f19f9dc8d7b565df6055a0f251f1b0'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def prior_seals():
    if digest(BASE/'g1-attempt-2/host-seal.json')!=G1_SEAL or \
            digest(G2/'host-seal.json')!=G2_SEAL:
        raise ValueError('G1/G2 sealed evidence changed')
    verify_seal(BASE/'g1-attempt-2');verify_seal(G2)


def prepare(repo: Path):
    prior_seals()
    if G3.exists():raise FileExistsError('G3 root already exists; no retry or overwrite')
    source=repo/'scripts/eat_nav_p1_g3_pilot_sim.py'
    protocol=repo/'src/visualnav_transformer/eat_nav/p1_longhorizon_protocol.py'
    host=repo/'scripts/eat_nav_p1_g3_pilot_host.py'
    if not all(p.is_file() for p in (source,protocol,host)):
        raise FileNotFoundError('G3 frozen source missing')
    old=json.loads((G2/'input/assignment.json').read_text())
    geometry=json.loads((G2/'input/source-contract.json').read_text())
    path_by_bearing={x['bearing_deg']:x for x in geometry['paths']}
    if set(BEARINGS)-set(path_by_bearing):raise ValueError('frozen geometry missing')
    trials=[];schedule=[]
    for group in GROUPS:
        for block in (0,1):
            batch_seed=old['batch_seeds'][group][block]
            schedule.append({'batch_number':len(schedule),'anchor_id':group,
                             'batch_index':block,'batch_seed':batch_seed})
            for slot,(name,bearing) in enumerate(zip(SLOTS,BEARINGS)):
                trials.append({'trial_id':f'{group}-block{block}-{name}',
                    'anchor_id':group,'batch_index':block,'batch_seed':batch_seed,
                    'slot':slot,'candidate_slot':name,'condition':'A',
                    'bearing_deg':bearing,'path_sha256':path_by_bearing[bearing]['path_sha256']})
    G3.mkdir(mode=0o775)
    write_json(G3/'g3-assignment.json',{'schema':'eatnav_p1_longhorizon_g3_assignment_v1',
                                       'trials':trials,'schedule':schedule})
    contract={'schema':'eatnav_p1_longhorizon_g3_pilot_v1',
        'g3_simulator_source_sha256':digest(source),
        'assignment_sha256':digest(G3/'g3-assignment.json'),
        'episode_length_s':40.0,'control_dt_s':0.05,
        'unreached_control_step_cap':600,'post_arrival_hold_steps':40,
        'max_control_steps':640,'one_Path_publication_per_trial':True,
        'follower_command':'unmodified_A_passthrough',
        'direct_duplicate_gates':{'arrival_time_spread_s':0.10,
            'tracking_mean_spread_m':0.02,'common_goal_progress_2s_spread_m_strict':0.025,
            'dem_height_rms_m':1e-5,'dem_height_max_abs_m':1e-4,
            'dem_mask_iou_min':1.0,'actor_max_abs':1e-4},
        'resource_limits_from':'p1-long-horizon-runtime-freeze-20261011.json',
        'stage_wall_cap_s':3000,'batch_wall_cap_s':600,
        'source_reference_g2_seal_sha256':G2_SEAL,'source_reference_g1_seal_sha256':G1_SEAL}
    write_json(G3/'g3-contract.json',contract)
    shutil.copy2(source,G3/'source.py')
    shutil.copy2(host,G3/'host-runner.py')
    for number,_ in enumerate(schedule):
        batch=G3/f'batch-{number}'
        batch.mkdir()
        shutil.copy2(source,batch/'source.py')
        shutil.copytree(G2/'src',batch/'src',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(protocol,batch/'src/visualnav_transformer/eat_nav/p1_longhorizon_protocol.py')
        inputs=batch/'input';inputs.mkdir()
        for name in ('assignment.json','freeze.json','statistical-config.json',
                     'source-contract.json','anchor-fixtures.json','eat_nav.yaml'):
            shutil.copy2(G2/'input'/name,inputs/name)
        shutil.copy2(G3/'g3-assignment.json',inputs/'g3-assignment.json')
        shutil.copy2(G3/'g3-contract.json',inputs/'g3-contract.json')
        write_json(batch/'input-sha-ledger.json',{'source.py':digest(batch/'source.py'),
            **{f'input/{p.name}':digest(p) for p in sorted(inputs.iterdir())}})
    write_json(G3/'stage-status.json',{'stage':'prepared','completed_batches':0,
               'schedule':schedule,'assignment_sha256':contract['assignment_sha256'],
               'source_sha256':contract['g3_simulator_source_sha256']})
    return G3


def run(repo: Path):
    prior_seals()
    stage=json.loads((G3/'stage-status.json').read_text())
    if stage['stage']!='prepared' or stage['completed_batches']!=0:
        raise ValueError('G3 stage is not fresh; no retry or resume')
    if digest(G3/'host-runner.py')!=digest(repo/'scripts/eat_nav_p1_g3_pilot_host.py'):
        raise ValueError('G3 host code drift')
    start=time.monotonic();results=[]
    for number,item in enumerate(stage['schedule']):
        elapsed=time.monotonic()-start
        if elapsed+600+120+120>3000:
            write_json(G3/'stage-status.json',{'stage':'stopped_stage_wall_budget',
                'completed_batches':number,'elapsed_s':elapsed,'results':results})
            return 1
        batch=G3/f'batch-{number}'
        if (batch/'output').exists() or (batch/'host-seal.json').exists():
            raise ValueError('G3 batch has prior execution evidence; no retry')
        cmd=[sys.executable,str(repo/'scripts/eat_nav_p1_g3_pilot_host.py'),'--root',str(batch),
            '--remote',f'{REMOTE_BASE}/batch-{number}',
            '--anchor-id',item['anchor_id'],'--batch-index',str(item['batch_index'])]
        write_json(G3/'stage-status.json',{'stage':'running','active_batch':number,
            'completed_batches':number,'elapsed_s':elapsed,'results':results})
        with (G3/f'batch-{number}-host.log').open('x') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,
                                  env={**os.environ,'PYTHONPATH':str(repo/'src')+':'+str(repo/'scripts')})
            log.flush();os.fsync(log.fileno())
        status=json.loads((batch/'g3-host-status.json').read_text()) if \
            (batch/'g3-host-status.json').exists() else {'failure':'host_status_missing'}
        results.append({'batch':number,'exit':result.returncode,'host_status':status,
            'host_seal_sha256':digest(batch/'host-seal.json') if (batch/'host-seal.json').exists() else None})
        failure=None
        try:verify_seal(batch)
        except BaseException as exc:failure=type(exc).__name__+':'+str(exc)
        if result.returncode or status.get('failure') or failure or \
                status.get('lifecycle_phase')!='COMPLETE':
            write_json(G3/'stage-status.json',{'stage':'stopped_on_batch_gate',
                'active_batch':number,'completed_batches':number,
                'elapsed_s':time.monotonic()-start,'results':results,'seal_error':failure})
            return 1
        write_json(G3/'stage-status.json',{'stage':'batch_pass',
            'completed_batches':number+1,'elapsed_s':time.monotonic()-start,'results':results})
    prior_seals()
    write_json(G3/'stage-status.json',{'stage':'four_batches_protocol_valid',
        'completed_batches':4,'elapsed_s':time.monotonic()-start,'results':results})
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    if args.action=='prepare':print(prepare(args.repo))
    else:sys.exit(run(args.repo))
