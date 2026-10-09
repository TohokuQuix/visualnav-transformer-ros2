"""One-shot owned candidate-free P1 statistical preflight launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from visualnav_transformer.eat_nav.p1_minimal_prepath import resource_violation
from visualnav_transformer.eat_nav.p1_statistical_freeze import sha


CONTAINER='isaac-lab-onix'
BASE_PYTHONPATH=':'.join((
    '/tmp/eatnav-train/src',
    '/tmp/eatnav-extended/isaac-source',
    '/workspace/isaaclab/source/isaaclab',
    '/workspace/isaaclab/source/isaaclab_rl',
    '/workspace/isaaclab/source/isaaclab_tasks',
    '/workspace/isaaclab_onix/source/onix_isaac_lab',
))


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def docker(*args,timeout=30):
    return subprocess.run(['docker','exec',CONTAINER,*args],capture_output=True,text=True,timeout=timeout)


def prepare_container_parent(root: Path, docker_fn=docker):
    """Docker cp cannot create missing intermediate destination directories."""
    result=docker_fn('mkdir','-p',str(Path(root).parent))
    if result.returncode:
        raise RuntimeError('owned Docker destination parent creation failed: '+result.stderr)


def owned_identity(root):
    r=docker('cat',f'{root}/output/owned-process.json')
    if r.returncode:
        return None
    owner=json.loads(r.stdout)
    stat=docker('cat',f'/proc/{owner["pid"]}/stat')
    if stat.returncode:
        return None
    tick=int(stat.stdout.rsplit(')',1)[1].split()[19])
    if tick!=owner['start_tick']:
        raise ValueError('owned PID/start tick changed')
    return owner


def sample(root,started,heartbeat_state):
    gpu=docker('nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits')
    gpu_mib=int(gpu.stdout.splitlines()[0]) if gpu.returncode==0 else None
    mem_kib=next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                 if line.startswith('MemAvailable:'))
    host_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    remote=docker('du','-sb',f'{root}/output')
    remote_bytes=int(remote.stdout.split()[0]) if remote.returncode==0 else 0
    heartbeat=docker('cat',f'{root}/output/heartbeat.json')
    if heartbeat.returncode==0:
        step=json.loads(heartbeat.stdout)['completed_control_steps']
        if heartbeat_state is None or heartbeat_state[0]!=step:
            heartbeat_state=(step,time.monotonic())
    age=0 if heartbeat_state is None else time.monotonic()-heartbeat_state[1]
    return {'gpu_mib':gpu_mib,'host_available_gib':mem_kib/1024**2,
            'active_storage_gib':(host_bytes+remote_bytes)/1024**3,
            'heartbeat_age_s':age,'elapsed_s':time.monotonic()-started},heartbeat_state


def run(root,anchor_id,batch_index):
    root=Path(root).resolve()
    if not (root/'input/freeze.json').is_file() or not (root/'source.py').is_file():
        raise ValueError('frozen source or input missing')
    if (root/'output').exists() or (root/'host-seal.json').exists():
        raise FileExistsError('one-shot output already exists')
    freeze=json.loads((root/'input/freeze.json').read_text())
    if sha(root/'source.py')!=freeze['simulator_source_sha256']:
        raise ValueError('simulator source SHA changed')
    if sha(root/'input/assignment.json')!=freeze['assignment_sha256']:
        raise ValueError('assignment SHA changed')
    cfg=json.loads((root/'input/statistical-config.json').read_text())
    if cfg['primary_control_steps']!=40 or cfg['diagnostic_control_steps']!=120:
        raise ValueError('P1 primary/diagnostic window contract')
    cap=cfg['resource_cap']
    limits={'gpu_device_total_mib':cap['gpu_device_total_mib'],
            'host_available_min_gib':cap['host_available_min_gib'],
            'active_storage_gib':cap['active_storage_max_gib'],
            'heartbeat_stale_s':cap['heartbeat_stale_s'],
            'process_timeout_s':cap['process_timeout_s']}
    baseline_gpu=docker('nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits')
    if baseline_gpu.returncode:
        raise ValueError('prelaunch GPU query failed')
    baseline_mib=int(baseline_gpu.stdout.splitlines()[0])
    available_kib=next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:'))
    baseline={'gpu_mib':baseline_mib,'host_available_gib':available_kib/1024**2,
              'active_storage_gib':sum(p.stat().st_size for p in root.rglob('*') if p.is_file())/1024**3,
              'heartbeat_age_s':0,'elapsed_s':0}
    atomic(root/'resource-baseline.json',baseline)
    if violation:=resource_violation(baseline,limits):
        raise ValueError('prelaunch resource cap: '+violation)
    container_root=str(root)
    exists=docker('test','-e',container_root)
    if exists.returncode==0:
        raise FileExistsError('container destination already exists')
    prepare_container_parent(root)
    copied=subprocess.run(['docker','cp',str(root),f'{CONTAINER}:{container_root}'],capture_output=True,text=True,timeout=120)
    if copied.returncode:
        raise RuntimeError('input Docker copy failed: '+copied.stderr)
    remote=docker('sha256sum',f'{container_root}/source.py')
    if remote.returncode or remote.stdout.split()[0]!=freeze['simulator_source_sha256']:
        raise ValueError('host/container source SHA mismatch')
    started=time.monotonic()
    cmd=['docker','exec','-e',f'PYTHONPATH={root}/src:{BASE_PYTHONPATH}',CONTAINER,
         'timeout','600s','/isaac-sim/python.sh',f'{container_root}/source.py','--headless',
         '--input',f'{container_root}/input','--output',f'{container_root}/output',
         '--anchor-id',anchor_id,'--batch-index',str(batch_index)]
    samples=[];heartbeat_state=None;failure=None
    atomic(root/'host-status.json',{'stage':'running','Path_publications':0})
    with (root/'runner.log').open('w') as logfile:
        process=subprocess.Popen(cmd,stdout=logfile,stderr=subprocess.STDOUT)
        while process.poll() is None:
            measured,heartbeat_state=sample(root,started,heartbeat_state)
            samples.append(measured)
            failure=resource_violation(measured,limits)
            if failure:
                owner=owned_identity(container_root)
                if owner: docker('kill','-TERM',str(owner['pid']))
                break
            time.sleep(2)
        rc=process.wait(timeout=30)
    atomic(root/'resource-series.json',samples)
    atomic(root/'host-status.json',{'stage':'exited','native_exit':rc,
           'resource_failure':failure,'Path_publications':0,
           'elapsed_s':time.monotonic()-started,
           'peak_gpu_mib':max((r['gpu_mib'] or 0 for r in samples),default=None)})
    copied=subprocess.run(['docker','cp',f'{CONTAINER}:{container_root}/output',str(root)],
                          capture_output=True,text=True,timeout=120)
    if copied.returncode:
        raise RuntimeError('output Docker copy failed: '+copied.stderr)
    if rc or failure:
        raise RuntimeError(f'candidate-free process failed: native={rc}, resource={failure}')
    complete=root/'output/complete.json'
    if not complete.is_file():
        raise ValueError('simulator completion marker missing')
    marker=json.loads(complete.read_text())
    if marker['Path_publications']!=0 or marker['settling_control_steps']!=20 or marker['control_steps']!=120:
        raise ValueError('candidate-free completion contract')
    for key,name in (('initial_state_sha256','initial-state.json'),
                     ('terrain_sha256','terrain.json'),('prepath_gate_sha256','prepath-gate.json')):
        if sha(root/'output'/name)!=marker[key]:
            raise ValueError('host output SHA mismatch: '+name)
        remote=docker('sha256sum',f'{container_root}/output/{name}')
        if remote.returncode or remote.stdout.split()[0]!=marker[key]:
            raise ValueError('container output SHA mismatch: '+name)
    if owned_identity(container_root) is not None:
        raise ValueError('owned process residual')
    if len(marker.get('trace_sha256',[]))!=7:
        raise ValueError('candidate-free trace coverage')
    for index,digest in enumerate(marker['trace_sha256']):
        path=root/'output'/f'trace-slot-{index}.jsonl'
        if sha(path)!=digest:
            raise ValueError('candidate-free host trace SHA mismatch')
        remote=docker('sha256sum',f'{container_root}/output/{path.name}')
        if remote.returncode or remote.stdout.split()[0]!=digest:
            raise ValueError('candidate-free host/container trace SHA mismatch')
    atomic(root/'host-seal.json',{'schema':'p1_statistical_candidate_free_seal_v1',
           'freeze_sha256':sha(root/'input/freeze.json'),
           'complete_sha256':sha(complete),
           'host_status_sha256':sha(root/'host-status.json'),
           'Path_publications':0})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--anchor-id',required=True)
    p.add_argument('--batch-index',type=int,choices=(0,1),required=True)
    a=p.parse_args()
    run(a.root,a.anchor_id,a.batch_index)
