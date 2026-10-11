"""One-shot limited P1 pilot with owned production follower bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from visualnav_transformer.eat_nav.p1_minimal_prepath import resource_violation
from visualnav_transformer.eat_nav.p1_statistical_freeze import sha
from eat_nav_ros_subprocess_env import build_ros_host_env, env_audit


CONTAINER='isaac-lab-onix'
BASE_PYTHONPATH=':'.join((
    '/tmp/eatnav-train/src',
    '/tmp/eatnav-extended/isaac-source',
    '/workspace/isaaclab/source/isaaclab',
    '/workspace/isaaclab/source/isaaclab_rl',
    '/workspace/isaaclab/source/isaaclab_tasks',
    '/workspace/isaaclab_onix/source/onix_isaac_lab',
))
FOLLOWER=Path('/home/kojima/eatnav-backups/2026-10-08-pre-shutdown/eatnav-candidate-coverage/exact-stamp-long-horizon/exact-follower-build/simple_trajectory_follower_node')
PARAMS=Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml')
ROUTER=Path('/home/kojima/eatnav-router-logs/eatnav-p1-paired-noise-20261010-v7')


def process_tick(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:
        return None


def router_audit():
    launcher_pid=int((ROUTER/'router-launch.pid').read_text().strip())
    pid=int((ROUTER/'router.pid').read_text().strip())
    def identity(target):
        proc=Path(f'/proc/{target}')
        if not proc.exists():
            raise ValueError(f'external router process absent: {target}')
        parts=(proc/'stat').read_text().rsplit(')',1)[1].split()
        state,ppid,tick=parts[0],int(parts[1]),int(parts[19])
        command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
        if state in ('Z','X') or not command:
            raise ValueError(f'external router process not alive: {target}')
        return {'pid':target,'ppid':ppid,'start_tick':tick,'state':state,'cmdline':command}
    launcher=identity(launcher_pid)
    router=identity(pid)
    if 'ros2 run rmw_zenoh_cpp rmw_zenohd' not in launcher['cmdline']:
        raise ValueError('external router launcher cmdline mismatch')
    if '/rmw_zenohd' not in router['cmdline'] or router['ppid']!=launcher_pid:
        raise ValueError('external router child cmdline/parent mismatch')
    stdout=ROUTER/'router.stdout.log';stderr=ROUTER/'router.stderr.log'
    data=stdout.read_text()+'\n'+stderr.read_text()
    zid=re.search(r'Started Zenoh router with id ([0-9a-f]+)',data)
    endpoints=re.findall(r'Zenoh can be reached at: tcp/([^\s]+):7447',data)
    if zid is None or not endpoints:
        raise ValueError('router log lacks ZID/7447 endpoint')
    reachable=[]
    for endpoint in endpoints:
        try:
            with socket.create_connection((endpoint,7447),timeout=2):
                reachable.append(endpoint)
        except OSError:
            pass
    if not reachable:
        raise ValueError('external router 7447 unreachable')
    return {'pid':pid,'start_tick':router['start_tick'],'launcher_pid':launcher_pid,
            'launcher_start_tick':launcher['start_tick'],'router_process':router,
            'launcher_process':launcher,'zid':zid.group(1),
            'endpoints':endpoints,'reachable_endpoints':reachable,
            'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr)}


def owned_group_members(pgid):
    members=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            parts=(entry/'stat').read_text().rsplit(')',1)[1].split()
            if int(parts[2])==pgid and parts[0]!='Z':
                members.append(int(entry.name))
        except (FileNotFoundError,ProcessLookupError,ValueError):
            pass
    return sorted(members)


def stop_owned_bridge(process,tick):
    if process is None:return []
    if process_tick(process.pid)!=tick:
        raise ValueError('bridge PID/start tick changed before cleanup')
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if process_tick(process.pid)!=tick:
            raise ValueError('bridge PID/start tick changed during cleanup')
        os.killpg(process.pid,signal.SIGKILL)
        process.wait(timeout=10)
    remaining=owned_group_members(process.pid)
    if remaining:
        raise ValueError('owned bridge/follower residual: '+repr(remaining))
    return remaining


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


def run(root,anchor_id,batch_index,*,branches=5,params=PARAMS):
    root=Path(root).resolve()
    if not (root/'input/freeze.json').is_file() or not (root/'source.py').is_file():
        raise ValueError('frozen source or input missing')
    if (root/'output').exists() or (root/'host-seal.json').exists():
        raise FileExistsError('one-shot output already exists')
    freeze=json.loads((root/'input/freeze.json').read_text())
    if sha(root/'source.py')!=freeze['pilot_simulator_source_sha256']:
        raise ValueError('simulator source SHA changed')
    if sha(root/'input/assignment.json')!=freeze['assignment_sha256']:
        raise ValueError('assignment SHA changed')
    cfg=json.loads((root/'input/statistical-config.json').read_text())
    if cfg['primary_control_steps']!=40 or cfg['diagnostic_control_steps']!=120:
        raise ValueError('P1 primary/diagnostic window contract')
    cap=cfg['resource_cap']
    router_before=router_audit()
    router_freeze_path=root.parent/'router-identity-freeze.json'
    if not router_freeze_path.is_file():
        raise ValueError('router identity freeze missing')
    router_freeze=json.loads(router_freeze_path.read_text())
    if router_freeze.get('design_freeze_sha256')!=sha(root.parent/'design-freeze.json'):
        raise ValueError('router/design freeze mismatch')
    for key in ('pid','start_tick','launcher_pid','launcher_start_tick','zid','endpoints'):
        if router_before[key]!=router_freeze['router_identity'][key]:
            raise ValueError('external router identity differs from frozen manifest: '+key)
    if (sha(FOLLOWER)!=cfg['follower_sha256'] or sha(params)!=cfg['follower_params_sha256']
            or sha(Path(__file__).parent/'eat_nav_p1_paired_bridge.py')!=cfg['bridge_sha256']):
        raise ValueError('follower/bridge source SHA mismatch')
    atomic(root/'router-before.json',router_before)
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
    if remote.returncode or remote.stdout.split()[0]!=freeze['pilot_simulator_source_sha256']:
        raise ValueError('host/container source SHA mismatch')
    remote_checkpoint=docker('sha256sum',cfg['rl_checkpoint'])
    if remote_checkpoint.returncode or remote_checkpoint.stdout.split()[0]!=cfg['rl_checkpoint_sha256']:
        raise ValueError('container RL checkpoint SHA mismatch')
    started=time.monotonic()
    cmd=['docker','exec','-e',f'PYTHONPATH={root}/src:{BASE_PYTHONPATH}',CONTAINER,
         'timeout','600s','/isaac-sim/python.sh',f'{container_root}/source.py','--headless',
         '--input',f'{container_root}/input','--output',f'{container_root}/output',
         '--anchor-id',anchor_id,'--batch-index',str(batch_index)]
    samples=[];heartbeat_state=None;failure=None
    atomic(root/'host-status.json',{'stage':'running','Path_publications':'unknown_until_sim_marker'})
    bridge=None;bridge_tick=None;rc=None
    try:
        ros_env=build_ros_host_env(repo=Path(__file__).resolve().parents[1])
        ros_env['RMW_IMPLEMENTATION']='rmw_zenoh_cpp'
        atomic(root/'bridge-environment.json',env_audit(ros_env,'/usr/bin/python3'))
        bridge_cmd=['/usr/bin/python3',str(Path(__file__).parent/'eat_nav_p1_paired_bridge.py'),
                    '--branches',str(branches),'--follower',str(FOLLOWER),'--params',str(params),
                    '--diagnostic-dir',str(root/'follower-logs')]
        with (root/'bridge.log').open('x') as bridge_log:
            bridge=subprocess.Popen(bridge_cmd,env=ros_env,stdout=bridge_log,
                                    stderr=subprocess.STDOUT,start_new_session=True)
            bridge_tick=process_tick(bridge.pid)
            atomic(root/'bridge-owner.json',{'pid':bridge.pid,'start_tick':bridge_tick,
                   'command':bridge_cmd})
            deadline=time.monotonic()+20
            while True:
                if bridge.poll() is not None:
                    raise RuntimeError('production follower bridge exited before listening')
                try:
                    with socket.create_connection(('127.0.0.1',47819),timeout=.2):break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('bridge listening timeout')
                    time.sleep(.1)
            with (root/'runner.log').open('x') as logfile:
                process=subprocess.Popen(cmd,stdout=logfile,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if bridge.poll() is not None:
                        failure='bridge_exited'
                    measured,heartbeat_state=sample(root,started,heartbeat_state)
                    samples.append(measured)
                    failure=failure or resource_violation(measured,limits)
                    if failure:
                        owner=owned_identity(container_root)
                        if owner:docker('kill','-TERM',str(owner['pid']))
                        break
                    time.sleep(2)
                rc=process.wait(timeout=30)
    finally:
        residual=stop_owned_bridge(bridge,bridge_tick) if bridge is not None else []
        atomic(root/'bridge-cleanup.json',{'owned_residual':residual,
               'router_pid_unmodified':router_before['pid']})
        router_after=router_audit()
        if any(router_after[key]!=router_before[key] for key in
               ('pid','start_tick','launcher_pid','launcher_start_tick','zid','endpoints')):
            raise ValueError('external router identity changed during pilot')
        atomic(root/'router-after.json',router_after)
    atomic(root/'resource-series.json',samples)
    atomic(root/'host-status.json',{'stage':'exited','native_exit':rc,
           'resource_failure':failure,
           'elapsed_s':time.monotonic()-started,
           'peak_gpu_mib':max((r['gpu_mib'] or 0 for r in samples),default=None)})
    copied=subprocess.run(['docker','cp',f'{CONTAINER}:{container_root}/output',str(root)],
                          capture_output=True,text=True,timeout=120)
    if copied.returncode:
        raise RuntimeError('output Docker copy failed: '+copied.stderr)
    if rc or failure:
        raise RuntimeError(f'pilot process failed: native={rc}, resource={failure}')
    complete=root/'output/complete.json'
    if not complete.is_file():
        raise ValueError('simulator completion marker missing')
    marker=json.loads(complete.read_text())
    if marker['Path_publications']!=1 or marker['settling_control_steps']!=20 or marker['control_steps']!=cfg['diagnostic_control_steps']:
        raise ValueError('pilot completion contract')
    for key,name in (('initial_state_sha256','initial-state.json'),
                     ('terrain_sha256','terrain.json'),('prepath_gate_sha256','prepath-gate.json'),
                     ('path_publication_sha256','path-publication.json'),
                     ('pilot_summary_sha256','pilot-summary.json'),
                     ('contact_body_names_sha256','contact-body-names.json')):
        if sha(root/'output'/name)!=marker[key]:
            raise ValueError('host output SHA mismatch: '+name)
        remote=docker('sha256sum',f'{container_root}/output/{name}')
        if remote.returncode or remote.stdout.split()[0]!=marker[key]:
            raise ValueError('container output SHA mismatch: '+name)
    if owned_identity(container_root) is not None:
        raise ValueError('owned process residual')
    for index,digest in enumerate(marker['trace_sha256']):
        path=root/'output'/f'trace-slot-{index}.jsonl'
        if sha(path)!=digest:
            raise ValueError('host trace SHA mismatch')
        remote=docker('sha256sum',f'{container_root}/output/{path.name}')
        if remote.returncode or remote.stdout.split()[0]!=digest:
            raise ValueError('host/container trace SHA mismatch')
    atomic(root/'host-seal.json',{'schema':'p1_statistical_pilot_seal_v1',
           'freeze_sha256':sha(root/'input/freeze.json'),
           'complete_sha256':sha(complete),
           'host_status_sha256':sha(root/'host-status.json'),
           'Path_publications':1,'router_before_sha256':sha(root/'router-before.json'),
           'router_after_sha256':sha(root/'router-after.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--anchor-id',required=True)
    p.add_argument('--batch-index',type=int,choices=(0,1,2),required=True)
    a=p.parse_args()
    run(a.root,a.anchor_id,a.batch_index)
