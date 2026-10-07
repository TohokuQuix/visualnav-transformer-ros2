"""One-shot monitored Docker lifecycle for the candidate-free second-seed scan."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path('/tmp/eatnav-r3-second-seed-20261008')
WORKTREE=Path('/tmp/eatnav-vnt-r3-second-seed')
CONTAINER='isaac-lab-onix'
SCAN=ROOT/'scan-v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)


def docker(*args,check=True):
    return subprocess.run(['docker',*map(str,args)],capture_output=True,text=True,check=check)


def remote_sha(path):
    p=docker('exec',CONTAINER,'sha256sum',path,check=False)
    return p.stdout.split()[0] if p.returncode==0 else None


def remote_json(path):
    p=docker('exec',CONTAINER,'cat',path,check=False)
    return json.loads(p.stdout) if p.returncode==0 else None


def owned_alive(identity):
    if not identity:return False
    p=docker('exec',CONTAINER,'cat',f"/proc/{identity['pid']}/stat",check=False)
    if p.returncode!=0:return False
    tick=int(p.stdout.split(') ',1)[1].split()[19])
    if tick!=identity['proc_start_ticks']:
        raise RuntimeError('owned PID/start tick mismatch')
    return True


def stop_owned(client):
    identity=remote_json(SCAN/'owned-process.json')
    if owned_alive(identity):
        docker('exec',CONTAINER,'kill','-TERM',identity['pid'],check=False)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline and owned_alive(identity):time.sleep(.25)
        if owned_alive(identity):docker('exec',CONTAINER,'kill','-KILL',identity['pid'],check=False)
        if owned_alive(identity):raise RuntimeError('owned process residual after targeted cleanup')
    if client.poll() is None:
        client.terminate()
        try:client.wait(timeout=20)
        except subprocess.TimeoutExpired:client.kill();client.wait(timeout=5)


def copy_inputs(inputs):
    files=[(ROOT/'seed-freeze.json',ROOT/'seed-freeze.json'),
           (ROOT/'robustness-freeze.json',ROOT/'robustness-freeze.json'),
           (ROOT/'inputs/scan-inputs.json',ROOT/'inputs/scan-inputs.json'),
           (WORKTREE/'scripts/eat_nav_r3_second_scan_sim.py',ROOT/'inputs/eat_nav_r3_second_scan_sim.py')]
    for name in ('config','dem','radial_inventory','radial_interaction',
                 'anchor_placement_geometry','r3_robustness'):
        local=WORKTREE/'src/visualnav_transformer/eat_nav'/f'{name}.py'
        remote=ROOT/'src/visualnav_transformer/eat_nav'/f'{name}.py'
        files.append((local,remote))
    for local,remote in files:
        docker('exec',CONTAINER,'mkdir','-p',remote.parent)
        docker('cp',local,f'{CONTAINER}:{remote}')
        if sha(local)!=remote_sha(remote):raise RuntimeError('input copy SHA mismatch: '+str(local))
    for label,path in inputs['frozen_source_paths'].items():
        if remote_sha(path)!=inputs['frozen_source_sha256'][label]:
            raise RuntimeError('frozen source SHA drift: '+label)


def sample(start):
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True)
    used=max(int(x.strip()) for x in gpu.stdout.splitlines() if x.strip())
    mem=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    stat=docker('exec',CONTAINER,'cat','/sys/fs/cgroup/memory.stat').stdout
    cg={x.split()[0]:int(x.split()[1]) for x in stat.splitlines() if len(x.split())==2}
    remote=docker('exec',CONTAINER,'du','-sb',ROOT,check=False)
    rem_size=int(remote.stdout.split()[0]) if remote.returncode==0 else 0
    host_size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    return {'elapsed_sec':time.monotonic()-start,'gpu_device_used_mib':used,
        'host_available_bytes':mem,'container_anon_shmem_bytes':cg.get('anon',0)+cg.get('shmem',0),
        'storage_bytes':host_size+rem_size,'heartbeat':remote_json(SCAN/'heartbeat.json'),
        'owned_process':remote_json(SCAN/'owned-process.json'),'wall_ns':time.time_ns()}


def main():
    pre=json.loads((ROOT/'scan-preflight.json').read_text())
    inputs=json.loads((ROOT/'inputs/scan-inputs.json').read_text())
    if (ROOT/'scan-status.json').exists() or SCAN.exists():
        raise RuntimeError('second scan already launched; no automatic retry')
    if sha(ROOT/'inputs/scan-inputs.json')!=pre['scan_inputs_sha256']:
        raise RuntimeError('scan inputs SHA drift')
    copy_inputs(inputs)
    atomic(ROOT/'scan-status.json',{'stage':'running','candidate_free':True,'physics_steps':0})
    started=time.monotonic();client=None
    try:
        with (ROOT/'scan-runner.log').open('wb') as log:
            client=subprocess.Popen(pre['command'],stdout=log,stderr=subprocess.STDOUT)
            while client.poll() is None:
                item=sample(started)
                atomic(ROOT/'scan-resource-latest.json',item)
                with (ROOT/'scan-resource-series.jsonl').open('a') as f:
                    f.write(json.dumps(item,sort_keys=True)+'\n')
                reasons=[];hb=item['heartbeat']
                if item['gpu_device_used_mib']>12288:reasons.append('gpu_cap')
                if item['container_anon_shmem_bytes']>10*1024**3:reasons.append('anon_shmem_cap')
                if item['host_available_bytes']<8*1024**3:reasons.append('host_memory_floor')
                if item['storage_bytes']>500_000_000:reasons.append('storage_cap')
                if item['elapsed_sec']>900:reasons.append('process_timeout')
                if hb is None and item['elapsed_sec']>90:reasons.append('startup_heartbeat_timeout')
                if hb and (time.monotonic_ns()-hb['monotonic_ns'])/1e9>90:reasons.append('heartbeat_stale')
                if item['owned_process'] and not owned_alive(item['owned_process']):reasons.append('ownership')
                if reasons:
                    stop_owned(client)
                    raise RuntimeError('resource/lifecycle stop: '+repr(reasons))
                time.sleep(2)
        SCAN.mkdir(exist_ok=False)
        docker('cp',f'{CONTAINER}:{SCAN}/.',SCAN)
        summary=json.loads((SCAN/'scan-summary.json').read_text())
        complete=json.loads((SCAN/'complete.json').read_text())
        if (client.returncode!=0 or summary['candidate_count']!=448 or
            summary['dem_complete']!=448 or summary['robustness_complete']!=448 or
            summary['physics_steps_after_scene_init']!=0 or
            complete['physics_step_counter_start']!=0 or complete['physics_step_counter_end']!=0):
            raise RuntimeError('offline scan completeness/zero-step gate failed')
        identity=remote_json(SCAN/'owned-process.json')
        if not identity or owned_alive(identity):raise RuntimeError('owned process residual or identity missing')
        files=[p for p in SCAN.rglob('*') if p.is_file()]
        mismatches=[str(p) for p in files if sha(p)!=remote_sha(p)]
        if mismatches:raise RuntimeError('host/container SHA mismatch: '+repr(mismatches))
        atomic(ROOT/'scan-sha-ledger.json',{'files':[{'path':str(p.relative_to(ROOT)),
            'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]})
        atomic(ROOT/'scan-seal.json',{'schema':'eatnav_r3_second_scan_seal_v1',
            'native_exit_code':client.returncode,'owned_process':identity,'owned_residual':False,
            'sha_ledger_sha256':sha(ROOT/'scan-sha-ledger.json'),
            'scan_inputs_sha256':pre['scan_inputs_sha256'],
            'terrain_mesh_sha256':summary['terrain_mesh_sha256'],
            'resource_latest':json.loads((ROOT/'scan-resource-latest.json').read_text())})
        atomic(ROOT/'scan-status.json',{'stage':'complete','candidate_free':True,
            'physics_steps':0,'candidate_count':448,'native_exit_code':0,'owned_residual':False,
            'seal_sha256':sha(ROOT/'scan-seal.json')})
    except BaseException as exc:
        if client and client.poll() is None:stop_owned(client)
        if not SCAN.exists():SCAN.mkdir(exist_ok=True)
        docker('cp',f'{CONTAINER}:{SCAN}/.',SCAN,check=False)
        atomic(ROOT/'scan-status.json',{'stage':'stopped','decision':'r3_second_seed_measurement_invalid',
            'error':repr(exc),'traceback':traceback.format_exc(),
            'native_exit_code':None if client is None else client.poll(),
            'owned_process':remote_json(SCAN/'owned-process.json')})
        raise

if __name__=='__main__':main()
