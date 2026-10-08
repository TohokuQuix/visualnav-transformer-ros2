"""One-shot candidate-free replay of the three frozen R3 settling shards."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(os.environ.get('EATNAV_RADIAL_FIXTURE_ROOT', '/tmp/eatnav-radial-anchor-contract-20261008-r1'))
CONTAINER = 'isaac-lab-onix'
_spec = importlib.util.spec_from_file_location('contract', Path(__file__).with_name('eat_nav_radial_anchor_contract.py'))
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def docker(*args, check=False):
    return subprocess.run(['docker', *map(str, args)], capture_output=True, text=True, check=check)


def remote_json(path):
    proc = docker('exec', CONTAINER, 'cat', path)
    return json.loads(proc.stdout) if proc.returncode == 0 else None


def remote_sha(path):
    proc = docker('exec', CONTAINER, 'sha256sum', path)
    return proc.stdout.split()[0] if proc.returncode == 0 else None


def gpu_mib():
    proc = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True)
    return max(int(line) for line in proc.stdout.splitlines())


def host_available():
    line = next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    return int(line.split()[1]) * 1024


def anon_shmem():
    proc = docker('exec', CONTAINER, 'cat', '/sys/fs/cgroup/memory.stat', check=True)
    entries = dict(line.split() for line in proc.stdout.splitlines() if len(line.split()) == 2)
    return int(entries.get('anon', 0)) + int(entries.get('shmem', 0))


def owned_alive(identity):
    if not identity:
        return False
    proc = docker('exec', CONTAINER, 'cat', f"/proc/{identity['pid']}/stat")
    if proc.returncode != 0:
        return False
    tick = int(proc.stdout.split(') ', 1)[1].split()[19])
    if tick != identity['proc_start_ticks']:
        raise RuntimeError('PID/start tick ownership ambiguity')
    return True


def cleanup(client, identity):
    if owned_alive(identity):
        docker('exec', CONTAINER, 'kill', '-TERM', identity['pid'])
        for _ in range(40):
            if not owned_alive(identity):
                break
            time.sleep(.25)
        if owned_alive(identity):
            docker('exec', CONTAINER, 'kill', '-KILL', identity['pid'])
        if owned_alive(identity):
            raise RuntimeError('owned process residual after targeted cleanup')
    if client.poll() is None:
        client.terminate()
        client.wait(timeout=20)


def verify_source(source):
    freeze = read(source / 'input-freeze.json')
    prelaunch = read(source / 'prelaunch.json')
    if contract.sha256(source/'input-freeze.json') != prelaunch['freeze_sha256']:
        raise RuntimeError('freeze SHA mismatch')
    if contract.sha256(source/'pose-manifest.json') != freeze['pose_manifest_sha256']:
        raise RuntimeError('pose manifest SHA mismatch')
    for label, path in freeze['source_paths'].items():
        if remote_sha(path) != freeze['source_sha256'][label]:
            raise RuntimeError('frozen remote source SHA mismatch: '+label)
    cmd=prelaunch['command']
    if remote_sha(cmd[cmd.index('--freeze')+1]) != prelaunch['freeze_sha256']:
        raise RuntimeError('frozen container input-freeze SHA mismatch')
    if remote_sha(cmd[cmd.index('--pose-manifest')+1]) != freeze['pose_manifest_sha256']:
        raise RuntimeError('frozen container pose-manifest SHA mismatch')
    return freeze, prelaunch


def run_one(entry, caps):
    source = Path(entry['source_shard'])
    target = Path(entry['fixture_subroot'])
    if target.exists():
        raise RuntimeError('fixture subroot already exists: '+str(target))
    freeze, prelaunch = verify_source(source)
    cmd = list(prelaunch['command'])
    assert cmd[0:2] == ['docker','exec'] and cmd[cmd.index('--mode')+1] == 'full'
    assert cmd[cmd.index('--batch-seed')+1] == str(entry['batch_seed'])
    assert cmd[cmd.index('--output')+1].startswith('/tmp/eatnav-r3-')
    cmd[cmd.index('--output')+1] = str(target/'run')
    if 'Path' in ' '.join(cmd) or 'follower' in ' '.join(cmd) or 'bridge' in ' '.join(cmd):
        raise RuntimeError('candidate command in fixture invocation')
    target.mkdir(parents=True)
    docker('exec', CONTAINER, 'mkdir', '-p', target, check=True)
    atomic(target/'invocation.json', {'candidate_free':True,'command':cmd,
                                     'source_freeze_sha256':contract.sha256(source/'input-freeze.json'),
                                     'source_pose_manifest_sha256':contract.sha256(source/'pose-manifest.json')})
    started = time.monotonic()
    samples = []
    client = None
    identity = None
    try:
        with (target/'runner.log').open('wb') as log:
            client = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            while client.poll() is None:
                heartbeat = remote_json(target/'run/heartbeat.json')
                identity = remote_json(target/'run/owned-process.json') or identity
                sample = {'elapsed_sec':time.monotonic()-started,'gpu_device_used_mib':gpu_mib(),
                          'host_available_bytes':host_available(),'container_anon_shmem_bytes':anon_shmem(),
                          'host_output_storage_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
                          'heartbeat':heartbeat,'owned_process':identity,'wall_ns':time.time_ns()}
                samples.append(sample)
                atomic(target/'resource-latest.json',sample)
                with (target/'resource-series.jsonl').open('a') as f:
                    f.write(json.dumps(sample,sort_keys=True)+'\n')
                reasons = []
                if sample['gpu_device_used_mib'] > caps['gpu_device_total_hard_cap_mib']:reasons.append('gpu_cap')
                if sample['container_anon_shmem_bytes'] > caps['container_anon_plus_shmem_cap_bytes']:reasons.append('anon_shmem_cap')
                if sample['host_available_bytes'] < caps['minimum_host_available_bytes']:reasons.append('host_memory_floor')
                if sample['elapsed_sec'] > caps['per_process_timeout_sec']:reasons.append('process_timeout')
                if sample['host_output_storage_bytes'] > caps['storage_cap_bytes']:reasons.append('storage_cap')
                if heartbeat and (time.monotonic_ns()-heartbeat['monotonic_ns'])/1e9 > caps['heartbeat_stale_sec']:
                    reasons.append('heartbeat_stale')
                if identity and not owned_alive(identity) and (not heartbeat or heartbeat.get('physics_step',0)<200):
                    reasons.append('owned_process_disappeared_early')
                if reasons:raise RuntimeError('resource_lifecycle_stop: '+','.join(reasons))
                time.sleep(2)
        docker('cp', f'{CONTAINER}:{target}/run/.', target/'run', check=True)
        marker=read(target/'run/sim-finished.json') if (target/'run/sim-finished.json').exists() else None
        if client.returncode != 0 or not marker or marker['physics_steps'] != 200 or marker['physical_termination_pose_ids']:
            raise RuntimeError('native_exit_or_completion_marker_invalid')
        if marker['sim_source_sha256'] != freeze['sim_source_sha256']:
            raise RuntimeError('sim source SHA mismatch')
        if contract.sha256(target/'run/terrain-mesh-identity.json') != contract.sha256(source/'run/terrain-mesh-identity.json'):
            raise RuntimeError('terrain mesh identity mismatch')
        identity = read(target/'run/owned-process.json')
        if owned_alive(identity):
            raise RuntimeError('owned residual')
        file_rows=[]
        for file in sorted((target/'run').rglob('*')):
            if file.is_file():
                local=contract.sha256(file)
                if remote_sha(file) != local:
                    raise RuntimeError('host/container SHA mismatch: '+str(file))
                file_rows.append({'path':str(file),'sha256':local,'bytes':file.stat().st_size})
        if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) > caps['storage_cap_bytes']:
            raise RuntimeError('storage_cap')
        atomic(target/'sha-ledger.json',{'files':file_rows})
        atomic(target/'seal.json',{'source_shard':str(source),'physics_steps':200,'native_exit_code':0,
                                   'owned_process':identity,'owned_residual':False,'host_container_sha_match':True,
                                   'resource_peak':{'gpu_mib':max(s['gpu_device_used_mib'] for s in samples),
                                                    'anon_shmem_bytes':max(s['container_anon_shmem_bytes'] for s in samples)},
                                   'sha_ledger_sha256':contract.sha256(target/'sha-ledger.json')})
        atomic(target/'status.json',{'stage':'protocol_valid','source_shard':str(source),
                                     'seal_sha256':contract.sha256(target/'seal.json')})
        return read(target/'seal.json')
    except BaseException as exc:
        if client is not None:
            cleanup(client, identity)
        docker('cp',f'{CONTAINER}:{target}/run/.',target/'run')
        atomic(target/'status.json',{'stage':'stopped','error':repr(exc),'native_exit':None if client is None else client.poll(),
                                     'owned_process':identity,'owned_residual':owned_alive(identity) if identity else None})
        raise


def main():
    manifest=read(ROOT/'fixture-manifest.json')
    caps=read(ROOT/'resource-forecast.json')['caps']
    results=[]
    for entry in manifest['shards']:
        results.append(run_one(entry,caps))
        atomic(ROOT/'fixture-shard-status.json',{'completed':len(results),'total':len(manifest['shards']),'results':results})


if __name__=='__main__':
    main()
