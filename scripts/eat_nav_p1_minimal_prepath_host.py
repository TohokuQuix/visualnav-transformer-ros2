"""Owned, capped Docker launcher for one sealed candidate-free P1 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from visualnav_transformer.eat_nav.p1_minimal_prepath import resource_violation


CONTAINER = 'isaac-lab-onix'
PYTHONPATH = ':'.join((
    '/tmp/eatnav-p1-radial-candidate-20261009-r4-continuation/inputs/src',
    '/tmp/eatnav-r3-settling-pilot-20261008/shard-B/src',
    '/tmp/eatnav-train/src',
    '/tmp/eatnav-extended/isaac-source',
    '/workspace/isaaclab/source/isaaclab',
    '/workspace/isaaclab/source/isaaclab_rl',
    '/workspace/isaaclab/source/isaaclab_tasks',
    '/workspace/isaaclab_onix/source/onix_isaac_lab',
))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def docker(*parts, timeout=20):
    return subprocess.run(['docker', 'exec', CONTAINER, *parts], capture_output=True,
                          text=True, timeout=timeout)


def owned_identity(root):
    p = docker('cat', f'{root}/output/owned-process.json')
    if p.returncode:
        return None
    owner = json.loads(p.stdout)
    current = docker('cat', f'/proc/{owner["pid"]}/stat')
    if current.returncode:
        return None
    tick = int(current.stdout.rsplit(')', 1)[1].split()[19])
    if tick != owner['start_tick']:
        raise ValueError('owned PID start tick mismatch')
    return owner


def measure(root, started, heartbeat_state):
    gpu = docker('nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits')
    gpu_mib = int(gpu.stdout.strip().splitlines()[0]) if gpu.returncode == 0 else None
    mem_kib = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:'))
    host_bytes = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    remote = docker('du', '-sb', f'{root}/output')
    remote_bytes = int(remote.stdout.split()[0]) if remote.returncode == 0 else 0
    heartbeat = docker('cat', f'{root}/output/heartbeat.json')
    if heartbeat.returncode == 0:
        step = int(json.loads(heartbeat.stdout)['completed_control_steps'])
        if heartbeat_state is None or heartbeat_state[0] != step:
            heartbeat_state = (step, time.monotonic())
        heartbeat_age = time.monotonic()-heartbeat_state[1]
    else:
        heartbeat_age = 0.0 if heartbeat_state is None else time.monotonic()-heartbeat_state[1]
    return {'gpu_mib': gpu_mib, 'host_available_gib': mem_kib/1024**2,
            'active_storage_gib': (host_bytes+remote_bytes)/1024**3,
            'heartbeat_age_s': heartbeat_age, 'elapsed_s': time.monotonic()-started}, heartbeat_state


def main(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if sha(__file__) != manifest['host_source_sha256']:
        raise ValueError('host monitor source SHA drift')
    if sha(root/'source.py') != manifest['source_sha256'] or sha(root/'anchor-fixtures.json') != manifest['fixture_sha256']:
        raise ValueError('frozen local source/fixture SHA drift')
    container_root = str(root)
    files = ['source.py', 'anchor-fixtures.json', 'manifest.json']
    if 'contact_helper_sha256' in manifest:
        if sha(root/'p1_contact_summary.py') != manifest['contact_helper_sha256']:
            raise ValueError('local contact helper SHA drift')
        files.append('p1_contact_summary.py')
    for filename in files:
        remote = docker('sha256sum', f'{container_root}/{filename}')
        if remote.returncode or remote.stdout.split()[0] != sha(root/filename):
            raise ValueError(f'container input SHA mismatch: {filename}')
    if (root/'output').exists():
        raise FileExistsError('output root already exists')
    started = time.monotonic()
    atomic(root/'host-status.json', {'stage': 'running', 'mode': manifest['mode'], 'Path_publications': 0})
    cmd = ['docker', 'exec', '-e', f'PYTHONPATH={PYTHONPATH}', CONTAINER,
           'timeout', '600s', '/isaac-sim/python.sh', f'{container_root}/source.py',
           '--headless', '--manifest', f'{container_root}/manifest.json',
           '--output', f'{container_root}/output']
    samples = []
    heartbeat_state = None
    failure = None
    with (root/'runner.log').open('w') as stream:
        process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            sample, heartbeat_state = measure(root, started, heartbeat_state)
            samples.append(sample)
            failure = resource_violation(sample, manifest['resource_cap'])
            if failure:
                owner = owned_identity(container_root)
                if owner:
                    docker('kill', '-TERM', str(owner['pid']))
                break
            time.sleep(2)
        rc = process.wait(timeout=30)
    atomic(root/'resource-series.json', samples)
    atomic(root/'host-status.json', {'stage': 'exited', 'native_exit': rc, 'resource_failure': failure,
                                     'mode': manifest['mode'], 'Path_publications': 0,
                                     'peak_gpu_mib': max((x['gpu_mib'] or 0 for x in samples), default=None),
                                     'elapsed_s': time.monotonic()-started})
    if rc != 0 or failure:
        raise RuntimeError(f'probe failed: exit {rc}, resource {failure}')
    copied = subprocess.run(['docker', 'cp', f'{CONTAINER}:{container_root}/output', str(root)],
                            capture_output=True, text=True, timeout=30)
    if copied.returncode:
        raise RuntimeError(f'output host copy failed: {copied.stderr}')
    if not (root/'output/complete.json').exists():
        atomic(root/'host-status.json', {'stage': 'protocol_invalid_missing_complete',
                                         'native_exit': rc, 'resource_failure': failure,
                                         'Path_publications': 0})
        raise RuntimeError('simulator exited without complete marker')
    complete = json.loads((root/'output/complete.json').read_text())
    for key, filename in (('trace_sha256','physics-trace.json'), ('actors_sha256','actor-observations.json'),
                          ('dem_sha256','dem-frames.json')):
        if complete[key] != sha(root/'output'/filename):
            raise ValueError(f'host output SHA mismatch: {filename}')
        remote = docker('sha256sum', f'{container_root}/output/{filename}')
        if remote.returncode or remote.stdout.split()[0] != complete[key]:
            raise ValueError(f'container output SHA mismatch: {filename}')
    if complete['physics_steps'] != 200 or complete['Path_publications'] != 0:
        raise ValueError('probe completion contract failed')
    if owned_identity(container_root) is not None:
        raise ValueError('owned process residual after cleanup')
    atomic(root/'host-seal.json', {'stage': 'sealed', 'manifest_sha256': sha(root/'manifest.json'),
                                   'complete_sha256': sha(root/'output/complete.json'),
                                   'host_status_sha256': sha(root/'host-status.json'),
                                   'Path_publications': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root)
