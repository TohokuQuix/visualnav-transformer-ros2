"""One-shot host lifecycle for candidate-free radial inventory; no ROS/follower."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/tmp/eatnav-radial-terrain-inventory-20261007')
CONTAINER = 'isaac-lab-onix'
REMOTE = str(ROOT)
PYTHONPATH = ':'.join(['/tmp/eatnav-train/src', '/tmp/eatnav-extended/isaac-source',
    '/workspace/isaaclab/source/isaaclab', '/workspace/isaaclab/source/isaaclab_rl',
    '/workspace/isaaclab/source/isaaclab_tasks', '/workspace/isaaclab_onix/source/onix_isaac_lab'])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def docker(*args, check=True):
    return subprocess.run(['docker', *map(str, args)], capture_output=True, text=True, check=check)


def remote_sha(path):
    result = docker('exec', CONTAINER, 'sha256sum', path, check=False)
    return result.stdout.split()[0] if result.returncode == 0 else None


def source_setup(root, freeze):
    repository = Path(__file__).resolve().parents[1]
    copies = {
        repository / 'config/eat_nav.yaml': freeze['source_paths']['eat_config'],
        repository / 'src/visualnav_transformer/eat_nav/radial_inventory.py': freeze['source_paths']['radial_module'],
        repository / 'scripts/eat_nav_radial_inventory_sim.py': freeze['source_paths']['sim_source'],
        root / 'input-freeze.json': str(root / 'input-freeze.json'),
    }
    docker('exec', CONTAINER, 'mkdir', '-p', str(root / 'inputs'))
    for source, destination in copies.items():
        if remote_sha(destination) == sha(source):
            continue
        docker('cp', source, f'{CONTAINER}:{destination}')
        if remote_sha(destination) != sha(source):
            raise RuntimeError('Docker copy SHA mismatch: ' + str(source))
    for label, path in freeze['source_paths'].items():
        if remote_sha(path) != freeze['source_sha256'][label]:
            raise RuntimeError('remote source SHA drift: ' + label)
    if remote_sha(freeze['source_paths']['sim_source']) != freeze['sim_source_sha256']:
        raise RuntimeError('remote simulator source drift')


def gpu_used():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    return max(int(s.strip()) for s in result.stdout.splitlines() if s.strip())


def host_available():
    line = next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    return int(line.split()[1]) * 1024


def container_anon_shmem():
    stat = docker('exec', CONTAINER, 'cat', '/sys/fs/cgroup/memory.stat').stdout.splitlines()
    fields = {s.split()[0]: int(s.split()[1]) for s in stat if len(s.split()) == 2}
    return fields.get('anon', 0) + fields.get('shmem', 0)


def remote_bytes(path):
    r = docker('exec', CONTAINER, 'du', '-sb', path, check=False)
    return int(r.stdout.split()[0]) if r.returncode == 0 else 0


def owned_process(root, batch_seed):
    file = root / f'runs/batch-{batch_seed}/owned-process.json'
    r = docker('exec', CONTAINER, 'cat', file, check=False)
    return json.loads(r.stdout) if r.returncode == 0 else None


def terminate_owned(root, batch_seed, client):
    owned = owned_process(root, batch_seed)
    if owned:
        pid = owned['pid']
        current = docker('exec', CONTAINER, 'cat', f'/proc/{pid}/stat', check=False)
        if current.returncode == 0:
            tick = int(current.stdout.split(') ', 1)[1].split()[19])
            if tick == owned['proc_start_ticks']:
                docker('exec', CONTAINER, 'kill', '-TERM', str(pid), check=False)
    client.terminate()
    try:
        client.wait(timeout=20)
    except subprocess.TimeoutExpired:
        client.kill()
        client.wait(timeout=5)


def run_batch(root, freeze, seed, started):
    out = root / 'runs' / f'batch-{seed}'
    out.mkdir(parents=True, exist_ok=False)
    docker('exec', CONTAINER, 'mkdir', '-p', out)
    command = ['docker', 'exec', '-e', 'PYTHONPATH=' + PYTHONPATH, CONTAINER,
        '/isaac-sim/python.sh', freeze['source_paths']['sim_source'], '--headless',
        '--freeze', str(root / 'input-freeze.json'), '--batch-seed', str(seed), '--output', str(out)]
    atomic(root / 'status.json', {'stage': 'running', 'batch_seed': seed, 'candidate_free': True,
                                  'started_ns': time.time_ns(), 'completed_batches': freeze['batch_seeds'].index(seed)})
    log_path = root / f'batch-{seed}.runner.log'
    batch_started = time.monotonic()
    with log_path.open('wb') as log:
        client = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while client.poll() is None:
            elapsed = time.monotonic() - started
            gpu = gpu_used()
            host = host_available()
            anon = container_anon_shmem()
            storage = remote_bytes(root) + sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            hb = docker('exec', CONTAINER, 'cat', out / 'heartbeat.json', check=False)
            heartbeat = json.loads(hb.stdout) if hb.returncode == 0 else None
            sample = {'batch_seed': seed, 'elapsed_total_sec': elapsed,
                      'batch_elapsed_sec': time.monotonic() - batch_started,
                      'gpu_device_used_mib': gpu, 'host_mem_available_bytes': host,
                      'container_anon_shmem_bytes': anon, 'storage_bytes': storage,
                      'heartbeat': heartbeat, 'wall_ns': time.time_ns()}
            atomic(root / 'resource-latest.json', sample)
            reasons = []
            limits = freeze['limits']
            if gpu > limits['gpu_device_total_cap_mib']: reasons.append('gpu_cap')
            if host < limits['host_mem_available_floor_bytes']: reasons.append('host_memory_floor')
            if anon > limits['anon_plus_shmem_cap_bytes']: reasons.append('anon_shmem_cap')
            if storage > limits['storage_cap_bytes']: reasons.append('storage_cap')
            if elapsed > limits['total_runtime_cap_sec']: reasons.append('runtime_cap')
            if heartbeat is None and time.monotonic() - batch_started > limits['per_batch_startup_cap_sec']:
                reasons.append('startup_cap')
            if heartbeat is not None and heartbeat.get('elapsed_sec', 0) > 0:
                age = (time.monotonic_ns() - heartbeat['monotonic_ns']) / 1e9
                if age > limits['heartbeat_cap_sec']: reasons.append('heartbeat_cap')
            if reasons:
                terminate_owned(root, seed, client)
                atomic(root / 'status.json', {'stage': 'stopped_resource_or_lifecycle',
                    'batch_seed': seed, 'reasons': reasons, 'sample': sample, 'retry_count': 0})
                raise RuntimeError('resource/lifecycle stop: ' + ','.join(reasons))
            time.sleep(5)
        if client.returncode:
            docker('cp', f'{CONTAINER}:{out}', str(root / 'failed-batch-copy'), check=False)
            atomic(root / 'status.json', {'stage': 'failed_simulator', 'batch_seed': seed,
                                          'exit_code': client.returncode, 'retry_count': 0})
            raise RuntimeError(f'candidate-free simulator failed, batch={seed}, code={client.returncode}')
    if remote_sha(out / 'sim-finished.json') is None:
        raise RuntimeError('missing simulator completion marker')
    docker('cp', f'{CONTAINER}:{out}/.', out)
    poses = list((out / 'poses').glob('*.json'))
    if len(poses) != 112:
        raise RuntimeError(f'incomplete candidate-free pose count: {len(poses)}')
    for local in [out / 'env-origins.json', out / 'sim-finished.json', *poses]:
        if sha(local) != remote_sha(local):
            raise RuntimeError('host/container output SHA mismatch: ' + str(local))
    owned = json.loads((out / 'owned-process.json').read_text())
    if docker('exec', CONTAINER, 'cat', f"/proc/{owned['pid']}/stat", check=False).returncode == 0:
        raise RuntimeError('owned simulator process remains after completion')
    atomic(root / f'batch-{seed}.complete.json', {'batch_seed': seed, 'pose_count': 112,
        'sim_finished_sha256': sha(out / 'sim-finished.json'), 'cleanup_ok': True,
        'owned_pid': owned['pid'], 'owned_start_ticks': owned['proc_start_ticks']})


def main(root):
    root = Path(root)
    freeze = json.loads((root / 'input-freeze.json').read_text())
    if any((root / 'runs').glob('*')) if (root / 'runs').exists() else False:
        raise RuntimeError('one-shot root already has a run; no automatic retry')
    source_setup(root, freeze)
    started = time.monotonic()
    atomic(root / 'inventory-started.json', {'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(),
        'candidate_free': True, 'retry_count': 0, 'freeze_sha256': sha(root / 'input-freeze.json')})
    for seed in freeze['batch_seeds']:
        run_batch(root, freeze, seed, started)
        # Stop on the first batch's integrity/anchor-gate failure before starting the next batch.
        from visualnav_transformer.eat_nav.radial_inventory import build_pose_schedule, validate_record
        by_id = {r['pose_id']: r for r in build_pose_schedule(freeze['batch_seeds'])}
        folder = root / 'runs' / f'batch-{seed}' / 'poses'
        invalid = []
        for path in sorted(folder.glob('*.json')):
            item = json.loads(path.read_text())
            item['cleanup_ok'] = True
            result = validate_record(item, by_id[item['pose_id']], freeze['validity_limits'])
            if not result['valid']:
                invalid.append({'pose_id': item['pose_id'], 'reasons': result['reasons']})
        atomic(root / f'batch-{seed}.anchor-gate.json', {'batch_seed': seed, 'invalid_count': len(invalid),
            'invalid': invalid, 'candidate_free': True, 'outcome': None})
        if invalid:
            atomic(root / 'status.json', {'stage': 'stopped_anchor_gate', 'batch_seed': seed,
                'invalid_count': len(invalid), 'completed_batches': freeze['batch_seeds'].index(seed) + 1,
                'retry_count': 0})
            return
    atomic(root / 'status.json', {'stage': 'inventory_complete', 'completed_batches': len(freeze['batch_seeds']),
                                  'pose_count': 224, 'candidate_free': True})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    main(ap.parse_args().root)
