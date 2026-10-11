"""Prepared G1 host launcher with fail-closed per-PID NVML attribution.

This source has only static/synthetic validation. It must not be run until approved.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time
from eat_nav_g0e_gpu_probe import Nvml, proc_table
from eat_nav_p1_gpu_attribution import capture_live
from eat_nav_p1_longhorizon_gpu import capture_sample, resolve_simulator
from eat_nav_p1_owner_host import initial_graphics_identities

CONTAINER = 'isaac-lab-onix'
ROOT = None
REMOTE = None
BASE = ':'.join(('/tmp/eatnav-train/src', '/tmp/eatnav-extended/isaac-source',
                 '/workspace/isaaclab/source/isaaclab', '/workspace/isaaclab/source/isaaclab_rl',
                 '/workspace/isaaclab/source/isaaclab_tasks',
                 '/workspace/isaaclab_onix/source/onix_isaac_lab'))


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = pathlib.Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


GPU_UUID = 'GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26'
GPU_CONTRACT = {'required_gpu_uuid': GPU_UUID,
                'required_total_memory_mib_min': 16000,
                'gpu_device_total_hard_cap_mib': 7680,
                'bracket_nonatomic_sampling_gap_hard_cap_mib': 256,
                'absolute_attribution_gap_hard_cap_mib': 256,
                'unresolved_gpu_bytes_hard_cap': 0}


def host_available():
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing')


def anon_shmem():
    raw = command(['docker', 'exec', CONTAINER, 'cat', '/sys/fs/cgroup/memory.stat'])
    stats = {row.split()[0]: int(row.split()[1]) for row in raw.splitlines()}
    return stats['anon'] + stats['shmem']


def root_bytes():
    return sum(p.stat().st_size for p in ROOT.parent.rglob('*') if p.is_file())


def sample(started, nvml, owned, baseline, container_id, source_path):
    gpu = capture_sample(nvml, owned, baseline, GPU_CONTRACT,
                         container_id=container_id, source_path=source_path)
    # Keep the bracket evidence small enough to stay inside the frozen storage cap.
    audit = {key: gpu[key] for key in ('pass','failures','gpu_uuid','device_before_bytes',
             'device_after_bytes','device_total_bytes','processes','owned_gpu_pids',
             'bracket_gap_bytes','process_sum_bytes','resolved_owned','wall_ns',
             'sample_finished_monotonic_ns')}
    return {'elapsed_s': time.monotonic() - started,
            'gpu_device_total_mib': max(gpu['device_before_bytes'],gpu['device_after_bytes'])/1048576,
            'gpu_attribution': audit,
            'host_available_bytes': host_available(), 'container_anon_plus_shmem_bytes': anon_shmem(),
            'persistent_root_bytes': root_bytes()}


def validate_sample(row):
    if not row['gpu_attribution']['pass']: return 'gpu_attribution:' + ','.join(row['gpu_attribution']['failures'])
    if row['gpu_device_total_mib'] > 7680: return 'gpu_device_total_cap'
    if row['host_available_bytes'] < 10095421850: return 'host_available_floor'
    if row['container_anon_plus_shmem_bytes'] > 6442450944: return 'container_anon_shmem_cap'
    if row['persistent_root_bytes'] > 1000000000: return 'persistent_storage_cap'
    return None


def seal(stage, *, reason=None, physics_steps=None):
    files = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and p.name not in ('host-seal.json', 'sha-ledger.json'))
    ledger = {str(p.relative_to(ROOT)): digest(p) for p in files}
    atomic(ROOT / 'sha-ledger.json', {'file_sha256': ledger})
    atomic(ROOT / 'host-seal.json', {'stage': stage, 'reason': reason,
           'physics_steps': physics_steps, 'file_sha256': ledger,
           'sha_ledger_sha256': digest(ROOT / 'sha-ledger.json')})


def main():
    global ROOT, REMOTE
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--remote', required=True)
    arguments=parser.parse_args()
    ROOT=arguments.root.resolve(); REMOTE=arguments.remote
    if ROOT.name == 'g1-attempt-0':
        raise ValueError('G1 attempt-0 is immutable')
    if (ROOT / 'output').exists() or (ROOT / 'host-seal.json').exists():
        raise FileExistsError('G1 output already exists')
    freeze = json.loads((ROOT / 'input/freeze.json').read_text())
    assert digest(ROOT / 'source.py') == freeze['pilot_simulator_source_sha256']
    inspect = json.loads(command(['docker', 'inspect', CONTAINER]))[0]
    if inspect['State']['Status'] != 'running':
        raise RuntimeError('container not running')
    if subprocess.run(['docker', 'exec', CONTAINER, 'test', '-d', REMOTE]).returncode != 0:
        raise RuntimeError('host-mounted G1 root not visible')
    host_free = shutil.disk_usage(ROOT).free
    nvml=Nvml()
    initial_devices, initial_rows=nvml.sample()
    initial_table=proc_table()
    baseline_identities=initial_graphics_identities(initial_rows, initial_table)
    # Baseline is sampled before launch, with no expected simulator owner.
    baseline_gpu=capture_live(nvml, {}, baseline_identities, GPU_CONTRACT,
                              container_id=inspect['Id'], expected_simulator_alive=False,
                              device_cap_mib=1024)
    if not baseline_gpu['pass']:
        raise RuntimeError('GPU prelaunch attribution: '+','.join(baseline_gpu['failures']))
    baseline = {'gpu_device_total_mib': max(d['used_bytes'] for d in initial_devices)/1048576,
                'gpu_uuid':initial_devices[0]['uuid'],
                'known_graphics_identities':sorted(baseline_identities)}
    if baseline['gpu_uuid'] != GPU_UUID or not baseline_identities:
        raise RuntimeError('GPU baseline identity invalid')
    if baseline['gpu_device_total_mib'] > 1024:
        raise RuntimeError('GPU baseline cap')
    if host_free < 2000000000:
        raise RuntimeError('host disk free floor')
    atomic(ROOT / 'g1-prelaunch.json', {
        'container_id': inspect['Id'], 'image_id': inspect['Image'],
        'source_sha256': digest(ROOT / 'source.py'),
        'freeze_sha256': digest(ROOT / 'input/freeze.json'),
        'baseline': baseline, 'host_free_disk_bytes': host_free,
        'baseline_attribution': {key:baseline_gpu[key] for key in
                                 ('gpu_uuid','device_before_bytes','device_after_bytes',
                                  'processes','bracket_gap_bytes','owned_gpu_pids')},
        'path_publications': 0,
    })
    cmd = ['docker', 'exec', '-e', f'PYTHONPATH={REMOTE}/src:{BASE}', CONTAINER,
           'timeout', '600s', '/isaac-sim/python.sh', f'{REMOTE}/source.py', '--headless',
           '--input', f'{REMOTE}/input', '--output', f'{REMOTE}/output',
           '--anchor-id', 'first-differential', '--batch-index', '0', '--gate1-probe']
    atomic(ROOT / 'g1-command.json', {'argv': cmd})
    started = time.monotonic()
    failure = None
    with (ROOT / 'runner.log').open('x') as log, (ROOT / 'resource-series.jsonl').open('x') as resources:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        owner=None
        owner_deadline=time.monotonic()+10
        while proc.poll() is None and time.monotonic()<owner_deadline:
            try:
                owner=resolve_simulator(proc_table(),source_path=f'{REMOTE}/source.py',
                                        container_id=inspect['Id'])
                break
            except ValueError:
                time.sleep(.02)
        if owner is None:
            failure='simulator_pid_start_tick_unresolved_at_startup'
            proc.terminate()
        samples=0
        while proc.poll() is None and failure is None:
            try:
                row = sample(started,nvml,owner,baseline_identities,inspect['Id'],f'{REMOTE}/source.py')
                resources.write(json.dumps(row, sort_keys=True) + '\n')
                resources.flush(); os.fsync(resources.fileno())
                samples+=1
                failure = validate_sample(row)
                if row['elapsed_s'] > 600:
                    failure = 'process_wall_timeout'
                heartbeat = ROOT / 'output/heartbeat.json'
                if heartbeat.is_file() and time.time() - heartbeat.stat().st_mtime > 90:
                    failure = 'heartbeat_stale'
            except Exception as exc:
                failure = 'resource_monitor_error:' + repr(exc)
            if failure:
                # Preserve ownership evidence and stop only the Docker exec launched here.
                proc.terminate()
                break
            time.sleep(2)
        try: rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill(); rc = proc.wait(timeout=10)
    if samples==0 and failure is None:
        failure='no_live_GPU_bracket'
    survivor=proc_table().get(owner['simulator']['host_pid']) if owner else None
    if survivor and survivor.get('start_ticks')==owner['simulator']['start_ticks']:
        failure=failure or 'owned_simulator_residual'
    atomic(ROOT / 'g1-host-status.json', {'native_exit': rc, 'failure': failure,
                                         'elapsed_s': time.monotonic() - started,
                                         'gpu_bracket_count':samples,
                                         'owned_simulator':owner,
                                         'owned_simulator_residual':bool(survivor and owner and
                                             survivor.get('start_ticks')==owner['simulator']['start_ticks'])})
    if failure or rc:
        seal('G1_failed_or_inconclusive', reason=failure or f'native_exit_{rc}')
        raise RuntimeError(f'G1 process failed: native_exit={rc} failure={failure}')
    try:
        output = ROOT / 'output'
        marker = json.loads((output / 'gate1-probe-complete.json').read_text())
        if marker['Path_publications'] != 0:
            raise RuntimeError('G1 Path publication')
        rows = [json.loads(x) for x in (output / 'g1-physics-trace.jsonl').read_text().splitlines()]
        if [r['physics_step'] for r in rows] != list(range(20)):
            raise RuntimeError('G1 physics trace incomplete')
        if any(len(r['root_state']) != 4 or len(r['actor_policy']) != 4 or
               len(r['contact_net_forces_w']) != 4 or not all(any(mask) for mask in r['dem_finite_mask'])
               for r in rows):
            raise RuntimeError('G1 Actor/DEM/contact incomplete')
    except Exception as exc:
        seal('G1_failed_or_inconclusive', reason='artifact_validation:'+repr(exc))
        raise
    seal('G1_protocol_valid', physics_steps=20)
    print('G1_PASS', str(ROOT / 'host-seal.json'))


if __name__ == '__main__':
    main()
