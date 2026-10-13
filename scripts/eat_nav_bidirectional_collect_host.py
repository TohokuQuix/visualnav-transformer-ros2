"""Stage one immutable direction/angle process and preserve raw result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from visualnav_transformer.eat_nav.bidirectional_bank import ANGLES
from visualnav_transformer.eat_nav.mi_a_v3 import validate_collection_manifest


REPO = Path(__file__).resolve().parents[1]
RL = Path('/home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt')
CAP = {'gpu_device_total_mib': 6656, 'host_mem_available_min_kib': 2 * 1024 * 1024,
       'batch_storage_bytes': 800_000_000, 'wall_seconds': 600}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def container_stage(root):
    """Tie Docker staging to the full immutable host attempt, not its basename."""
    digest = hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:20]
    return f'/tmp/eatnav-bidirectional-{digest}'


def validate_output_identity(output, expected):
    output = Path(output)
    if (output/'failure.json').exists():
        raise ValueError('simulator diagnostic failure marker present')
    actual = json.loads((output/'input-manifest.json').read_text())
    for host_key, sim_key in (('seed', 'seed'), ('direction', 'direction'),
                              ('angle_deg', 'arc_angle_deg'),
                              ('source_sha256', 'source_sha256'),
                              ('rl_checkpoint_sha256', 'checkpoint_sha256')):
        if actual.get(sim_key) != expected[host_key]:
            raise ValueError(f'simulator output {sim_key} identity mismatch')


def durable_json(path, value):
    path = Path(path)
    with path.open('x') as out:
        json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())


def run(*argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True)


def sample_resources(root):
    gpu = run('nvidia-smi', '--query-gpu=uuid,memory.used,memory.total',
              '--format=csv,noheader,nounits').stdout.strip().split(', ')
    if len(gpu) != 3:
        raise ValueError('NVIDIA GPU query incomplete')
    mem = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
               if line.startswith('MemAvailable:'))
    used = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
    return {'wall_time_ns': time.time_ns(), 'gpu_uuid': gpu[0],
            'gpu_device_total_mib': int(gpu[1]), 'gpu_capacity_mib': int(gpu[2]),
            'host_mem_available_kib': mem, 'host_batch_storage_bytes': used}


def verify_sim_seal(output):
    output = Path(output)
    seal = json.loads((output/'host-seal.json').read_text())
    ledger = output/'sha-ledger.json'
    if sha(ledger) != seal['ledger_sha256']:
        raise ValueError('simulator ledger seal mismatch')
    expected = json.loads(ledger.read_text())
    actual = {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
              if p.is_file() and p.name not in ('sha-ledger.json', 'host-seal.json')}
    if actual != expected or len(actual) != seal['file_count']:
        raise ValueError('simulator raw file SHA mismatch')
    complete = json.loads((output/'complete.json').read_text())
    if complete['status'] != 'complete' or complete['control_steps'] != 40 or \
            complete['env_count'] != 4:
        raise ValueError('simulator completion contract')
    for index in range(4):
        path = output/f'trace-env-{index}.jsonl'
        if sha(path) != complete['trace_sha256'][index] or \
                len(path.read_text().splitlines()) != 40:
            raise ValueError('trace SHA/length mismatch')
    return {'simulator_seal_sha256': sha(output/'host-seal.json'),
            'file_count': len(actual), 'trace_count': 4}


def execute(root, *, direction, angle, seed, terrain_type, terrain_level, reference,
            run_mode='demo', state_group_id=None, flipper_assignment_manifest=None):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError('fresh persistent Arc attempt required')
    if direction not in (-1, 0, 1) or angle not in ANGLES or (direction == 0 and angle != 0):
        raise ValueError('Path direction/angle not frozen')
    if run_mode == 'collection':
        if not state_group_id or flipper_assignment_manifest is None:
            raise ValueError('collection requires frozen StateGroup assignment')
        assignment_path = Path(flipper_assignment_manifest).resolve()
        allocation = validate_collection_manifest(json.loads(assignment_path.read_text()))
        if state_group_id not in allocation:
            raise ValueError('StateGroup absent from collection assignment')
    elif run_mode == 'demo' and state_group_id is None and flipper_assignment_manifest is None:
        assignment_path = None
    else:
        raise ValueError('invalid run mode/assignment')
    source = REPO/'scripts/eat_nav_bidirectional_collect_sim.py'
    eat_config = REPO/'config/eat_nav.yaml'
    reference = Path(reference).resolve() if reference else None
    if reference and (not (reference/'prepath.npz').is_file() or
                      not (reference/'prepath-identity.json').is_file()):
        raise ValueError('sealed prepath reference unavailable')
    root.mkdir(parents=True)
    stage = container_stage(root)
    manifest = {'schema': 'eatnav_forward_reverse_r0_host_v1', 'direction': direction,
                'angle_deg': angle,
                'seed': seed, 'terrain_type': terrain_type, 'terrain_level': terrain_level,
                'run_mode': run_mode, 'state_group_id': state_group_id,
                'flipper_assignment_sha256': sha(assignment_path) if assignment_path else None,
                'source_sha256': sha(source), 'host_source_sha256': sha(__file__),
                'bank_source_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/bidirectional_bank.py'),
                'rl_checkpoint_sha256': sha(RL), 'eat_config_sha256': sha(eat_config),
                'reference_prepath_sha256': sha(reference/'prepath.npz') if reference else None,
                'resource_cap': CAP, 'container_stage': stage,
                'output_root': str(root)}
    durable_json(root/'manifest.json', manifest)
    sim = None
    failure = None
    start = time.monotonic()
    try:
        run('docker', 'exec', 'isaac-lab-onix', 'test', '!', '-e', stage)
        run('docker', 'exec', 'isaac-lab-onix', 'mkdir', '-p', stage+'/src', stage+'/config')
        run('docker', 'cp', str(REPO/'src/visualnav_transformer'), 'isaac-lab-onix:'+stage+'/src/')
        run('docker', 'cp', str(source), 'isaac-lab-onix:'+stage+'/sim.py')
        run('docker', 'cp', str(eat_config), 'isaac-lab-onix:'+stage+'/config/eat_nav.yaml')
        if assignment_path:
            run('docker', 'cp', str(assignment_path),
                'isaac-lab-onix:'+stage+'/config/flipper-assignment.json')
        if reference:
            run('docker', 'exec', 'isaac-lab-onix', 'mkdir', '-p', stage+'/reference')
            for name in ('prepath.npz', 'prepath-identity.json'):
                run('docker', 'cp', str(reference/name), 'isaac-lab-onix:'+stage+'/reference/'+name)
        staged_source_sha = run('docker', 'exec', 'isaac-lab-onix', 'sha256sum', stage+'/sim.py').stdout.split()[0]
        if staged_source_sha != manifest['source_sha256']:
            raise ValueError('host/container staged simulator SHA mismatch')
        durable_json(root/'staged.json', {'source_sha256': staged_source_sha,
                     'source_tree_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/bidirectional_bank.py')})
        command = ['docker', 'exec', '-e',
                   f'PYTHONPATH={stage}/src:/workspace/isaaclab_onix/source/onix_isaac_lab:/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_rl:/workspace/isaaclab/source/isaaclab_tasks',
                   'isaac-lab-onix', '/isaac-sim/python.sh', stage+'/sim.py',
                   '--output', stage+'/output', '--rl-checkpoint',
                   '/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
                   '--eat-config', stage+'/config/eat_nav.yaml', '--arc-angle', str(angle),
                   '--direction', str(direction),
                   '--seed', str(seed), '--terrain-type', terrain_type,
                   '--terrain-level', str(terrain_level), '--headless']
        command.extend(['--run-mode', run_mode])
        if assignment_path:
            command.extend(['--state-group-id', state_group_id,
                            '--flipper-assignment-manifest', stage+'/config/flipper-assignment.json'])
        if reference:
            command.extend(['--reference-prepath', stage+'/reference'])
        durable_json(root/'launch-command.json', command)
        with (root/'simulator.log').open('x') as logfile, \
                (root/'resource-series.jsonl').open('x') as resources:
            sim = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT)
            while sim.poll() is None:
                row = sample_resources(root)
                resources.write(json.dumps(row, sort_keys=True) + '\n')
                resources.flush()
                os.fsync(resources.fileno())
                if row['gpu_device_total_mib'] > CAP['gpu_device_total_mib'] or \
                        row['host_mem_available_kib'] < CAP['host_mem_available_min_kib'] or \
                        row['host_batch_storage_bytes'] > CAP['batch_storage_bytes'] or \
                        time.monotonic() - start > CAP['wall_seconds']:
                    raise RuntimeError('resource cap exceeded')
                time.sleep(1)
            sim.wait()
        if sim.returncode != 0:
            raise RuntimeError(f'Isaac native exit {sim.returncode}')
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        if sim is not None and sim.poll() is None:
            sim.terminate()
            try:
                sim.wait(timeout=15)
            except subprocess.TimeoutExpired:
                sim.kill()
                sim.wait()
    finally:
        durable_json(root/'native-exit.json', {'exit_code': sim.returncode if sim else None,
                     'failure': failure, 'wall_seconds': time.monotonic() - start})
        try:
            run('docker', 'cp', 'isaac-lab-onix:'+stage+'/output', str(root/'sim-output'))
        except Exception as exc:
            if failure is None:
                failure = f'artifact_copy: {exc}'
        if failure is None:
            try:
                validate_output_identity(root/'sim-output', manifest)
                verification = verify_sim_seal(root/'sim-output')
                durable_json(root/'verification.json', verification)
            except Exception as exc:
                failure = f'seal_verification: {exc}'
        durable_json(root/'decision.json', {'schema': manifest['schema'],
                     'pass': failure is None, 'failure': failure})
        files = {str(path.relative_to(root)): sha(path) for path in root.rglob('*')
                 if path.is_file() and path.name not in ('outer-sha-ledger.json', 'outer-seal.json')}
        durable_json(root/'outer-sha-ledger.json', files)
        durable_json(root/'outer-seal.json', {'ledger_sha256': sha(root/'outer-sha-ledger.json'),
                     'file_count': len(files)})
    if failure:
        raise RuntimeError(failure)
    return root/'verification.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arc-angle', type=int, required=True)
    parser.add_argument('--direction', type=int, choices=(-1, 0, 1), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--terrain-type', required=True)
    parser.add_argument('--terrain-level', type=int, required=True)
    parser.add_argument('--reference-prepath', type=Path)
    parser.add_argument('--run-mode', choices=('demo', 'collection'), default='demo')
    parser.add_argument('--state-group-id')
    parser.add_argument('--flipper-assignment-manifest', type=Path)
    args = parser.parse_args()
    print(execute(args.root, direction=args.direction, angle=args.arc_angle, seed=args.seed,
                  terrain_type=args.terrain_type, terrain_level=args.terrain_level,
                  reference=args.reference_prepath, run_mode=args.run_mode,
                  state_group_id=args.state_group_id,
                  flipper_assignment_manifest=args.flipper_assignment_manifest))


if __name__ == '__main__':
    main()
