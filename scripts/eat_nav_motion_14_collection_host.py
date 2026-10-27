"""Independent 14-Path paired collection pre-Path or candidate process."""

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
from visualnav_transformer.eat_nav.mi_fi_v1 import CONDITIONS
from visualnav_transformer.eat_nav.motion_14_collection import COLLECTION_SCHEMA as SCHEMA


REPO = Path(__file__).resolve().parents[1]
RL = Path('/home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt')
SHARED_RL = Path('/home/ubuntu/eatnav-student-share-20261018/rl_actor_model_10000.pt')
ACTOR_ENV = Path('/home/ubuntu/Projects/onix_isaac_lab/logs/rsl_rl/onix_rough/'
                 '2026-04-09_10-51-54/params/env.yaml')
CAP = {'gpu_device_total_mib': 6656, 'host_mem_available_min_kib': 2 * 1024 * 1024,
       'batch_storage_bytes': 800_000_000, 'wall_seconds': 600}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def container_stage(root):
    """Tie Docker staging to the full immutable host attempt, not its basename."""
    digest = hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:20]
    return f'/tmp/eatnav-motion-14-collection-{digest}'


def validate_output_identity(output, expected):
    output = Path(output)
    if (output/'failure.json').exists():
        raise ValueError('simulator diagnostic failure marker present')
    actual = json.loads((output/'input-manifest.json').read_text())
    for host_key, sim_key in (('seed', 'seed'), ('direction', 'direction'),
                              ('angle_deg', 'arc_angle_deg'),
                              ('condition', 'condition'), ('repeat', 'repeat'),
                              ('mode', 'mode'),
                              ('control_steps', 'control_steps'),
                              ('replan_every_steps', 'replan_every_steps'),
                              ('actor_env_sha256', 'actor_env_sha256'),
                              ('protocol_sha256', 'protocol_sha256'),
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


def verify_sim_seal(output, *, expected_control_steps=None, expected_replan_every_steps=None):
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
    if complete.get('raw_dem_sha256') != sha(output/'raw-dem.npz'):
        raise ValueError('raw DEM source SHA mismatch')
    if complete['status'] != 'complete' or complete['env_count'] != 4 or \
            complete['mode'] not in ('prepath_only', 'candidate'):
        raise ValueError('simulator completion contract')
    if complete['mode'] == 'prepath_only':
        if complete['control_steps'] != 0 or complete['trace_sha256'] or \
                (output/'path-activation.json').exists():
            raise ValueError('candidate Path published during pre-Path identity run')
    else:
        lengths = complete.get('trace_lengths')
        terminated = complete.get('terminated_envs')
        censored = complete.get('censored_envs')
        if not isinstance(lengths, list) or len(lengths) != 4 or \
                len(complete['trace_sha256']) != 4 or \
                any(not isinstance(length, int) or length <= 0 or length > complete['requested_control_steps']
                    for length in lengths) or len(set(lengths)) != 1:
            raise ValueError('variable-length candidate trace contract')
        if expected_control_steps is not None and \
                complete['requested_control_steps'] != expected_control_steps:
            raise ValueError('requested control horizon mismatch')
        if expected_replan_every_steps is not None and \
                complete.get('replan_every_steps') != expected_replan_every_steps:
            raise ValueError('replanning period mismatch')
        if not isinstance(terminated, list) or not isinstance(censored, list) or \
                set(terminated) & set(censored) or any(not 0 <= i < 4 for i in terminated + censored):
            raise ValueError('terminal/censor env identity mismatch')
        for index in range(4):
            path = output/f'trace-env-{index}.jsonl'
            if sha(path) != complete['trace_sha256'][index] or \
                    len(path.read_text().splitlines()) != lengths[index]:
                raise ValueError('trace SHA/length mismatch')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            terminal = rows[-1].get('terminal')
            if index in terminated and not (terminal and terminal.get('pre_reset_state')):
                raise ValueError('natural termination lacks reset-preceding state')
            if index in censored and not (terminal and terminal.get('termination_reason') ==
                                           'parallel_env_censored_by_other_env'):
                raise ValueError('parallel env censor reason missing')
        event_path = output/'long-horizon-event-labels.json'
        if not event_path.is_file():
            raise ValueError('long-horizon event label artifact missing')
        event_labels = json.loads(event_path.read_text())
        if len(event_labels.get('trials', [])) != 4 or any(
                event_labels['trials'][index].get('window_count') != lengths[index]
                for index in range(4)):
            raise ValueError('event label window/trace length mismatch')
    return {'simulator_seal_sha256': sha(output/'host-seal.json'),
            'file_count': len(actual),
            'trace_count': 0 if complete['mode'] == 'prepath_only' else 4,
            'terminated_envs': [] if complete['mode'] == 'prepath_only' else complete['terminated_envs'],
            'censored_envs': [] if complete['mode'] == 'prepath_only' else complete['censored_envs'],
            'mode': complete['mode']}


def execute(root, *, direction, angle, seed, terrain_type, terrain_level,
            condition, repeat, protocol, reference, prepath_only=False,
            rl_checkpoint=None, actor_env_yaml=None, control_steps=40,
            replan_every_steps=0):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError('fresh persistent Arc attempt required')
    if direction not in (-1, 1) or angle not in ANGLES or \
            condition not in CONDITIONS or repeat not in (0, 1):
        raise ValueError('MI-FI Path/condition/repeat not frozen')
    source = REPO/'scripts/eat_nav_motion_14_collection_sim.py'
    protocol = Path(protocol).resolve()
    rl_checkpoint = Path(rl_checkpoint or (RL if RL.is_file() else SHARED_RL)).resolve()
    actor_env_yaml = Path(actor_env_yaml or ACTOR_ENV).resolve()
    if not protocol.is_file():
        raise ValueError('frozen protocol unavailable')
    if not rl_checkpoint.is_file() or not actor_env_yaml.is_file():
        raise ValueError('RL checkpoint and matching Actor env YAML are required')
    if not 40 <= control_steps <= 780 or replan_every_steps < 0 or \
            (replan_every_steps and replan_every_steps % 5):
        raise ValueError('invalid control horizon or DEM-aligned replan period')
    eat_config = REPO/'config/eat_nav.yaml'
    reference = Path(reference).resolve() if reference else None
    if reference and (not (reference/'prepath.npz').is_file() or
                      not (reference/'prepath-identity.json').is_file()):
        raise ValueError('sealed prepath reference unavailable')
    root.mkdir(parents=True)
    stage = container_stage(root)
    manifest = {'schema': SCHEMA, 'direction': direction,
                'angle_deg': angle,
                'seed': seed, 'terrain_type': terrain_type, 'terrain_level': terrain_level,
                'condition': condition, 'repeat': repeat,
                'control_steps': control_steps,
                'replan_every_steps': replan_every_steps,
                'mode': 'prepath_only' if prepath_only else 'candidate',
                'protocol_sha256': sha(protocol),
                'source_sha256': sha(source), 'host_source_sha256': sha(__file__),
                'gate_source_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/mi_fi_v1.py'),
                'bank_source_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/bidirectional_bank.py'),
                'rl_checkpoint_sha256': sha(rl_checkpoint),
                'actor_env_sha256': sha(actor_env_yaml),
                'eat_config_sha256': sha(eat_config),
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
        run('docker', 'cp', str(rl_checkpoint), 'isaac-lab-onix:'+stage+'/config/rl_actor.pt')
        run('docker', 'cp', str(actor_env_yaml), 'isaac-lab-onix:'+stage+'/config/actor_env.yaml')
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
                   '--output', stage+'/output', '--rl-checkpoint', stage+'/config/rl_actor.pt',
                   '--actor-env-yaml', stage+'/config/actor_env.yaml',
                   '--eat-config', stage+'/config/eat_nav.yaml', '--arc-angle', str(angle),
                   '--direction', str(direction),
                   '--seed', str(seed), '--terrain-type', terrain_type,
                   '--terrain-level', str(terrain_level), '--condition', condition,
                   '--repeat', str(repeat), '--control-steps', str(control_steps),
                   '--replan-every-steps', str(replan_every_steps),
                   '--protocol-sha256', sha(protocol), '--headless']
        if prepath_only:
            command.append('--prepath-only')
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
                verification = verify_sim_seal(root/'sim-output',
                    expected_control_steps=0 if prepath_only else control_steps,
                    expected_replan_every_steps=replan_every_steps)
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
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--repeat', type=int, choices=(0, 1), required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--rl-checkpoint', type=Path)
    parser.add_argument('--actor-env-yaml', type=Path)
    parser.add_argument('--control-steps', type=int, default=40)
    parser.add_argument('--replan-every-steps', type=int, default=0)
    parser.add_argument('--prepath-only', action='store_true')
    parser.add_argument('--reference-prepath', type=Path)
    args = parser.parse_args()
    print(execute(args.root, direction=args.direction, angle=args.arc_angle, seed=args.seed,
                  terrain_type=args.terrain_type, terrain_level=args.terrain_level,
                  condition=args.condition, repeat=args.repeat, protocol=args.protocol,
                  reference=args.reference_prepath, prepath_only=args.prepath_only,
                  rl_checkpoint=args.rl_checkpoint, actor_env_yaml=args.actor_env_yaml,
                  control_steps=args.control_steps,
                  replan_every_steps=args.replan_every_steps))


if __name__ == '__main__':
    main()
