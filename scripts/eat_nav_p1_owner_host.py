"""Host-side P1 startup ownership handshake and one-shot resource gates."""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import subprocess
import time

from eat_nav_g0e_gpu_probe import Nvml, proc_table, proc_info
from eat_nav_p1_gpu_attribution import capture_live
from eat_nav_p1_handshake_prepare import WORKTREE, R5, atomic, sha
from eat_nav_radial_anchor_fixture import verify_source


CONTAINER = 'isaac-lab-onix'
ROOT = Path('/tmp/eatnav-p1-handshake-20261008-r1')
PREVIOUS = Path('/tmp/eatnav-p1-resource-rollout-20261008-attempt2/attribution-smoke-manifest.json')


def resolve_owner(table: dict, ready: dict, container_id: str) -> dict:
    token = ready.get('run_token')
    matches = []
    if not isinstance(token, str) or len(token) != 64 or not ready.get('nspid'):
        raise ValueError('owner_ready_identity_invalid')
    for row in table.values():
        chain = row.get('namespace_pid_chain') or []
        argv = row.get('command_line', '').split()
        if (container_id in row.get('cgroup', '') and row.get('container_pid') == ready.get('container_pid')
                and chain and chain[-1] == ready['nspid'][-1]
                and row.get('start_ticks') == ready.get('start_ticks')
                and '--run-token' in argv and argv[argv.index('--run-token') + 1:argv.index('--run-token') + 2] == [token]):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError('owner_pid_mapping_missing_or_ambiguous')
    return {'host_pid': matches[0]['host_pid'], 'start_ticks': matches[0]['start_ticks']}


def exact_owned_alive(table: dict, owner: dict) -> bool:
    row = table.get(owner['host_pid'])
    return bool(row and row.get('start_ticks') == owner['start_ticks'])


def assert_pre_authorization_gpu_clear(sample: dict) -> None:
    if sample.get('owned_gpu_pids'):
        raise ValueError('gpu_before_authorization')
    if not sample.get('pass'):
        raise ValueError('preauthorization_gpu_gate:' + ','.join(sample.get('failures', [])))


def initial_graphics_identities(gpu_rows: list[dict], table: dict) -> set[tuple[int, int]]:
    return {(row['host_pid'], table[row['host_pid']]['start_ticks'])
            for row in gpu_rows if row.get('kind') == 'graphics'
            and row['host_pid'] in table and isinstance(table[row['host_pid']].get('start_ticks'), int)}


def validate_smoke_artifacts(output: Path, expected_pose_count: int) -> dict:
    finished = json.loads((output / 'sim-finished.json').read_text())
    if finished.get('physics_steps') != 200 or finished.get('pose_count') != expected_pose_count:
        raise ValueError('smoke_completion_invalid')
    if finished.get('physical_termination_pose_ids'):
        raise ValueError('smoke_physical_termination')
    pose_files = sorted((output / 'poses').glob('*.json'))
    if len(pose_files) != expected_pose_count:
        raise ValueError('smoke_pose_count_invalid')
    for pose_file in pose_files:
        pose = pose_file.stem
        trace_file = output / 'trace' / (pose + '.jsonl')
        trace = [json.loads(line) for line in trace_file.read_text().splitlines()]
        if len(trace) != 200 or [row['physics_step'] for row in trace] != list(range(1, 201)):
            raise ValueError('smoke_trace_incomplete:' + pose)
        if any(row.get('reset_counter') != 0 or row.get('post_reset_state')
               or not row['contact'].get('bounds_valid') or not row['contact'].get('finite')
               or row['contact'].get('buffer_full_or_truncation_suspected') for row in trace):
            raise ValueError('smoke_contact_or_reset_invalid:' + pose)
        for index, step in enumerate((50, 100, 150, 200)):
            dem = json.loads((output / 'dem' / pose / f'frame-{index:02d}.json').read_text())
            if dem.get('physics_step') != step or not dem.get('finite'):
                raise ValueError('smoke_DEM_invalid:' + pose)
        for phase in ('prephysics', 'settled_step_200'):
            actor = json.loads((output / 'actor' / pose / (phase + '.json')).read_text())
            if actor.get('shape') != [22] or not actor.get('finite'):
                raise ValueError('smoke_Actor_invalid:' + pose)
    return {'physics_steps': 200, 'pose_count': expected_pose_count,
            'DEM_frame_count': expected_pose_count * 4, 'Actor_record_count': expected_pose_count * 2}


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(['docker', *map(str, args)], capture_output=True, text=True, check=check)


def remote_json(path: Path) -> dict | None:
    result = docker('exec', CONTAINER, 'cat', str(path), check=False)
    return json.loads(result.stdout) if result.returncode == 0 else None


def remote_sha(path: Path) -> str | None:
    result = docker('exec', CONTAINER, 'sha256sum', str(path), check=False)
    return result.stdout.split()[0] if result.returncode == 0 else None


def remote_atomic(local: Path, remote: Path) -> None:
    temporary = Path(str(remote) + '.host-tmp')
    docker('cp', str(local), f'{CONTAINER}:{temporary}')
    docker('exec', CONTAINER, 'mv', str(temporary), str(remote))
    if remote_sha(remote) != sha(local):
        raise RuntimeError('remote_atomic_SHA_mismatch')


def sample_gpu(nvml: Nvml, owned: dict, baseline: set, contract: dict, container_id: str,
               expected_alive: bool, path: Path, phase: str, cap: int = 7680,
               resource_root: Path | None = None) -> dict:
    sample = capture_live(nvml, owned, baseline, contract, container_id=container_id,
                          expected_simulator_alive=expected_alive, device_cap_mib=cap)
    sample['phase'] = phase
    if resource_root is not None:
        stat = docker('exec', CONTAINER, 'cat', '/sys/fs/cgroup/memory.stat').stdout
        memory = {parts[0]: int(parts[1]) for line in stat.splitlines()
                  if len(parts := line.split()) == 2}
        sample['container_anon_shmem_bytes'] = memory.get('anon', 0) + memory.get('shmem', 0)
        memline = next(line for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:'))
        sample['host_available_bytes'] = int(memline.split()[1]) * 1024
        remote = docker('exec', CONTAINER, 'du', '-sb', str(resource_root), check=False)
        remote_bytes = int(remote.stdout.split()[0]) if remote.returncode == 0 else 0
        host_bytes = sum(file.stat().st_size for file in resource_root.rglob('*') if file.is_file())
        sample['storage_bytes'] = remote_bytes + host_bytes
        if sample['container_anon_shmem_bytes'] > contract['container_anon_plus_shmem_cap_bytes']:
            sample['failures'].append('anon_shmem_cap')
        if sample['host_available_bytes'] < contract['minimum_host_available_bytes']:
            sample['failures'].append('host_memory_floor')
        if sample['storage_bytes'] > contract['storage_cap_bytes']:
            sample['failures'].append('storage_cap')
        sample['failures'] = sorted(set(sample['failures']))
        sample['pass'] = not sample['failures']
    with path.open('a') as stream:
        stream.write(json.dumps(sample, sort_keys=True, allow_nan=False) + '\n')
    return sample


def no_prior_owner_live(root: Path) -> None:
    table = proc_table()
    for marker in root.glob('*/coord/registered-owner.json'):
        if exact_owned_alive(table, json.loads(marker.read_text())):
            raise RuntimeError('prior_owned_pid_still_live:' + str(marker))


def build_wrapper_command(original: list[str], wrapper_path: str,
                          wrapper_args: list[str], mode: str) -> list[str]:
    if original[:2] != ['docker', 'exec'] or original[5] != '/isaac-sim/python.sh':
        raise ValueError('frozen_launcher_prefix_invalid')
    if mode not in ('preflight', 'smoke'):
        raise ValueError('invalid_mode')
    command = original[:6] + [wrapper_path] + wrapper_args
    if mode == 'smoke':
        command += ['--'] + original[7:]
    return command


def one_run(root: Path, mode: str) -> dict:
    if mode not in ('preflight', 'smoke'):
        raise ValueError('invalid_mode')
    no_prior_owner_live(root)
    target = root / mode
    if target.exists():
        raise ValueError('one_shot_target_exists')
    previous = json.loads(PREVIOUS.read_text())
    source = Path(previous['source_shard'])
    freeze, prelaunch = verify_source(source)
    if previous['selected_prior_subroot'] != '/tmp/eatnav-radial-anchor-contract-20261008-r2/shard-00':
        raise RuntimeError('highest_GPU_shard_selection_drift')
    original = list(prelaunch['command'])
    original[original.index('--output') + 1] = str(target / 'run')
    if mode == 'smoke' and ('--mode' not in original or original[original.index('--mode') + 1] != 'full'):
        raise RuntimeError('frozen_candidate_free_sim_mode_drift')
    inspect = docker('inspect', '-f', '{{.Id}} {{.State.Running}}', CONTAINER).stdout.split()
    if len(inspect) != 2 or inspect[1] != 'true':
        raise RuntimeError('container_not_running')
    container_id = inspect[0]
    contract = json.loads((R5 / 'p1-resource-contract-v2.json').read_text())
    if sha(R5 / 'p1-resource-contract-v2.json') != previous['resource_contract_sha256']:
        raise RuntimeError('resource_contract_drift')
    if contract['gpu_device_total_hard_cap_mib'] != 7680:
        raise RuntimeError('resource_cap_drift')
    wrapper = WORKTREE / 'scripts/eat_nav_p1_owner_wrapper.py'
    target.mkdir(parents=True)
    coord = target / 'coord'
    coord.mkdir()
    docker('exec', CONTAINER, 'mkdir', '-p', str(coord), str(root / 'inputs'))
    remote_wrapper = root / 'inputs/eat_nav_p1_owner_wrapper.py'
    if remote_sha(remote_wrapper) is None:
        docker('cp', str(wrapper), f'{CONTAINER}:{remote_wrapper}')
    if remote_sha(remote_wrapper) != sha(wrapper):
        raise RuntimeError('wrapper_remote_SHA_mismatch')
    token = secrets.token_hex(32)
    output = target / 'run'
    intent = {'schema': 'eatnav_p1_owner_v1', 'run_token': token, 'container_id': container_id,
              'output_root': str(output), 'source_sha256': freeze['sim_source_sha256'],
              'config_sha256': prelaunch['freeze_sha256'], 'mode': mode,
              'wrapper_sha256': sha(wrapper), 'resource_contract_sha256': sha(R5 / 'p1-resource-contract-v2.json')}
    atomic(coord / 'launch-intent.json', intent)
    remote_atomic(coord / 'launch-intent.json', coord / 'launch-intent.json')
    nvml = Nvml()
    initial_devices, initial_rows = nvml.sample()
    initial_table = proc_table()
    baseline = initial_graphics_identities(initial_rows, initial_table)
    atomic(target / 'prebaseline-process-table.json', initial_table)
    atomic(target / 'prebaseline-gpu-rows.json', {'devices': initial_devices, 'rows': initial_rows,
           'known_external_graphics_identities': sorted(baseline)})
    baseline_sample = sample_gpu(nvml, {}, baseline, contract, container_id, False,
                                 target / 'gpu-brackets.jsonl', 'baseline', cap=1024,
                                 resource_root=root)
    if not baseline_sample['pass']:
        raise RuntimeError('prelaunch_GPU_baseline_invalid:' + ','.join(baseline_sample['failures']))
    baseline = initial_graphics_identities(baseline_sample['raw_nvml_rows'],
                                           baseline_sample['process_table_before']) & baseline
    wrapper_args = ['--coord', str(coord), '--run-token', token, '--container-id', container_id,
                    '--source', original[6], '--source-sha', freeze['sim_source_sha256'],
                    '--config', original[original.index('--freeze') + 1], '--config-sha', prelaunch['freeze_sha256'],
                    '--output', str(output), '--mode', mode]
    # Preserve docker exec -e environment and the original /isaac-sim/python.sh.
    wrapper_cmd = build_wrapper_command(original, str(remote_wrapper), wrapper_args, mode)
    atomic(target / 'wrapper-invocation.json', {'command': wrapper_cmd, 'source_original': original,
           'wrapper_sha256': sha(wrapper), 'run_token': token})
    client = None
    owner = None
    authorized = False
    started = time.monotonic()
    failure = None
    with (target / 'runner.log').open('wb') as log:
        try:
            client = subprocess.Popen(wrapper_cmd, stdout=log, stderr=subprocess.STDOUT)
            while time.monotonic() - started < 45:
                ready = remote_json(coord / 'owner-ready.json')
                if ready is not None:
                    break
                if client.poll() is not None:
                    raise RuntimeError('wrapper_exit_before_owner_ready')
                time.sleep(0.1)
            else:
                raise RuntimeError('owner_ready_timeout')
            if (ready.get('run_token') != token or ready.get('container_id') != container_id
                    or ready.get('output_root') != str(output)
                    or ready.get('source_sha256') != freeze['sim_source_sha256']
                    or ready.get('config_sha256') != prelaunch['freeze_sha256']):
                raise RuntimeError('owner_ready_frozen_identity_mismatch')
            table = proc_table()
            atomic(target / 'owner-resolution-process-table.json', table)
            owner = resolve_owner(table, ready, container_id)
            atomic(coord / 'registered-owner.json', owner)
            sample = sample_gpu(nvml, {'simulator': owner}, baseline, contract, container_id, True,
                                target / 'gpu-brackets.jsonl', 'preauthorization', resource_root=root)
            assert_pre_authorization_gpu_clear(sample)
            atomic(coord / 'monitor-ready.json', {'run_token': token, 'owner': owner,
                   'bracket_monotonic_ns': sample['monotonic_ns'], 'sample_pass': True})
            authorization = {key: ready[key] for key in ('schema', 'run_token', 'container_pid',
                              'start_ticks', 'output_root', 'source_sha256', 'config_sha256')}
            authorization.update({'host_pid': owner['host_pid'], 'monitor_ready': True,
                                  'bracket_monotonic_ns': sample['monotonic_ns']})
            atomic(coord / 'owner-authorized.json', authorization)
            remote_atomic(coord / 'owner-authorized.json', coord / 'owner-authorized.json')
            authorized = True
            while client.poll() is None:
                sample = sample_gpu(nvml, {'simulator': owner}, baseline, contract, container_id, True,
                                    target / 'gpu-brackets.jsonl', 'authorized', resource_root=root)
                if not sample['pass']:
                    raise RuntimeError('resource_attribution_gate:' + ','.join(sample['failures']))
                if time.monotonic() - started > contract['per_process_timeout_sec']:
                    raise RuntimeError('per_process_timeout')
                if mode == 'smoke':
                    heartbeat = remote_json(output / 'heartbeat.json')
                    if heartbeat is not None and (time.monotonic_ns() - heartbeat['monotonic_ns']) / 1e9 > contract['heartbeat_stale_sec']:
                        raise RuntimeError('heartbeat_stale')
                time.sleep(1)
            sample_gpu(nvml, {}, baseline, contract, container_id, False,
                       target / 'gpu-brackets.jsonl', 'shutdown', resource_root=root)
            docker('cp', f'{CONTAINER}:{coord}/.', str(coord))
            if output.exists():
                raise RuntimeError('host_output_preexisting')
            if mode == 'smoke':
                output.mkdir()
                docker('cp', f'{CONTAINER}:{output}/.', str(output))
            completion = json.loads((coord / 'completion.json').read_text()) if mode == 'preflight' else json.loads((output / 'sim-finished.json').read_text())
            if client.returncode != 0 or completion.get('physics_steps') != (0 if mode == 'preflight' else 200):
                raise RuntimeError('native_exit_or_completion_invalid')
            if mode == 'smoke' and completion.get('physical_termination_pose_ids'):
                raise RuntimeError('physical_termination_present')
            if mode == 'smoke':
                artifact_check = validate_smoke_artifacts(output, expected_pose_count=4)
                atomic(target / 'artifact-completeness.json', artifact_check)
            if exact_owned_alive(proc_table(), owner):
                raise RuntimeError('owned_residual_after_exit')
            if mode == 'smoke':
                ledger = []
                for path in sorted(output.rglob('*')):
                    if path.is_file():
                        if remote_sha(path) != sha(path):
                            raise RuntimeError('host_container_SHA_mismatch:' + str(path))
                        ledger.append({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size})
                atomic(target / 'raw-sha-ledger.json', {'files': ledger})
            result = {'decision': 'p1_startup_preflight_pass' if mode == 'preflight' else 'p1_resource_attribution_pass',
                      'owner': owner, 'run_token': token, 'native_exit_code': client.returncode,
                      'completion_sha256': sha(coord / 'completion.json' if mode == 'preflight' else output / 'sim-finished.json'),
                      'owned_residual_count': 0, 'authorized': authorized}
            atomic(target / 'result.json', result)
            return result
        except BaseException as exc:
            failure = {'decision': 'startup_ownership_handshake_fail' if mode == 'preflight' else 'p1_resource_attribution_fail',
                       'first_failing_signal': str(exc), 'authorized': authorized,
                       'owner': owner, 'run_token': token}
            atomic(target / 'result.json', failure)
            raise
        finally:
            if client is not None and client.poll() is None:
                # Only terminate the exact registered PID. A changed start tick is
                # a different process and must never receive a signal.
                if owner and exact_owned_alive(proc_table(), owner):
                    os.kill(owner['host_pid'], 15)
                    for _ in range(40):
                        if not exact_owned_alive(proc_table(), owner):
                            break
                        time.sleep(0.25)
                client.terminate()
                try:
                    client.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    client.kill()
                    client.wait(timeout=10)
            atomic(target / 'cleanup.json', {'owner': owner,
                   'owner_alive': bool(owner and exact_owned_alive(proc_table(), owner)),
                   'client_exit': None if client is None else client.poll(), 'authorized': authorized,
                   'failure': failure})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--mode', choices=('preflight', 'smoke'), required=True)
    args = parser.parse_args()
    print(json.dumps(one_run(args.output_root, args.mode), sort_keys=True))
