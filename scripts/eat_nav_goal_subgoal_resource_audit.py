"""Freeze resource-aware timing inputs before changing the external monitor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/tmp/eatnav-vnt-goal-ood-simulator')
OLD = Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator')
PREV = Path('/tmp/eatnav-candidate-coverage/goal-ood-timing-20261001')
OUT = Path('/tmp/eatnav-candidate-coverage/goal-ood-resource-timing-20261001')
DOC = REPO / 'docs/quality-identifiability-gate/candidate-coverage/goal-ood-simulator'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def docker(*args):
    return run('docker', 'exec', 'isaac-lab-onix', *args)


def keyed(text):
    return {line.split()[0]: int(line.split()[1]) for line in text.splitlines() if len(line.split()) >= 2}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    status = run('git', 'status', '--porcelain')
    if status:
        raise RuntimeError(f'worktree not clean at R0: {status}')
    branch = run('git', 'branch', '--show-current')
    head = run('git', 'rev-parse', 'HEAD')
    if branch != 'exp/eatnav-goal-ood-simulator':
        raise RuntimeError(f'unexpected branch {branch}')
    prior = json.loads((PREV / 'timing-source-provenance.json').read_text())
    prev_source = json.loads((PREV / 'timing-probe-source.json').read_text())
    parity = json.loads((PREV / 'parity/parity-result.json').read_text())
    freeze = json.loads((DOC / 'pilot-input-freeze.json').read_text())
    states = json.loads((OLD / 'inputs/selected-states.json').read_text())['train']
    runs = json.loads((DOC / 'run-identities.json').read_text())['runs']
    source = REPO / 'scripts/eat_nav_goal_subgoal_sim.py'
    if sha(source) != prev_source['source_sha256']:
        raise ValueError('unexpected simulator source SHA change')
    if sha(OLD / 'inputs/selected-states.json') != freeze['selected_state_sha256'] or sha(DOC / 'run-identities.json') != freeze['run_identities_sha256']:
        raise ValueError('fixed input/run order SHA mismatch')
    if len(states) != 4 or len(runs) != 44 or [x['order'] for x in runs] != list(range(44)):
        raise ValueError('state/run identity or order changed')
    for row in runs:
        path = OLD / 'inputs' / f"state-{row['state_index']:03d}" / f"{row['candidate_id']}-base-path.json"
        if sha(path) != row['base_path_sha256']:
            raise ValueError(f'Path SHA mismatch {row["run_id"]}')
    checks = {
        'bridge': (REPO / 'scripts/eat_nav_command_delivery_bridge.py', prior['source_sha256']['bridge']),
        'exact_stamp_protocol': (REPO / 'src/visualnav_transformer/eat_nav/exact_stamp_protocol.py', prior['source_sha256']['exact_stamp_protocol']),
        'exact_follower_binary': (Path('/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon/exact-follower-build/simple_trajectory_follower_node'), prior['source_sha256']['exact_stamp_follower']),
        'follower_config': (Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml'), prior['source_sha256']['follower_config']),
    }
    for name, (path, expected) in checks.items():
        if sha(path) != expected:
            raise ValueError(f'{name} SHA changed')
    if sha(Path('/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt')) != freeze['reference_checkpoint_sha256']:
        raise ValueError('NoMaD checkpoint SHA changed')
    if sha(Path('/tmp/eatnav-candidate-coverage/goal-ood/input-copy/manifest.json')) != freeze['manifest_sha256']:
        raise ValueError('manifest SHA changed')
    rl_sha = docker('sha256sum', '/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt').split()[0]
    if rl_sha != prior['rl_checkpoint_sha256']:
        raise ValueError('RL checkpoint SHA changed')
    host_ps = run('ps', '-eo', 'pid,args')
    docker_ps = docker('ps', '-eo', 'pid,args')
    keywords = ('sim-timing.py', 'sim-parity.py', 'eat_nav_command_delivery_bridge.py',
                'simple_trajectory_follower_node', 'rmw_zenohd', 'eat_nav_goal_subgoal_timing_monitor.py')
    active_host = [line for line in host_ps.splitlines() if any(word in line for word in keywords)
                   and 'ps -eo' not in line and 'resource_audit.py' not in line]
    active_docker = [line for line in docker_ps.splitlines() if any(word in line for word in keywords)
                     and 'ps -eo' not in line]
    if active_host or active_docker:
        raise RuntimeError(f'residual processes: {active_host} {active_docker}')
    import numpy
    if Path(sys.executable) != Path('/usr/bin/python3'):
        raise ValueError('use the fixed /usr/bin/python3 NumPy environment')
    memory_stat = keyed(docker('cat', '/sys/fs/cgroup/memory.stat'))
    memory_events = keyed(docker('cat', '/sys/fs/cgroup/memory.events'))
    meminfo = keyed(Path('/proc/meminfo').read_text())
    tmux = subprocess.run(['tmux', 'ls'], text=True, capture_output=True)
    result = {
        'status': 'pass', 'captured_unix': time.time(), 'branch': branch, 'head': head,
        'worktree_status': status, 'previous_t0_sha256': sha(PREV / 'timing-source-provenance.json'),
        'previous_t1_parity_sha256': sha(PREV / 'parity/parity-result.json'),
        'previous_t1_trace_sha256': parity['new_trace_sha256'],
        'assertion_fix_source_sha256': prior['sim_source_after_assertion_sha256'],
        'current_simulator_source_sha256': sha(source),
        'previous_heartbeat_sha256': sha(PREV / 'runs/train-003-timing-anchor/timing-heartbeat.json'),
        'previous_failure_marker_sha256': sha(PREV / 'runs/train-003-timing-anchor/timing-stall.failure.json'),
        'nomad_checkpoint_sha256': freeze['reference_checkpoint_sha256'], 'rl_checkpoint_sha256': rl_sha,
        'manifest_sha256': freeze['manifest_sha256'], 'split_hash': freeze['split_hash'],
        'fixed_input_freeze_sha256': sha(DOC / 'pilot-input-freeze.json'),
        'fixed_run_order_sha256': sha(DOC / 'run-identities.json'),
        'selected_states': states, 'ordered_runs': runs,
        'train003': {'state_id': states[0]['state_id'], 'episode_key': states[0]['episode_key'],
                     'global_prefix_steps': states[0]['step'] + 1, 'episode_local_prefix_steps': states[0]['local_step']},
        'source_sha256': {name: sha(path) for name, (path, _) in checks.items()},
        'old_heartbeat_resource': json.loads((PREV / 'runs/train-003-timing-anchor/timing-heartbeat.json').read_text()),
        'old_failure_reason': json.loads((PREV / 'runs/train-003-timing-anchor/timing-stall.failure.json').read_text())['reason'],
        'memory_current_bytes': int(docker('cat', '/sys/fs/cgroup/memory.current')),
        'memory_peak_bytes': int(docker('cat', '/sys/fs/cgroup/memory.peak')),
        'memory_max': docker('cat', '/sys/fs/cgroup/memory.max'),
        'memory_events': memory_events, 'memory_stat': memory_stat,
        'host_meminfo_kib': meminfo, 'python_executable': sys.executable,
        'numpy_version': numpy.__version__, 'active_host_processes': active_host,
        'active_docker_processes': active_docker,
        'tmux_before': tmux.stdout if tmux.returncode == 0 else tmux.stderr,
        'docker_ps': run('docker', 'ps', '--format', '{{.Names}} {{.Status}}'),
        'previous_6_3_hour_forecast_used_for_gate': False,
    }
    OUT.mkdir(parents=True)
    path = OUT / 'resource-probe-provenance.json'
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print('R0_PASS', path, sha(path))


if __name__ == '__main__':
    main()
