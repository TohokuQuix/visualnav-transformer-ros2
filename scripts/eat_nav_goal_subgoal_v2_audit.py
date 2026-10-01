"""Read-only provenance gate before the independent cost-aware v2 pilot."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from eat_nav_goal_subgoal_v2_select import REPO, DOC, OUT, MANIFEST, CONFIG, REF, sha, atomic

OLD = Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator')
TIMING = Path('/tmp/eatnav-candidate-coverage/goal-ood-resource-timing-20261001')
SOURCE_DOC = REPO / 'docs/quality-identifiability-gate/candidate-coverage/goal-ood'
EVAL_DOC = REPO / 'docs/quality-identifiability-gate/candidate-coverage/exact-stamp-long-horizon/source-provenance.json'


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, wanted, name):
    actual = sha(path)
    if actual != wanted:
        raise ValueError(f'{name} SHA mismatch: {actual} != {wanted}')
    return {'path': str(path), 'sha256': actual, 'bytes': Path(path).stat().st_size}


def output(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def process_rows(command):
    lines = output(command).splitlines()
    tokens = ('eat_nav_goal_subgoal_sim.py', 'sim-resource.py', 'eat_nav_command_delivery_bridge.py',
              'simple_trajectory_follower_node', 'eat_nav_goal_subgoal_resource_monitor.py', 'zenohd')
    return [line for line in lines if any(token in line for token in tokens)
            and 'ps -eo' not in line and 'sh -lc' not in line and 'bash -lc' not in line]


def main():
    config = __import__('yaml').safe_load(CONFIG.read_text())
    if OUT.exists():
        raise FileExistsError('v2 output root already exists; never overwrite evidence')
    branch = output(['git', 'branch', '--show-current']).strip()
    head = output(['git', 'rev-parse', 'HEAD']).strip()
    worktree_status = output(['git', 'status', '--porcelain'])
    if branch != 'exp/eatnav-goal-ood-simulator-v2' or worktree_status:
        raise ValueError('unexpected branch or uncommitted change before V0')
    old_status = read(OLD / 'status.json')
    timing_status = read(TIMING / 'status.json')
    old_runs = list((OLD / 'runs').glob('*/live-*.trace.json.gz')) if (OLD / 'runs').exists() else []
    if old_status.get('candidate_runs') != 0 or old_runs:
        raise ValueError('previous candidate outcome exists')
    if timing_status['stage'] != 'runtime_gate_fail' or timing_status['candidate_runs'] != 0:
        raise ValueError('resource timing result differs')
    tmux = subprocess.run(['tmux', 'ls'], capture_output=True, text=True)
    host_processes = process_rows(['ps', '-eo', 'pid,args'])
    docker_processes = process_rows(['docker', 'exec', 'isaac-lab-onix', 'ps', '-eo', 'pid,args'])
    if tmux.returncode == 0 or host_processes or docker_processes:
        raise RuntimeError('residual tmux/simulator/bridge/follower/router/monitor process')
    old_freeze = read(DOC / 'pilot-input-freeze.json')
    old_schedule = read(DOC / 'run-identities.json')['runs']
    old_states = read(OLD / 'inputs/selected-states.json')['train']
    if len(old_states) != 4 or len(old_schedule) != 44 or len([r for r in old_schedule if r['kind'] == 'candidate']) != 32:
        raise ValueError('old pilot identity count differs')
    support_path = SOURCE_DOC / 'goal-support-thresholds.json'
    old_source = read(DOC / 'source-provenance.json')['source_file_sha256']
    recent_source = read(TIMING / 'resource-probe-source.json')
    eval_source = read(EVAL_DOC)['files']
    paths = {
        'manifest': checked(MANIFEST, config['manifest_sha256'], 'manifest'),
        'complete_window_ids': checked(REF / 'complete-window-ids.jsonl.gz', config['complete_window_ids_sha256'], 'complete ids'),
        'reference_config': checked(REF / 'reference-eat.yaml', 'caa88735e765fc0358930dfa41579ecaac9eb96dc4052dc01a81390377279992', 'reference config'),
        'nomad_checkpoint': checked('/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt', config['reference_checkpoint_sha256'], 'NoMaD'),
        'simulator_source': checked(REPO / 'scripts/eat_nav_goal_subgoal_sim.py', recent_source['simulator_sha256'], 'simulator'),
        'runner_source': checked(REPO / 'scripts/eat_nav_goal_subgoal_run.py', recent_source['runner_sha256'], 'runner'),
        'bridge_source': checked(REPO / 'scripts/eat_nav_command_delivery_bridge.py', old_source['bridge_source'], 'bridge'),
        'exact_stamp_protocol': checked(REPO / 'src/visualnav_transformer/eat_nav/exact_stamp_protocol.py', old_source['exact_clock_protocol'], 'exact stamp'),
        'follower_binary': checked(eval_source['exact_binary']['path'], old_source['exact_follower_binary'], 'follower binary'),
        'pilot_follower_binary': checked('/tmp/eatnav-candidate-coverage/evaluator-readiness-recovery/follower-build/simple_trajectory_follower_node', old_source['pilot_follower_binary'], 'pilot follower binary'),
        'follower_cpp': checked('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/src/simple_trajectory_follower.cpp', old_source['production_follower_cpp'], 'follower source'),
        'follower_header': checked('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/include/simple_trajectory_follower/simple_trajectory_follower.hpp', old_source['production_follower_header'], 'follower header'),
        'follower_config': checked('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml', old_source['follower_config'], 'follower config'),
        'logging_parity': checked(TIMING / 'parity/parity-result.json', recent_source['parity_sha256'], 'parity'),
        'runtime_forecast': checked(TIMING / 'runtime-forecast.json', '7159bbf7d8769ae6625f82e808ed0d78a6087dede265ee5d36b453f23fdcf3e3', 'timing forecast'),
        'resource_gate_result': checked(TIMING / 'resource-gate-result.json', timing_status['gate_result_sha256'], 'timing gate'),
        'resource_config': checked(TIMING / 'resource-thresholds.json', config['resource_config_sha256'], 'resource config'),
        'goal_support_thresholds': {'path': str(support_path), 'sha256': sha(support_path), 'bytes': support_path.stat().st_size},
        'old_selected_states': checked(OLD / 'inputs/selected-states.json', old_freeze['selected_state_sha256'], 'old states'),
        'old_run_schedule': checked(DOC / 'run-identities.json', old_freeze['run_identities_sha256'], 'old schedule'),
    }
    for key in ('evaluator_GT_11', 'evaluator_GT_29', 'evaluator_GT_47'):
        item = eval_source[key]
        if Path(item['path']).exists():
            paths[key] = checked(item['path'], item['sha256'], key)
        else:
            digest, container_path = output(['docker', 'exec', 'isaac-lab-onix', 'sha256sum', item['path']]).split()
            if digest != item['sha256'] or container_path != item['path']:
                raise ValueError(f'{key} missing on host and Docker SHA mismatch')
            size = int(output(['docker', 'exec', 'isaac-lab-onix', 'stat', '-c', '%s', item['path']]).strip())
            paths[key] = {'container_path': container_path, 'sha256': digest, 'bytes': size,
                          'host_path_missing': True}
    rl_path = '/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt'
    rl_sha = output(['docker', 'exec', 'isaac-lab-onix', 'sha256sum', rl_path]).split()[0]
    if rl_sha != 'b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d':
        raise ValueError('RL checkpoint SHA drift')
    proposal = read(SOURCE_DOC / 'simulator-pilot-proposal.json')
    if [s['terrain'] for s in proposal['states']] != config['terrain_order']:
        raise ValueError('old proposal terrain order drift')
    OUT.mkdir(parents=True)
    provenance = {'status': 'pass', 'branch': branch, 'head': head, 'worktree': str(REPO),
                  'worktree_status': worktree_status, 'source_selection_config_sha256': sha(CONFIG),
                  'split_hash': old_freeze['split_hash'], 'files': paths,
                  'rl_checkpoint': {'container_path': rl_path, 'sha256': rl_sha},
                  'old_pilot': {'state_ids': [s['state_id'] for s in old_states], 'state_count': 4,
                                'goal_count': 16, 'path_count': 32, 'run_count': 44,
                                'candidate_outcome_count': len(old_runs), 'status': old_status},
                  'timing_status': timing_status,
                  'processes': {'tmux_returncode': tmux.returncode, 'tmux_output': tmux.stdout + tmux.stderr,
                                'host': host_processes, 'docker': docker_processes}}
    atomic(OUT / 'v2-source-provenance.json', provenance)
    atomic(OUT / 'v0.done.json', {'stage': 'V0_PASS', 'provenance_sha256': sha(OUT / 'v2-source-provenance.json')})
    atomic(OUT / 'status.json', {'stage': 'V0_PASS', 'candidate_runs': 0})
    print('V0_PASS', sha(OUT / 'v2-source-provenance.json'))


if __name__ == '__main__':
    main()
