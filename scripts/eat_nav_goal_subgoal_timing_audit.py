"""Read-only frozen-input audit for the goal-subgoal timing rerun."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import time


REPO = pathlib.Path('/tmp/eatnav-vnt-goal-ood-simulator')
OLD = pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator')
OUT = pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood-timing-20261001')
DOC = REPO / 'docs/quality-identifiability-gate/candidate-coverage/goal-ood-simulator'


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def remote_digest(path: str) -> str:
    output = subprocess.check_output(['docker', 'exec', 'isaac-lab-onix', 'sha256sum', path], text=True)
    return output.split()[0]


def read(path: pathlib.Path):
    return json.loads(path.read_text())


def assert_sha(path: pathlib.Path, expected: str) -> str:
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'SHA mismatch: {path}: expected {expected}, actual {actual}')
    return actual


def main() -> None:
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
    if status.strip():
        raise RuntimeError('unexpected tracked/untracked worktree changes before T0: ' + status)
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if branch != 'exp/eatnav-goal-ood-simulator':
        raise ValueError(f'wrong branch: {branch}')
    freeze = read(DOC / 'pilot-input-freeze.json')
    revision = read(OLD / 'source-revision.json')
    parity = read(DOC / 'parity-smoke.json')
    source = REPO / 'scripts/eat_nav_goal_subgoal_sim.py'
    if assert_sha(source, revision['revised_sim_sha256']) != remote_digest(str(OLD / 'sim.py')):
        raise ValueError('container simulator source differs from latest assertion fix')
    if revision['previous_sim_sha256'] != parity['logging_sim_source_sha256']:
        raise ValueError('source revision chain differs')
    assert_sha(OLD / 'inputs/selected-states.json', freeze['selected_state_sha256'])
    assert_sha(DOC / 'run-identities.json', freeze['run_identities_sha256'])
    assert_sha(REPO / 'docs/quality-identifiability-gate/candidate-coverage/goal-ood/simulator-pilot-proposal.json', freeze['proposal_sha256'])
    assert_sha(REPO / 'docs/quality-identifiability-gate/candidate-coverage/goal-ood/goal-sweep-inputs.json', freeze['offline_goal_sweep_inputs_sha256'])
    assert_sha(REPO / 'docs/quality-identifiability-gate/candidate-coverage/goal-ood/artifact-sha-ledger.json', freeze['offline_ledger_sha256'])
    assert_sha(pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood/input-copy/manifest.json'), freeze['manifest_sha256'])
    assert_sha(pathlib.Path('/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt'), freeze['reference_checkpoint_sha256'])
    inputs = read(OLD / 'inputs/selected-states.json')['train']
    runs = read(DOC / 'run-identities.json')['runs']
    metadata = read(DOC / 'goal-metadata-sha.json')['runs']
    if len(inputs) != 4 or len(runs) != 44 or [x['order'] for x in runs] != list(range(44)):
        raise ValueError('frozen four states / 44 ordered runs are incomplete')
    if [x['kind'] for x in runs] != (['duplicate'] * 3 + ['candidate'] * 8) * 4:
        raise ValueError('duplicate and candidate order changed')
    for index, row in enumerate(inputs):
        frozen = freeze['states'][index]
        folder = OLD / 'inputs' / f'state-{index:03d}'
        if row['state_id'] != frozen['state_id'] or row['step'] + 1 != frozen['prefix_steps'] or row['episode_key'] != frozen['episode_key']:
            raise ValueError(f'state identity mismatch: {index}')
        for name, key in [('prefix-actions.npz', 'prefix_sha256'), ('suffix-actions-120.npz', 'suffix_sha256'),
                          ('logged-anchor.json', 'logged_anchor_sha256'), ('config-120.yaml', 'config_sha256')]:
            assert_sha(folder / name, frozen[key])
    for index, row in enumerate(runs):
        folder = OLD / 'inputs' / f"state-{row['state_index']:03d}"
        assert_sha(folder / f"{row['candidate_id']}-base-path.json", row['base_path_sha256'])
        assert_sha(folder / f"{row['run_id']}-goal-metadata.json", metadata[index]['metadata_sha256'])
        if row['state_id'] != inputs[row['state_index']]['state_id']:
            raise ValueError(f'run {index} points to another state')
    rl = remote_digest('/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt')
    if rl != read(DOC / 'source-provenance.json')['rl_checkpoint_sha256']:
        raise ValueError('RL checkpoint SHA differs')
    reference_config = remote_digest('/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml')
    prior_sources = read(DOC / 'source-provenance.json')['source_file_sha256']
    checked_sources = {
        'bridge': REPO / 'scripts/eat_nav_command_delivery_bridge.py',
        'exact_stamp_protocol': REPO / 'src/visualnav_transformer/eat_nav/exact_stamp_protocol.py',
        'runner': REPO / 'scripts/eat_nav_goal_subgoal_run.py',
        'exact_stamp_follower': pathlib.Path('/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon/exact-follower-build/simple_trajectory_follower_node'),
        'follower_config': pathlib.Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml'),
    }
    expected = {'bridge': 'bridge_source', 'exact_stamp_protocol': 'exact_clock_protocol',
                'runner': 'new_host_runner', 'exact_stamp_follower': 'exact_follower_binary',
                'follower_config': 'follower_config'}
    source_hashes = {}
    for key, path in checked_sources.items():
        source_hashes[key] = assert_sha(path, prior_sources[expected[key]])
    old_artifacts = [DOC / 'goal-subgoal-simulator-report.md', DOC / 'parity-smoke.json',
                     OLD / 'runs/train-003-anchor/simulator.log', OLD / 'runs/train-003-anchor-v2/simulator.log',
                     OLD / 'runs/train-003-anchor/final-status.json', OLD / 'runs/train-003-anchor-v2/final-status.json',
                     OLD / 'pilot-stop.done.json']
    processes = subprocess.check_output(['ps', '-eo', 'pid,ppid,etimes,args'], text=True)
    docker_processes = subprocess.check_output(['docker', 'exec', 'isaac-lab-onix', 'ps', '-eo', 'pid,ppid,etimes,args'], text=True)
    relevant = ('eat_nav_goal_subgoal', 'eat_nav_command_delivery_bridge', 'simple_trajectory_follower_node', 'rmw_zenohd', 'isaac-sim/python.sh')
    active = [line for line in processes.splitlines() if any(term in line for term in relevant)
              and 'ps -eo' not in line and 'eat_nav_goal_subgoal_timing_audit.py' not in line]
    active_docker = [line for line in docker_processes.splitlines() if any(term in line for term in relevant) and 'ps -eo' not in line]
    if active or active_docker:
        raise RuntimeError(f'existing pilot processes active: host={active}, docker={active_docker}')
    tmux = subprocess.run(['tmux', 'ls'], text=True, capture_output=True)
    payload = {
        'status': 'pass', 'checked_unix': time.time(), 'branch': branch, 'head': head,
        'worktree_status': status, 'manifest_sha256': freeze['manifest_sha256'],
        'split_hash': freeze['split_hash'], 'nomad_checkpoint_sha256': freeze['reference_checkpoint_sha256'],
        'rl_checkpoint_sha256': rl, 'reference_config_sha256': reference_config,
        'selected_states': [{'state_id': row['state_id'], 'terrain': row['terrain'],
                             'episode_key': row['episode_key'], 'global_prefix_steps': row['step'] + 1,
                             'episode_local_prefix_steps': row['local_step'],
                             'input_sha256': freeze['states'][index]}
                            for index, row in enumerate(inputs)],
        'goals': [{'run_id': row['run_id'], 'state_id': row['state_id'], 'goal_id': row['goal_id'],
                   'goal_class': row['goal_class'], 'goal_xy': row['goal_xy']}
                  for row in runs if row['kind'] == 'candidate'],
        'ordered_runs': runs, 'run_order_sha256': freeze['run_identities_sha256'],
        'sim_source_before_assertion_sha256': revision['previous_sim_sha256'],
        'sim_source_after_assertion_sha256': revision['revised_sim_sha256'],
        'source_sha256': source_hashes, 'prior_parity': parity,
        'old_artifact_sha256': {str(path): digest(path) for path in old_artifacts},
        'tmux_before': tmux.stdout if tmux.returncode == 0 else tmux.stderr,
        'docker_status': subprocess.check_output(['docker', 'ps', '--format', '{{.Names}} {{.Status}}'], text=True),
        'active_host_processes': active, 'active_docker_processes': active_docker,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    path = OUT / 'timing-source-provenance.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(f'T0_PASS {path} {digest(path)}')


if __name__ == '__main__':
    main()
