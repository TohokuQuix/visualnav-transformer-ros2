"""Freeze an offline-only v4 protocol template without staging a rollout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = 'eatnav_p1_g3_path_crossover_v4_router_lifecycle'
PRIOR = ('batch-first-differential-0-period-0',
         'batch-first-differential-0-period-1',
         'batch-first-differential-0-period-2',
         'batch-first-differential-1-period-0')


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_id(row: dict) -> str:
    return f"batch-{row['anchor_id']}-{row['batch_index']}-period-{row['execution_index']}"


def make_manifest(repo: Path) -> dict:
    repo = Path(repo)
    parent_path = repo/'docs/eat-nav/p1-g3-crossover-v3-manifest.json'
    parent = json.loads(parent_path.read_text())
    if (len(parent['processes']) != 18 or
            [process_id(row) for row in parent['processes'][:4]] != list(PRIOR) or
            sha(repo/'scripts/eat_nav_p1_g3_crossover_v3_host.py') !=
            parent['g3_host_source_sha256'] or
            sha(repo/'scripts/eat_nav_p1_g3_crossover_v2_sim.py') !=
            parent['g3_simulator_source_sha256'] or
            sha(repo/'docs/eat-nav/p1-g3-crossover-v2-assignment.json') !=
            parent['assignment_sha256']):
        raise ValueError('frozen_v3_parent_drift')
    result = {key: value for key, value in parent.items() if key not in (
        'authorized_process_id', 'authorized_process_count',
        'pilot_authorized_max_processes', 'pilot_process_ids',
        'automatic_continuation_after_pilot', 'runtime_scope')}
    result.update({
        'host_protocol_version': SCHEMA,
        'parent_v3_manifest_sha256': sha(parent_path),
        'parent_v3_host_source_sha256': parent['g3_host_source_sha256'],
        'parent_v3_simulator_source_sha256': parent['g3_simulator_source_sha256'],
        'parent_v3_assignment_sha256': parent['assignment_sha256'],
        'g3_host_source_sha256': sha(repo/'scripts/eat_nav_p1_g3_crossover_v4_inner_host.py'),
        'v4_outer_host_source_sha256': sha(repo/'scripts/eat_nav_p1_g3_crossover_v4_outer.py'),
        'router_observer_source_sha256': sha(repo/'scripts/eat_nav_p1_g3_router_observer_v4.py'),
        'runtime_evidence_source_sha256': sha(repo/'scripts/eat_nav_p1_g3_runtime_evidence_v4.py'),
        'monitoring_contract_sha256': sha(repo/'docs/eat-nav/p1-g3-crossover-v4-monitoring-contract.json'),
        'preparation_source_sha256': sha(Path(__file__)),
        'runtime_scope': 'offline_preparation_only',
        'runtime_authorized': False,
        'authorized_process_ids': [],
        'authorized_process_count': 0,
        'automatic_continuation_after_pilot': False,
        'prior_process_ids': list(PRIOR),
        'prior_process_evidence_class': [
            'host_protocol_valid', 'host_protocol_valid',
            'offline_audited_conditional', 'protocol_invalid_not_in_dataset'],
        'remaining_process_ids': [process_id(row) for row in parent['processes'][4:]],
        'resume_blocker': {
            'first_remaining_process': 'batch-first-differential-1-period-1',
            'required_predecessor': 'batch-first-differential-1-period-0',
            'predecessor_status': 'protocol_invalid',
            'gate': 'frozen_host_requires_previous_and_period0_protocol_valid_seals',
            'disposition': 'no_automatic_continuation_or_gate_relaxation',
        },
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result = make_manifest(repo)
    if args.output.exists():
        raise FileExistsError('v4 manifest already frozen')
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(args.output, sha(args.output), len(result['remaining_process_ids']))


if __name__ == '__main__':
    main()
