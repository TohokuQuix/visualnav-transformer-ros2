"""Sealed, resumable MI-A v3 multi-seed schedule; no historical evidence writes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from visualnav_transformer.eat_nav.mi_a_v3 import SCHEMA, validate_evaluation_schedule
try:
    from .eat_nav_mi_fi_v1_host import execute, sha, RL, verify_sim_seal
except ImportError:  # direct script execution
    from eat_nav_mi_fi_v1_host import execute, sha, RL, verify_sim_seal


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO/'docs/eat-nav/motion-predictor-vnext/mi_a_v3_assignment_manifest.json'
PROTOCOL = REPO/'docs/eat-nav/motion-predictor-vnext/mi_a_v3_protocol.md'
EVIDENCE_PARENT = Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence')
SOURCES = {
    'runner': 'scripts/eat_nav_mi_a_v3_run.py',
    'analyzer': 'scripts/eat_nav_mi_a_v3_analyze.py',
    'mode': 'src/visualnav_transformer/eat_nav/mi_a_v3.py',
    'simulator': 'scripts/eat_nav_mi_fi_v1_sim.py',
    'host': 'scripts/eat_nav_mi_fi_v1_host.py',
    'gate': 'src/visualnav_transformer/eat_nav/mi_fi_v1.py',
    'bank': 'src/visualnav_transformer/eat_nav/bidirectional_bank.py',
    'reset': 'src/visualnav_transformer/eat_nav/demo_reset_policy.py',
    'demo': 'scripts/eat_nav_demo_isaac.py',
    'bidirectional_collection_sim': 'scripts/eat_nav_bidirectional_collect_sim.py',
    'bidirectional_collection_host': 'scripts/eat_nav_bidirectional_collect_host.py',
    'arc_collection_sim': 'scripts/eat_nav_motion_collect_sim.py',
    'arc_collection_host': 'scripts/eat_nav_motion_collect_host.py',
    'eat_config': 'config/eat_nav.yaml',
}


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_frozen() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest['schema'] != SCHEMA:
        raise ValueError('schema mismatch')
    validate_evaluation_schedule(manifest['assignment'], manifest['state_groups'])
    for name, relative in SOURCES.items():
        if sha(REPO/relative) != manifest['source_sha256'][name]:
            raise ValueError(f'frozen source mismatch: {name}')
    if sha(PROTOCOL) != manifest['protocol_sha256'] or \
            sha(RL) != manifest['checkpoint_sha256']:
        raise ValueError('protocol/checkpoint mismatch')
    if manifest['num_processes'] != len(manifest['assignment']) or \
            manifest['independent_seed_count'] != len(manifest['state_groups']):
        raise ValueError('frozen counts mismatch')
    return manifest


def row_name(row: dict) -> str:
    suffix = f"prepath-{row['repeat_index']}" if row['stage'] == 'prepath' else row['candidate_id']
    return f"{row['index']:03d}-{row['state_group_id']}-{row['flipper_initialization_mode']}-{suffix}"


def verify_outer(attempt: Path) -> None:
    ledger = attempt/'outer-sha-ledger.json'
    seal = json.loads((attempt/'outer-seal.json').read_text())
    expected = json.loads(ledger.read_text())
    actual = {str(path.relative_to(attempt)): sha(path) for path in attempt.rglob('*')
              if path.is_file() and path.name not in ('outer-sha-ledger.json', 'outer-seal.json')}
    if sha(ledger) != seal['ledger_sha256'] or expected != actual or \
            len(actual) != seal['file_count']:
        raise ValueError(f'outer seal mismatch: {attempt.name}')


def verify_attempt(attempt: Path, row: dict) -> None:
    if not json.loads((attempt/'decision.json').read_text())['pass']:
        raise ValueError(f'host-invalid {attempt.name}')
    verify_outer(attempt)
    verify_sim_seal(attempt/'sim-output')
    host = json.loads((attempt/'manifest.json').read_text())
    for key, expected in (
        ('seed', row['terrain_seed']), ('terrain_type', row['terrain_type']),
        ('terrain_level', row['terrain_level']),
        ('condition', row['flipper_initialization_mode']),
        ('repeat', row['repeat_index']),
        ('mode', 'prepath_only' if row['stage'] == 'prepath' else 'candidate'),
        ('protocol_sha256', sha(PROTOCOL)),
    ):
        if host[key] != expected:
            raise ValueError(f'attempt identity mismatch: {key}')
    gate = json.loads((attempt/'sim-output'/'prepath-pair-gate.json').read_text())
    if gate['pass'] is not True:
        raise ValueError(f'Gate P failed: {attempt.name}')


def run(root: Path, phase: str) -> None:
    manifest = load_frozen()
    root = Path(root).absolute()
    if root.parent != EVIDENCE_PARENT or root.is_symlink():
        raise ValueError('new persistent evidence root required')
    if phase == 'smoke':
        if root.exists():
            raise FileExistsError('smoke requires fresh root')
        root.mkdir()
        atomic_json(root/'frozen-provenance.json', {
            'schema': SCHEMA, 'manifest_sha256': sha(MANIFEST),
            'protocol_sha256': sha(PROTOCOL), 'started_ns': time.time_ns(),
            'smoke_state_groups': manifest['smoke_state_groups']})
    else:
        if not root.is_dir():
            raise ValueError('formal continuation requires smoke root')
        provenance = json.loads((root/'frozen-provenance.json').read_text())
        if provenance['manifest_sha256'] != sha(MANIFEST):
            raise ValueError('manifest changed after smoke')
        smoke = json.loads((root/'mi_a_v3_smoke_test_results.json').read_text())
        if smoke.get('gate') != 'pass':
            raise ValueError('smoke gate not passed')
    rows = manifest['assignment']
    selected = [r for r in rows if phase == 'formal' or
                r['state_group_id'] in manifest['smoke_state_groups']]
    completed = 0
    for row in selected:
        attempt = root/row_name(row)
        if attempt.exists():
            verify_attempt(attempt, row)
            completed += 1
            continue
        condition = row['flipper_initialization_mode']
        group = row['state_group_id']
        baseline = next(r for r in rows if r['state_group_id'] == group and
                        r['flipper_initialization_mode'] == condition and
                        r['stage'] == 'prepath' and r['repeat_index'] == 0)
        reference = root/row_name(baseline)/'sim-output'
        role = row['candidate_id'] or 'forward_direct'
        direction = 1 if role.startswith('forward') else -1
        angle = 0 if role.endswith('direct') else int(role.rsplit('_', 1)[1])
        atomic_json(root/'status.json', {'phase': phase, 'state': 'running',
                    'completed': completed, 'selected': len(selected), 'current': row_name(row)})
        try:
            execute(attempt, direction=direction, angle=angle,
                    seed=row['terrain_seed'], terrain_type=row['terrain_type'],
                    terrain_level=row['terrain_level'], condition=condition,
                    repeat=row['repeat_index'], protocol=PROTOCOL,
                    reference=None if row == baseline else reference,
                    prepath_only=row['stage'] == 'prepath')
            verify_attempt(attempt, row)
        except Exception as exc:
            atomic_json(root/'status.json', {'phase': phase, 'state': 'stopped',
                        'completed': completed, 'selected': len(selected),
                        'current': row_name(row), 'first_failure': f'{type(exc).__name__}: {exc}'})
            raise
        completed += 1
    atomic_json(root/'status.json', {'phase': phase, 'state': 'completed',
                'completed': completed, 'selected': len(selected), 'current': None})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'formal'), required=True)
    args = parser.parse_args()
    run(args.root, args.phase)
