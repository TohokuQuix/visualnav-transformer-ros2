"""Sealed MI-A v3.1 seed preflight and multi-seed smoke schedule."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import numpy as np

from visualnav_transformer.eat_nav.mi_a_v3 import validate_evaluation_schedule, validate_collection_manifest
try:
    from .eat_nav_mi_fi_v1_host import execute, sha, RL, verify_sim_seal
except ImportError:  # direct script execution
    from eat_nav_mi_fi_v1_host import execute, sha, RL, verify_sim_seal


REPO = Path(__file__).resolve().parents[1]
SCHEMA = 'eatnav_mi_a_v31_multi_seed'
MANIFEST = REPO/'docs/eat-nav/motion-predictor-vnext/mi_a_v31_assignment_manifest.json'
PROTOCOL = REPO/'docs/eat-nav/motion-predictor-vnext/mi_a_v31_protocol.md'
COLLECTION = REPO/'docs/eat-nav/motion-predictor-vnext/collection_randomization_audit_v31.json'
EVIDENCE_PARENT = Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence')
SOURCES = {
    'runner': 'scripts/eat_nav_mi_a_v31_run.py',
    'analyzer': 'scripts/eat_nav_mi_a_v31_analyze.py',
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
    collection = json.loads(COLLECTION.read_text())
    if validate_collection_manifest(collection) != collection['assignment'] or \
            collection['state_groups'] != manifest['state_groups'] or \
            sha(COLLECTION) != manifest['collection_assignment_sha256']:
        raise ValueError('collection allocation mismatch')
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


def selected_rows(manifest: dict, phase: str) -> list[dict]:
    smoke = [row for row in manifest['assignment']
             if row['state_group_id'] in manifest['smoke_state_groups']]
    if phase == 'preflight':
        return smoke[:1]
    if phase == 'smoke':
        return smoke
    raise ValueError('formal evaluation is not enabled in this runner')


def verify_prepath_snapshot(output: Path, *, seed: int, condition: str) -> dict:
    identity = json.loads((output/'prepath-identity.json').read_text())
    if identity['seed'] != seed or identity['condition'] != condition or \
            identity['dem_frame'] != 'robot_base' or identity['dem_resolution_m'] != .1 or \
            identity['dem_stamps_ns'] != [250_000_000, 500_000_000, 750_000_000, 1_000_000_000]:
        raise ValueError('pre-Path snapshot identity mismatch')
    shapes = {'root': (4, 13), 'actor_pre_noise': (4, 22),
              'actor_input': (4, 22), 'dem': (4, 4, 2, 20, 20),
              'command': (4, 3), 'flipper_reset_joint_pos': (4, 4),
              'flipper_prepath_joint_pos': (4, 4)}
    required = set(shapes) | {'joint', 'critic', 'contact_force',
                              'contact_history', 'latency'}
    with np.load(output/'prepath.npz', allow_pickle=False) as arrays:
        if not required <= set(arrays.files):
            raise ValueError('incomplete pre-Path snapshot arrays')
        for key in required:
            value = arrays[key]
            if (key in shapes and value.shape != shapes[key]) or \
                    (key not in shapes and (value.shape[0] != 4 or value.ndim < 2)):
                raise ValueError(f'pre-Path snapshot shape mismatch: {key}')
            if not np.isfinite(value).all():
                raise ValueError(f'nonfinite pre-Path snapshot: {key}')
    return {'dem_shape': list(shapes['dem']), 'required_array_count': len(required),
            'snapshot_sha256': sha(output/'prepath.npz')}


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
    if phase == 'preflight':
        if root.exists():
            raise FileExistsError('preflight requires fresh root')
        root.mkdir()
        atomic_json(root/'frozen-provenance.json', {
            'schema': SCHEMA, 'manifest_sha256': sha(MANIFEST),
            'protocol_sha256': sha(PROTOCOL), 'started_ns': time.time_ns(),
            'smoke_state_groups': manifest['smoke_state_groups']})
    elif phase == 'smoke':
        if not root.is_dir():
            raise ValueError('smoke continuation requires preflight root')
        provenance = json.loads((root/'frozen-provenance.json').read_text())
        if provenance['manifest_sha256'] != sha(MANIFEST) or \
                provenance['protocol_sha256'] != sha(PROTOCOL):
            raise ValueError('manifest/protocol changed after preflight')
        preflight = json.loads((root/'preflight-result.json').read_text())
        if preflight.get('gate') != 'pass':
            raise ValueError('preflight gate not passed')
    else:
        raise ValueError('formal evaluation is not enabled')
    rows = manifest['assignment']
    selected = selected_rows(manifest, phase)
    completed = 0
    for row in selected:
        attempt = root/row_name(row)
        if attempt.exists():
            verify_attempt(attempt, row)
            if phase == 'preflight':
                raise FileExistsError('preflight cannot reuse an attempt')
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
            if phase == 'preflight':
                snapshot = verify_prepath_snapshot(
                    attempt/'sim-output', seed=row['terrain_seed'], condition=condition)
                atomic_json(root/'preflight-result.json', {
                    'schema': SCHEMA, 'gate': 'pass', 'row_index': row['index'],
                    'attempt_outer_seal_sha256': sha(attempt/'outer-seal.json'),
                    'snapshot': snapshot})
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
    parser.add_argument('--phase', choices=('preflight', 'smoke'), required=True)
    args = parser.parse_args()
    run(args.root, args.phase)
