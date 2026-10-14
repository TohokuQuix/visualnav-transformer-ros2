"""Frozen 32-process Stage A run; stop on the first failed protocol gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from visualnav_transformer.eat_nav.motion_14_stage_a import (
    SCHEMA, build_pilot_schedule, candidate_role, validate_schedule, geometry_audit,
)
from eat_nav_motion_14_stage_a_host import execute, sha, verify_sim_seal


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO/'docs/eat-nav/motion-predictor-vnext/motion_14_stage_a_manifest.json'
PROTOCOL = REPO/'docs/eat-nav/motion-predictor-vnext/motion_14_stage_a_protocol.md'
EVIDENCE_PARENT = Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence')
SOURCES = {
    'coordinator': 'scripts/eat_nav_motion_14_stage_a_run.py',
    'host': 'scripts/eat_nav_motion_14_stage_a_host.py',
    'simulator': 'scripts/eat_nav_motion_14_stage_a_sim.py',
    'schedule': 'src/visualnav_transformer/eat_nav/motion_14_stage_a.py',
    'bank': 'src/visualnav_transformer/eat_nav/bidirectional_bank.py',
    'gate': 'src/visualnav_transformer/eat_nav/mi_fi_v1.py',
    'core': 'src/visualnav_transformer/eat_nav/p1_follower_core.py',
    'observation_adapter': 'src/visualnav_transformer/eat_nav/p1_g3_rosfree_control.py',
    'config': 'config/eat_nav.yaml',
}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def source_sha() -> dict[str, str]:
    return {name: sha(REPO/path) for name, path in SOURCES.items()}


def freeze_manifest() -> dict:
    geometry = geometry_audit()
    if not geometry['pass']:
        raise ValueError('offline geometry audit failed')
    return {'schema': SCHEMA, 'assignment': build_pilot_schedule(),
            'process_count': 32, 'candidate_trial_count': 112,
            'pilot_gate': 'all process/seal/Gate P pass; 40 finite steps and labels '
                          'per candidate tile; no minimum motion threshold',
            'protocol_sha256': sha(PROTOCOL), 'source_sha256': source_sha(),
            'geometry_sha256': hashlib.sha256(json.dumps(
                geometry, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            'legacy_reference_root': geometry['legacy_reference_root']}


def load_frozen() -> dict:
    data = json.loads(MANIFEST.read_text())
    validate_schedule(data['assignment'])
    expected = freeze_manifest()
    if data != expected:
        raise ValueError('frozen Stage A manifest/source/geometry mismatch')
    return data


def row_name(row: dict) -> str:
    suffix = f"prepath-{row['repeat_index']}" if row['stage'] == 'prepath' else row['candidate_id']
    return f"{row['index']:03d}-{row['flipper_initialization_mode']}-{suffix}"


def verify_outer(attempt: Path) -> None:
    ledger = attempt/'outer-sha-ledger.json'
    seal = json.loads((attempt/'outer-seal.json').read_text())
    expected = json.loads(ledger.read_text())
    actual = {str(path.relative_to(attempt)): sha(path) for path in attempt.rglob('*')
              if path.is_file() and path.name not in ('outer-sha-ledger.json', 'outer-seal.json')}
    if sha(ledger) != seal['ledger_sha256'] or expected != actual or len(actual) != seal['file_count']:
        raise ValueError(f'outer seal mismatch: {attempt}')


def verify_attempt(attempt: Path, row: dict) -> None:
    verify_outer(attempt)
    verify_sim_seal(attempt/'sim-output')
    if not json.loads((attempt/'decision.json').read_text())['pass']:
        raise ValueError(f'host decision failed: {attempt}')
    host = json.loads((attempt/'manifest.json').read_text())
    for key, expected in (('seed', row['terrain_seed']),
                          ('terrain_type', row['terrain_type']),
                          ('terrain_level', row['terrain_level']),
                          ('condition', row['flipper_initialization_mode']),
                          ('repeat', row['repeat_index']),
                          ('protocol_sha256', sha(PROTOCOL)),
                          ('mode', 'prepath_only' if row['stage'] == 'prepath' else 'candidate')):
        if host.get(key) != expected:
            raise ValueError(f'host row identity mismatch: {key}')
    gate = json.loads((attempt/'sim-output/prepath-pair-gate.json').read_text())
    if gate['pass'] is not True:
        raise ValueError(f'Gate P failed: {attempt}')
    if row['stage'] == 'candidate':
        role = candidate_role(host['direction'], host['angle_deg'])
        if role != row['candidate_id']:
            raise ValueError('candidate role mismatch')
        output = attempt/'sim-output'
        labels = json.loads((output/'motion-labels.json').read_text())['labels']
        if len(labels) != 4 or not all(label['available'] for label in labels):
            raise ValueError('missing 40-step label')
        activation = json.loads((output/'path-activation.json').read_text())
        if activation['count_per_env'] != [1] * 4:
            raise ValueError('Path publication count mismatch')


def run(root: Path) -> None:
    data = load_frozen()
    root = root.absolute()
    if root.parent != EVIDENCE_PARENT or root.is_symlink() or root.resolve().parent != EVIDENCE_PARENT.resolve():
        raise ValueError('persistent direct-child Stage A evidence root required')
    if not root.exists():
        root.mkdir()
        atomic_json(root/'frozen-provenance.json', {'schema': SCHEMA,
                    'manifest_sha256': sha(MANIFEST), 'protocol_sha256': sha(PROTOCOL),
                    'started_ns': time.time_ns()})
        atomic_json(root/'geometry-audit.json', geometry_audit())
    elif json.loads((root/'frozen-provenance.json').read_text())['manifest_sha256'] != sha(MANIFEST):
        raise ValueError('resume manifest mismatch')
    rows = data['assignment']
    completed = 0
    for row in rows:
        attempt = root/row_name(row)
        if attempt.exists():
            verify_attempt(attempt, row)
            completed += 1
            continue
        posture = row['flipper_initialization_mode']
        baseline = next(r for r in rows if r['flipper_initialization_mode'] == posture
                        and r['stage'] == 'prepath' and r['repeat_index'] == 0)
        reference = root/row_name(baseline)/'sim-output'
        role = row['candidate_id'] or 'forward_direct'
        direction = 1 if role.startswith('forward') else -1
        angle = 0 if role.endswith('direct') else int(role.rsplit('_', 1)[1])
        atomic_json(root/'status.json', {'state': 'running', 'completed': completed,
                    'total': len(rows), 'current': row_name(row), 'updated_ns': time.time_ns()})
        try:
            execute(attempt, direction=direction, angle=angle,
                    seed=row['terrain_seed'], terrain_type=row['terrain_type'],
                    terrain_level=row['terrain_level'], condition=posture,
                    repeat=row['repeat_index'], protocol=PROTOCOL,
                    reference=None if row == baseline else reference,
                    prepath_only=row['stage'] == 'prepath')
            verify_attempt(attempt, row)
        except Exception as exc:
            atomic_json(root/'status.json', {'state': 'stopped', 'completed': completed,
                        'total': len(rows), 'current': row_name(row),
                        'failure': f'{type(exc).__name__}: {exc}', 'updated_ns': time.time_ns()})
            raise
        completed += 1
    atomic_json(root/'status.json', {'state': 'completed', 'completed': completed,
                'total': len(rows), 'updated_ns': time.time_ns()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
