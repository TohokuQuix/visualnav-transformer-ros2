"""Frozen 1152-process 14-Path collection, gated smoke then continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

from visualnav_transformer.eat_nav.motion_14_stage_a import candidate_role, geometry_audit
from visualnav_transformer.eat_nav.motion_14_collection import (
    COLLECTION_SCHEMA as SCHEMA, SEED_DERIVATION, build_collection_schedule,
    seed_groups, split_seeds, validate_collection_schedule,
)
from eat_nav_motion_14_collection_host import execute, sha, verify_sim_seal
from eat_nav_motion_14_stage_a_run import (
    MANIFEST as STAGE_A_MANIFEST, load_frozen as load_stage_a,
    row_name as stage_a_row_name, verify_attempt as verify_stage_a_attempt,
)


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO/'docs/eat-nav/motion-predictor-vnext/motion_14_collection_manifest.json'
PROTOCOL = REPO/'docs/eat-nav/motion-predictor-vnext/motion_14_collection_protocol.md'
EVIDENCE_PARENT = Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence')
SOURCES = {
    'coordinator': 'scripts/eat_nav_motion_14_collection_run.py',
    'host': 'scripts/eat_nav_motion_14_collection_host.py',
    'simulator': 'scripts/eat_nav_motion_14_collection_sim.py',
    'schedule': 'src/visualnav_transformer/eat_nav/motion_14_collection.py',
    'bank': 'src/visualnav_transformer/eat_nav/bidirectional_bank.py',
    'gate': 'src/visualnav_transformer/eat_nav/mi_fi_v1.py',
    'gate_numeric': 'src/visualnav_transformer/eat_nav/mi_a_v2.py',
    'core': 'src/visualnav_transformer/eat_nav/p1_follower_core.py',
    'core_cpp': 'src/visualnav_transformer/eat_nav/p1_follower_core.cpp',
    'observation_adapter': 'src/visualnav_transformer/eat_nav/p1_g3_rosfree_control.py',
    'dem': 'src/visualnav_transformer/eat_nav/dem.py',
    'environment': 'src/visualnav_transformer/eat_nav/demo_isaac_env_cfg.py',
    'terrain': 'src/visualnav_transformer/eat_nav/demo_terrain.py',
    'reset': 'src/visualnav_transformer/eat_nav/demo_reset_policy.py',
    'path_transform': 'src/visualnav_transformer/eat_nav/motion_collector.py',
    'labels': 'src/visualnav_transformer/eat_nav/bidirectional_labels.py',
    'label_base': 'src/visualnav_transformer/eat_nav/motion_vnext.py',
    'command_and_capture': 'src/visualnav_transformer/eat_nav/phase_c_sim.py',
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
    groups = seed_groups()
    return {'schema': SCHEMA, 'assignment': build_collection_schedule(groups),
            'state_groups': groups, 'split_seeds': split_seeds(groups),
            'seed_derivation': SEED_DERIVATION,
            'process_count': 1152, 'candidate_trial_count': 4032,
            'smoke_process_count': 32,
            'collection_gate': 'all process/seal/Gate P and raw DEM SHA pass; '
                               '40 finite steps and labels per candidate tile; '
                               'no minimum motion threshold',
            'protocol_sha256': sha(PROTOCOL), 'source_sha256': source_sha(),
            'geometry_sha256': hashlib.sha256(json.dumps(
                geometry, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            'legacy_reference_root': geometry['legacy_reference_root']}


def load_frozen() -> dict:
    data = json.loads(MANIFEST.read_text())
    validate_collection_schedule(data['assignment'], data['state_groups'])
    expected = freeze_manifest()
    if data != expected:
        raise ValueError('frozen Stage A manifest/source/geometry mismatch')
    return data


def row_name(row: dict) -> str:
    suffix = f"prepath-{row['repeat_index']}" if row['stage'] == 'prepath' else row['candidate_id']
    return f"{row['index']:04d}-{row['state_group_id']}-{row['flipper_initialization_mode']}-{suffix}"


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
    with np.load(attempt/'sim-output/raw-dem.npz', allow_pickle=False) as raw:
        if raw['ray_hits_w'].shape != (4, 4, 400, 3) or \
                raw['ray_valid'].shape != (4, 4, 400) or \
                raw['logical_stamp_ns'].tolist() != [250_000_000, 500_000_000,
                                                      750_000_000, 1_000_000_000]:
            raise ValueError('native ray history shape/stamp mismatch')
        if not np.array_equal(raw['ray_valid'], np.isfinite(raw['ray_hits_w'][..., 2])) or \
                not np.isfinite(raw['ray_starts_w']).all() or \
                not np.isfinite(raw['root_state_w']).all() or \
                not np.isfinite(raw['env_origins_w']).all():
            raise ValueError('native ray validity/pose mismatch')
    metadata = json.loads((attempt/'sim-output/raw-dem-metadata.json').read_text())
    if metadata['logical_stamp_ns'] != [250_000_000, 500_000_000,
                                        750_000_000, 1_000_000_000] or \
            len(metadata['sensor_update_stamps']) != 4:
        raise ValueError('native sensor stamp provenance missing')
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


def verify_stage_a_pass(root: Path) -> None:
    stage_a = load_stage_a()
    status = json.loads((root/'status.json').read_text())
    if status.get('state') != 'completed' or status.get('completed') != 32:
        raise ValueError('Stage A has not passed')
    for row in stage_a['assignment']:
        verify_stage_a_attempt(root/stage_a_row_name(row), row)


def run(root: Path, phase: str, *, stage_a_root: Path) -> None:
    data = load_frozen()
    verify_stage_a_pass(stage_a_root)
    if phase not in ('smoke', 'formal'):
        raise ValueError('unknown collection phase')
    if phase == 'smoke' and (root/'smoke-pass.json').exists():
        raise FileExistsError('protocol-valid smoke must not be rerun')
    root = root.absolute()
    if root.parent != EVIDENCE_PARENT or root.is_symlink() or root.resolve().parent != EVIDENCE_PARENT.resolve():
        raise ValueError('persistent direct-child Stage A evidence root required')
    if not root.exists():
        root.mkdir()
        atomic_json(root/'frozen-provenance.json', {'schema': SCHEMA,
                    'manifest_sha256': sha(MANIFEST), 'protocol_sha256': sha(PROTOCOL),
                    'stage_a_manifest_sha256': sha(STAGE_A_MANIFEST),
                    'stage_a_root': str(stage_a_root),
                    'stage_a_status_sha256': sha(stage_a_root/'status.json'),
                    'started_ns': time.time_ns()})
        atomic_json(root/'geometry-audit.json', geometry_audit())
    elif json.loads((root/'frozen-provenance.json').read_text())['manifest_sha256'] != sha(MANIFEST):
        raise ValueError('resume manifest mismatch')
    rows = data['assignment']
    if phase == 'formal':
        if not (root/'smoke-pass.json').is_file():
            raise ValueError('collection smoke has not passed')
        smoke = json.loads((root/'smoke-pass.json').read_text())
        if smoke['manifest_sha256'] != sha(MANIFEST):
            raise ValueError('smoke manifest mismatch')
    selected = rows[:32] if phase == 'smoke' else rows
    completed = 0
    for row in selected:
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
                    'total': len(rows), 'phase': phase,
                    'current': row_name(row), 'updated_ns': time.time_ns()})
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
                        'total': len(rows), 'phase': phase,
                        'current': row_name(row),
                        'failure': f'{type(exc).__name__}: {exc}', 'updated_ns': time.time_ns()})
            raise
        completed += 1
    if phase == 'smoke':
        atomic_json(root/'smoke-pass.json', {'schema': SCHEMA,
                    'manifest_sha256': sha(MANIFEST), 'completed': completed,
                    'last_attempt_outer_seal_sha256': sha(root/row_name(selected[-1])/'outer-seal.json')})
    atomic_json(root/'status.json', {'state': 'completed' if phase == 'formal' else 'smoke_pass',
                'phase': phase, 'completed': completed, 'total': len(rows),
                'updated_ns': time.time_ns()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'formal'), required=True)
    parser.add_argument('--stage-a-root', type=Path, required=True)
    options = parser.parse_args()
    run(options.root, options.phase, stage_a_root=options.stage_a_root)
