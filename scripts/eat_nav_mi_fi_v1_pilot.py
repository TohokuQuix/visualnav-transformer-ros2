"""Run the frozen 16-process MI-FI v1 pilot, stopping at each gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from visualnav_transformer.eat_nav.mi_fi_v1 import SCHEMA, pilot_plan
try:
    from .eat_nav_mi_fi_v1_host import execute, sha, RL
except ImportError:  # direct script execution
    from eat_nav_mi_fi_v1_host import execute, sha, RL


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO/'docs/eat-nav/motion-predictor-vnext/mi_fi_v1_frozen_manifest.json'
PROTOCOL = REPO/'docs/eat-nav/motion-predictor-vnext/mi_fi_v1_protocol.md'


def validate_pilot_manifest(manifest: dict) -> list[dict]:
    if manifest.get('schema') != SCHEMA or manifest.get('num_envs') != 4 or \
            manifest.get('settling_steps') != 20 or manifest.get('candidate_control_steps') != 40 or \
            manifest.get('step_seconds') != .05 or manifest.get('flipper_fold_mode') != 0:
        raise ValueError('unfrozen MI-FI pilot contract')
    expected = pilot_plan(manifest['terrain_seed'])
    rows = manifest['pilot_order']
    if len(rows) != 16:
        raise ValueError('sixteen process pilot only')
    for actual, plan in zip(rows, expected):
        if {key: actual[key] for key in ('index', 'stage', 'candidate', 'condition', 'repeat')} != \
                {key: plan[key] for key in ('index', 'stage', 'candidate', 'condition', 'repeat')}:
            raise ValueError('pilot order/candidate mismatch')
    return rows


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_pilot(root: Path):
    manifest = json.loads(MANIFEST.read_text())
    rows = validate_pilot_manifest(manifest)
    if sha(REPO/'config/eat_nav.yaml') != manifest['eat_config_sha256'] or \
            sha(REPO/'src/visualnav_transformer/eat_nav/bidirectional_bank.py') != manifest['bank_source_sha256'] or \
            sha(RL) != manifest['checkpoint_sha256']:
        raise ValueError('frozen source/input SHA mismatch')
    for key, path in (
            ('sim_source_sha256', REPO/'scripts/eat_nav_mi_fi_v1_sim.py'),
            ('host_source_sha256', REPO/'scripts/eat_nav_mi_fi_v1_host.py'),
            ('gate_source_sha256', REPO/'src/visualnav_transformer/eat_nav/mi_fi_v1.py'),
            ('analysis_source_sha256', REPO/'scripts/eat_nav_mi_fi_v1_analyze.py'),
            ('protocol_sha256', PROTOCOL)):
        if sha(path) != manifest[key]:
            raise ValueError(f'frozen {key} mismatch')
    root = root.resolve()
    if root.exists():
        raise FileExistsError('new persistent MI-A v2 pilot root required')
    root.mkdir(parents=True)
    provenance = {'schema': SCHEMA, 'manifest_sha256': sha(MANIFEST),
                  'protocol_sha256': sha(PROTOCOL),
                  'pilot_source_sha256': sha(__file__),
                  'host_source_sha256': sha(REPO/'scripts/eat_nav_mi_fi_v1_host.py'),
                  'sim_source_sha256': sha(REPO/'scripts/eat_nav_mi_fi_v1_sim.py'),
                  'gate_source_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/mi_fi_v1.py'),
                  'checkpoint_sha256': manifest['checkpoint_sha256'],
                  'assignment': rows, 'started_ns': time.time_ns()}
    atomic_json(root/'frozen-provenance.json', provenance)
    baselines = {'front_folded': root/'00-front_folded-prepath-0'/'sim-output',
                 'rear_folded': root/'08-rear_folded-prepath-0'/'sim-output'}
    for row in rows:
        name = f"{row['index']:02d}-{row['condition']}-" + (
            f"prepath-{row['repeat']}" if row['stage'] == 'prepath' else row['candidate'])
        attempt = root/name
        role = row['candidate'] or 'forward_direct'
        direction = 1 if role.startswith('forward') else -1
        angle = 0 if role.endswith('direct') else int(role.rsplit('_', 1)[1])
        atomic_json(root/'status.json', {'state': 'running', 'completed': row['index'],
                                         'expected': 16, 'current': name})
        try:
            execute(attempt, direction=direction, angle=angle,
                    seed=manifest['terrain_seed'], terrain_type=manifest['terrain_type'],
                    terrain_level=manifest['terrain_level'], condition=row['condition'],
                    repeat=row['repeat'], protocol=PROTOCOL,
                    reference=baselines[row['condition']] if row['repeat'] or row['stage'] != 'prepath' else None,
                    prepath_only=row['stage'] == 'prepath')
        except Exception as exc:
            atomic_json(root/'status.json', {'state': 'stopped', 'completed': row['index'],
                                             'expected': 16, 'current': name,
                                             'first_failure': f'{type(exc).__name__}: {exc}'})
            raise
        atomic_json(root/'status.json', {'state': 'running', 'completed': row['index']+1,
                                         'expected': 16, 'current': None})
    atomic_json(root/'status.json', {'state': 'pilot_complete', 'completed': 16,
                                     'expected': 16, 'current': None})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run_pilot(args.root)
