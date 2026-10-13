"""Run the frozen B-MI stage-one schedule one sealed fresh process at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from visualnav_transformer.eat_nav.motion_identifiability import SCHEMA, CANDIDATES


REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO/'docs/eat-nav/motion-predictor-vnext/motion_identifiability_manifest.json'
PROTOCOL = REPO/'docs/eat-nav/motion-predictor-vnext/motion_identifiability_protocol.md'
HOST = REPO/'scripts/eat_nav_motion_identifiability_host.py'
SIM = REPO/'scripts/eat_nav_motion_identifiability_sim.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assignment_rows(manifest):
    if manifest['schema'] != SCHEMA or tuple(manifest['candidate_order']) != (
            'forward_direct', 'reverse_direct', 'forward_-30', 'forward_+30',
            'reverse_-30', 'reverse_+30') or tuple(manifest['repeats']) != (0, 1):
        raise ValueError('unfrozen B-MI assignment')
    groups = manifest['terrain_groups']
    if len(groups) != 6 or len({int(g['seed']) for g in groups}) != 6 or \
            [g['condition'] for g in groups].count('mirror') != 2 or \
            [g['condition'] for g in groups].count('rough') != 4:
        raise ValueError('six independent groups/mirror support required')
    rows = []
    for group in groups:
        for repeat in manifest['repeats']:
            for role in manifest['candidate_order']:
                if role not in CANDIDATES:
                    raise ValueError('unknown candidate')
                direction = 1 if role.startswith('forward') else -1
                angle = 0 if role.endswith('direct') else int(role.rsplit('_', 1)[1])
                rows.append({**group, 'repeat': repeat, 'candidate': role,
                             'direction': direction, 'angle': angle})
    if len(rows) != 72:
        raise ValueError('expected 72 fresh candidate processes')
    return rows


def atomic_json(path, value):
    path = Path(path)
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


def run_schedule(root: Path):
    manifest = json.loads(FROZEN.read_text())
    rows = assignment_rows(manifest)
    if sha(REPO/'config/eat_nav.yaml') != manifest['eat_config_sha256'] or \
            sha(REPO/'src/visualnav_transformer/eat_nav/bidirectional_bank.py') != manifest['path_bank_sha256']:
        raise ValueError('config or Path bank changed after Stage 0 freeze')
    root = root.resolve()
    if root.exists():
        raise FileExistsError('fresh persistent schedule root required')
    root.mkdir(parents=True)
    provenance = {'schema': SCHEMA, 'manifest_sha256': sha(FROZEN),
                  'protocol_sha256': sha(PROTOCOL), 'sim_source_sha256': sha(SIM),
                  'host_source_sha256': sha(HOST), 'schedule_source_sha256': sha(__file__),
                  'analysis_source_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/motion_identifiability.py'),
                  'assignment_count': len(rows), 'assignments': rows,
                  'started_wall_ns': time.time_ns()}
    atomic_json(root/'frozen-provenance.json', provenance)
    atomic_json(root/'status.json', {'state': 'running', 'sealed_processes': 0,
                                     'expected_processes': len(rows), 'current': None})
    for ordinal, row in enumerate(rows):
        name = f"{ordinal:02d}-seed{row['seed']}-r{row['repeat']}-{row['candidate']}"
        attempt = root/name
        reference = root/f"{ordinal - manifest['candidate_order'].index(row['candidate']):02d}-seed{row['seed']}-r{row['repeat']}-{manifest['candidate_order'][0]}"
        # Reverse contrasts use the reverse Direct, not the forward Direct.
        if row['direction'] == -1:
            reference = root/f"{ordinal - manifest['candidate_order'].index(row['candidate']) + 1:02d}-seed{row['seed']}-r{row['repeat']}-reverse_direct"
        reference_output = reference/'sim-output' if row['angle'] else None
        argv = [sys.executable, str(HOST), '--root', str(attempt),
                '--direction', str(row['direction']), '--arc-angle', str(row['angle']),
                '--seed', str(row['seed']), '--terrain-type', row['terrain_type'],
                '--terrain-level', str(row['terrain_level']), '--condition', row['condition'],
                '--repeat', str(row['repeat']), '--protocol', str(PROTOCOL)]
        if reference_output:
            argv += ['--reference-prepath', str(reference_output)]
        atomic_json(root/'status.json', {'state': 'running', 'sealed_processes': ordinal,
                                         'expected_processes': len(rows), 'current': name})
        with (root/'schedule.log').open('a') as stream:
            stream.write(f'START {name} {time.time_ns()}\n')
            stream.flush()
            os.fsync(stream.fileno())
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'PYTHONPATH': str(REPO/'src')+':'+str(REPO)})
            stream.write(f'EXIT {name} {result.returncode} {time.time_ns()}\n')
            stream.flush()
            os.fsync(stream.fileno())
        if result.returncode != 0:
            atomic_json(root/'status.json', {'state': 'stopped', 'sealed_processes': ordinal,
                                             'expected_processes': len(rows), 'current': name,
                                             'reason': 'process protocol/resource/seal failure'})
            return 1
        if not json.loads((attempt/'decision.json').read_text())['pass']:
            raise ValueError('host returned success without pass decision')
        atomic_json(root/'status.json', {'state': 'running', 'sealed_processes': ordinal+1,
                                         'expected_processes': len(rows), 'current': None})
    atomic_json(root/'status.json', {'state': 'complete', 'sealed_processes': len(rows),
                                     'expected_processes': len(rows), 'current': None})
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(run_schedule(arguments.root))
