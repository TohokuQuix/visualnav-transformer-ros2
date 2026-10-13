"""Authorized, fresh-root formal execution of the immutable MI-A v3.1 schedule.

The smoke runner and every source listed in its frozen manifest remain unchanged.
This coordinator adds no simulator or scientific logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

try:
    from .eat_nav_mi_a_v31_run import (
        EVIDENCE_PARENT, MANIFEST, PROTOCOL, SCHEMA, atomic_json,
        load_frozen, row_name, verify_attempt,
    )
    from .eat_nav_mi_fi_v1_host import execute, sha
except ImportError:
    from eat_nav_mi_a_v31_run import (
        EVIDENCE_PARENT, MANIFEST, PROTOCOL, SCHEMA, atomic_json,
        load_frozen, row_name, verify_attempt,
    )
    from eat_nav_mi_fi_v1_host import execute, sha


def formal_rows(manifest: dict) -> list[dict]:
    rows = manifest['assignment']
    if len(rows) != 192 or [row['index'] for row in rows] != list(range(192)) or \
            sum(row['stage'] == 'prepath' for row in rows) != 48 or \
            sum(row['stage'] == 'candidate' for row in rows) != 144 or \
            len({row['terrain_seed'] for row in rows}) != 12:
        raise ValueError('formal assignment is not the frozen 192-row schedule')
    return rows


def baseline_for(rows: list[dict], row: dict) -> dict:
    matches = [r for r in rows if r['state_group_id'] == row['state_group_id']
               and r['flipper_initialization_mode'] == row['flipper_initialization_mode']
               and r['stage'] == 'prepath' and r['repeat_index'] == 0]
    if len(matches) != 1 or matches[0]['index'] > row['index']:
        raise ValueError('missing preceding same-condition pre-Path reference')
    return matches[0]


def validate_new_root(root: Path) -> Path:
    root = Path(root).absolute()
    if root.parent != EVIDENCE_PARENT or root.name in ('', '.', '..') or \
            root.is_symlink() or root.resolve().parent != EVIDENCE_PARENT.resolve():
        raise ValueError('fresh direct child of persistent evidence parent required')
    if root.exists():
        raise FileExistsError('formal evidence root must be new')
    return root


def run(root: Path) -> None:
    manifest = load_frozen()
    rows = formal_rows(manifest)
    root = validate_new_root(root)
    root.mkdir()
    source = Path(__file__).resolve()
    atomic_json(root/'frozen-provenance.json', {
        'schema': SCHEMA, 'phase': 'formal',
        'manifest_sha256': sha(MANIFEST), 'protocol_sha256': sha(PROTOCOL),
        'formal_coordinator_sha256': sha(source),
        'authorization': 'user approval of 192-process MI-A v3.1 formal evaluation, 2026-10-14',
        'source_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source.parents[1],
                                        check=True, capture_output=True, text=True).stdout.strip(),
        'started_ns': time.time_ns(), 'expected_processes': len(rows),
        'smoke_reuse': False,
    })
    completed = 0
    for row in rows:
        name = row_name(row)
        attempt = root/name
        if attempt.exists():
            raise FileExistsError(f'unexpected existing attempt: {name}')
        baseline = baseline_for(rows, row)
        reference = root/row_name(baseline)/'sim-output'
        role = row['candidate_id'] or 'forward_direct'
        direction = 1 if role.startswith('forward') else -1
        angle = 0 if role.endswith('direct') else int(role.rsplit('_', 1)[1])
        atomic_json(root/'status.json', {'phase': 'formal', 'state': 'running',
                    'completed': completed, 'selected': len(rows), 'current': name,
                    'updated_ns': time.time_ns()})
        try:
            execute(attempt, direction=direction, angle=angle,
                    seed=row['terrain_seed'], terrain_type=row['terrain_type'],
                    terrain_level=row['terrain_level'],
                    condition=row['flipper_initialization_mode'],
                    repeat=row['repeat_index'], protocol=PROTOCOL,
                    reference=None if row == baseline else reference,
                    prepath_only=row['stage'] == 'prepath')
            verify_attempt(attempt, row)
        except Exception as exc:
            atomic_json(root/'status.json', {'phase': 'formal', 'state': 'stopped',
                        'completed': completed, 'selected': len(rows), 'current': name,
                        'first_failure': f'{type(exc).__name__}: {exc}',
                        'updated_ns': time.time_ns()})
            raise
        completed += 1
        atomic_json(root/'status.json', {'phase': 'formal', 'state': 'running',
                    'completed': completed, 'selected': len(rows), 'current': None,
                    'last_outer_seal_sha256': sha(attempt/'outer-seal.json'),
                    'updated_ns': time.time_ns()})
    atomic_json(root/'status.json', {'phase': 'formal', 'state': 'completed',
                'completed': completed, 'selected': len(rows), 'current': None,
                'updated_ns': time.time_ns()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
