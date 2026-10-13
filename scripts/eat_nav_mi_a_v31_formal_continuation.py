"""Continue a sealed MI-A formal prefix after a pre-Python Kit startup crash.

Earlier attempts are symlinked read-only; the failed attempt is never reused.
All scientific inputs and the frozen assignment remain the original v3.1 files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

try:
    from .eat_nav_mi_a_v31_formal_run import formal_rows, baseline_for, validate_new_root
    from .eat_nav_mi_a_v31_run import (
        MANIFEST, PROTOCOL, SCHEMA, atomic_json, load_frozen, row_name,
        verify_attempt, verify_outer,
    )
    from .eat_nav_mi_fi_v1_host import execute, sha
except ImportError:
    from eat_nav_mi_a_v31_formal_run import formal_rows, baseline_for, validate_new_root
    from eat_nav_mi_a_v31_run import (
        MANIFEST, PROTOCOL, SCHEMA, atomic_json, load_frozen, row_name,
        verify_attempt, verify_outer,
    )
    from eat_nav_mi_fi_v1_host import execute, sha


STOP_INDEX = 43


def classify_prelaunch_failure(failed: Path) -> str:
    log = (failed/'simulator.log').read_text()
    native = json.loads((failed/'native-exit.json').read_text())
    if native['exit_code'] != 1 or 'XOpenDisplay' not in log or \
            'Segmentation fault' not in log or (failed/'sim-output').exists() or \
            (failed/'path-activation.json').exists():
        raise ValueError('failure is not isolated to Kit/X11 startup before Python main')
    return 'kit_xopen_display_before_python_main'


def link_prefix(prior: Path, target: Path, rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        name = row_name(row)
        source = prior/name
        link = target/name
        if link.exists() or link.is_symlink():
            raise FileExistsError(f'prior result already linked: {name}')
        verify_attempt(source, row)
        seal_sha = sha(source/'outer-seal.json')
        link.symlink_to(source, target_is_directory=True)
        verify_attempt(link, row)
        records.append({'row_index': row['index'], 'source_root': str(source),
                        'outer_seal_sha256': seal_sha})
    return records


def run(root: Path, prior: Path) -> None:
    manifest = load_frozen()
    rows = formal_rows(manifest)
    prior = Path(prior).resolve(strict=True)
    previous = json.loads((prior/'frozen-provenance.json').read_text())
    status = json.loads((prior/'status.json').read_text())
    if previous['manifest_sha256'] != sha(MANIFEST) or \
            previous['protocol_sha256'] != sha(PROTOCOL) or \
            previous['phase'] != 'formal' or \
            status['state'] != 'stopped' or status['completed'] != STOP_INDEX or \
            status['current'] != row_name(rows[STOP_INDEX]):
        raise ValueError('previous run identity or stopped prefix differs')
    if previous['formal_coordinator_sha256'] != sha(Path(__file__).with_name(
            'eat_nav_mi_a_v31_formal_run.py')):
        raise ValueError('previous formal coordinator source changed')
    failed = prior/row_name(rows[STOP_INDEX])
    verify_outer(failed)
    classification = classify_prelaunch_failure(failed)
    root = validate_new_root(root)
    root.mkdir()
    source = Path(__file__).resolve()
    links = link_prefix(prior, root, rows[:STOP_INDEX])
    atomic_json(root/'frozen-provenance.json', {
        'schema': SCHEMA, 'phase': 'formal',
        'manifest_sha256': sha(MANIFEST), 'protocol_sha256': sha(PROTOCOL),
        'formal_coordinator_sha256': sha(source),
        'prior_root': str(prior), 'prior_provenance_sha256': sha(prior/'frozen-provenance.json'),
        'prior_status_sha256': sha(prior/'status.json'),
        'excluded_failed_attempt': str(failed),
        'excluded_failed_outer_seal_sha256': sha(failed/'outer-seal.json'),
        'failure_classification': classification,
        'reused_valid_prefix': links,
        'authorization': 'user approval of 192-process MI-A v3.1 formal evaluation, 2026-10-14',
        'source_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source.parents[1],
                                        check=True, capture_output=True, text=True).stdout.strip(),
        'started_ns': time.time_ns(), 'expected_processes': len(rows),
    })
    completed = STOP_INDEX
    for row in rows[STOP_INDEX:]:
        name = row_name(row)
        attempt = root/name
        if attempt.exists() or attempt.is_symlink():
            raise FileExistsError(f'unexpected existing new attempt: {name}')
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
    parser.add_argument('--prior-root', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.root, arguments.prior_root)
