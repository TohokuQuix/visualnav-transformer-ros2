"""Resume the frozen MI-A schedule from row 44 after process-43 stability checks.

The prior 43 successful attempts and the designated new process-43 attempt
are linked read-only. Neither failed process-43 attempt is reused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

try:
    from .eat_nav_mi_a_v31_formal_run import formal_rows, baseline_for, validate_new_root
    from .eat_nav_mi_a_v31_formal_continuation import link_prefix
    from .eat_nav_mi_a_v31_run import (
        MANIFEST, PROTOCOL, SCHEMA, atomic_json, load_frozen, row_name,
        verify_attempt, verify_outer,
    )
    from .eat_nav_mi_fi_v1_host import execute, sha
except ImportError:
    from eat_nav_mi_a_v31_formal_run import formal_rows, baseline_for, validate_new_root
    from eat_nav_mi_a_v31_formal_continuation import link_prefix
    from eat_nav_mi_a_v31_run import (
        MANIFEST, PROTOCOL, SCHEMA, atomic_json, load_frozen, row_name,
        verify_attempt, verify_outer,
    )
    from eat_nav_mi_fi_v1_host import execute, sha


RESUME_INDEX = 44
MEASUREMENT_FILES = ('prepath.npz', 'motion-labels.json',
                     *(f'trace-env-{i}.jsonl' for i in range(4)))


def source_for_row(row: dict, original: Path, stability: Path) -> Path:
    if row['index'] < 43:
        return original/row_name(row)
    if row['index'] == 43:
        return stability/'attempt-a'
    raise ValueError('later rows require new execution')


def verify_identical_measurements(a: Path, b: Path) -> dict[str, str]:
    hashes = {}
    for name in MEASUREMENT_FILES:
        first = sha(a/'sim-output'/name)
        second = sha(b/'sim-output'/name)
        if first != second:
            raise ValueError(f'process-43 stability mismatch: {name}')
        hashes[name] = first
    return hashes


def verify_prior_stop(root: Path, row: dict) -> str:
    status = json.loads((root/'status.json').read_text())
    if status['state'] != 'stopped' or status['completed'] != 43 or \
            status['current'] != row_name(row):
        raise ValueError('prior formal stop identity mismatch')
    failed = root/row_name(row)
    verify_outer(failed)
    if json.loads((failed/'decision.json').read_text())['pass'] is not False:
        raise ValueError('prior failed attempt was changed to valid')
    return sha(failed/'outer-seal.json')


def run(root: Path, original: Path, prior_continuation: Path,
        stability: Path) -> None:
    manifest = load_frozen()
    rows = formal_rows(manifest)
    original = Path(original).resolve(strict=True)
    prior_continuation = Path(prior_continuation).resolve(strict=True)
    stability = Path(stability).resolve(strict=True)
    failure_seals = [verify_prior_stop(old, rows[43])
                     for old in (original, prior_continuation)]
    original_provenance = json.loads((original/'frozen-provenance.json').read_text())
    if original_provenance['manifest_sha256'] != sha(MANIFEST) or \
            original_provenance['protocol_sha256'] != sha(PROTOCOL):
        raise ValueError('prior scientific provenance differs')
    for name in ('attempt-a', 'attempt-b'):
        verify_attempt(stability/name, rows[43])
    measurement_hashes = verify_identical_measurements(
        stability/'attempt-a', stability/'attempt-b')
    root = validate_new_root(root)
    root.mkdir()
    linked = link_prefix(original, root, rows[:43])
    designated = source_for_row(rows[43], original, stability)
    (root/row_name(rows[43])).symlink_to(designated, target_is_directory=True)
    verify_attempt(root/row_name(rows[43]), rows[43])
    source = Path(__file__).resolve()
    atomic_json(root/'frozen-provenance.json', {
        'schema': SCHEMA, 'phase': 'formal',
        'manifest_sha256': sha(MANIFEST), 'protocol_sha256': sha(PROTOCOL),
        'formal_coordinator_sha256': sha(source),
        'source_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source.parents[1],
                                        check=True, capture_output=True, text=True).stdout.strip(),
        'prior_valid_prefix': linked,
        'prior_failed_roots': [str(original), str(prior_continuation)],
        'excluded_failed_outer_seals_sha256': failure_seals,
        'designated_process43_root': str(designated),
        'designated_process43_outer_seal_sha256': sha(designated/'outer-seal.json'),
        'excluded_stability_control_root': str(stability/'attempt-b'),
        'excluded_stability_control_outer_seal_sha256': sha(stability/'attempt-b'/'outer-seal.json'),
        'process43_measurement_sha256': measurement_hashes,
        'scientific_runtime_change': False,
        'display_mode': 'unchanged Docker DISPLAY=:1 and XAUTHORITY; original --headless',
        'started_ns': time.time_ns(), 'expected_processes': len(rows),
    })
    completed = RESUME_INDEX
    for row in rows[RESUME_INDEX:]:
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
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--prior-continuation-root', type=Path, required=True)
    parser.add_argument('--stability-root', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.original_root, args.prior_continuation_root,
        args.stability_root)
