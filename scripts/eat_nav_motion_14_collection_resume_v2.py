"""Protocol-preserving continuation after the cross-StateGroup reference bug.

The frozen scientific assignment, host, simulator, Path, gates and resource
limits are imported unchanged. Only the coordinator's reference lookup changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from eat_nav_motion_14_collection_host import execute, sha
from eat_nav_motion_14_collection_run import (
    EVIDENCE_PARENT, MANIFEST, PROTOCOL, atomic_json,
    load_frozen, row_name, verify_attempt, verify_stage_a_pass,
)


REPO = Path(__file__).resolve().parents[1]
CONTINUATION = REPO/'docs/eat-nav/motion-predictor-vnext/motion_14_collection_continuation_v2.json'
FAILED_INDEX = 32
TARGET_ROOT = EVIDENCE_PARENT/'motion-14-collection-20261014-02'


def baseline_for(rows: list[dict], row: dict) -> dict:
    matches = [item for item in rows if
               item['state_group_id'] == row['state_group_id'] and
               item['flipper_initialization_mode'] == row['flipper_initialization_mode'] and
               item['stage'] == 'prepath' and item['repeat_index'] == 0]
    if len(matches) != 1 or matches[0]['index'] > row['index']:
        raise ValueError('missing preceding same-group/posture pre-Path reference')
    return matches[0]


def frozen_continuation(source_root: Path, stage_a_root: Path) -> dict:
    rows = load_frozen()['assignment']
    failed = source_root/row_name(rows[FAILED_INDEX])
    failure = json.loads((failed/'sim-output/failure.json').read_text())
    status = json.loads((source_root/'status.json').read_text())
    if status.get('state') != 'stopped' or status.get('completed') != FAILED_INDEX or \
            status.get('current') != row_name(rows[FAILED_INDEX]) or \
            failure.get('path_published') is not False or \
            failure.get('message') != 'cross-candidate pre-Path gate failed':
        raise ValueError('failed pre-Path attempt does not match diagnosed stop')
    verify_stage_a_pass(stage_a_root)
    prefix_seals = []
    for row in rows[:FAILED_INDEX]:
        source = source_root/row_name(row)
        verify_attempt(source, row)
        prefix_seals.append({'index': row['index'],
                             'outer_seal_sha256': sha(source/'outer-seal.json')})
    if not (source_root/'smoke-pass.json').is_file():
        raise ValueError('valid sealed smoke prefix missing')
    return {'schema': 'eatnav_motion_14_collection_continuation_v2',
            'scientific_manifest_sha256': sha(MANIFEST),
            'scientific_protocol_sha256': sha(PROTOCOL),
            'continuation_source_sha256': sha(Path(__file__)),
            'host_source_sha256': sha(REPO/'scripts/eat_nav_motion_14_collection_host.py'),
            'simulator_source_sha256': sha(REPO/'scripts/eat_nav_motion_14_collection_sim.py'),
            'source_root': str(source_root), 'stage_a_root': str(stage_a_root),
            'continuation_output_root': str(TARGET_ROOT),
            'smoke_pass_sha256': sha(source_root/'smoke-pass.json'),
            'excluded_failed_attempt_index': FAILED_INDEX,
            'excluded_failed_outer_seal_sha256': sha(failed/'outer-seal.json'),
            'failure_classification': 'coordinator reference lookup omitted state_group_id; '
                                      'Path unpublished; not a candidate outcome',
            'reused_valid_prefix_count': FAILED_INDEX,
            'reused_valid_prefix_outer_seals': prefix_seals,
            'reference_rule': 'same state_group_id, same flipper posture, prepath repeat 0',
            'assignment_unchanged': True,
            'path_physics_follower_actor_gate_unchanged': True}


def run(root: Path, *, source_root: Path, stage_a_root: Path) -> None:
    rows = load_frozen()['assignment']
    source_root = source_root.resolve()
    stage_a_root = stage_a_root.resolve()
    expected = frozen_continuation(source_root, stage_a_root)
    if json.loads(CONTINUATION.read_text()) != expected:
        raise ValueError('continuation source/manifest/evidence SHA mismatch')
    root = root.absolute()
    if root != TARGET_ROOT or root.parent != EVIDENCE_PARENT or root.is_symlink() or \
            root.resolve().parent != EVIDENCE_PARENT.resolve() or root == source_root:
        raise ValueError('fresh persistent continuation root required')
    if not root.exists():
        root.mkdir()
        atomic_json(root/'continuation-provenance.json', {
            'continuation_manifest_sha256': sha(CONTINUATION),
            'scientific_manifest_sha256': sha(MANIFEST),
            'source_root': str(source_root), 'started_ns': time.time_ns()})
        for row in rows[:FAILED_INDEX]:
            source = source_root/row_name(row)
            target = root/row_name(row)
            shutil.copytree(source, target, copy_function=shutil.copy2)
            verify_attempt(target, row)
            if sha(target/'outer-seal.json') != sha(source/'outer-seal.json'):
                raise ValueError('valid prefix copy differs from original seal')
        atomic_json(root/'copied-prefix.json', {
            'count': FAILED_INDEX, 'source_root': str(source_root),
            'first_outer_seal_sha256': expected['reused_valid_prefix_outer_seals'][0]['outer_seal_sha256'],
            'last_outer_seal_sha256': expected['reused_valid_prefix_outer_seals'][-1]['outer_seal_sha256']})
    elif json.loads((root/'continuation-provenance.json').read_text())[
            'continuation_manifest_sha256'] != sha(CONTINUATION):
        raise ValueError('continuation root provenance mismatch')
    for row in rows[:FAILED_INDEX]:
        verify_attempt(root/row_name(row), row)
    completed = FAILED_INDEX
    for row in rows[FAILED_INDEX:]:
        attempt = root/row_name(row)
        if attempt.exists():
            verify_attempt(attempt, row)
            completed += 1
            continue
        baseline = baseline_for(rows, row)
        reference = root/row_name(baseline)/'sim-output'
        role = row['candidate_id'] or 'forward_direct'
        direction = 1 if role.startswith('forward') else -1
        angle = 0 if role.endswith('direct') else int(role.rsplit('_', 1)[1])
        atomic_json(root/'status.json', {'state': 'running', 'completed': completed,
                    'total': len(rows), 'current': row_name(row),
                    'continuation_manifest_sha256': sha(CONTINUATION),
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
            atomic_json(root/'status.json', {'state': 'stopped', 'completed': completed,
                        'total': len(rows), 'current': row_name(row),
                        'failure': f'{type(exc).__name__}: {exc}',
                        'continuation_manifest_sha256': sha(CONTINUATION),
                        'updated_ns': time.time_ns()})
            raise
        completed += 1
    atomic_json(root/'status.json', {'state': 'completed', 'completed': completed,
                'total': len(rows), 'continuation_manifest_sha256': sha(CONTINUATION),
                'updated_ns': time.time_ns()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--stage-a-root', type=Path, required=True)
    options = parser.parse_args()
    run(options.root, source_root=options.source_root, stage_a_root=options.stage_a_root)
