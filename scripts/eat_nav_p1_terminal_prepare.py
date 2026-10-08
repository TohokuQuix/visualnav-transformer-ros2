"""Freeze P1 terminal-monitor provenance before any real process starts."""
from __future__ import annotations

import json
from pathlib import Path

from eat_nav_p1_handshake_prepare import (WORKTREE, R5, PARITY, atomic,
                                          freeze_overlay, sha, verify_sha_entry)
from eat_nav_radial_anchor_contract import verify_output_overlay


BASE = '305bdc27eec19296a3316fed33b70fbc3d5068ee'
PRIOR = Path('/tmp/eatnav-p1-handshake-20261008-r2')
BACKUP = Path('/home/kojima/eatnav-backups/2026-10-08-p1-handshake-r2')
PRIOR_REPORT = Path('/tmp/eatnav-vnt-p1-handshake/docs/eat-nav/p1-handshake-resource-stop-20261008')
CODE_FILES = (
    'scripts/eat_nav_p1_owner_wrapper.py', 'scripts/eat_nav_p1_owner_host.py',
    'scripts/eat_nav_p1_gpu_attribution.py', 'scripts/eat_nav_p1_terminal_lifecycle.py',
    'scripts/eat_nav_p1_terminal_prepare.py', 'tests/test_p1_terminal_lifecycle.py',
    'tests/test_p1_terminal_host.py', 'tests/test_p1_terminal_prepare.py',
    'tests/test_p1_owner_wrapper.py', 'tests/test_p1_owner_host.py',
    'tests/test_p1_attribution_smoke.py')


def compare_sha_trees(source: Path, backup: Path) -> dict:
    def ledger(root):
        return {str(path.relative_to(root)): sha(path) for path in root.rglob('*') if path.is_file()}
    a, b = ledger(source), ledger(backup)
    if a != b:
        raise ValueError('backup_SHA_mismatch')
    return {'file_count': len(a), 'files': a}


def verify_frozen_code(root: Path) -> dict:
    freeze = json.loads((root / 't0-frozen-code.json').read_text())
    for name, digest in freeze['files'].items():
        path = Path(name)
        try:
            verify_sha_entry(path if path.is_absolute() else WORKTREE / path, digest)
        except ValueError as exc:
            raise ValueError('frozen_code_SHA_drift:' + name) from exc
    if 'resource_contract_sha256' in freeze:
        verify_sha_entry(R5 / 'p1-resource-contract-v2.json', freeze['resource_contract_sha256'])
    if 'overlay_sha256' in freeze:
        verify_sha_entry(root / 'p1-run-manifest-output-overlay.json', freeze['overlay_sha256'])
    return {'file_count': len(freeze['files'])}


def prepare(root: Path) -> dict:
    if root.exists():
        raise ValueError('fresh_terminal_output_root_required')
    compared = compare_sha_trees(PRIOR, BACKUP)
    prior_result = json.loads((PRIOR / 'smoke/result.json').read_text())
    if (prior_result['decision'] != 'p1_resource_attribution_fail'
            or 'simulator_disappeared_in_bracket' not in prior_result['first_failing_signal']):
        raise ValueError('prior_H5_decision_drift')
    brackets = (PRIOR / 'smoke/gpu-brackets.jsonl').read_text().splitlines()
    if len(brackets) != 33:
        raise ValueError('prior_33_brackets_missing')
    if json.loads((PRIOR / 'preflight/result.json').read_text())['decision'] != 'p1_startup_preflight_pass':
        raise ValueError('prior_H4_marker_invalid')
    archive_ledger = json.loads((PRIOR / 'smoke/poststop-archive/poststop-sha-ledger.json').read_text())
    if archive_ledger.get('file_count') != 94 or not archive_ledger.get('match'):
        raise ValueError('prior_archive_ledger_invalid')
    for item in archive_ledger['files']:
        verify_sha_entry(Path(item['local']), item['sha256'])
    previous_ledger = json.loads((PRIOR_REPORT / 'artifact-sha-ledger.json').read_text())
    for item in previous_ledger['files']:
        verify_sha_entry(Path(item['path']), item['sha256'])
    parity_ledger = json.loads((PARITY / 'artifact-sha-ledger.json').read_text())
    for item in parity_ledger['small_report_files']:
        verify_sha_entry(Path(item['path']), item['sha256'])
    frozen = json.loads((R5 / 'input-freeze.json').read_text())
    for key, name in (('manifest_sha256', 'p1-run-manifest-r3.json'),
                      ('primitive_paths_sha256', 'frozen-primitive-paths-r3.json'),
                      ('selected_four_sha256', 'selected-four-anchors-r3.json')):
        verify_sha_entry(R5 / name, frozen[key])
    previous_manifest = json.loads((PRIOR / 'smoke/wrapper-invocation.json').read_text())
    verify_sha_entry(R5 / 'p1-resource-contract-v2.json',
                     json.loads((PRIOR / 'smoke/coord/launch-intent.json').read_text())['resource_contract_sha256'])
    manifest = json.loads((R5 / 'p1-run-manifest-r3.json').read_text())
    overlay, diff = freeze_overlay(manifest, root)
    verify_output_overlay(manifest, overlay)
    code_freeze = {path: sha(WORKTREE / path) for path in CODE_FILES}
    root.mkdir(parents=True)
    atomic(root / 't0-provenance.json', {'base_commit': BASE, 'prior_H5_decision': prior_result,
           'prior_H4_SHA': sha(PRIOR / 'preflight/result.json'),
           'prior_33_bracket_SHA': sha(PRIOR / 'smoke/gpu-brackets.jsonl'),
           'prior_archive_ledger_SHA': sha(PRIOR / 'smoke/poststop-archive/poststop-sha-ledger.json'),
           'prior_backup_file_count': compared['file_count'],
           'prior_backup_matching': True, 'prior_roots_modified': False,
           'parity_ledger_SHA': sha(PARITY / 'artifact-sha-ledger.json'),
           'source_manifest_SHA': sha(R5 / 'p1-run-manifest-r3.json'),
           'source_paths_SHA': sha(R5 / 'frozen-primitive-paths-r3.json'),
           'selected_anchors_SHA': sha(R5 / 'selected-four-anchors-r3.json'),
           'resource_contract_SHA': sha(R5 / 'p1-resource-contract-v2.json'),
           'previous_wrapper_SHA': previous_manifest['wrapper_sha256']})
    atomic(root / 'p1-run-manifest-output-overlay.json', overlay)
    atomic(root / 'p1-run-manifest-output-diff.json', diff)
    atomic(root / 't0-frozen-code.json', {'base_commit': BASE, 'files': code_freeze,
           'resource_contract_sha256': sha(R5 / 'p1-resource-contract-v2.json'),
           'overlay_sha256': sha(root / 'p1-run-manifest-output-overlay.json')})
    atomic(root / 't0-prepared.json', {'schema': 'eatnav_p1_terminal_t0_v1',
           'status': 'prepared', 'code_file_count': len(code_freeze),
           'overlay_row_count': len(overlay['rows']), 'real_process_count': 0})
    return {'root': str(root), 'prior_brackets': len(brackets),
            'backup_files_verified': compared['file_count'],
            'code_files_frozen': len(code_freeze), 'overlay_rows': len(overlay['rows'])}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_root), sort_keys=True))
