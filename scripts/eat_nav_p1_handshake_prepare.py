"""Read-only provenance audit and frozen P1 output overlay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


WORKTREE = Path(__file__).resolve().parents[1]
R5 = WORKTREE / 'docs/eat-nav/local-primitive-p1-r5-20261008-attempt2'
PARITY = WORKTREE / 'docs/eat-nav/eatnav-radial-anchor-contract-20261008-r2'
BACKUP = Path('/home/kojima/eatnav-backups/2026-10-08-p1-resource-attribution-stop')
ATTEMPTS = [Path(f'/tmp/eatnav-p1-resource-rollout-20261008-attempt{i}') for i in (1, 2)]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha_entry(path: Path, expected: str) -> dict:
    path = Path(path)
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f'sha_mismatch:{path}')
    return {'path': str(path), 'sha256': expected, 'bytes': path.stat().st_size}


def atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def freeze_overlay(manifest: dict, root: Path) -> tuple[dict, list[dict]]:
    if len(manifest['rows']) != 28 or any(row['order'] != i for i, row in enumerate(manifest['rows'])):
        raise ValueError('frozen_28_row_order_invalid')
    overlay = json.loads(json.dumps(manifest))
    diff = []
    for before, after in zip(manifest['rows'], overlay['rows'], strict=True):
        after['output_directory'] = str(root / 'runs' / before['physical_run_id'])
        changed = {key: {'old': before[key], 'new': after[key]} for key in before if before[key] != after[key]}
        if set(changed) != {'output_directory'}:
            raise ValueError('manifest_logical_identity_changed')
        diff.append({'physical_run_id': before['physical_run_id'], 'changed': changed})
    return overlay, diff


def prepare(root: Path) -> dict:
    if root.exists():
        raise ValueError('new_output_root_required')
    files = []
    parity_ledger = json.loads((PARITY / 'artifact-sha-ledger.json').read_text())
    for item in parity_ledger['small_report_files']:
        files.append(verify_sha_entry(Path(item['path']), item['sha256']))
    for item in parity_ledger['raw_shard_seals']:
        shard = Path(item['subroot'])
        files.append(verify_sha_entry(shard / 'seal.json', item['seal_sha256']))
        files.append(verify_sha_entry(shard / 'sha-ledger.json', item['raw_sha_ledger_sha256']))
    freeze = json.loads((R5 / 'input-freeze.json').read_text())
    pinned = {'manifest_sha256': 'p1-run-manifest-r3.json',
              'primitive_paths_sha256': 'frozen-primitive-paths-r3.json',
              'selected_four_sha256': 'selected-four-anchors-r3.json',
              'gate_config_sha256': 'minimal-p1-gate-v2.json',
              'source_provenance_sha256': 'source-provenance.json'}
    for key, name in pinned.items():
        files.append(verify_sha_entry(R5 / name, freeze[key]))
    # The earlier r3 input freeze records a pre-v2 resource SHA. The immediately
    # preceding failed smoke pinned the current v2 file separately; never infer
    # a new threshold from that stale historical field.
    preceding = json.loads((ATTEMPTS[1] / 'attribution-smoke-manifest.json').read_text())
    files.append(verify_sha_entry(R5 / 'p1-resource-contract-v2.json', preceding['resource_contract_sha256']))
    provenance = json.loads((R5 / 'source-provenance.json').read_text())
    for item in provenance['source_paths'].values():
        files.append(verify_sha_entry(Path(item['path']), item['sha256']))
    for old in ATTEMPTS:
        if not old.exists():
            raise ValueError(f'prior_attempt_missing:{old}')
        for path in sorted(old.rglob('*')):
            if path.is_file():
                relative = path.relative_to(old)
                files.append(verify_sha_entry(BACKUP / old.name / relative, sha(path)))
    files.append({'path': str(PARITY / 'artifact-sha-ledger.json'), 'sha256': sha(PARITY / 'artifact-sha-ledger.json')})
    files.append({'path': str(WORKTREE / 'docs/eat-nav/p1-resource-attribution-stop-20261008/report.md'),
                  'sha256': sha(WORKTREE / 'docs/eat-nav/p1-resource-attribution-stop-20261008/report.md')})
    manifest = json.loads((R5 / 'p1-run-manifest-r3.json').read_text())
    overlay, diff = freeze_overlay(manifest, root)
    # Existing contract performs the independent field-by-field check.
    from eat_nav_radial_anchor_contract import verify_output_overlay
    verify_output_overlay(manifest, overlay)
    root.mkdir(parents=True)
    atomic(root / 'h0-provenance-audit.json', {'base_commit': 'a2915537c124ea59061061c8625273a4310c3b30',
           'verified_files': files, 'verified_count': len(files), 'previous_smokes_promoted': False,
           'historical_resource_sha': freeze['resource_contract_sha256'],
           'current_v2_resource_sha': preceding['resource_contract_sha256'],
           'resource_sha_history_note': 'r3 input-freeze predates current v2 contract; attempt2 manifest pins current v2'})
    atomic(root / 'p1-run-manifest-output-overlay.json', overlay)
    atomic(root / 'p1-run-manifest-output-diff.json', diff)
    atomic(root / 'h1-root-cause.json', {'cause': 'GPU process visible before owner resolution',
           'attempt2': 'in-bracket lookup still raced; prior attempt entered physics before second attempt',
           'solution_boundary': 'owner identity and GPU monitor established before authorization'})
    return {'root': str(root), 'verified_count': len(files), 'overlay_sha256': sha(root / 'p1-run-manifest-output-overlay.json')}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_root), sort_keys=True))
