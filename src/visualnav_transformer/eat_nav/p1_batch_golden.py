"""Read-only verifier for the SHA-sealed P1 28-run reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class GoldenIntegrityError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    except (OSError, ValueError) as exc:
        raise GoldenIntegrityError(f'missing or unreadable {path}') from exc
    return digest.hexdigest()


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise GoldenIntegrityError(f'missing or invalid {path.name}') from exc


def _require(condition: bool, description: str) -> None:
    if not condition:
        raise GoldenIntegrityError(description)


def verify_backup(root: Path) -> dict:
    """Verify every backed-up file and the per-run source seal before use."""
    root = Path(root)
    seal = _json(root / 'backup-seal.json')
    ledger_path = root / 'backup-sha-ledger.json'
    manifest_path = root / 'integrated-manifest.json'
    _require(_sha(ledger_path) == seal['backup_sha_ledger_sha256'], 'backup ledger SHA mismatch')
    _require(_sha(manifest_path) == seal['integrated_manifest_sha256'], 'integrated manifest SHA mismatch')
    ledger = _json(ledger_path)
    manifest = _json(manifest_path)
    _require(manifest.get('old_tenth_excluded') is True, 'old tenth must be excluded')
    _require(manifest.get('row_count') == 28 and len(manifest.get('rows', [])) == 28,
             'expected exactly 28 rows')
    _require(ledger.get('file_count') == seal.get('verified_file_count') == len(ledger['files']),
             'backup file count mismatch')
    for entry in ledger['files']:
        rel = Path(entry['path'])
        _require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe ledger path')
        file = root / rel
        try:
            size = file.stat().st_size
        except OSError as exc:
            raise GoldenIntegrityError(f'missing backup file: {rel}') from exc
        _require(size == entry['bytes'], f'backup size mismatch: {rel}')
        _require(_sha(file) == entry['sha256'], f'backup SHA mismatch: {rel}')
    run_ids = []
    trace_steps = 0
    for row in manifest['rows']:
        run_id = row['physical_run_id']
        _require(run_id not in run_ids and Path(run_id).name == run_id, 'duplicate/unsafe run ID')
        run_ids.append(run_id)
        run_root = root / 'runs' / run_id
        _require(_sha(run_root / 'seal.json') == row['source_seal_sha256'],
                 f'run seal mismatch: {run_id}')
        _require(_sha(run_root / 'raw-sha-ledger.json') == row['original_raw_sha_ledger_sha256'],
                 f'raw ledger mismatch: {run_id}')
        _require(_sha(run_root / 'run' / 'candidate-trace.jsonl') == row['source_trace_sha256'],
                 f'trace mismatch: {run_id}')
        run_seal = _json(run_root / 'seal.json')
        result = _json(run_root / 'result.json')
        _require(run_seal['native_exit_code'] == result['native_exit_code'] == 0,
                 f'native exit: {run_id}')
        _require(run_seal['owned_residual_count'] == result['owned_residual_count'] == 0,
                 f'owned residual: {run_id}')
        _require(result['decision'] == 'p1_candidate_protocol_valid',
                 f'protocol invalid: {run_id}')
        with (run_root / 'run' / 'candidate-trace.jsonl').open() as stream:
            steps = sum(1 for line in stream if line.strip())
        _require(steps == 120, f'trace step count: {run_id}')
        trace_steps += steps
    return {
        'schema': 'p1_batch_golden_audit_v1',
        'backup_seal_sha256': _sha(root / 'backup-seal.json'),
        'integrated_manifest_sha256': _sha(manifest_path),
        'run_count': len(run_ids),
        'run_ids': run_ids,
        'verified_file_count': ledger['file_count'],
        'trace_steps': trace_steps,
        'old_tenth_excluded': manifest['old_tenth_excluded'],
    }
