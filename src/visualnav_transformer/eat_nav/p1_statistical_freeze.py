"""SHA verification and prospective P1 assignment freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .p1_statistical_protocol import build_assignments


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def verify_ledger(root: Path, ledger_path: Path) -> dict:
    root = Path(root)
    ledger = json.loads(Path(ledger_path).read_text())
    files = ledger['files']
    if len(files) != ledger['file_count']:
        raise ValueError('ledger file count mismatch')
    for row in files:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe ledger path')
        file = root/relative
        if not file.is_file() or file.stat().st_size != row['bytes'] or sha(file) != row['sha256']:
            raise ValueError(f"SHA mismatch: {relative}")
    return {'file_count':len(files), 'ledger_sha256':sha(ledger_path)}


def _atomic(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def freeze_manifest(source_contract: Path, config: Path, output: Path) -> dict:
    source_contract, config, output = Path(source_contract), Path(config), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = json.loads(source_contract.read_text())
    cfg = json.loads(config.read_text())
    if cfg['policy_action_mode'] != 'act_inference' or cfg['paired_policy_noise']:
        raise ValueError('P1 inference/noise contract')
    assignment = build_assignments(source,cfg['batch_seeds'],cfg['assignment_randomization_seed'])
    output.mkdir(parents=True)
    _atomic(output/'assignment.json',assignment)
    result = {'schema':'p1_statistical_freeze_v1',
              'source_contract_path':str(source_contract.resolve()),
              'source_contract_sha256':sha(source_contract),
              'config_path':str(config.resolve()),'config_sha256':sha(config),
              'assignment_sha256':sha(output/'assignment.json'),
              'trial_count':len(assignment['trials']),
              'policy_action_mode':'act_inference','paired_policy_noise':False}
    _atomic(output/'freeze.json',result)
    return result
