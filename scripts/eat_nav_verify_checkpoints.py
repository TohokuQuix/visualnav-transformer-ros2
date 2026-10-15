#!/usr/bin/env python3
"""Verify the local EAT-Nav GUI checkpoint package without loading PyTorch."""

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO/'checkpoints/eat_nav'
REQUIRED = {
    'rl_actor/model_10000.pt',
    'motion14/training/full-seed123/best.pt',
    'motion14_heading/h2a-seed123/best.pt',
    'motion14/training/source.json',
    'motion14/results.json',
    'motion14/results-ledger.json',
    'motion14/normalization.json',
    'motion14/dataset-ledger.json',
    'motion14/grouped.npz',
    'motion14/contract.json',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(package=PACKAGE):
    package = Path(package)
    manifest = json.loads((package/'manifest.json').read_text())
    if manifest.get('schema') != 'eatnav_gui_checkpoints_v2':
        raise ValueError('checkpoint package schema mismatch')
    files = manifest.get('files_sha256')
    if not isinstance(files, dict) or not REQUIRED.issubset(files):
        raise ValueError('checkpoint package file list missing')
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or len(expected) != 64:
            raise ValueError(f'unsafe checkpoint package entry: {name}')
        path = package/relative
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f'checkpoint package SHA mismatch: {name}')
    return manifest


if __name__ == '__main__':
    result = verify()
    print(f"checkpoint package valid: {len(result['files_sha256'])} files")
