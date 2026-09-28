"""Standard-library-only atomic status and SHA marker helpers for the host runner."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        json.dump(payload,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno());name=stream.name
    os.replace(name,path)


def make_stage_marker(stage: str, inputs: dict[str,str], config_sha: str,
                      outputs: dict[str,Path], seeds: list[int]) -> dict:
    return {'stage':stage,'input_sha256':inputs,'config_sha256':config_sha,'seeds':seeds,
            'output_sha256':{name:{'path':str(path),'sha256':sha256_file(path)} for name,path in outputs.items()}}


def verify_stage_marker(marker: dict, inputs: dict[str,str], config_sha: str) -> bool:
    if marker.get('input_sha256')!=inputs or marker.get('config_sha256')!=config_sha:return False
    outputs=marker.get('output_sha256',{})
    if not outputs:return False
    try:return all(sha256_file(Path(value['path']))==value['sha256'] for value in outputs.values())
    except (OSError,KeyError,TypeError):return False
