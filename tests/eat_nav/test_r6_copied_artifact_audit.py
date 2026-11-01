import gzip
import hashlib
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / 'docs/eat-nav/r6-2s-pose-20261030/execution-records/copied_artifact_audit.py'


def audit(base, proof):
    spec = importlib.util.spec_from_file_location('copied_artifact_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_copied_artifacts(base, proof)


def fixture(tmp_path):
    original = {
        'complete.json': b'{"complete":true}',
        'group-000/result.json': b'{"tip_count":0}',
        'group-000/decision-0000.json': b'{"candidate":0}',
        'group-000/initial-state.npz': b'initial binary bytes',
        'group-000/trace.jsonl': b'{"control_step":0}\n',
    }
    for name, content in original.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('trace.jsonl'):
            p.with_suffix('.jsonl.gz').write_bytes(gzip.compress(content))
        else:
            p.write_bytes(content)
    proof = {'verified': True, 'files_sha256': {
        name: hashlib.sha256(content).hexdigest() for name, content in original.items()
    }}
    return proof


def test_retained_originals_and_lossless_compressed_trace_verified(tmp_path):
    proof = fixture(tmp_path)
    assert audit(tmp_path, proof) == {'verified_original_files': 5, 'decompressed_trace_files': 1}


@pytest.mark.parametrize('name', ['complete.json', 'group-000/result.json', 'group-000/decision-0000.json', 'group-000/initial-state.npz'])
def test_changed_consumed_artifact_rejected(tmp_path, name):
    proof = fixture(tmp_path)
    (tmp_path / name).write_bytes(b'changed but structurally plausible')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        audit(tmp_path, proof)


def test_changed_decompressed_trace_rejected(tmp_path):
    proof = fixture(tmp_path)
    (tmp_path / 'group-000/trace.jsonl.gz').write_bytes(gzip.compress(b'changed\n'))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        audit(tmp_path, proof)


def test_missing_retained_original_rejected(tmp_path):
    proof = fixture(tmp_path)
    (tmp_path / 'complete.json').unlink()
    with pytest.raises(ValueError, match='Missing original'):
        audit(tmp_path, proof)


@pytest.mark.parametrize('name', ['../outside', '/tmp/outside'])
def test_unsafe_proof_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError, match='Unsafe'):
        audit(tmp_path, {'verified': True, 'files_sha256': {name: '0' * 64}})


@pytest.mark.parametrize('proof', [{'verified': False, 'files_sha256': {'complete.json': '0' * 64}}, {'verified': True, 'files_sha256': {}}])
def test_unverified_or_empty_proof_rejected(tmp_path, proof):
    with pytest.raises(ValueError, match='proof'):
        audit(tmp_path, proof)
