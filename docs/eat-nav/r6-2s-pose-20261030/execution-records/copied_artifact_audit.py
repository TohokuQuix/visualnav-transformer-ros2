"""Verify retained original bytes against the completed simulator copy manifest.

Losslessly compressed trace.jsonl files are checked after decompression. This
checks consistency with the saved manifest, not independent manifest authenticity.
"""
import gzip
import hashlib
from pathlib import Path, PurePosixPath


def verify_copied_artifacts(base, proof):
    files = proof.get('files_sha256')
    if proof.get('verified') is not True or not isinstance(files, dict) or not files:
        raise ValueError('Unverified or empty copy proof')
    base = Path(base).resolve()
    compressed = 0
    for name, expected in files.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError(f'Unsafe copy proof path: {name}')
        original = base / name
        use_gzip = not original.is_file() and original.name == 'trace.jsonl'
        actual = original.with_suffix('.jsonl.gz') if use_gzip else original
        if not actual.resolve().is_relative_to(base):
            raise ValueError(f'Unsafe copy proof path: {name}')
        if not actual.is_file():
            raise ValueError(f'Missing original copy artifact: {name}')
        digest = hashlib.sha256()
        opener = gzip.open if use_gzip else open
        with opener(actual, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'SHA256 mismatch for original copy artifact: {name}')
        compressed += int(use_gzip)
    return {'verified_original_files': len(files), 'decompressed_trace_files': compressed}
