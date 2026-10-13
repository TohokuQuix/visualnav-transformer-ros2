"""Bounded typed chunks with per-chunk SHA and explicit finalization."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SCHEMA = 'eatnav_p1_teacher_chunks_v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class ChunkWriter:
    def __init__(self, root, *, schema, fixed_shapes, chunk_rows=256, metadata=None):
        if schema != SCHEMA or not isinstance(chunk_rows, int) or not 1 <= chunk_rows <= 4096:
            raise ValueError('chunk_schema_or_size')
        self.root = Path(root)
        if self.root.exists():
            raise FileExistsError(self.root)
        self.root.mkdir(parents=True)
        (self.root / 'chunks').mkdir()
        self.fields = fixed_shapes
        for name, spec in self.fields.items():
            if not name.isidentifier() or np.dtype(spec['dtype']).hasobject:
                raise ValueError('field_schema')
        self.chunk_rows = chunk_rows
        self.buffer = []
        self.ragged = []
        self.ledger = []
        self.sealed = False
        self.closed = False
        _write_new_json(self.root / 'manifest.json',
                        {'schema': SCHEMA, 'fixed_fields': fixed_shapes,
                         'chunk_rows': chunk_rows, 'ragged_table': 'contact_records',
                         'metadata': metadata or {}})
        _atomic_json(self.root / 'chunk-ledger.json', {'schema': SCHEMA, 'chunks': []})

    @property
    def buffered_rows(self):
        return len(self.buffer)

    def append(self, row, ragged_contacts=None):
        if self.sealed or self.closed:
            raise ValueError('sealed_or_closed')
        if set(row) != set(self.fields):
            raise ValueError('field_names')
        normalized = {}
        for name, spec in self.fields.items():
            array = np.asarray(row[name], dtype=np.dtype(spec['dtype']))
            if list(array.shape) != spec['shape']:
                raise ValueError('shape_mismatch:' + name)
            if array.dtype.kind == 'f' and not np.isfinite(array).all():
                raise ValueError('nonfinite:' + name)
            normalized[name] = array.copy()
        self.buffer.append(normalized)
        self.ragged.append(list(ragged_contacts or []))
        if len(self.buffer) == self.chunk_rows:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return
        index = len(self.ledger)
        stem = f'chunk-{index:06d}'
        data_tmp = self.root / 'chunks' / (stem + '.tmp.npz')
        data_final = self.root / 'chunks' / (stem + '.npz')
        with data_tmp.open('xb') as stream:
            np.savez_compressed(stream, **{name: np.stack([row[name] for row in self.buffer])
                                           for name in self.fields})
            stream.flush(); os.fsync(stream.fileno())
        os.replace(data_tmp, data_final)
        ragged_tmp = self.root / 'chunks' / (stem + '.tmp.jsonl.gz')
        ragged_final = self.root / 'chunks' / (stem + '.jsonl.gz')
        with ragged_tmp.open('xb') as stream:
            with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as zipped:
                for events in self.ragged:
                    zipped.write((json.dumps(events, sort_keys=True, allow_nan=False) + '\n').encode())
            stream.flush(); os.fsync(stream.fileno())
        os.replace(ragged_tmp, ragged_final)
        self.ledger.append({'index': index, 'rows': len(self.buffer),
                            'data': data_final.name, 'data_sha256': _sha(data_final),
                            'ragged': ragged_final.name,
                            'ragged_sha256': _sha(ragged_final)})
        _atomic_json(self.root / 'chunk-ledger.json', {'schema': SCHEMA, 'chunks': self.ledger})
        self.buffer.clear()
        self.ragged.clear()

    def seal(self):
        if self.sealed or self.closed:
            raise ValueError('sealed_or_closed')
        self._flush()
        evidence = verify_chunks(self.root, require_complete=False)
        _write_new_json(self.root / 'complete.json',
                        {'schema': SCHEMA, 'rows': evidence['rows'],
                         'chunk_ledger_sha256': _sha(self.root / 'chunk-ledger.json'),
                         'manifest_sha256': _sha(self.root / 'manifest.json')})
        self.sealed = True

    def close(self):
        self.closed = True
        self.buffer.clear()
        self.ragged.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def verify_chunks(root, *, require_complete=True):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA:
        raise ValueError('chunk_schema')
    ledger = json.loads((root / 'chunk-ledger.json').read_text())
    if manifest.get('schema') != SCHEMA or ledger.get('schema') != SCHEMA:
        raise ValueError('chunk_schema')
    counts = []
    for index, item in enumerate(ledger['chunks']):
        if item['index'] != index or Path(item['data']).name != item['data'] or Path(item['ragged']).name != item['ragged']:
            raise ValueError('chunk_index_or_path')
        data = root / 'chunks' / item['data']
        ragged = root / 'chunks' / item['ragged']
        if _sha(data) != item['data_sha256'] or _sha(ragged) != item['ragged_sha256']:
            raise ValueError('chunk_sha_mismatch')
        with np.load(data, allow_pickle=False) as arrays:
            if set(arrays.files) != set(manifest['fixed_fields']):
                raise ValueError('chunk_field_names')
            for name, spec in manifest['fixed_fields'].items():
                array = arrays[name]
                if list(array.shape) != [item['rows'], *spec['shape']] or array.dtype != np.dtype(spec['dtype']):
                    raise ValueError('chunk_shape_or_dtype')
        with gzip.open(ragged, 'rt') as stream:
            if sum(1 for _ in stream) != item['rows']:
                raise ValueError('ragged_row_count')
        counts.append(item['rows'])
    if require_complete:
        complete = json.loads((root / 'complete.json').read_text())
        if complete.get('schema') != SCHEMA or complete['rows'] != sum(counts) or \
                complete['chunk_ledger_sha256'] != _sha(root / 'chunk-ledger.json') or \
                complete['manifest_sha256'] != _sha(root / 'manifest.json'):
            raise ValueError('chunk_complete_or_sha')
    return {'rows': sum(counts), 'chunks': len(counts), 'chunk_rows': counts}
