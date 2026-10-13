import json
from pathlib import Path

import numpy as np
import pytest

from visualnav_transformer.eat_nav.p1_teacher_chunks import ChunkWriter, verify_chunks


FIELDS = {'reward': {'shape': [], 'dtype': 'float32'},
          'torque': {'shape': [2], 'dtype': 'float32'},
          'stamp_ns': {'shape': [], 'dtype': 'int64'}}


def sample(i):
    return {'reward': 0.1 * i, 'torque': [i, i + 1], 'stamp_ns': i * 50_000_000}


def test_chunk_roundtrip_and_seal(tmp_path):
    with ChunkWriter(tmp_path / 'new', schema='eatnav_p1_teacher_chunks_v1',
                     fixed_shapes=FIELDS, chunk_rows=2) as writer:
        for i in range(5):
            writer.append(sample(i), ragged_contacts=[{'body': 0, 'force_N': i}])
        writer.seal()
    result = verify_chunks(tmp_path / 'new', require_complete=True)
    assert result['rows'] == 5
    assert result['chunks'] == 3
    assert result['chunk_rows'] == [2, 2, 1]
    with np.load(tmp_path / 'new/chunks/chunk-000000.npz', allow_pickle=False) as data:
        assert data['torque'].shape == (2, 2)
        assert data['torque'].dtype == np.float32
        assert data['stamp_ns'].tolist() == [0, 50_000_000]


def test_interrupted_run_preserves_only_completed_chunks(tmp_path):
    root = tmp_path / 'interrupted'
    writer = ChunkWriter(root, schema='eatnav_p1_teacher_chunks_v1',
                         fixed_shapes=FIELDS, chunk_rows=2)
    writer.append(sample(0))
    writer.append(sample(1))
    writer.append(sample(2))
    writer.close()
    assert not (root / 'complete.json').exists()
    result = verify_chunks(root, require_complete=False)
    assert result['rows'] == 2
    assert result['chunks'] == 1


def test_corrupt_sha_and_legacy_schema_fail_closed(tmp_path):
    root = tmp_path / 'sealed'
    with ChunkWriter(root, schema='eatnav_p1_teacher_chunks_v1',
                     fixed_shapes=FIELDS, chunk_rows=1) as writer:
        writer.append(sample(0))
        writer.seal()
    path = root / 'chunks/chunk-000000.npz'
    path.write_bytes(path.read_bytes() + b'bad')
    with pytest.raises(ValueError, match='chunk_sha'):
        verify_chunks(root, require_complete=True)
    other = tmp_path / 'old'; other.mkdir()
    (other / 'manifest.json').write_text(json.dumps({'schema': 'old_json'}))
    with pytest.raises(ValueError, match='schema'):
        verify_chunks(other, require_complete=False)


def test_bounded_buffer_rejects_shape_and_new_rows_after_seal(tmp_path):
    root = tmp_path / 'bounded'
    with ChunkWriter(root, schema='eatnav_p1_teacher_chunks_v1',
                     fixed_shapes=FIELDS, chunk_rows=2) as writer:
        with pytest.raises(ValueError, match='shape'):
            writer.append(dict(sample(0), torque=[1, 2, 3]))
        writer.append(sample(0))
        assert writer.buffered_rows == 1
        writer.seal()
        with pytest.raises(ValueError, match='sealed'):
            writer.append(sample(1))
