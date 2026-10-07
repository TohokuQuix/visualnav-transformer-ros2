import hashlib
from pathlib import Path

import pytest

from eat_nav_corridor_audit import (check_sha, validate_folds, validate_records,
                                    source_frames_path, validate_live_identity,
                                    validate_label_trace, validate_manifest_counts,
                                    validate_frame_sequence)


def test_sha_mismatch_is_rejected(tmp_path):
    source = tmp_path / 'sealed.json'
    source.write_text('original')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    check_sha(source, expected)
    source.write_text('altered')
    with pytest.raises(ValueError, match='SHA mismatch'):
        check_sha(source, expected)


def test_records_require_three_direct_and_six_distinct_non_direct():
    rows = [{'group': 'g', 'family': 'f', 'candidate': 'direct_reference', 'repeat': i,
             'process_id': f'd{i}', 'terrain': 'boxes'} for i in range(3)]
    rows += [{'group': 'g', 'family': 'f', 'candidate': f'other_{i}', 'repeat': 0,
              'process_id': f'o{i}', 'terrain': 'boxes'} for i in range(6)]
    validate_records(rows, expected_groups=1)
    rows[-1]['candidate'] = 'other_0'
    with pytest.raises(ValueError, match='candidate composition'):
        validate_records(rows, expected_groups=1)


def test_fold_rejects_family_leakage():
    rows = [{'group': 'g1', 'family': 'f', 'terrain': 'boxes'},
            {'group': 'g2', 'family': 'f', 'terrain': 'boxes'}]
    folds = [{'fold_id': 'f0', 'mode': 'E-anchor', 'train': ['g1'], 'test': ['g2']}]
    with pytest.raises(ValueError, match='family leakage'):
        validate_folds(rows, folds, expected_folds=1)


def test_source_frames_mapping_preserves_old_and_new_families(tmp_path):
    assert str(source_frames_path(tmp_path, 'indep-train-013-goal_right')).endswith(
        '/inputs/state-013/recorded-frames.json')
    assert 'eatnav-expanded-feasibility-ros-continuation-20261005' in str(
        source_frames_path(tmp_path, 'expanded-train-004'))
    assert 'eatnav-e1-artifact-recovered-continuation-20261004' in str(
        source_frames_path(tmp_path, 'g0e-train-001'))


def test_live_identity_rejects_wrong_path_and_goal():
    import numpy as np
    summary = {'goal_metadata': {'base_path': [[0, 0]] + [[.1 * i, 0] for i in range(1, 9)],
                                 'goal_xy': [.8, 0]},
               'anchor': {'root_state': [0, 0, 0]}}
    path = np.array(summary['goal_metadata']['base_path'][1:])
    validate_live_identity(summary, path, np.array([.8, 0]), [0, 0, 0])
    with pytest.raises(ValueError, match='Path identity'):
        validate_live_identity(summary, path + .1, np.array([.8, 0]), [0, 0, 0])


def test_label_must_reference_sealed_trace_sha():
    validate_label_trace({'process_id': 'p', 'source_trace_sha256': 'abc', 'valid': True},
                         'p', 'abc')
    with pytest.raises(ValueError, match='label trace SHA'):
        validate_label_trace({'process_id': 'p', 'source_trace_sha256': 'wrong', 'valid': True},
                             'p', 'abc')


def test_manifest_requires_fourteen_anchor_and_189_live_processes():
    manifest = {'anchor_count': 14, 'live_count': 189, 'process_count': 203,
                'rows': [{'role': 'anchor_only'}] * 14 + [{'role': 'candidate'}] * 189}
    validate_manifest_counts(manifest)
    manifest['rows'].pop()
    with pytest.raises(ValueError, match='manifest count'):
        validate_manifest_counts(manifest)


def test_frame_history_rejects_reset_or_nonmonotonic_steps():
    frames = [{'step': i * 5, 'timestamp_ns': i * 10, 'pre_reset': False,
               'terminated': False, 'truncated': False} for i in range(4)]
    validate_frame_sequence(frames)
    frames[-1]['pre_reset'] = True
    with pytest.raises(ValueError, match='reset/termination'):
        validate_frame_sequence(frames)
