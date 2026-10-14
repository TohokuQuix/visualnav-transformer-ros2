"""Read-only importer for complete, sealed fourteen-candidate paired groups."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .bidirectional_labels import motion_label_with_yaw
from .motion_14_predictor import CANDIDATE_IDS, validate_grouped_arrays
from .mi_b_pilot import split_indices


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_grouped_dataset(evidence_root: Path, *, smoke_only: bool = False):
    """Verify every input attempt and construct 288 groups (or eight smoke groups)."""
    from scripts.eat_nav_motion_14_collection_run import (
        MANIFEST, load_frozen, row_name, verify_attempt,
    )

    evidence_root = Path(evidence_root)
    frozen = load_frozen()
    continuation_audit = None
    continuation_file = evidence_root/'continuation-provenance.json'
    if continuation_file.is_file():
        from scripts.eat_nav_motion_14_collection_resume_v2 import CONTINUATION
        continuation = json.loads(CONTINUATION.read_text())
        provenance = json.loads(continuation_file.read_text())
        if provenance.get('continuation_manifest_sha256') != sha(CONTINUATION) or \
                provenance.get('scientific_manifest_sha256') != sha(MANIFEST) or \
                continuation['scientific_manifest_sha256'] != sha(MANIFEST) or \
                continuation['reused_valid_prefix_count'] != 32:
            raise ValueError('continuation provenance/manifest mismatch')
        source = Path(continuation['source_root'])
        for item in continuation['reused_valid_prefix_outer_seals']:
            row = frozen['assignment'][item['index']]
            name = row_name(row)
            if sha(source/name/'outer-seal.json') != item['outer_seal_sha256'] or \
                    sha(evidence_root/name/'outer-seal.json') != item['outer_seal_sha256']:
                raise ValueError('copied valid prefix diverged from original')
        failed = source/row_name(frozen['assignment'][continuation['excluded_failed_attempt_index']])
        if sha(failed/'outer-seal.json') != continuation['excluded_failed_outer_seal_sha256']:
            raise ValueError('excluded failed attempt changed')
        continuation_audit = {'continuation_manifest_sha256': sha(CONTINUATION),
                              'original_source_root': str(source),
                              'reused_valid_prefix_count': 32,
                              'excluded_failed_attempt_index': 32,
                              'failed_attempt_retained_outside_dataset': True}
    status = json.loads((evidence_root/'status.json').read_text())
    if smoke_only:
        if status.get('state') != 'smoke_pass' or status.get('completed') != 32:
            raise ValueError('smoke not completed')
        rows = frozen['assignment'][:32]
    else:
        if status.get('state') != 'completed' or status.get('completed') != 1152:
            raise ValueError('full collection not completed')
        rows = frozen['assignment']
    by_group = {}
    seals = []
    for row in rows:
        attempt = evidence_root/row_name(row)
        verify_attempt(attempt, row)
        seals.append({'row_index': row['index'], 'outer_seal_sha256': sha(attempt/'outer-seal.json')})
        key = (row['terrain_seed'], row['flipper_initialization_mode'])
        by_group.setdefault(key, {})[(row['stage'], row['repeat_index'], row['candidate_id'])] = row
    arrays = {name: [] for name in ('dem', 'state', 'paths', 'direction', 'mask', 'target', 'yaw')}
    metadata = []
    for (seed, posture), group in sorted(by_group.items()):
        if len(group) != 16:
            raise ValueError('incomplete paired 14-Path group')
        reference_row = group['prepath', 0, None]
        reference_path = evidence_root/row_name(reference_row)/'sim-output/prepath.npz'
        with np.load(reference_path, allow_pickle=False) as z:
            reference = {key: z[key].copy() for key in
                         ('root', 'joint', 'actor_pre_noise', 'dem')}
        paths, labels, yaw_labels = [], [], []
        for role in CANDIDATE_IDS:
            row = group['candidate', 0, role]
            output = evidence_root/row_name(row)/'sim-output'
            input_manifest = json.loads((output/'input-manifest.json').read_text())
            path = np.asarray(input_manifest['path_body_xy_m'], dtype='<f8')
            if path.shape != (8, 2) or sha_bytes(path.tobytes()) != input_manifest['path_body_sha256']:
                raise ValueError('Path coordinates/SHA mismatch')
            with np.load(output/'prepath.npz', allow_pickle=False) as z:
                for field, baseline in reference.items():
                    current = z[field]
                    if current.shape != baseline.shape or not np.isfinite(current).all() or \
                            np.max(np.abs(current.astype(np.float64) -
                                          baseline.astype(np.float64))) > 1e-6:
                        raise ValueError(f'cross-Path pre-Path mismatch: {field}')
            stored = json.loads((output/'motion-labels.json').read_text())['labels']
            xy, yaw = [], []
            for env in range(4):
                trace = [json.loads(line) for line in
                         (output/f'trace-env-{env}.jsonl').read_text().splitlines()]
                derived = motion_label_with_yaw(trace)
                if len(trace) != 40 or not derived['available'] or \
                        not np.allclose(derived['delta_body_xy_m'], stored[env]['delta_body_xy_m'],
                                        atol=1e-9, rtol=0) or \
                        abs(derived['delta_body_yaw_rad'] - stored[env]['delta_body_yaw_rad']) > 1e-9:
                    raise ValueError('sealed 40-step label mismatch')
                xy.append(stored[env]['delta_body_xy_m'])
                yaw.append(stored[env]['delta_body_yaw_rad'])
            paths.append(path.astype(np.float32))
            labels.append(np.asarray(xy, np.float32))
            yaw_labels.append(np.asarray(yaw, np.float32))
        path_array = np.stack(paths)
        target = np.stack(labels, axis=1)
        yaw = np.stack(yaw_labels, axis=1)
        direction = np.asarray([1] * 7 + [-1] * 7, np.float32)[:, None]
        for env in range(4):
            arrays['dem'].append(reference['dem'][env].astype(np.float32))
            arrays['state'].append(reference['actor_pre_noise'][env].astype(np.float32))
            arrays['paths'].append(path_array)
            arrays['direction'].append(direction)
            arrays['mask'].append(np.ones(14, bool))
            arrays['target'].append(target[env])
            arrays['yaw'].append(yaw[env])
            metadata.append({'terrain_seed': seed, 'posture': posture, 'env_tile_id': env,
                             'terrain_type': reference_row['terrain_type'],
                             'terrain_level': reference_row['terrain_level'],
                             'state_group_id': reference_row['state_group_id'],
                             'split': reference_row['split']})
    arrays = {name: np.stack(values) for name, values in arrays.items()}
    validate_grouped_arrays(arrays)
    if arrays['yaw'].shape != (len(metadata), 14) or not np.isfinite(arrays['yaw']).all():
        raise ValueError('yaw diagnostic shape/finite contract')
    split = split_indices(metadata, frozen['split_seeds']) if not smoke_only else None
    audit = {'schema': 'eatnav_motion_14_dataset_v1', 'smoke_only': smoke_only,
             'source_manifest_sha256': sha(MANIFEST), 'source_root': str(evidence_root),
             'verified_process_count': len(seals), 'candidate_trial_count': len(metadata) * 14,
             'group_count': len(metadata), 'source_outer_seals': seals,
             'split_group_count': {name: len(ids) for name, ids in split.items()} if split else None,
             'continuation': continuation_audit}
    return arrays, metadata, audit


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
