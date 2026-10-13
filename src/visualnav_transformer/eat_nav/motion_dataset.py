"""Independent-StateGroup split and train-only motion-label normalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .motion_collector import compare_prepath
from .motion_vnext import motion_label_from_trace


def validate_state_group_split(samples, split, *, extra_terrain_seeds=None):
    if set(split) != {'train', 'validation', 'test'}:
        raise ValueError('train/validation/test split required')
    sample_for = {sample['id']: sample for sample in samples}
    if len(sample_for) != len(samples):
        raise ValueError('duplicate sample ID')
    ids = [item for part in ('train', 'validation', 'test') for item in split[part]]
    if len(ids) != len(set(ids)) or set(ids) != set(sample_for):
        raise ValueError('split sample coverage/duplicate')
    by_group = {part: {sample_for[item]['group'] for item in split[part]} for part in split}
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if by_group[left] & by_group[right]:
            raise ValueError('StateGroup split leakage')
    def seed(sample):
        group = sample['group']
        return (extra_terrain_seeds or {}).get(group, sample['terrain_seed'])
    by_seed = {part: {seed(sample_for[item]) for item in split[part]} for part in split}
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if by_seed[left] & by_seed[right]:
            raise ValueError('terrain seed split leakage')


def fit_motion_normalization(train_labels) -> dict:
    labels = np.asarray(train_labels, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] != 2 or len(labels) < 2 or \
            not np.isfinite(labels).all():
        raise ValueError('train-only motion labels required')
    mean = labels.mean(axis=0)
    std = labels.std(axis=0)
    if np.any(std < 1e-8):
        raise ValueError('motion dimension has insufficient train variation')
    return {'mean': mean.tolist(), 'std': std.tolist(), 'fitted_from': 'train_only'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_arc_root(root):
    """Verify both immutable inner and outer ledgers before reading any label."""
    root = Path(root)
    outer = json.loads((root/'outer-seal.json').read_text())
    ledger_path = root/'outer-sha-ledger.json'
    ledger = json.loads(ledger_path.read_text())
    actual = {str(path.relative_to(root)): _sha(path) for path in root.rglob('*')
              if path.is_file() and path.name not in ('outer-sha-ledger.json', 'outer-seal.json')}
    if actual != ledger or _sha(ledger_path) != outer['ledger_sha256'] or \
            not json.loads((root/'decision.json').read_text())['pass']:
        raise ValueError('Arc outer seal/decision mismatch')
    output = root/'sim-output'
    inner = json.loads((output/'host-seal.json').read_text())
    inner_path = output/'sha-ledger.json'
    inner_ledger = json.loads(inner_path.read_text())
    inner_actual = {str(path.relative_to(output)): _sha(path) for path in output.rglob('*')
                    if path.is_file() and path.name not in ('sha-ledger.json', 'host-seal.json')}
    if inner_actual != inner_ledger or _sha(inner_path) != inner['ledger_sha256']:
        raise ValueError('Arc inner seal mismatch')
    return {'outer_seal_sha256': _sha(root/'outer-seal.json'),
            'inner_seal_sha256': _sha(output/'host-seal.json')}


def load_sealed_arc_bank(roots, *, required_angles):
    """Assemble four paired groups from one seed; never regard Paths as independent."""
    roots = [Path(root) for root in roots]
    if len(roots) != len(required_angles) or len(set(required_angles)) != len(required_angles):
        raise ValueError('Arc Bank root/angle count')
    records = []
    reference = None
    for root, expected_angle in zip(roots, required_angles):
        provenance = verify_arc_root(root)
        output = root/'sim-output'
        manifest = json.loads((output/'input-manifest.json').read_text())
        identity = json.loads((output/'prepath-identity.json').read_text())
        with np.load(output/'prepath.npz', allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in ('root', 'joint', 'actor_pre_noise', 'dem')}
        state = {**identity, **arrays}
        if manifest['arc_angle_deg'] != expected_angle or \
                not json.loads((output/'prepath-pair-gate.json').read_text())['pass']:
            raise ValueError('Arc role/prepath gate mismatch')
        if reference is None:
            reference = state
        elif not compare_prepath(reference, state)['pass']:
            raise ValueError('Arc Bank cross-candidate pre-Path mismatch')
        labels = json.loads((output/'motion-labels.json').read_text())['labels']
        if len(labels) != 4:
            raise ValueError('Arc label/env count')
        for env in range(4):
            trace_path = output/f'trace-env-{env}.jsonl'
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            derived = motion_label_from_trace(trace)
            if derived != labels[env]:
                raise ValueError('Arc motion label differs from sealed trace')
        records.append({'angle_deg': expected_angle, 'path': np.asarray(
                        manifest['path_body_xy_m'], dtype=np.float32),
                        'labels': labels, 'traces': [str(output/f'trace-env-{i}.jsonl')
                                                   for i in range(4)],
                        'provenance': provenance})
    groups = []
    for env in range(4):
        tile = reference['terrain_tiles'][env]
        group_id = f"{reference['seed']}:{env}:{tile['seed']}:{tile['geometry_sha256']}"
        groups.append({'id': group_id, 'group': group_id,
                       'terrain_seed': reference['seed'], 'tile_seed': tile['seed'],
                       'terrain_type': tile['terrain_type'], 'difficulty': tile['difficulty'],
                       'env_id': env, 'dem': reference['dem'][env].astype(np.float32),
                       'actor_state': reference['actor_pre_noise'][env].astype(np.float32),
                       'paths': np.stack([record['path'] for record in records]),
                       'candidate_angles_deg': list(required_angles),
                       'labels': np.asarray([record['labels'][env]['delta_body_xy_m']
                           if record['labels'][env]['available'] else [np.nan, np.nan]
                           for record in records], dtype=np.float32),
                       'label_mask': np.asarray([record['labels'][env]['available']
                                                for record in records], dtype=bool),
                       'provenance': [{'trace': record['traces'][env],
                                      **record['provenance']} for record in records]})
    return groups
