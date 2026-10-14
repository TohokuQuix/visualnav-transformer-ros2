"""Small, goal-independent MI-B pilot model and paired-motion metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .motion_vnext import motion_label_from_trace


CANDIDATES = ('forward_direct', 'forward_+30', 'forward_-30',
              'reverse_direct', 'reverse_+30', 'reverse_-30')


def contrast_indices() -> tuple[tuple[int, int], ...]:
    return ((1, 0), (2, 0), (4, 3), (5, 3))


def validate_grouped_arrays(arrays: dict) -> None:
    n = len(arrays['dem'])
    shapes = {'dem': (n, 4, 2, 20, 20), 'state': (n, 22),
              'paths': (n, 6, 8, 2), 'direction': (n, 6, 1),
              'mask': (n, 6), 'target': (n, 6, 2)}
    for name, shape in shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or (name != 'mask' and not np.isfinite(value).all()):
            raise ValueError(f'{name} shape/finite contract')
    if np.asarray(arrays['mask']).dtype != np.bool_ or not arrays['mask'].all():
        raise ValueError('candidate mask contract')
    expected = np.array([1, 1, 1, -1, -1, -1])
    if not np.array_equal(arrays['direction'][:, :, 0], np.tile(expected, (n, 1))):
        raise ValueError('direction contract')
    masks = arrays['dem'][:, :, 1]
    if not np.isin(masks, [0, 1]).all():
        raise ValueError('DEM valid-mask contract')


class PilotPredictor(nn.Module):
    """Matched-capacity CNN/MLP with a fixed modality mask for each ablation."""

    def __init__(self, variant: str, feature_dim: int = 32):
        super().__init__()
        if variant not in ('full', 'no_dem', 'no_body', 'no_path'):
            raise ValueError('unknown MI-B variant')
        self.variant = variant
        self.dem_encoder = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(24, feature_dim), nn.ReLU())
        self.state_encoder = nn.Sequential(nn.Linear(22, 32), nn.ReLU(),
                                           nn.Linear(32, 32), nn.ReLU())
        self.path_encoder = nn.Sequential(nn.Linear(17, 48), nn.ReLU(),
                                          nn.Linear(48, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(feature_dim + 64, 64), nn.ReLU(),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, dem: torch.Tensor, state: torch.Tensor,
                paths: torch.Tensor, direction: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        n = dem.shape[0]
        if tuple(dem.shape) != (n, 4, 2, 20, 20) or state.shape != (n, 22) or \
                paths.shape != (n, 6, 8, 2) or direction.shape != (n, 6, 1) or \
                mask.shape != (n, 6) or mask.dtype != torch.bool:
            raise ValueError('MI-B model shape contract')
        if self.variant == 'no_dem':
            dem = torch.zeros_like(dem)
        if self.variant == 'no_body':
            state = torch.zeros_like(state)
        if self.variant == 'no_path':
            paths = torch.zeros_like(paths)
        dem_features = self.dem_encoder(dem.reshape(n, 8, 20, 20))
        state_features = self.state_encoder(state)
        path_features = self.path_encoder(torch.cat((paths.reshape(n, 6, 16), direction), -1))
        shared = torch.cat((dem_features, state_features), -1)[:, None].expand(-1, 6, -1)
        return self.head(torch.cat((shared, path_features), -1)).masked_fill(~mask[..., None], 0)


def normalized_pair_loss(pred: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1:] != (6, 2) or \
            mask.shape != pred.shape[:2] or mask.dtype != torch.bool or not mask.any():
        raise ValueError('paired loss shape')
    absolute = (pred[mask] - target[mask]).square().sum(-1).mean()
    contrasts = []
    for arc, direct in contrast_indices():
        valid = mask[:, arc] & mask[:, direct]
        if valid.any():
            difference = (pred[:, arc] - pred[:, direct]) - (target[:, arc] - target[:, direct])
            contrasts.append(difference[valid].square().sum(-1))
    if not contrasts:
        raise ValueError('no paired Arc/Direct samples')
    return absolute + torch.cat(contrasts).mean()


def paired_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                   seed_ids: np.ndarray) -> dict:
    pred, target, mask, seed_ids = map(np.asarray, (pred, target, mask, seed_ids))
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1:] != (6, 2) or \
            mask.shape != pred.shape[:2] or seed_ids.shape != (len(pred),):
        raise ValueError('paired metric shape')
    absolute_error = np.sum((pred - target)**2, axis=-1)[mask]
    contrast_errors = []
    for arc, direct in contrast_indices():
        difference = (pred[:, arc] - pred[:, direct]) - (target[:, arc] - target[:, direct])
        contrast_errors.append(np.sum(difference**2, axis=-1))
    paired = np.stack(contrast_errors, axis=1)
    by_seed = {str(int(seed)): float(np.sqrt(paired[seed_ids == seed].mean()))
               for seed in sorted(set(seed_ids.tolist()))}
    return {'group_count': len(pred), 'seed_count': len(by_seed),
            'absolute_xy_mse_m2': float(absolute_error.mean()),
            'absolute_xy_rmse_m': float(np.sqrt(absolute_error.mean())),
            'contrast_xy_mse_m2': float(paired.mean()),
            'contrast_rmse_m': float(np.sqrt(paired.mean())),
            'seed_contrast_rmse_m': by_seed}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_trial(attempt: Path, row: dict) -> tuple[dict, dict, list[dict]]:
    """Verify a sealed candidate and independently recompute all four labels."""
    from scripts.eat_nav_mi_a_v31_run import verify_attempt
    verify_attempt(attempt, row)
    output = attempt / 'sim-output'
    manifest = json.loads((output / 'input-manifest.json').read_text())
    path = np.asarray(manifest['path_body_xy_m'], dtype='<f8')
    if path.shape != (8, 2) or hashlib.sha256(path.tobytes()).hexdigest() != manifest['path_body_sha256']:
        raise ValueError('candidate Path SHA/shape mismatch')
    with np.load(output / 'prepath.npz', allow_pickle=False) as z:
        snapshot = {key: z[key].copy() for key in ('root', 'joint', 'actor_pre_noise', 'dem')}
    stored = json.loads((output / 'motion-labels.json').read_text())['labels']
    if len(stored) != 4:
        raise ValueError('four stored labels required')
    for env in range(4):
        trace = [json.loads(line) for line in (output / f'trace-env-{env}.jsonl').read_text().splitlines()]
        derived = motion_label_from_trace(trace)
        if len(trace) != 40 or not derived['available'] or \
                not np.allclose(derived['delta_body_xy_m'], stored[env]['delta_body_xy_m'], atol=1e-9, rtol=0) or \
                stored[env]['termination_reason'] is not None:
            raise ValueError('40-step sealed motion label mismatch')
    return manifest, snapshot, stored


def split_indices(metadata: list[dict], split_seeds: dict) -> dict[str, np.ndarray]:
    if set(split_seeds) != {'train', 'validation', 'test'}:
        raise ValueError('three split names required')
    sets = [set(map(int, split_seeds[name])) for name in ('train', 'validation', 'test')]
    if any(not seeds for seeds in sets) or sum(map(len, sets)) != len(set.union(*sets)):
        raise ValueError('terrain seed overlap or empty split')
    present = {int(row['terrain_seed']) for row in metadata}
    if set.union(*sets) != present:
        raise ValueError('terrain seed coverage mismatch')
    return {name: np.asarray([i for i, row in enumerate(metadata)
                              if int(row['terrain_seed']) in seeds], dtype=np.int64)
            for name, seeds in zip(('train', 'validation', 'test'), sets)}


def build_grouped_dataset(evidence_root: Path, expected_manifest_sha: str,
                          split_seeds: dict) -> tuple[dict, list[dict], dict]:
    """Read-only audit of 192 sealed rows; return 96 groups of paired candidates."""
    from scripts.eat_nav_mi_a_v31_run import MANIFEST, load_frozen, row_name, verify_attempt
    evidence_root = Path(evidence_root)
    if sha(MANIFEST) != expected_manifest_sha:
        raise ValueError('frozen MI-A manifest SHA mismatch')
    assignment = load_frozen()['assignment']
    rows_by_group = {}
    seals = []
    for row in assignment:
        attempt = evidence_root / row_name(row)
        verify_attempt(attempt, row)
        seals.append({'row_index': row['index'], 'outer_seal_sha256': sha(attempt / 'outer-seal.json')})
        key = (row['terrain_seed'], row['flipper_initialization_mode'])
        rows_by_group.setdefault(key, {})[(row['stage'], row['repeat_index'], row['candidate_id'])] = row
    arrays = {name: [] for name in ('dem', 'state', 'paths', 'direction', 'mask', 'target', 'yaw')}
    metadata = []
    for (seed, posture), rows in sorted(rows_by_group.items()):
        if len(rows) != 8:
            raise ValueError('eight prepath/candidate rows required per seed/posture')
        ref_row = rows['prepath', 0, None]
        ref_attempt = evidence_root / row_name(ref_row)
        with np.load(ref_attempt / 'sim-output' / 'prepath.npz', allow_pickle=False) as z:
            reference = {key: z[key].copy() for key in ('root', 'joint', 'actor_pre_noise', 'dem')}
        candidate_paths, candidate_labels, candidate_yaw = [], [], []
        for candidate in CANDIDATES:
            row = rows['candidate', 0, candidate]
            manifest, snapshot, labels = read_trial(evidence_root / row_name(row), row)
            if manifest['seed'] != seed or manifest['condition'] != posture or \
                    manifest['direction'] != (1 if candidate.startswith('forward') else -1):
                raise ValueError('candidate metadata mismatch')
            for field in reference:
                if reference[field].shape != snapshot[field].shape or \
                        not np.isfinite(reference[field]).all() or \
                        not np.isfinite(snapshot[field]).all() or \
                        np.max(np.abs(reference[field].astype(np.float64) -
                                      snapshot[field].astype(np.float64))) > 1e-6:
                    raise ValueError(f'cross-Path pre-Path mismatch: {field}')
            candidate_paths.append(np.asarray(manifest['path_body_xy_m'], np.float32))
            candidate_labels.append(np.asarray([v['delta_body_xy_m'] for v in labels], np.float32))
            candidate_yaw.append(np.asarray([v['delta_body_yaw_rad'] for v in labels], np.float32))
        paths = np.stack(candidate_paths)
        target = np.stack(candidate_labels, axis=1)
        yaw = np.stack(candidate_yaw, axis=1)
        for env in range(4):
            arrays['dem'].append(reference['dem'][env].astype(np.float32))
            arrays['state'].append(reference['actor_pre_noise'][env].astype(np.float32))
            arrays['paths'].append(paths)
            arrays['direction'].append(np.asarray([1, 1, 1, -1, -1, -1], np.float32)[:, None])
            arrays['mask'].append(np.ones(6, bool))
            arrays['target'].append(target[env])
            arrays['yaw'].append(yaw[env])
            metadata.append({'terrain_seed': seed, 'posture': posture, 'env_tile_id': env,
                             'terrain_type': ref_row['terrain_type'],
                             'terrain_level': ref_row['terrain_level'],
                             'state_group_id': ref_row['state_group_id']})
    arrays = {name: np.stack(values) for name, values in arrays.items()}
    validate_grouped_arrays(arrays)
    split = split_indices(metadata, split_seeds)
    terrain_types = {row['terrain_type'] for row in metadata}
    if any({metadata[i]['terrain_type'] for i in indices} != terrain_types
           for indices in split.values()):
        raise ValueError('terrain type missing from split')
    audit = {'schema': 'eatnav_mi_b_grouped_dataset_v1',
             'source_manifest_sha256': expected_manifest_sha,
             'verified_process_count': len(seals), 'candidate_trial_count': len(metadata) * 6,
             'group_count': len(metadata), 'source_outer_seals': seals,
             'split_group_count': {name: len(indices) for name, indices in split.items()},
             'split_seed_count': {name: len({metadata[i]['terrain_seed'] for i in indices})
                                  for name, indices in split.items()},
             'source_root': str(evidence_root)}
    return arrays, metadata, audit


def fit_normalization(arrays: dict, train_indices: np.ndarray) -> dict:
    """Fit only the selected terrain-seed training groups."""
    idx = np.asarray(train_indices, dtype=np.int64)
    if idx.ndim != 1 or len(idx) == 0 or len(set(idx.tolist())) != len(idx) or \
            np.any(idx < 0) or np.any(idx >= len(arrays['dem'])):
        raise ValueError('invalid normalization train indices')
    valid = arrays['dem'][idx, :, 1].astype(bool)
    heights = arrays['dem'][idx, :, 0][valid].astype(np.float64)
    if len(heights) == 0:
        raise ValueError('no valid train DEM cells')
    state = arrays['state'][idx].astype(np.float64)
    target = arrays['target'][idx].reshape(-1, 2).astype(np.float64)
    return {'fitted_from': 'train_only', 'train_group_count': len(idx),
            'dem_height_mean': float(heights.mean()),
            'dem_height_std': float(max(heights.std(), 1e-4)),
            'state_mean': state.mean(0).tolist(),
            'state_std': np.maximum(state.std(0), 1e-3).tolist(),
            'target_mean': target.mean(0).tolist(),
            'target_std': np.maximum(target.std(0), 1e-3).tolist()}


def transform_arrays(arrays: dict, norm: dict) -> dict:
    result = {key: value.copy() for key, value in arrays.items()}
    valid = result['dem'][:, :, 1].astype(bool)
    height = result['dem'][:, :, 0]
    height[valid] = ((height[valid] - norm['dem_height_mean']) /
                     norm['dem_height_std']).astype(np.float32)
    height[~valid] = 0
    result['state'] = ((result['state'] - np.asarray(norm['state_mean'], np.float32)) /
                       np.asarray(norm['state_std'], np.float32)).astype(np.float32)
    result['target'] = ((result['target'] - np.asarray(norm['target_mean'], np.float32)) /
                        np.asarray(norm['target_std'], np.float32)).astype(np.float32)
    return result

