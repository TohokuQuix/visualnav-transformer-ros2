"""Reproducible independent-seed, paired fourteen-Path collection schedule."""

from __future__ import annotations

import hashlib
import json

from .bidirectional_bank import bank
from .motion_14_stage_a import candidate_role


COLLECTION_SCHEMA = 'eatnav_motion_14_collection_v1'
TERRAINS = ('pyramid_stairs', 'pyramid_stairs_inv', 'random_rough', 'k_rail_step_field')
LEVELS = (1, 2, 3)
SPLITS = ('train', 'validation', 'test')
SEED_DERIVATION = 'first_u32_sha256(eatnav-motion14-v1:terrain:level:split), big-endian'


def seed_groups() -> list[dict]:
    groups = []
    for terrain in TERRAINS:
        for level in LEVELS:
            for split in SPLITS:
                token = f'eatnav-motion14-v1:{terrain}:{level}:{split}'
                seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], 'big')
                groups.append({'state_group_id': f'{terrain}-level{level}-seed{seed}',
                               'terrain_type': terrain, 'terrain_level': level,
                               'terrain_seed': seed, 'split': split,
                               'env_tile_ids': [0, 1, 2, 3],
                               'seed_derivation_token': token})
    if len({group['terrain_seed'] for group in groups}) != 36:
        raise ValueError('derived terrain seed collision')
    return groups


def split_seeds(groups: list[dict]) -> dict[str, list[int]]:
    return {split: [g['terrain_seed'] for g in groups if g['split'] == split]
            for split in SPLITS}


def build_collection_schedule(groups: list[dict] | None = None) -> list[dict]:
    groups = seed_groups() if groups is None else groups
    if json.dumps(groups, sort_keys=True) != json.dumps(seed_groups(), sort_keys=True):
        raise ValueError('collection seed/group allocation changed')
    rows = []
    for group in groups:
        for posture in ('front_folded', 'rear_folded'):
            for repeat in (0, 1):
                rows.append({**group, 'index': len(rows), 'stage': 'prepath',
                             'candidate_id': None, 'repeat_index': repeat,
                             'flipper_initialization_mode': posture})
            for path in bank():
                rows.append({**group, 'index': len(rows), 'stage': 'candidate',
                             'candidate_id': candidate_role(path.direction, path.arc_angle_deg),
                             'repeat_index': 0, 'flipper_initialization_mode': posture,
                             'travel_direction': 'forward' if path.direction == 1 else 'reverse'})
    return rows


def validate_collection_schedule(rows: list[dict], groups: list[dict]) -> None:
    if json.dumps(rows, sort_keys=True) != json.dumps(build_collection_schedule(groups), sort_keys=True):
        raise ValueError('collection assignment mismatch')
