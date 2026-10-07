"""Outcome-blind second batch seed selection for the frozen R3 protocol."""
from __future__ import annotations

import hashlib

from .radial_inventory import TERRAIN_ORDER, terrain_seed


def choose_second_seed(existing_seeds: set[int]) -> dict:
    """Use the first collision-free big-endian SHA-256 uint32 candidate."""
    forbidden = set(existing_seeds) | {20261007}
    base = 'eatnav-r3-second-seed-v1'
    for index in range(100_000):
        salt = base if index == 0 else f'{base}|{index}'
        digest = hashlib.sha256(salt.encode()).digest()
        batch = int.from_bytes(digest[:4], 'big')
        seeds = {name: terrain_seed(batch, col) for col, name in enumerate(TERRAIN_ORDER)}
        if batch not in forbidden and not set(seeds.values()) & forbidden:
            return {'salt': salt, 'sha256': digest.hex(), 'uint32_rule': 'first 4 SHA-256 bytes, big endian',
                    'batch_seed': batch, 'terrain_seeds': seeds,
                    'source_family_ids': {name: f'{batch}:{name}' for name in TERRAIN_ORDER}}
    raise RuntimeError('no collision-free second seed within deterministic salt search')
