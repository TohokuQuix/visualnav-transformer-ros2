"""MI-A v3 execution-mode separation and frozen multi-seed assignment contract."""

from __future__ import annotations

import hashlib
import json

from .demo_reset_policy import configure_demo_reset_policy
from .mi_fi_v1 import CONDITIONS, configure_initialization
from .mi_a_v2 import CANDIDATES


SCHEMA = 'eatnav_mi_a_v3_multi_seed'
TERRAINS = ('pyramid_stairs', 'pyramid_stairs_inv', 'random_rough', 'k_rail_step_field')


def configure_run_mode(cfg, mode: str = 'demo', *, state_group_id: str | None = None,
                       assignment: dict[str, str] | None = None) -> str:
    """Use the original reset events; require an explicit group allocation outside demo."""
    if mode not in ('demo', 'collection', 'evaluation'):
        raise ValueError('unknown EAT run mode')
    configure_demo_reset_policy(cfg)
    if mode == 'demo':
        return 'front_folded'
    if not state_group_id or assignment is None or state_group_id not in assignment:
        raise ValueError('collection/evaluation requires a frozen StateGroup assignment')
    condition = assignment[state_group_id]
    configure_initialization(cfg, condition)
    return condition


def _validate_groups(groups: list[dict]) -> None:
    if not groups or len({g['state_group_id'] for g in groups}) != len(groups):
        raise ValueError('nonempty distinct StateGroup IDs required')
    seeds = set()
    for g in groups:
        if g['terrain_type'] not in TERRAINS or g['terrain_level'] not in range(5) or \
                not isinstance(g['terrain_seed'], int) or not 0 <= g['terrain_seed'] <= 2**32-1 or \
                g['env_tile_ids'] != [0, 1, 2, 3]:
            raise ValueError('terrain/seed/tile group contract')
        if g['terrain_seed'] in seeds:
            raise ValueError('independent terrain seed reused')
        seeds.add(g['terrain_seed'])


def build_collection_assignment(groups: list[dict], randomization_seed: int) -> dict[str, str]:
    """Balanced randomized allocation within terrain strata, once per StateGroup."""
    _validate_groups(groups)
    if not isinstance(randomization_seed, int) or randomization_seed < 0:
        raise ValueError('nonnegative randomization seed required')
    result = {}
    for terrain_index, terrain in enumerate(TERRAINS):
        subset = [g for g in groups if g['terrain_type'] == terrain]
        ordered = sorted(subset, key=lambda g: hashlib.sha256(
            f"{randomization_seed}:{terrain}:{g['state_group_id']}".encode()).hexdigest())
        # Alternate the odd stratum's extra assignment across terrain types.
        first = terrain_index % 2
        for index, g in enumerate(ordered):
            result[g['state_group_id']] = CONDITIONS[(first + index) % 2]
    return result


def validate_collection_manifest(manifest: dict) -> dict[str, str]:
    if manifest.get('schema') != 'eatnav_mi_a_v3_collection_assignment' or \
            manifest.get('assignment') != build_collection_assignment(
                manifest['state_groups'], manifest['randomization_seed']):
        raise ValueError('unfrozen collection StateGroup allocation')
    return manifest['assignment']


def build_evaluation_schedule(groups: list[dict]) -> list[dict]:
    """Pair all six Paths within each terrain seed and each flipper condition."""
    _validate_groups(groups)
    rows = []
    for group in groups:
        for condition in CONDITIONS:
            for repeat in (0, 1):
                rows.append({**group, 'index': len(rows), 'run_mode': 'evaluation',
                             'flipper_initialization_mode': condition,
                             'flipper_initialization_seed': group['terrain_seed'],
                             'stage': 'prepath', 'candidate_id': None, 'repeat_index': repeat})
            for candidate in CANDIDATES:
                rows.append({**group, 'index': len(rows), 'run_mode': 'evaluation',
                             'flipper_initialization_mode': condition,
                             'flipper_initialization_seed': group['terrain_seed'],
                             'stage': 'candidate', 'candidate_id': candidate, 'repeat_index': 0,
                             'travel_direction': 'forward' if candidate.startswith('forward') else 'reverse'})
    return rows


def validate_evaluation_schedule(rows: list[dict], groups: list[dict]) -> None:
    expected = build_evaluation_schedule(groups)
    if json.dumps(rows, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('MI-A v3 frozen assignment mismatch')
