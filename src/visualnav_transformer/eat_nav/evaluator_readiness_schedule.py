"""Immutable per-terrain rollout order and integrity checks."""
from __future__ import annotations

import hashlib
import json


SEEDS = tuple(range(1001, 1009))
BANK = ('nominal', 'slight_left', 'slight_right', 'strong_left', 'strong_right')


def schedule(selection, candidate_tables):
    selected = selection['selected']
    if len(selected) != 7 or len(candidate_tables) != 7:
        raise ValueError('seven frozen states required')
    runs = []
    for state_index, state in enumerate(selected):
        table = candidate_tables[state_index]
        ids = [x['candidate_id'] for x in table]
        expected = [f'nomad-seed-{seed}' for seed in SEEDS] + [f'bank-{name}' for name in BANK]
        if ids != expected or len({x['base_path_sha256'] for x in table}) < 1:
            raise ValueError('candidate order or identity changed')
        state_id = state['state_id']
        spec = [('anchor', 'bank-nominal', None)]
        spec += [('nominal', 'bank-nominal', repeat) for repeat in range(3)]
        spec += [('nomad', f'nomad-seed-{seed}', seed) for seed in SEEDS]
        spec += [('bank', f'bank-{name}', None) for name in BANK]
        for kind, candidate, marker in spec:
            entry = {'order': len(runs), 'run_id': f'readiness-{len(runs):03d}',
                     'state_index': state_index, 'state_id': state_id,
                     'terrain_type': state['terrain_type'], 'episode_key': state['episode_key'],
                     'kind': kind, 'candidate_id': candidate,
                     'seed': marker if kind == 'nomad' else None,
                     'repeat': marker if kind == 'nominal' else None,
                     'base_path_sha256': next(x['base_path_sha256'] for x in table if x['candidate_id'] == candidate),
                     'control_steps': 120,
                     'logical_t0_ns': (1000 + len(runs) * 10) * 1_000_000_000}
            runs.append(entry)
    if len(runs) != 119 or sum(x['kind'] != 'anchor' for x in runs) != 112:
        raise ValueError('119 total / 112 rollout schedule mismatch')
    return {'runs': runs, 'nomad_seeds': list(SEEDS), 'bank_order': list(BANK),
            'rollout_count': 112, 'anchor_count': 7, 'candidate_outcomes_used': False}


def canonical_sha(data):
    return hashlib.sha256((json.dumps(data, indent=2, sort_keys=True) + '\n').encode()).hexdigest()
