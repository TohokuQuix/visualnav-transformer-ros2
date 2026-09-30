"""Outcome-blind train state selection for the six-second readiness pilot."""
from __future__ import annotations

import hashlib
import json
import math


TERRAINS = ('boxes', 'hf_pyramid_slope', 'k_rail_field', 'k_rail_step_field',
            'pyramid_stairs', 'pyramid_stairs_inv', 'random_rough')
SEED = 20260930
INPUT_FIELDS = ('split', 'complete', 'terrain_type', 'episode_key', 'step',
                'local_step', 'view_index', 'goal', 'dem_sha256', 'dem_finite',
                'goal_finite', 'boundary_clean', 'terrain_seed')


def select_states(rows, terrains=TERRAINS, *, prior_episodes, seed=SEED):
    """Freeze one episode-0 state per terrain; all filters precede ranking.

    Episode 0 is a reproducible subset of the approved episode-local-prefix
    support. It avoids replaying unrelated earlier episodes.
    """
    if tuple(terrains) != TERRAINS or seed != SEED:
        raise ValueError('terrain order or ranking seed changed')
    prior = {tuple(x) for x in prior_episodes}
    support = {t: {'complete_train': 0, 'eligible': 0} for t in terrains}
    eligible = {t: [] for t in terrains}
    for source in rows:
        row = {key: source[key] for key in INPUT_FIELDS if key in source}
        terrain = row['terrain_type']
        if terrain not in support:
            raise ValueError('unexpected terrain')
        if row['split'] != 'train' or not row['complete']:
            continue
        support[terrain]['complete_train'] += 1
        episode = tuple(row['episode_key'])
        local = int(row['local_step'])
        goal = row['goal']
        if episode in prior or episode[2] != 0 or local < 0 or local >= 80:
            continue
        if int(row['step']) != local or not row['dem_finite'] or not row['goal_finite'] or not row['boundary_clean']:
            continue
        distance = math.hypot(*map(float, goal))
        if not math.isfinite(distance) or distance < .30:
            continue
        row['goal_distance_m'] = distance
        row['episode_local_prefix_steps'] = local + 1
        identity = [seed, terrain, row['episode_key'], row['view_index']]
        row['rank_sha256'] = hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()
        support[terrain]['eligible'] += 1
        eligible[terrain].append(row)
    selected = []
    used = set()
    for terrain in terrains:
        for row in sorted(eligible[terrain], key=lambda r: (r['rank_sha256'], r['view_index'])):
            episode = tuple(row['episode_key'])
            if episode not in used:
                selected.append(dict(row, state_id=f'readiness-{len(selected):03d}'))
                used.add(episode)
                break
        else:
            raise ValueError(f'support shortage: {terrain}')
    return {'seed': seed, 'terrain_order': list(terrains), 'selected': selected,
            'support': support, 'prior_episodes_excluded': [list(x) for x in sorted(prior)],
            'outcome_fields_used': [], 'input_fields': list(INPUT_FIELDS),
            'episode_cap': 1, 'max_episode_local_prefix_steps': 80,
            'minimum_goal_distance_m': .30, 'episode_zero_replay_subset': True}
