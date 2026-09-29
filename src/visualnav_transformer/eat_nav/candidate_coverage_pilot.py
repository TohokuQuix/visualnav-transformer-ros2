"""Frozen 119-run pilot order and wall-clock start gate."""
from __future__ import annotations

_BANK = ('nominal', 'slight_left', 'slight_right', 'strong_left', 'strong_right')


def build_schedule(terrains, *, seeds):
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError('pilot requires eight distinct nested NoMaD seeds')
    if len(set(terrains)) != len(terrains):
        raise ValueError('duplicate terrain')
    rows = []
    for terrain in terrains:
        entries = [('anchor', None, None, 'freeze anchor state and map Path before candidate execution')]
        entries.extend(('duplicate', repeat, None, 'measure nominal duplicate variation') for repeat in range(3))
        entries.extend(('nomad', None, seed, 'measure NoMaD proposal outcome') for seed in seeds)
        entries.extend(('bank', name, None, 'measure fixed maneuver bank outcome') for name in _BANK)
        for kind, name, seed, purpose in entries:
            rows.append({'run_id':f'pilot-{len(rows):03d}', 'terrain_type':terrain,
                         'kind':kind, 'name':name, 'seed':seed, 'purpose':purpose})
    return rows


def may_start_next(*, elapsed_sec, predicted_remaining_sec, approved_budget_sec):
    if min(elapsed_sec, predicted_remaining_sec) < 0 or approved_budget_sec <= 0:
        raise ValueError('invalid wall-clock budget')
    return elapsed_sec < approved_budget_sec and elapsed_sec + predicted_remaining_sec <= approved_budget_sec


def bind_schedule(schedule, selected):
    """Attach frozen state identity while preserving the predeclared run order."""
    by_terrain = {row['terrain_type']: row for row in selected}
    if len(by_terrain) != len(selected):
        raise ValueError('duplicate selected terrain')
    if {row['terrain_type'] for row in schedule} != set(by_terrain):
        raise ValueError('schedule/selection terrain mismatch')
    return [{**run,
             'state_id': by_terrain[run['terrain_type']]['state_id'],
             'view_index': by_terrain[run['terrain_type']]['view_index'],
             'episode_key': by_terrain[run['terrain_type']]['episode_key'],
             'anchor_step': by_terrain[run['terrain_type']]['step'],
             'full_prefix_steps': by_terrain[run['terrain_type']]['full_prefix_steps']}
            for run in schedule]
