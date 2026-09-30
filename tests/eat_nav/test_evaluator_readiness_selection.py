import numpy as np
import pytest

from visualnav_transformer.eat_nav.evaluator_readiness import select_states


TERRAINS = ('boxes', 'hf_pyramid_slope', 'k_rail_field',
            'k_rail_step_field', 'pyramid_stairs', 'pyramid_stairs_inv', 'random_rough')


def row(terrain, env, step=20, *, goal=(.4, 0.), episode=0, view=1):
    return {'split': 'train', 'complete': True, 'terrain_type': terrain,
            'episode_key': [0, env, episode], 'step': step, 'local_step': step,
            'view_index': view, 'goal': list(goal), 'dem_sha256': 'a' * 64,
            'dem_finite': True, 'goal_finite': True, 'boundary_clean': True}


def test_selects_seven_distinct_new_episodes_from_input_only():
    rows = [row(t, i, view=i) for i, t in enumerate(TERRAINS)]
    rows += [dict(row(TERRAINS[0], 99, view=99), critic_value=999999)]
    prior = {(0, 99, 0)}
    first = select_states(rows, TERRAINS, prior_episodes=prior)
    second = select_states([dict(r, critic_value=-999999) for r in rows],
                           TERRAINS, prior_episodes=prior)
    assert first == second
    assert len(first['selected']) == 7
    assert first['outcome_fields_used'] == []
    assert all(s['episode_key'] != [0, 99, 0] for s in first['selected'])
    assert len({tuple(s['episode_key']) for s in first['selected']}) == 7


@pytest.mark.parametrize('change', [
    {'goal': [.29, 0.]}, {'dem_finite': False}, {'goal_finite': False},
    {'boundary_clean': False}, {'local_step': 80}, {'episode_key': [0, 1, 1]},
    {'complete': False}, {'split': 'val'},
])
def test_rejects_unsupported_terrain_without_relaxing(change):
    rows = [row(t, i, view=i) for i, t in enumerate(TERRAINS)]
    rows[0].update(change)
    with pytest.raises(ValueError, match='support'):
        select_states(rows, TERRAINS, prior_episodes=set())
