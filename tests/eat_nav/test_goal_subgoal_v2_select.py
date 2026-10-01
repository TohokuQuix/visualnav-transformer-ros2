"""Selection is deterministic and independent of outcomes."""
import copy

import numpy as np
import yaml

from eat_nav_goal_subgoal_v2_select import CONFIG, exclusion_reasons, rank, select


def fixture_window(*, terrain='boxes', env=2, episode=0, step=30, goal=(.4, 0.)):
    return {'episode_key': (0, env, episode), 'terrain_seed': 100 + env,
            'terrain_type': terrain, 'step': step, 'local_step': step,
            'dem': np.zeros((4, 2, 10, 10), dtype=np.float32),
            'goal': np.asarray(goal, dtype=np.float32),
            'action': np.ones((8, 2), dtype=np.float32),
            'action_mask': np.ones(8, dtype=bool),
            'reward': -999., 'critic_value': 999., 'termination_reason': ('failure',)}


def config():
    return yaml.safe_load(CONFIG.read_text())


def test_fixed_prefix_complete_goal_and_episode_requirements():
    cfg = config()
    w = fixture_window()
    assert exclusion_reasons(w, cfg) == []
    for change, expected in [({'episode_key': (0, 2, 1)}, 'episode_not_zero'),
                             ({'step': 80, 'local_step': 80}, 'prefix_exceeds_80'),
                             ({'step': 30, 'local_step': 29}, 'global_local_step_differ'),
                             ({'goal': np.asarray([.29, 0.])}, 'goal_too_close'),
                             ({'action_mask': np.asarray([True] * 7 + [False])}, 'incomplete_dem_or_teacher')]:
        altered = copy.deepcopy(w)
        altered.update(change)
        assert expected in exclusion_reasons(altered, cfg)


def test_outcome_blind_rank_and_selection_stable_to_order():
    cfg = config()
    terrains = cfg['terrain_order']
    windows = [fixture_window(terrain=t, env=i * 2 + k) for i, t in enumerate(terrains) for k in (0, 1)]
    before, _ = select(windows, cfg)
    changed = copy.deepcopy(windows)
    for w in changed:
        w['reward'] = 1e9
        w['critic_value'] = -1e9
        w['termination_reason'] = ('timeout',)
        w['candidate_outcome'] = 'arbitrary'
    after, _ = select(list(reversed(changed)), cfg)
    assert [r['window_identity'] for r, _ in before] == [r['window_identity'] for r, _ in after]
    assert [rank(a, cfg) for a in windows] == [rank(b, cfg) for b in changed]
    assert len({tuple(r['episode_key']) for r, _ in before}) == 4


def test_no_silent_terrain_substitution():
    cfg = config()
    windows = [fixture_window(terrain=t, env=i) for i, t in enumerate(cfg['terrain_order'][:-1])]
    selected, _ = select(windows, cfg)
    assert len(selected) == 3
    assert [r['terrain_name'] for r, _ in selected] == cfg['terrain_order'][:-1]
