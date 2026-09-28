import gzip
import json

import numpy as np

from visualnav_transformer.eat_nav.config import EATConfig
from visualnav_transformer.eat_nav.dataset import _records, _build_window
from visualnav_transformer.eat_nav.window_waterfall import count_window_waterfall


def test_terminal_window_history_goal_and_teacher_never_use_next_episode(tmp_path):
    dem = np.zeros((2, 20, 20), dtype=np.float32).tolist()
    records = []
    for episode in (0, 1):
        for local in range(61):
            step = episode * 61 + local
            terminal = local == 60
            records.append(dict(session_id=1, env_id=0, episode_id=episode,
                                timestamp_ns=step * 50_000_000, step=step,
                                pose=[episode * 100.0 + local * 0.01, 0, 0],
                                yaw=0.0, critic_value=0.5, reward=0.0,
                                terminated=terminal, truncated=False,
                                termination_reason=['time_out'] if terminal else [],
                                terrain_seed=1, terrain_type='boxes',
                                **({'dem': {'transformed': dem}} if local % 5 == 0 or terminal else {})))
    with gzip.open(tmp_path/'rows.jsonl.gz', 'wt') as stream:
        for row in records:
            stream.write(json.dumps(row)+'\n')
    (tmp_path/'manifest.json').write_text(json.dumps({'chunks':[{'file':'rows.jsonl.gz'}]}))
    grouped = _records(tmp_path/'manifest.json')
    cfg = EATConfig()
    windows = []
    for key, rows in grouped.items():
        for i, row in enumerate(rows):
            if row['dem'] is not None:
                w = _build_window(key, rows, i, cfg)
                if w is not None:
                    windows.append(w)
                    assert w['episode_key'] == key
                    assert np.max(np.abs(w['action'])) < 0.1
                    assert np.linalg.norm(w['goal']) < 1.0
                    assert np.array_equal(w['action'][~w['action_mask']], np.zeros(((~w['action_mask']).sum(), 2)))
    assert any(not w['action_mask'].all() for w in windows)
    summary = count_window_waterfall(grouped, cfg)
    assert summary['cross_episode_sources'] == 0
    assert summary['valid_training_windows'] == len(windows)
