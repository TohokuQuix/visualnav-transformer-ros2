import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from eat_nav_goal_subgoal_dem_gate import verify_dem_history


def fixture(tmp_path):
    ref = tmp_path / 'reference.yaml'
    ref.write_text('context_size: 3\ndem_height_cells: 20\ndem_width_cells: 20\ndem_resolution_m: 0.1\nreference_frame: map\nbase_frame: chassis/body\n')
    state = tmp_path / 'state.yaml'
    state.write_text('anchor_limits:\n  dem_rms_m: 0.03\n  mask_iou_min: 0.95\n')
    dem = np.zeros((4, 2, 20, 20), dtype=np.float32)
    dem[:, 1] = 1
    recorded = []
    for i, step in enumerate((45, 50, 55, 60)):
        recorded.append({'session_id': 0, 'env_id': 60, 'episode_id': 0,
                         'step': step, 'timestamp_ns': (step + 1) * 50_000_000,
                         'pre_reset': False,
                         'dem': {'axis': 'row=+world_y,col=+world_x',
                                 'unknown': 'mask=0,height=0', 'resolution_m': .1,
                                 'transformed': dem[i].tolist()}})
    replay = dem.copy()
    replay[0, 0] += 1e-6
    digest = lambda x: hashlib.sha256(x.tobytes()).hexdigest()
    summary = {'state': {'state_id': 'v2-train-000', 'episode_key': [0, 60, 0],
                         'dem_sha256': digest(dem)},
               'source_sha256': {'runner_source': 'fixed-source'},
               'anchor': {'dem_history': replay.tolist(),
                          'dem_history_sha256': digest(replay)}}
    entry = {'state_id': 'v2-train-000', 'episode_key': [0, 60, 0], 'anchor_step': 60}
    return {'entry': entry, 'summary': summary, 'recorded_frames': recorded,
            'selected_dem': dem, 'reference_config': ref, 'state_config': state,
            'source_sha256': 'fixed-source', 'expected_source_sha256': 'fixed-source'}


def test_small_numeric_difference_passes_with_sha_mismatch(tmp_path):
    kwargs = fixture(tmp_path)
    result = verify_dem_history(**kwargs)
    assert result['status'] == 'pass'
    assert not result['history_raw_sha_match']
    assert not result['frames'][0]['raw_sha_match']
    assert result['frames'][0]['rms_m'] < .03
    assert all(frame['mask_iou'] == 1 for frame in result['frames'])


@pytest.mark.parametrize('mutation', ['timestamp_ns', 'step', 'axis', 'resolution_m', 'nan', 'mask', 'height'])
def test_metadata_and_numeric_fail_closed(tmp_path, mutation):
    kwargs = fixture(tmp_path)
    kwargs = copy.deepcopy(kwargs)
    row = kwargs['recorded_frames'][0]
    if mutation == 'timestamp_ns': row['timestamp_ns'] += 1
    elif mutation == 'step': row['step'] += 1
    elif mutation == 'axis': row['dem']['axis'] = 'wrong'
    elif mutation == 'resolution_m': row['dem']['resolution_m'] = .2
    elif mutation == 'nan': row['dem']['transformed'][0][0][0] = float('nan')
    elif mutation == 'mask':
        replay = np.asarray(kwargs['summary']['anchor']['dem_history'], dtype=np.float32)
        replay[0, 1] = 0
        kwargs['summary']['anchor']['dem_history'] = replay.tolist()
        kwargs['summary']['anchor']['dem_history_sha256'] = hashlib.sha256(replay.tobytes()).hexdigest()
    else:
        replay = np.asarray(kwargs['summary']['anchor']['dem_history'], dtype=np.float32)
        replay[0, 0] = .1
        kwargs['summary']['anchor']['dem_history'] = replay.tolist()
        kwargs['summary']['anchor']['dem_history_sha256'] = hashlib.sha256(replay.tobytes()).hexdigest()
    with pytest.raises(ValueError):
        verify_dem_history(**kwargs)
