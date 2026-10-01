"""Check the v2 frozen inputs before any candidate outcome exists."""
import gzip
import hashlib
import json

import numpy as np

from eat_nav_goal_subgoal_v2_select import OUT, sha


def read(path):
    return json.loads(path.read_text())


def test_frozen_48_run_order_and_input_identity():
    schedule = read(OUT / 'run-schedule-v2.json')
    runs = schedule['runs']
    assert len(runs) == 48
    assert [r['order'] for r in runs] == list(range(48))
    assert [r['kind'] for r in runs] == (['anchor'] + ['duplicate'] * 3 + ['candidate'] * 8) * 4
    assert len({tuple(runs[i * 12]['episode_key']) for i in range(4)}) == 4
    assert read(OUT / 'v4.done.json')['schedule_sha256'] == sha(OUT / 'run-schedule-v2.json')
    for i in range(4):
        group = runs[i * 12:(i + 1) * 12]
        assert group[0]['episode_key'][2] == 0 and group[0]['prefix_steps'] <= 80
        assert len({r['logical_t0_ns'] for r in group}) == 1
        assert len({r['base_path_sha256'] for r in group[1:4]}) == 1
        assert [(r['goal_id'], r['seed']) for r in group[4:]] == [
            (goal, seed) for goal in ('logged', 'id_forward_long', 'near_left_lateral', 'far_diag_right')
            for seed in (1001, 1002)]
        folder = OUT / 'inputs' / f'state-{i:03d}'
        with np.load(folder / 'prefix-actions.npz') as prefix, np.load(folder / 'suffix-actions-120.npz') as suffix:
            assert prefix['actions'].shape == (group[0]['prefix_steps'], 100, 6)
            assert prefix['selected_env_commands'].shape == (group[0]['prefix_steps'], 3)
            assert suffix['actions'].shape == (120, 100, 6)
        for row in group:
            path = folder / f"{row['candidate_id']}-base-path.json"
            assert sha(path) == row['base_path_sha256']
            points = np.asarray(read(path), dtype=np.float32)
            assert points.shape == (9, 2) and np.isfinite(points).all()
            assert np.array_equal(points[0], [0, 0])


def test_paths_reuse_original_samples_and_all_goals_screen_valid():
    old = OUT / 'paths-v2-sampler-output-pre-correction.jsonl.gz'
    new = OUT / 'paths-v2.jsonl.gz'
    with gzip.open(old, 'rt') as stream:
        original = list(map(json.loads, stream))
    with gzip.open(new, 'rt') as stream:
        corrected = list(map(json.loads, stream))
    assert len(original) == len(corrected) == 32
    assert len(read(OUT / 'goals-v2.json')['goals']) == 16
    for before, after in zip(original, corrected):
        sample = np.asarray(before['path_increments'], dtype=np.float32)
        waypoints = np.asarray(after['cumulative_waypoints'], dtype=np.float32)
        increments = np.asarray(after['path_increments'], dtype=np.float32)
        assert np.array_equal(sample, waypoints)
        assert np.allclose(increments.cumsum(axis=0), waypoints, atol=1e-6)
        assert hashlib.sha256(sample.tobytes()).hexdigest() == after['path_sha256']
        assert after['geometry_screen']['geometry_screen_valid']
    assert read(OUT / 'v3-paths.done.json')['resampling_performed'] is False


def test_short_prefix_forecast_and_current_source_parity():
    forecast = read(OUT / 'runtime-forecast-v2.json')
    assert forecast['decision'] == 'pass'
    assert forecast['full_prefix_step_total'] == 12 * (61 + 61 + 36 + 36)
    assert forecast['conservative_plus_20_percent_sec'] <= 14400
    assert forecast['worst_observed']['total_sec'] <= 14400
    assert read(OUT / 'v6.done.json')['status'] == 'pass'
