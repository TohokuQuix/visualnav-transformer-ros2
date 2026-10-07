import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pytest

from visualnav_transformer.eat_nav.radial_lean_log import (
    LeanContactRecorder, append_step, read_steps, save_dem_frame, save_actor,
    write_complete_before_close, process_matches,
)


def arrays(count=1, start=0, force=2.0, separation=-0.01, capacity=4096):
    f = np.zeros((capacity, 1), dtype=np.float32)
    p = np.zeros((capacity, 3), dtype=np.float32)
    n = np.zeros((capacity, 3), dtype=np.float32)
    s = np.zeros((capacity, 1), dtype=np.float32)
    c = np.zeros((1, 1), dtype=np.int32)
    st = np.zeros((1, 1), dtype=np.int32)
    c[0, 0], st[0, 0] = count, start
    if count and start + count <= capacity:
        f[start:start+count, 0] = force
        s[start:start+count, 0] = separation
    return [f, p, n, s, c, st]


def test_append_is_non_cumulative_and_200_by_5_recoverable(tmp_path):
    for pose in range(5):
        path = tmp_path / f'{pose}.jsonl'
        for step in range(1, 201):
            append_step(path, {'physics_step': step, 'pose_id': pose})
        rows, tail = read_steps(path)
        assert tail is None
        assert [x['physics_step'] for x in rows] == list(range(1, 201))
        assert path.stat().st_size < 100_000


def test_interrupted_tail_is_detected_without_losing_complete_rows(tmp_path):
    path = tmp_path / 'steps.jsonl'
    append_step(path, {'physics_step': 1})
    with path.open('ab') as stream:
        stream.write(b'{"physics_step":2')
    rows, tail = read_steps(path)
    assert rows == [{'physics_step': 1}]
    assert tail is not None


def test_raw_only_events_and_first_contact_fixed_step_deduplicated(tmp_path):
    recorder = LeanContactRecorder(tmp_path, {'pose': (['/World/robot/chassis'], ['/World/ground/terrain/mesh'])})
    assert recorder.record('pose', 1, arrays(count=0), 4096)['contact_count'] == 0
    assert recorder.record('pose', 2, arrays(), 4096)['contact_count'] == 1
    recorder.record('pose', 3, arrays(), 4096)
    recorder.record('pose', 50, arrays(), 4096)
    assert len(list((tmp_path/'raw').rglob('*.npz'))) == 2
    recorder2 = LeanContactRecorder(tmp_path/'second', {'pose': (['/World/robot/chassis'], ['/World/ground/terrain/mesh'])})
    recorder2.record('pose', 50, arrays(), 4096)
    assert len(list((tmp_path/'second'/'raw').rglob('*.npz'))) == 1


@pytest.mark.parametrize('bad', ['bounds', 'nonfinite', 'full'])
def test_invalid_contact_saves_raw_then_stops(tmp_path, bad):
    recorder = LeanContactRecorder(tmp_path, {'pose': (['/World/robot/chassis'], ['/World/ground/terrain/mesh'])})
    a = arrays()
    if bad == 'bounds':
        a[5][0, 0] = 4096
    elif bad == 'nonfinite':
        a[0][0, 0] = np.nan
    else:
        a[4][0, 0] = 4096
    with pytest.raises((ValueError, RuntimeError)):
        recorder.record('pose', 7, a, 4096)
    assert len(list((tmp_path/'raw').rglob('*.npz'))) == 1


def test_capacity_is_fixed_at_4096(tmp_path):
    recorder = LeanContactRecorder(tmp_path, {'pose': (['s'], ['c'])})
    with pytest.raises(ValueError, match='4096'):
        recorder.record('pose', 1, arrays(capacity=256), 256)


def test_dem_atomic_immediate_and_actor_phase_shape(tmp_path):
    dem = {'source_height': [[0.0]*20 for _ in range(20)], 'source_mask': [[1]*20 for _ in range(20)],
           'transformed': [[[0.0]*20 for _ in range(20)], [[1.0]*20 for _ in range(20)]],
           'frame_id': '/map', 'logical_stamp_ns': 50_000_000, 'resolution_m': 0.1, 'axis': 'xy'}
    path = save_dem_frame(tmp_path, 'pose', 0, dem)
    assert json.loads(path.read_text())['finite'] is True
    assert not list(tmp_path.rglob('*.tmp'))
    save_actor(tmp_path, 'pose', 'prephysics', [0.0]*22, ['field']*22, 'cfg')
    save_actor(tmp_path, 'pose', 'settled_step_200', [1.0]*22, ['field']*22, 'cfg')
    assert len(list((tmp_path/'actor').rglob('*.json'))) == 2
    with pytest.raises(ValueError):
        save_actor(tmp_path, 'pose', 'bad', [0.0]*21, ['field']*22, 'cfg')


def test_completion_is_written_before_close_and_read_back(tmp_path):
    events = []
    def close():
        events.append('close')
    path = write_complete_before_close(tmp_path, {'physics_steps': 200}, close)
    assert events == ['close']
    assert json.loads(path.read_text())['physics_steps'] == 200


def test_process_match_requires_start_tick():
    assert process_matches({'pid': 123, 'proc_start_ticks': 456}, 123, 456)
    assert not process_matches({'pid': 123, 'proc_start_ticks': 456}, 123, 457)
    assert not process_matches({'pid': 123, 'proc_start_ticks': 456}, 124, 456)


def test_host_runner_imports_with_system_python_without_numpy():
    work = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'EATNAV_FIVE_POSE_ROOT': '/tmp/import-only',
           'PYTHONPATH': str(work/'src')}
    result = subprocess.run([shutil.which('python'), '-c',
        f'import runpy;runpy.run_path({str(work/"scripts/eat_nav_radial_five_pose_run.py")!r},run_name="not_main")'],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
