import pytest

from eat_nav_p1_owner_host import load_terminal_marker, validate_smoke_artifacts


OWNER = {'host_pid': 900, 'start_ticks': 77}
TOKEN = 'a' * 64
SOURCE = 'b' * 64
CONFIG = 'c' * 64


def test_load_frozen_smoke_marker_attests_identity_and_preclose_order():
    raw = {'data': {'physics_steps': 200, 'pose_count': 4,
                    'physical_termination_pose_ids': [], 'sim_source_sha256': SOURCE},
           'sha256': 'd' * 64, 'mtime_ns': 150}
    marker = load_terminal_marker('smoke', raw, OWNER, TOKEN, SOURCE, CONFIG,
                                  '/tmp/run', before_close_source_verified=True)
    assert marker['host_pid'] == 900
    assert marker['written_wall_ns'] == 150
    assert marker['marker_before_close'] is True


def test_missing_marker_stays_missing_and_wrong_source_fails():
    assert load_terminal_marker('smoke', None, OWNER, TOKEN, SOURCE, CONFIG,
                                '/tmp/run', before_close_source_verified=True) is None
    raw = {'data': {'physics_steps': 200, 'physical_termination_pose_ids': [],
                    'sim_source_sha256': 'wrong'}, 'sha256': 'd' * 64, 'mtime_ns': 150}
    with pytest.raises(ValueError, match='completion_source'):
        load_terminal_marker('smoke', raw, OWNER, TOKEN, SOURCE, CONFIG,
                             '/tmp/run', before_close_source_verified=True)


def test_smoke_control_count_must_match_marker_envelope(tmp_path):
    import json
    import shutil
    from pathlib import Path
    prior = Path('/tmp/eatnav-radial-anchor-contract-20261008-r2/shard-00/run')
    if not prior.exists():
        pytest.skip('prior fixture unavailable')
    for name in ('sim-finished.json', 'trace', 'dem', 'actor'):
        (tmp_path / name).symlink_to(prior / name)
    shutil.copytree(prior / 'poses', tmp_path / 'poses')
    pose = next((tmp_path / 'poses').glob('*.json'))
    data = json.loads(pose.read_text())
    data['control_step_count'] = 19
    pose.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='smoke_control_count_invalid'):
        validate_smoke_artifacts(tmp_path, expected_pose_count=4)
