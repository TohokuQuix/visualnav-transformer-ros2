import json
from pathlib import Path

import pytest

from eat_nav_p1_owner_wrapper import validate_authorization, wait_for_authorization, forbidden_pre_auth_imports


IDENTITY = {'schema': 'eatnav_p1_owner_v1', 'run_token': 'a' * 64,
            'container_pid': 42, 'start_ticks': 1234, 'output_root': '/tmp/run',
            'source_sha256': 'b' * 64, 'config_sha256': 'c' * 64}


def test_matching_authorization(tmp_path):
    marker = {**IDENTITY, 'host_pid': 987, 'monitor_ready': True}
    assert validate_authorization(marker, IDENTITY) == marker


@pytest.mark.parametrize('mutation', [
    {'run_token': 'z' * 64}, {'container_pid': 99}, {'start_ticks': 2},
    {'output_root': '/tmp/other'}, {'source_sha256': 'd' * 64},
    {'config_sha256': 'e' * 64}, {'monitor_ready': False},
])
def test_mismatch_rejected(mutation):
    with pytest.raises(ValueError):
        validate_authorization({**IDENTITY, 'monitor_ready': True, **mutation}, IDENTITY)


def test_timeout_and_rejection_never_import_gpu(tmp_path):
    with pytest.raises(TimeoutError):
        wait_for_authorization(tmp_path, IDENTITY, timeout_sec=0.01, poll_sec=0.001)
    (tmp_path / 'owner-rejected.json').write_text(json.dumps({'run_token': IDENTITY['run_token']}))
    with pytest.raises(RuntimeError, match='rejected'):
        wait_for_authorization(tmp_path, IDENTITY, timeout_sec=0.01, poll_sec=0.001)


def test_wrapper_does_not_import_gpu_before_authorization():
    source = Path(__file__).resolve().parents[1] / 'scripts/eat_nav_p1_owner_wrapper.py'
    assert forbidden_pre_auth_imports(source.read_text()) == []


def test_stale_authorization_rejected_before_ready(tmp_path, monkeypatch):
    import sys
    from eat_nav_p1_owner_wrapper import main
    source = tmp_path / 'sim.py'
    source.write_text('pass\n')
    import hashlib
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    coord = tmp_path / 'coord'
    coord.mkdir()
    (coord / 'launch-intent.json').write_text(json.dumps({'run_token': 'a' * 64,
        'output_root': str(tmp_path / 'run')}))
    (coord / 'owner-authorized.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['wrapper.py', '--coord', str(coord), '--run-token', 'a' * 64,
        '--container-id', 'xyz', '--source', str(source), '--source-sha', digest,
        '--config', str(source), '--config-sha', digest, '--output', str(tmp_path / 'run'),
        '--mode', 'preflight'])
    with pytest.raises(ValueError, match='stale_coordination_marker'):
        main()


def test_preflight_identity_marker_is_written_before_close(tmp_path, monkeypatch):
    import sys
    import types
    from eat_nav_p1_owner_wrapper import preflight
    identity = {**IDENTITY, 'host_pid': 900, 'source_sha256': 'b' * 64, 'config_sha256': 'c' * 64}
    observed = []
    class App:
        def close(self):
            marker = json.loads((tmp_path / 'completion.json').read_text())
            observed.append(marker)
    package = types.ModuleType('isaaclab')
    module = types.ModuleType('isaaclab.app')
    module.AppLauncher = lambda headless: types.SimpleNamespace(app=App())
    monkeypatch.setitem(sys.modules, 'isaaclab', package)
    monkeypatch.setitem(sys.modules, 'isaaclab.app', module)
    preflight(tmp_path, identity)
    assert observed[0]['source_sha256'] == identity['source_sha256']
    assert observed[0]['config_sha256'] == identity['config_sha256']
    assert observed[0]['output_root'] == identity['output_root']
    assert observed[0]['completion_state'] == 'complete'
    assert observed[0]['written_wall_ns'] > 0
