import pytest

from eat_nav_p1_owner_host import resolve_owner, assert_pre_authorization_gpu_clear, exact_owned_alive, build_wrapper_command, no_prior_owner_live, initial_graphics_identities, validate_smoke_artifacts


READY = {'run_token': 'a' * 64, 'container_pid': 42, 'start_ticks': 77,
         'nspid': [42], 'cgroup': '0::/docker/xyz', 'container_id': 'xyz'}
ROW = {'host_pid': 900, 'container_pid': 42, 'namespace_pid_chain': [900, 42],
       'start_ticks': 77, 'cgroup': '0::/docker/xyz',
       'command_line': f'python wrapper.py --run-token {"a" * 64} --output /tmp/run',
       'parent_host_pid': 5}


def test_exact_one_to_one_nspid_token_cgroup_start_tick():
    assert resolve_owner({900: ROW}, READY, 'xyz') == {'host_pid': 900, 'start_ticks': 77}


@pytest.mark.parametrize('mutation', [
    {'container_pid': None}, {'namespace_pid_chain': []}, {'start_ticks': 78},
    {'cgroup': '0::/docker/other'}, {'command_line': 'python wrapper.py --output /tmp/run'},
])
def test_wrong_identity_rejected(mutation):
    with pytest.raises(ValueError):
        resolve_owner({900: {**ROW, **mutation}}, READY, 'xyz')


def test_duplicate_token_and_path_only_false_match_rejected():
    with pytest.raises(ValueError, match='ambiguous'):
        resolve_owner({900: ROW, 901: {**ROW, 'host_pid': 901}}, READY, 'xyz')
    with pytest.raises(ValueError):
        resolve_owner({900: {**ROW, 'command_line': 'python --output /tmp/run'}}, READY, 'xyz')


def test_gpu_before_authorization_rejected():
    with pytest.raises(ValueError, match='gpu_before_authorization'):
        assert_pre_authorization_gpu_clear({'owned_gpu_pids': [900], 'pass': True})


def test_pid_reuse_cannot_be_cleaned_as_owned():
    assert not exact_owned_alive({900: {**ROW, 'start_ticks': 78}}, {'host_pid': 900, 'start_ticks': 77})
    assert exact_owned_alive({900: ROW}, {'host_pid': 900, 'start_ticks': 77})


def test_wrapper_invocation_preserves_python_and_sim_arguments():
    original = ['docker', 'exec', '-e', 'PYTHONPATH=x', 'isaac-lab-onix',
                '/isaac-sim/python.sh', '/frozen/sim.py', '--headless', '--freeze', '/frozen/config.json',
                '--output', '/new/run', '--mode', 'full']
    cmd = build_wrapper_command(original, '/new/wrapper.py', ['--run-token', 'a' * 64], mode='smoke')
    assert cmd[:7] == original[:6] + ['/new/wrapper.py']
    assert cmd[cmd.index('--') + 1:] == original[7:]
    assert '/frozen/sim.py' not in cmd[:cmd.index('--')]


def test_no_retry_while_prior_exact_owner_live(tmp_path, monkeypatch):
    import json
    marker = tmp_path / 'preflight/coord/registered-owner.json'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({'host_pid': 900, 'start_ticks': 77}))
    monkeypatch.setattr('eat_nav_p1_owner_host.proc_table', lambda: {900: ROW})
    with pytest.raises(RuntimeError, match='prior_owned_pid_still_live'):
        no_prior_owner_live(tmp_path)


def test_prelaunch_baseline_registers_only_exact_graphics_identity():
    rows = [{'host_pid': 5, 'kind': 'graphics'}, {'host_pid': 6, 'kind': 'compute'}]
    table = {5: {'start_ticks': 11}, 6: {'start_ticks': 12}}
    assert initial_graphics_identities(rows, table) == {(5, 11)}


def test_prior_candidate_free_fixture_has_complete_artifacts():
    from pathlib import Path
    root = Path('/tmp/eatnav-radial-anchor-contract-20261008-r2/shard-00/run')
    if root.exists():
        result = validate_smoke_artifacts(root, expected_pose_count=4)
        assert result['pose_count'] == 4
        assert result['physics_steps'] == 200
