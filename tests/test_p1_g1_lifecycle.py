"""Synthetic G1 ownership, NVML and finalization tests; never starts Isaac."""

import json

import pytest

from eat_nav_p1_g1_lifecycle import (
    G1Lifecycle, diagnose_owner_resolution, residual_identities, resolve_ready_owner,
    seal_after_writers, verify_seal,
)


TOKEN = 'a' * 64
CONTAINER = 'container-id'
SOURCE = '/persistent/source.py'
OUTPUT = '/persistent/output'


def ready():
    return {'schema': 'eatnav_p1_owner_v1', 'run_token': TOKEN,
            'container_id': CONTAINER, 'container_pid': 42,
            'start_ticks': 1234, 'nspid': [900, 42],
            'source_sha256': 'b' * 64, 'config_sha256': 'c' * 64,
            'output_root': OUTPUT}


def table(*, pid=900, start=1234, cgroup=CONTAINER, token=TOKEN):
    return {pid: {'host_pid': pid, 'container_pid': 42,
                  'namespace_pid_chain': [pid, 42], 'start_ticks': start,
                  'cgroup': f'/docker/{cgroup}',
                  'command_line': f'python wrapper.py --run-token {token}'}}


def sample(*, owned_gpu=True, alive=True, nvml_pid=900):
    owner_table=table() if alive else {}
    return {'pass': True, 'failures': [], 'gpu_uuid': 'GPU-1',
            'bracket_gap_bytes': 100*1024*1024,
            'owned_gpu_pids': [nvml_pid] if owned_gpu else [],
            'processes': [{'host_pid':nvml_pid,'start_ticks':1234,
                           'classification':'owned_simulator_tree'}] if owned_gpu else [],
            'process_table_before': owner_table,
            'process_table_after': owner_table}


def new_gate():
    return G1Lifecycle(token=TOKEN, container_id=CONTAINER,
                       source_sha='b'*64, config_sha='c'*64,
                       output_root=OUTPUT, gpu_uuid='GPU-1')


def test_delayed_pid_ready_blocks_authorization_until_exact_owner():
    gate=new_gate()
    with pytest.raises(ValueError,match='owner_not_registered'):
        gate.authorize()
    owner=gate.register_ready(ready(),table())
    assert owner == {'host_pid':900,'start_ticks':1234}
    gate.pre_authorization(sample(owned_gpu=False))
    assert gate.authorize()['monitor_ready'] is True


@pytest.mark.parametrize('change', [
    {'cgroup':'other'}, {'start':999}, {'token':'wrong'},
])
def test_command_cgroup_and_namespace_match_fail_closed(change):
    with pytest.raises(ValueError):
        resolve_ready_owner(ready(),table(**change),token=TOKEN,
                            container_id=CONTAINER,source_sha='b'*64,
                            config_sha='c'*64,output_root=OUTPUT)


@pytest.mark.parametrize('change,expected', [
    ({'cgroup':'other'},'cgroup_mismatch'),
    ({'start':999},'start_tick_mismatch'),
    ({'token':'wrong'},'token_or_command_mismatch'),
])
def test_owner_resolution_diagnostic_is_specific(change,expected):
    assert diagnose_owner_resolution(ready(),table(**change),CONTAINER) == expected


def test_short_simulator_without_nvml_owned_sample_cannot_pass():
    gate=new_gate()
    gate.register_ready(ready(),table())
    gate.pre_authorization(sample(owned_gpu=False))
    gate.authorize()
    with pytest.raises(ValueError,match='owned_GPU_sample_missing'):
        gate.complete(native_exit=0, owner_alive=False, writers_closed=True)


def test_pid_disappears_before_completion_fails():
    gate=new_gate()
    gate.register_ready(ready(),table())
    gate.pre_authorization(sample(owned_gpu=False))
    gate.authorize()
    with pytest.raises(ValueError,match='owner_disappeared'):
        gate.observe(sample(alive=False))


def test_nvml_pid_mismatch_fails_without_memory_imputation():
    gate=new_gate()
    gate.register_ready(ready(),table())
    gate.pre_authorization(sample(owned_gpu=False))
    gate.authorize()
    with pytest.raises(ValueError,match='NVML_owner_mismatch'):
        gate.observe(sample(nvml_pid=901))


def test_gap_limit_stays_256_mib():
    gate=new_gate()
    gate.register_ready(ready(),table())
    bad=sample(owned_gpu=False)
    bad['bracket_gap_bytes']=257*1024*1024
    with pytest.raises(ValueError,match='attribution_gap'):
        gate.pre_authorization(bad)


def test_seal_waits_for_all_writers_and_detects_late_write(tmp_path):
    trace=tmp_path/'trace.jsonl'
    trace.write_text('{"step":0}\n')
    with pytest.raises(ValueError,match='writers_not_closed'):
        seal_after_writers(tmp_path,writer_state={'simulator':False,'heartbeat':True,
            'resource':True,'log':True},stage='G1_failed')
    assert not (tmp_path/'host-seal.json').exists()
    seal_after_writers(tmp_path,writer_state={'simulator':True,'heartbeat':True,
        'resource':True,'log':True},stage='G1_failed')
    assert verify_seal(tmp_path)['valid']
    with trace.open('a') as stream: stream.write('{"step":1}\n')
    with pytest.raises(ValueError,match='post_seal_mutation'):
        verify_seal(tmp_path)


def test_complete_requires_cleanup_and_writers():
    gate=new_gate()
    gate.register_ready(ready(),table())
    gate.pre_authorization(sample(owned_gpu=False))
    gate.authorize()
    gate.observe(sample())
    with pytest.raises(ValueError,match='owned_residual'):
        gate.complete(native_exit=0,owner_alive=True,writers_closed=True)
    with pytest.raises(ValueError,match='writers_not_closed'):
        gate.complete(native_exit=0,owner_alive=False,writers_closed=False)
    assert gate.complete(native_exit=0,owner_alive=False,writers_closed=True)


def test_unresolved_wrapper_token_prevents_seal():
    assert residual_identities(table(),owner=None,known={},token=TOKEN,
                               container_id=CONTAINER) == [
        {'host_pid':900,'start_ticks':1234,'reason':'unresolved_run_token_claimant'}]


def test_host_command_runs_paused_wrapper_before_simulator_source():
    from eat_nav_p1_g1_gpu_host import build_wrapper_command
    cmd=build_wrapper_command(remote='/persistent',container_id=CONTAINER,
        token=TOKEN,source_sha='b'*64,config_sha='c'*64)
    assert cmd[cmd.index('/isaac-sim/python.sh')+1]=='/persistent/owner-wrapper.py'
    assert cmd.index('/persistent/owner-wrapper.py') < cmd.index('/persistent/source.py')
    assert cmd[cmd.index('--mode')+1]=='smoke'
    assert cmd[cmd.index('--output')+1]=='/persistent/output'
