import copy

import pytest

from eat_nav_p1_longhorizon_gpu import assess_sample, capture_sample, resolve_simulator
from test_p1_gpu_attribution import case, MIB


def inputs():
    device, rows, table, owned, baseline, contract = copy.deepcopy(case())
    return device, rows, table, owned, baseline, contract


def test_full_identity_and_gap_are_recorded():
    d, rows, table, owned, baseline, contract = inputs()
    result = assess_sample(d, rows, table, d, table, owned, baseline, contract,
                           container_id='container-id')
    assert result['pass']
    assert result['gpu_uuid'] == d['uuid']
    assert result['device_total_bytes'] == d['total_bytes']
    assert result['bracket_gap_bytes'] == 100*MIB
    assert result['processes'][0]['start_ticks'] == 111


def test_any_unresolved_pid_fails_even_when_reported_memory_zero():
    d, rows, table, owned, baseline, contract = inputs()
    rows.append({'gpu_uuid': d['uuid'], 'host_pid': 999, 'kind': 'graphics',
                 'used_gpu_memory_bytes': 0})
    result = assess_sample(d, rows, table, d, table, owned, baseline, contract,
                           container_id='container-id')
    assert not result['pass']


def test_disappeared_pid_fails():
    d, rows, table, owned, baseline, contract = inputs()
    after = copy.deepcopy(table)
    del after[100]
    result = assess_sample(d, rows, table, d, after, owned, baseline, contract,
                           container_id='container-id')
    assert not result['pass']


def test_disappeared_graphics_pid_also_fails_closed():
    d, rows, table, owned, baseline, contract = inputs()
    after = copy.deepcopy(table)
    del after[20]
    result = assess_sample(d, rows, table, d, after, owned, baseline, contract,
                           container_id='container-id')
    assert 'gpu_pid_disappeared_or_reused' in result['failures']


def test_nvml_failure_fails_closed():
    class BrokenNvml:
        def sample(self):
            raise RuntimeError('NVML failure')
    with pytest.raises(RuntimeError, match='NVML'):
        capture_sample(BrokenNvml(), {}, set(), {}, container_id='x')


def test_source_owner_requires_unique_exact_pid_and_start_tick():
    _, _, table, _, _, _ = inputs()
    source = '/workspace/g1/source.py'
    table[100]['command_line'] = f'/isaac-sim/python.sh {source} --headless'
    assert resolve_simulator(table, source_path=source, container_id='container-id') == {
        'simulator': {'host_pid': 100, 'start_ticks': 111}}
    table[101]['command_line'] = f'/isaac-sim/python.sh {source} --headless'
    with pytest.raises(ValueError, match='ambiguous'):
        resolve_simulator(table, source_path=source, container_id='container-id')
