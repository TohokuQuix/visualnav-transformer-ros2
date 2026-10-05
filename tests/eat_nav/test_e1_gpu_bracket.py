"""Startup GPU attribution must use device totals around process enumeration."""

import eat_nav_e1_monitor as monitor
from eat_nav_g0e_formal_gate import runtime_gate
from eat_nav_g0e_gpu_probe import MIB


class RisingNvml:
    def __init__(self, after_mib=5380):
        self.after_mib = after_mib

    def sample(self):
        self.last_query_timing = {'device_before_start_ns': 1,
                                  'device_before_end_ns': 2,
                                  'compute_start_ns': 3,
                                  'compute_end_ns': 4,
                                  'graphics_start_ns': 5,
                                  'graphics_end_ns': 6}
        before = [{'index': 0, 'uuid': 'GPU-test', 'used_bytes': 3714 * MIB}]
        rows = [
            {'host_pid': 100, 'gpu_uuid': 'GPU-test', 'gpu_index': 0,
             'used_gpu_memory_bytes': 4257 * MIB, 'kind': 'compute'},
            {'host_pid': 100, 'gpu_uuid': 'GPU-test', 'gpu_index': 0,
             'used_gpu_memory_bytes': 4257 * MIB, 'kind': 'graphics'},
            {'host_pid': 200, 'gpu_uuid': 'GPU-test', 'gpu_index': 0,
             'used_gpu_memory_bytes': 297 * MIB, 'kind': 'graphics'},
        ]
        return before, rows

    def memory_snapshot(self):
        self.last_memory_snapshot_timing = {'device_after_start_ns': 9,
                                            'device_after_end_ns': 10}
        return [{'index': 0, 'uuid': 'GPU-test',
                 'used_bytes': self.after_mib * MIB}]


def _sample(monkeypatch, after_mib, *, return_snapshot=False):
    table = {
        100: {'host_pid': 100, 'parent_host_pid': 1, 'start_ticks': 11,
              'container_pid': 10, 'cgroup': 'docker-abc',
              'command_line': 'sim --output /tmp/run', 'rss_bytes': 0,
              'fd_count': 0, 'thread_count': 1},
        200: {'host_pid': 200, 'parent_host_pid': 1, 'start_ticks': 22,
              'container_pid': 200, 'cgroup': 'host',
              'command_line': 'Xorg', 'rss_bytes': 0,
              'fd_count': 0, 'thread_count': 1},
    }
    monkeypatch.setattr(monitor, 'proc_table', lambda: table)
    monkeypatch.setattr(monitor, 'cgroup_memory', lambda _: {
        'anon_bytes': 0, 'shmem_bytes': 0, 'events': {'oom': 0}})
    monkeypatch.setattr(monitor, 'container_proc_details', lambda _: {})
    monkeypatch.setattr(monitor, 'host_available_bytes', lambda: 100_000_000_000)
    snapshot = monitor.sample(RisingNvml(after_mib), {'container_id': 'abc'},
                              set(), '/tmp/run', source='sim')
    return snapshot if return_snapshot else monitor.gate_sample(snapshot, 0)


LIMITS = {'GPU_cap_mib': 6656, 'unattributed_cap_mib': 256,
          'anon_shmem_cap_bytes': 6_442_450_944,
          'host_memory_floor_bytes': 1_000_000_000}


def test_startup_allocation_inside_device_bracket_does_not_false_stop(monkeypatch):
    gate = _sample(monkeypatch, 5380)
    assert gate['device_total_used_mib'] == 5380
    assert runtime_gate(gate, LIMITS) is None


def test_process_sum_outside_bracket_still_stops(monkeypatch):
    assert runtime_gate(_sample(monkeypatch, 3900), LIMITS) == 'unattributed_gpu'


def test_after_device_total_above_hard_cap_still_stops(monkeypatch):
    assert runtime_gate(_sample(monkeypatch, 6700), LIMITS).startswith('device_GPU_cap')


def test_bracketed_snapshot_preserves_raw_rows_and_query_times(monkeypatch):
    snapshot = _sample(monkeypatch, 5380, return_snapshot=True)
    acquisition = snapshot['gpu_acquisition']
    assert acquisition['device_before'][0]['used_bytes'] == 3714 * MIB
    assert acquisition['device_after'][0]['used_bytes'] == 5380 * MIB
    assert len(acquisition['raw_process_rows']) == 3
    assert len(snapshot['gpu_accounting']['processes']) == 2
    assert acquisition['query_timing']['device_before_start_ns'] == 1
    assert acquisition['query_timing']['device_after_end_ns'] == 10
    assert acquisition['query_timing']['process_table_end_ns'] >= acquisition['query_timing']['process_table_start_ns']
