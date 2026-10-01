"""Resource gate fixtures use the same predicates as the external monitor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import eat_nav_goal_subgoal_resource_monitor as monitor
from visualnav_transformer.eat_nav.readiness_recovery import terminate_tree


GIB = 1024**3


def config():
    return {'memory_events_baseline': {key: 0 for key in monitor.EVENT_KEYS},
            'anon_shmem_cap_bytes': 6 * GIB,
            'host_mem_available_min_bytes': 10 * GIB,
            'gpu_cap_mib': 6656, 'step_stall_sec': 120,
            'storage_cap_bytes': 800_000_000,
            'file_cache_warn_bytes': 6 * GIB}


def snapshot():
    return {'memory_events': {key: 0 for key in monitor.EVENT_KEYS},
            'kernel_oom': False, 'docker_oom': False,
            'memory_max_bytes': None, 'memory_current_bytes': 10 * GIB,
            'memory_stat': {'anon': 2 * GIB, 'shmem': 0, 'file': 7 * GIB},
            'mem_available_bytes': 50 * GIB, 'gpu_used_mib': 5550,
            'gpu_xid': False, 'simulator_alive': True,
            'seconds_since_progress': 0, 'output_bytes': 1_000_000,
            'projected_output_bytes': 100_000_000,
            'router_alive': True, 'docker_exec_alive': True,
            'simulator_logical_time_sec': 1.0}


class ResourceMonitorTest(unittest.TestCase):
    def test_memory_stat_parser_and_anon_shmem(self):
        parsed = monitor.parse_pairs('anon 200\nfile 900\nshmem 30\n', wanted=('anon', 'file', 'shmem'))
        self.assertEqual(monitor.anon_shmem(parsed), 230)
        with self.assertRaises(ValueError):
            monitor.parse_pairs('anon 1\n', wanted=('anon', 'shmem'))

    def test_events_delta_and_oom_stop(self):
        row = snapshot()
        row['memory_events']['oom'] = 1
        self.assertEqual(monitor.event_delta(row['memory_events'], config()['memory_events_baseline'])['oom'], 1)
        self.assertIn('OOM', monitor.evaluate(row, config(), {})[0])
        with self.assertRaises(ValueError):
            monitor.event_delta({key: 0 for key in monitor.EVENT_KEYS},
                                {key: 1 for key in monitor.EVENT_KEYS})

    def test_reclaimable_file_cache_does_not_stop(self):
        row = snapshot()
        self.assertGreater(row['memory_current_bytes'], 8 * GIB)
        reason, warnings = monitor.evaluate(row, config(), {})
        self.assertIsNone(reason)
        self.assertTrue(any('file cache' in warning for warning in warnings))

    def test_anon_and_gpu_caps(self):
        row = snapshot()
        row['memory_stat']['anon'] = 6 * GIB + 1
        self.assertIn('anonymous', monitor.evaluate(row, config(), {})[0])
        row = snapshot()
        row['gpu_used_mib'] = 6657
        self.assertIn('GPU', monitor.evaluate(row, config(), {})[0])

    def test_host_memory_requires_three_consecutive_samples(self):
        row = snapshot()
        row['mem_available_bytes'] = 9 * GIB
        state = {}
        self.assertIsNone(monitor.evaluate(row, config(), state)[0])
        self.assertIsNone(monitor.evaluate(row, config(), state)[0])
        self.assertIn('three heartbeats', monitor.evaluate(row, config(), state)[0])
        row['mem_available_bytes'] = 50 * GIB
        self.assertIsNone(monitor.evaluate(row, config(), state)[0])
        self.assertEqual(state['host_low_streak'], 0)

    def test_stall_and_stale_heartbeat(self):
        row = snapshot()
        row['seconds_since_progress'] = 120
        self.assertIn('120 seconds', monitor.evaluate(row, config(), {})[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'heartbeat.json'
            monitor.atomic(path, {'step': 1})
            now = time.time()
            os.utime(path, (now - 91, now - 91))
            self.assertTrue(monitor.heartbeat_stale(path, now, 90))
            self.assertFalse(monitor.heartbeat_stale(path, now, 95))

    def test_cleanup_grace_allows_simulator_exit_only_after_suffix(self):
        row = snapshot()
        row['docker_exec_alive'] = False
        row['simulator_alive'] = False
        self.assertIn('Docker exec', monitor.evaluate(row, config(), {})[0])
        self.assertIsNone(monitor.evaluate(row, config(), {'cleanup_grace_active': True})[0])
        row['gpu_used_mib'] = 6657
        self.assertIn('GPU', monitor.evaluate(row, config(), {'cleanup_grace_active': True})[0])

    def test_cleanup_requires_sha_verified_complete_marker(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'live-0.json').write_bytes(b'completed')
            self.assertFalse(monitor.verified_complete_marker(folder))
            monitor.atomic(folder / 'run-complete.json', {
                'protocol': 'fresh_bridge_per_run',
                'output_sha256': {'live-0.json': hashlib.sha256(b'completed').hexdigest()}})
            self.assertTrue(monitor.verified_complete_marker(folder))
            (folder / 'live-0.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                monitor.verified_complete_marker(folder)

    def test_live_suffix_progress_uses_existing_step_return_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'simulator.log'
            self.assertIsNone(monitor.completed_suffix_step(path, 60))
            path.write_text('DIAG_ENV_STEP_ENTRY 0 111\nDIAG_ENV_STEP_RETURN 0 222\n'
                            'DIAG_ENV_STEP_ENTRY 1 333\nDIAG_ENV_STEP_RETURN 1 444\n')
            self.assertEqual(monitor.completed_suffix_step(path, 60), 62)
            path.write_text(path.read_text() + 'DIAG_ENV_STEP_ENTRY 2 555\n')
            self.assertEqual(monitor.completed_suffix_step(path, 60), 62)

    def test_atomic_heartbeat_and_failure_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'timing-heartbeat.json'
            monitor.atomic(path, {'step': 7})
            self.assertEqual(json.loads(path.read_text())['step'], 7)
            self.assertFalse(path.with_name(path.name + '.tmp').exists())
            fail = Path(tmp) / 'resource-stop.failure.json'
            monitor.write_failure_marker(fail, {'reason': 'fixture'})
            self.assertEqual(json.loads(fail.read_text())['reason'], 'fixture')
            with self.assertRaises(FileExistsError):
                monitor.write_failure_marker(fail, {'reason': 'overwrite'})

    def test_cleanup_terminates_detached_process_group(self):
        process = subprocess.Popen(['sleep', '30'], start_new_session=True)
        try:
            terminate_tree(process, term_wait_sec=2, kill_wait_sec=2)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                terminate_tree(process, term_wait_sec=2, kill_wait_sec=2)

    def test_fixed_python_has_numpy(self):
        self.assertEqual(sys.executable, '/usr/bin/python3')
        import numpy
        self.assertTrue(numpy.__version__)

    def test_v2_storage_uses_remaining_steps_and_frozen_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            active = root / 'runs' / 'run-1-attempt-1'
            active.mkdir(parents=True)
            (active / 'simulator.log').write_bytes(b'x' * 10_000)
            schedule = {'runs': [{'run_id': f'run-{i}', 'order': i,
                                  'kind': 'anchor' if i < 4 else 'candidate'}
                                 for i in range(48)]}
            result = monitor.v2_storage_projection(
                root=root, active=active, remote_root_bytes=50_000,
                remote_active_bytes=5_000, completed_remote_bytes={},
                schedule=schedule, completed_step=60,
                target_step=60, initial_active_bytes=5_000,
                baseline_bytes=239_907_072)
            self.assertEqual(result['active_raw_bytes'], 15_000)
            self.assertEqual(result['remaining_current_control_steps'], 120)
            self.assertEqual(result['remaining_future_run_count'], 47)
            self.assertGreaterEqual(result['projected_final_storage_bytes'], 239_907_072)
            self.assertLess(result['projected_final_storage_bytes'], 800_000_000)
            self.assertEqual(result['completed_live_count'], 0)
            self.assertEqual(result['actual_storage_bytes'], 60_000)
            row = snapshot()
            row['output_bytes'] = result['actual_storage_bytes']
            row['projected_output_bytes'] = result['projected_final_storage_bytes']
            self.assertIsNone(monitor.evaluate(row, config(), {})[0])

    def test_v2_storage_counts_completed_compressed_run_and_actual_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = root / 'runs' / 'run-4'
            completed.mkdir(parents=True)
            (completed / 'run-complete.json').write_text('{}')
            (completed / 'trace.json.gz').write_bytes(b'z' * 5_000)
            active = root / 'runs' / 'run-5'
            active.mkdir()
            (active / 'simulator.log').write_bytes(b'x' * 2_000)
            schedule = {'runs': [{'run_id': f'run-{i}', 'order': i,
                                  'kind': 'candidate'} for i in range(48)]}
            result = monitor.v2_storage_projection(
                root=root, active=active, remote_root_bytes=1_000,
                remote_active_bytes=500, completed_remote_bytes={'run-4': 2_000},
                schedule=schedule, completed_step=5,
                target_step=60, initial_active_bytes=1_000,
                baseline_bytes=239_907_072)
            self.assertEqual(result['completed_live_count'], 1)
            self.assertEqual(result['cleanup_compressed_bytes'], 5_000)
            self.assertEqual(result['completed_physical_run_bytes'], 7_002)
            self.assertEqual(result['remaining_future_run_count'], 46)
            row = snapshot()
            row['output_bytes'] = 800_000_001
            row['projected_output_bytes'] = 800_000_001
            self.assertIn('storage cap', monitor.evaluate(row, config(), {})[0])


if __name__ == '__main__':
    unittest.main()
