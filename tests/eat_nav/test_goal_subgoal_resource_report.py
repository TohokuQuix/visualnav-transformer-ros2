"""Synthetic completed anchor verifies report generation before the live probe."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import eat_nav_goal_subgoal_resource_report as report


class ResourceReportTest(unittest.TestCase):
    def test_complete_probe_writes_forecast(self):
        seconds = [0.01] * 1621
        resets = [399, 799, 1199, 1599]
        for step in resets:
            seconds[step] = 0.02
        events = [{'event': name, 'monotonic_ns': round(value * 1e9)} for name, value in (
            ('isaac_start', 1), ('scene_construction_complete', 1.8),
            ('initial_reset_complete', 1.9), ('prefix_replay_start', 2),
            ('target_anchor_reached', 20), ('anchor_verification_complete', 20.2),
            ('cleanup_start', 21))]
        events.extend({'event': 'episode_reset', 'global_step': step,
                       'monotonic_ns': round((3 + step * .01) * 1e9)} for step in resets)
        probe = report.summarize_probe(events, [round(value * 1e9) for value in seconds],
                                       host_start_ns=0, host_return_ns=22_000_000_000,
                                       run_elapsed_sec=22)
        self.assertEqual(probe['all_step']['count'], 1621)
        self.assertEqual(len(probe['by_episode']), 5)
        self.assertEqual(probe['reset_indices'], resets)
        suffix_log = '\n'.join((
            'DIAG_BRIDGE_REQUEST_ENTRY reset 100 1000000000',
            'DIAG_BRIDGE_REQUEST_RETURN reset 100 1500000000',
            'DIAG_ENV_STEP_ENTRY 0 1600000000',
            'DIAG_ENV_STEP_RETURN 119 7600000000',
            'DIAG_BRIDGE_REQUEST_RETURN finish None 7800000000',
            'DIAG_APP_CLOSE_ENTRY 8000000000'))
        suffix = report.suffix_timing(suffix_log, host_elapsed_sec=10, sim_elapsed_sec=8)
        self.assertEqual(suffix['follower_bridge_setup_sec'], .5)
        prediction = report.forecast(probe, suffix, storage_bytes_per_run=1_000_000,
                                     observed_gpu_peak_mib=5550, observed_anon_peak_bytes=2_500_000_000)
        self.assertEqual(prediction['live_runs'], 44)
        self.assertGreater(prediction['conservative_plus_20_percent_sec'], prediction['point']['total_sec'])
        with tempfile.TemporaryDirectory() as tmp:
            result = report.write_report(Path(tmp), probe, prediction)
            payload = json.loads(Path(result['path']).read_text())
            self.assertEqual(payload['forecast']['live_runs'], 44)
            with self.assertRaises(FileExistsError):
                report.write_report(Path(tmp), probe, prediction)


if __name__ == '__main__':
    unittest.main()
