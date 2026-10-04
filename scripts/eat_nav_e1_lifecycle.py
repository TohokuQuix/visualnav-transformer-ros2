"""E1 runtime/finalization state transitions with a frozen 30 second grace."""
from __future__ import annotations

import json
from pathlib import Path


def read_complete_json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def marker_matches_run(marker, run_dir, expected_map_path_sha):
    if not isinstance(marker, dict) or marker.get('fixed_map_path_sha256') != expected_map_path_sha:
        return False
    names = marker.get('outputs_sha256')
    if not isinstance(names, dict) or not names:
        return False
    root = Path(run_dir).resolve()
    return all(Path(name).resolve().parent == root for name in names)


def protocol_valid_count(runs):
    return sum((read_complete_json(p) or {}).get('status') == 'valid'
               for p in Path(runs).glob('*/protocol-valid.json'))


class LifecycleMachine:
    def __init__(self, expected_steps=120, grace_ns=30_000_000_000):
        self.expected_steps = expected_steps
        self.grace_ns = grace_ns
        self.state = 'STARTING'
        self.reason = None
        self.host_start_ticks = None
        self.finalizing_since_ns = None
        self.transitions = []

    def _move(self, state, now, reason=None):
        if state != self.state:
            self.transitions.append({'from': self.state, 'to': state,
                                     'monotonic_ns': now, 'reason': reason})
        self.state, self.reason = state, reason
        return self

    def observe(self, sample):
        now = sample['now_ns']
        if self.state in ('COMPLETE', 'FAIL'):
            return self
        if self.host_start_ticks is None:
            self.host_start_ticks = sample['host_start_ticks']
        elif (sample['host_alive'] and sample['host_start_ticks'] is not None
              and sample['host_start_ticks'] != self.host_start_ticks):
            return self._move('FAIL', now, 'host_pid_reused')
        stop = sample['resource_stop']
        if stop:
            return self._move('FAIL', now, stop)
        if sample['heartbeat_stale'] and self.state != 'FINALIZING':
            return self._move('FAIL', now, 'heartbeat_stall')
        if sample['router_alive'] is False:
            return self._move('FAIL', now, 'router_disconnected')
        for key, reason in (('bridge_alive', 'bridge_disconnected'),
                            ('followers_alive', 'follower_disconnected')):
            if sample[key] is False and self.state != 'FINALIZING':
                return self._move('FAIL', now, reason)
        if sample['protocol_error']:
            return self._move('FAIL', now, 'protocol_error')
        if sample['nonfinite']:
            return self._move('FAIL', now, 'nonfinite')
        if self.state == 'STARTING':
            if not sample['simulator_tree']:
                return self
            self._move('RUNNING', now)
        if self.state == 'RUNNING':
            completion = sample['completion']
            if not sample['simulator_tree'] or completion is not None:
                if sample['step_count'] < self.expected_steps:
                    return self._move('FAIL', now, 'before_expected_steps')
                if completion is None:
                    return self._move('FAIL', now, 'completion_marker_missing')
                if not completion.get('valid') or not completion.get('trace_closed') or not completion.get('trace_sha_match'):
                    return self._move('FAIL', now, 'trace_sha')
                if completion.get('step_count') != self.expected_steps or not completion.get('normal_completion'):
                    return self._move('FAIL', now, 'completion_marker_invalid')
                self.finalizing_since_ns = now
                self._move('FINALIZING', now)
        if self.state == 'FINALIZING':
            f = sample['final']
            if f is not None:
                for condition, reason in ((f['docker_exit'] not in (None, 0), 'docker_exit'),
                        (f['bridge_exit'] not in (None, 0), 'bridge_exit'),
                        (f['followers_exit_ok'] is False, 'follower_exit'),
                        (f['output_sha_match'] is False, 'output_sha'),
                        (f['residual_count'] > 0, 'residual'),
                        (f['zombie_count'] > 0, 'zombie')):
                    if condition:
                        return self._move('FAIL', now, reason)
                if (f['docker_exit'] == 0 and f['bridge_exit'] == 0 and f['followers_exit_ok']
                        and f['run_complete_exists'] and f['output_sha_match']
                        and f['residual_count'] == f['zombie_count'] == 0 and f['cleanup_marker']):
                    return self._move('COMPLETE', now)
            if now - self.finalizing_since_ns > self.grace_ns:
                return self._move('FAIL', now, 'finalization_grace')
        return self
