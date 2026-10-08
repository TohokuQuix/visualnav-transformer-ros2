"""Identity- and ordering-aware terminal GPU lifecycle for P1 processes."""
from __future__ import annotations

from dataclasses import dataclass, field


SCHEMA = 'eatnav_p1_terminal_marker_v1'
EXPECTED_EXIT_FLAGS = frozenset(('owned_start_tick_mismatch',
                                  'simulator_disappeared_in_bracket',
                                  'owned_process_disappeared_in_bracket'))


def completion_envelope(raw: dict, *, raw_sha: str, raw_mtime_ns: int,
                        owner: dict, token: str, source_sha: str, config_sha: str,
                        output_root: str, expected_physics_steps: int,
                        expected_control_steps: int,
                        before_close_source_verified: bool) -> dict:
    """Attest the frozen simulator's pre-close marker with handshake identity."""
    if not before_close_source_verified or raw.get('sim_source_sha256') != source_sha:
        raise ValueError('completion_source_or_preclose_proof_invalid')
    if raw.get('physics_steps') != expected_physics_steps or raw.get('physical_termination_pose_ids'):
        raise ValueError('completion_step_or_termination_invalid')
    if not isinstance(raw_mtime_ns, int) or raw_mtime_ns <= 0 or len(raw_sha) != 64:
        raise ValueError('completion_marker_metadata_invalid')
    return {'schema': SCHEMA, 'run_token': token,
            'host_pid': owner['host_pid'], 'start_ticks': owner['start_ticks'],
            'source_sha256': source_sha, 'config_sha256': config_sha,
            'output_root': output_root, 'completion_state': 'complete',
            'physics_steps': expected_physics_steps,
            'control_steps': expected_control_steps,
            'written_wall_ns': raw_mtime_ns, 'marker_before_close': True,
            'original_marker_sha256': raw_sha,
            'marker_origin': 'frozen_sim_preclose_attestation'}


def signal_if_exact(owner: dict, lookup, sender) -> bool:
    """Refuse to signal a missing or reused PID."""
    current = lookup(owner['host_pid'])
    if current is None or current.get('start_ticks') != owner['start_ticks']:
        return False
    sender(owner['host_pid'])
    return True


@dataclass
class TerminalLifecycle:
    owner: dict
    token: str
    source_sha: str
    config_sha: str
    output_root: str
    expected_physics_steps: int
    expected_control_steps: int
    authorization_wall_ns: int
    container_id: str = ''
    allowed_token_claimants: set[tuple[int, int]] = field(default_factory=set)
    phase: str = 'PREAUTH'
    marker: dict | None = None
    events: list[dict] = field(default_factory=list)

    def _fail(self, reason: str):
        self.phase = 'FAILED'
        raise ValueError(reason)

    def authorize(self) -> str:
        if self.phase != 'PREAUTH' or not self.events or self.events[-1]['phase'] != 'PREAUTH':
            self._fail('preauthorization_bracket_missing')
        self.phase = 'AUTHORIZED'
        return self.phase

    def _validate_marker(self, marker: dict, before_present: bool, after_present: bool,
                         before_wall_ns: int, after_wall_ns: int) -> None:
        identity = {'schema': SCHEMA, 'run_token': self.token,
                    'host_pid': self.owner['host_pid'], 'start_ticks': self.owner['start_ticks'],
                    'source_sha256': self.source_sha, 'config_sha256': self.config_sha,
                    'output_root': self.output_root, 'completion_state': 'complete'}
        if any(marker.get(key) != value for key, value in identity.items()):
            self._fail('completion_identity_mismatch')
        if (marker.get('physics_steps') != self.expected_physics_steps
                or marker.get('control_steps') != self.expected_control_steps):
            self._fail('completion_step_mismatch')
        written = marker.get('written_wall_ns')
        if not isinstance(written, int) or marker.get('marker_before_close') is not True:
            self._fail('completion_marker_order_unproven')
        if written < self.authorization_wall_ns:
            self._fail('stale_completion_marker')
        # A marker written after the first observation of an absent owner
        # cannot retrospectively justify that disappearance.
        if not before_present and written > before_wall_ns:
            self._fail('marker_after_owner_disappearance')
        if before_present and not after_present and written > after_wall_ns:
            self._fail('marker_after_owner_disappearance')

    def observe(self, sample: dict, marker: dict | None = None) -> str:
        before = sample.get('process_table_before')
        after = sample.get('process_table_after')
        if not isinstance(before, dict) or not isinstance(after, dict):
            self._fail('missing_process_table')
        pid = self.owner['host_pid']
        ticks = self.owner['start_ticks']
        b = before.get(pid, before.get(str(pid)))
        a = after.get(pid, after.get(str(pid)))
        if (b is not None and b.get('start_ticks') != ticks) or (a is not None and a.get('start_ticks') != ticks):
            self._fail('owned_pid_reused')
        before_present, after_present = b is not None, a is not None
        failures = set(sample.get('failures', []))
        if self.phase == 'PREAUTH':
            if marker is not None or not before_present or not after_present or sample.get('owned_gpu_pids'):
                self._fail('preauthorization_identity_or_GPU_invalid')
            if failures:
                self._fail(','.join(sorted(failures)))
            self.events.append({'phase': 'PREAUTH', 'effective_failures': [],
                                'before_present': True, 'after_present': True})
            return self.phase
        if self.phase not in ('AUTHORIZED', 'RUNNING', 'FINALIZING'):
            self._fail('lifecycle_invalid_phase')
        current_marker = marker or self.marker
        if current_marker is not None:
            for table in (before, after):
                for key, row in table.items():
                    if self.container_id and self.container_id not in row.get('cgroup', ''):
                        continue
                    if self.token not in row.get('command_line', '').split():
                        continue
                    identity = (int(row.get('host_pid', key)), row.get('start_ticks'))
                    if identity != (pid, ticks) and identity not in self.allowed_token_claimants:
                        self._fail('new_run_token_claimant')
        if current_marker is None:
            if not before_present or not after_present:
                self._fail('unexpected_owned_process_loss')
            if failures:
                self._fail(','.join(sorted(failures)))
            self.phase = 'RUNNING'
            self.events.append({'phase': self.phase, 'effective_failures': [],
                                'before_present': True, 'after_present': True})
            return self.phase
        before_time = sample.get('process_table_before_wall_ns')
        after_time = sample.get('process_table_after_wall_ns')
        if not isinstance(before_time, int) or not isinstance(after_time, int) or before_time > after_time:
            self._fail('bracket_timestamp_missing_or_invalid')
        self._validate_marker(current_marker, before_present, after_present, before_time, after_time)
        if not before_present or not after_present:
            failures.difference_update(EXPECTED_EXIT_FLAGS)
        if failures:
            self._fail(','.join(sorted(failures)))
        self.marker = current_marker
        self.phase = 'FINALIZING'
        self.events.append({'phase': self.phase, 'effective_failures': [],
                            'before_present': before_present, 'after_present': after_present,
                            'marker_written_wall_ns': current_marker['written_wall_ns'],
                            'before_wall_ns': before_time, 'after_wall_ns': after_time})
        return self.phase

    def complete(self, *, native_exit: int | None, output_sha_valid: bool,
                 token_absent: bool, descendants_absent: bool) -> str:
        if self.marker is None or self.phase not in ('FINALIZING', 'COMPLETE'):
            self._fail('completion_marker_missing')
        if native_exit is None:
            raise ValueError('native_exit_unknown')
        if native_exit != 0:
            self._fail('native_exit_nonzero')
        finalizing = [event for event in self.events if event['phase'] == 'FINALIZING']
        if (len(finalizing) < 2 or finalizing[-1]['before_present']
                or finalizing[-1]['after_present']):
            self._fail('post_exit_terminal_bracket_missing')
        if not output_sha_valid or not token_absent or not descendants_absent:
            self._fail('final_SHA_or_cleanup_invalid')
        self.phase = 'COMPLETE'
        self.events.append({'phase': 'COMPLETE', 'effective_failures': [],
                            'before_present': False, 'after_present': False})
        return self.phase
