"""Fail-closed, durable protocol checks for long-horizon diagnostic rollouts."""

from __future__ import annotations

import json
import os
from pathlib import Path


STEP_NS = 50_000_000
EVENTS = ('clock', 'tf', 'command', 'bridge_acceptance', 'physics')


def validate_step_order(row: dict) -> None:
    if row.get('events') != list(EVENTS):
        raise ValueError('clock_TF_command_acceptance_physics_order')
    stamp = row.get('stamp_ns')
    if not isinstance(stamp, int) or any(row.get(key) != stamp for key in
        ('clock_ns', 'tf_ns', 'command_ns', 'bridge_accept_ns', 'physics_input_ns')):
        raise ValueError('logical_stamp_mismatch')
    accepted = row.get('command_accepted_steady_ns')
    physics = row.get('physics_begin_steady_ns')
    if not isinstance(accepted, int) or not isinstance(physics, int) or accepted > physics:
        raise ValueError('command_after_physics_or_missing_steady_time')


class SameEpisodeGate:
    def __init__(self, steps: int, *, initial_episode_length: int, episode_cap_steps: int):
        if steps not in (600, 640) or initial_episode_length < 0:
            raise ValueError('diagnostic_step_contract')
        if initial_episode_length + steps >= episode_cap_steps:
            raise ValueError('insufficient_episode_timeout_margin')
        self.steps = steps
        self.initial = initial_episode_length
        self.next_step = 0
        self.last_stamp = None

    def accept(self, row: dict) -> None:
        if self.next_step >= self.steps:
            raise ValueError('diagnostic_step_limit_exceeded')
        validate_step_order(row)
        step = row['step']
        if step != self.next_step or (self.last_stamp is not None and
                                      row['stamp_ns'] != self.last_stamp + STEP_NS):
            raise ValueError('noncontiguous_control_step')
        if row.get('terminated') or row.get('truncated') or row.get('reset_called'):
            raise ValueError('episode_termination_or_reset')
        if (row.get('episode_length_before') != self.initial + step or
                row.get('episode_length_after') != self.initial + step + 1):
            raise ValueError('episode_length_discontinuity')
        self.next_step += 1
        self.last_stamp = row['stamp_ns']

    @property
    def complete(self) -> bool:
        return self.next_step == self.steps


class DurableTrace:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('x', encoding='utf-8')
        return self

    def append(self, row: dict) -> None:
        if self.stream is None:
            raise RuntimeError('trace_not_open')
        self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def __exit__(self, *_):
        if self.stream is not None:
            self.stream.close()
