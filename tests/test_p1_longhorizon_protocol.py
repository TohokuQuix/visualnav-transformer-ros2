import json

import pytest

from visualnav_transformer.eat_nav.p1_longhorizon_protocol import (
    DurableTrace, SameEpisodeGate, validate_step_order,
)


def event(step, *, bad=None):
    stamp = 1_000_000_000 + step * 50_000_000
    row = {
        'step': step, 'stamp_ns': stamp, 'clock_ns': stamp, 'tf_ns': stamp,
        'command_ns': stamp, 'bridge_accept_ns': stamp, 'physics_input_ns': stamp,
        'events': ['clock', 'tf', 'command', 'bridge_acceptance', 'physics'],
        'command_accepted_steady_ns': step * 100 + 10,
        'physics_begin_steady_ns': step * 100 + 20,
        'episode_length_before': 20 + step,
        'episode_length_after': 21 + step,
        'terminated': False, 'truncated': False, 'reset_called': False,
    }
    if bad:
        row.update(bad)
    return row


def test_400_boundary_and_600_contiguous_steps(tmp_path, monkeypatch):
    import os
    sync_calls = []
    real_fsync = os.fsync
    def counting_fsync(fd):
        sync_calls.append(fd)
        real_fsync(fd)
    monkeypatch.setattr(os, 'fsync', counting_fsync)
    gate = SameEpisodeGate(600, initial_episode_length=20, episode_cap_steps=800)
    with DurableTrace(tmp_path / 'trace.jsonl') as trace:
        for step in range(600):
            row = event(step)
            gate.accept(row)
            trace.append(row)
    assert gate.complete
    assert len(sync_calls) == 600
    assert len((tmp_path / 'trace.jsonl').read_text().splitlines()) == 600
    assert json.loads((tmp_path / 'trace.jsonl').read_text().splitlines()[399])['step'] == 399


@pytest.mark.parametrize('bad', [
    {'tf_ns': 0}, {'command_ns': 0}, {'events': ['tf','clock','command','bridge_acceptance','physics']},
    {'events': ['clock','tf','physics','command','bridge_acceptance']},
    {'bridge_accept_ns': 0},
    {'physics_begin_steady_ns': 0},
])
def test_order_and_stamp_fail_closed(bad):
    with pytest.raises(ValueError):
        validate_step_order(event(0, bad=bad))


@pytest.mark.parametrize('bad', [
    {'truncated': True}, {'terminated': True}, {'reset_called': True},
    {'episode_length_after': 0},
])
def test_no_reset_even_at_step_400(bad):
    gate = SameEpisodeGate(600, initial_episode_length=20, episode_cap_steps=800)
    for step in range(400):
        gate.accept(event(step))
    with pytest.raises(ValueError):
        gate.accept(event(400, bad=bad))


def test_g3_640_steps_require_same_episode():
    gate = SameEpisodeGate(640, initial_episode_length=20, episode_cap_steps=800)
    for step in range(640):
        gate.accept(event(step))
    assert gate.complete
    with pytest.raises(ValueError):
        gate.accept(event(640))
