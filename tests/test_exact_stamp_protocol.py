"""Regression for the 1.65/1.70 s TF delivery race."""
from pathlib import Path
import pytest

from visualnav_transformer.eat_nav.sim_clock_protocol import STEP_NS
from visualnav_transformer.eat_nav.exact_stamp_protocol import ExactClockProtocol


def test_only_exact_tf_can_emit_once_and_reset_clears_latch():
    from visualnav_transformer.eat_nav.exact_stamp_protocol import ExactStampLatch
    latch = ExactStampLatch()
    assert not latch.may_emit(1_700_000_000, 1_650_000_000)
    assert latch.may_emit(1_700_000_000, 1_700_000_000)
    latch.emitted(1_700_000_000)
    assert not latch.may_emit(1_700_000_000, 1_700_000_000)
    assert latch.may_emit(1_750_000_000, 1_750_000_000)
    latch.emitted(1_750_000_000)
    latch.reset()
    assert latch.may_emit(1_700_000_000, 1_700_000_000)


def test_bridge_rejects_prepath_stale_future_duplicate_without_relabel():
    p = ExactClockProtocol()
    p.advance(1_700_000_000, reset=True)
    p.set_tf(1_700_000_000)
    assert not p.accept(0, 1_700_000_000, (.3, 0., 0.))
    assert p.reject_reason(0, 1_700_000_000) == 'pre_path'
    p.set_path(1_700_000_000)
    assert p.reject_reason(0, 1_650_000_000) == 'stale'
    assert not p.accept(0, 1_650_000_000, (.3, 0., 0.))
    assert p.reject_reason(0, 1_750_000_000) == 'future'
    assert not p.accept(0, 1_750_000_000, (.3, 0., 0.))
    assert p.accept(0, 1_700_000_000, (.3, 0., 0.))
    assert p.reject_reason(0, 1_700_000_000) == 'duplicate'
    assert not p.accept(0, 1_700_000_000, (.3, 0., 0.))
    assert p.commands[0] == [(.3, 0., 0.)]
    p.advance(1_700_000_000 + STEP_NS)
    assert not p.accept(0, 1_700_000_000, (.3, 0., 0.))
    assert not p.ready(1)
    p.set_tf(1_750_000_000)
    assert p.accept(0, 1_750_000_000, (.2, 0., 0.))
    assert p.commands[0] == [(.2, 0., 0.)]


def test_generated_follower_uses_exact_nonblocking_tf_and_stamp_latch():
    from scripts.eat_nav_exact_stamp_build import render_exact
    cpp_path = Path('/tmp/eatnav-candidate-coverage/tf-race-diagnostic/diagnostic-follower-source-r1/src/simple_trajectory_follower.cpp')
    hpp_path = Path('/tmp/eatnav-candidate-coverage/tf-race-diagnostic/diagnostic-follower-source-r1/include/simple_trajectory_follower/simple_trajectory_follower.hpp')
    if not cpp_path.exists() or not hpp_path.exists():
        pytest.skip('external diagnostic follower source is not installed')
    source, header = cpp_path.read_text(), hpp_path.read_text()
    cpp, hpp = render_exact(source, header)
    assert 'tf2::TimePointZero' not in cpp
    assert 'canTransform(target_frame, robot_frame_, request_time, tf2::durationFromSec(0.0))' in cpp
    assert 'diag_tf_stamp_ns_ != request_ns' in cpp
    assert 'last_emitted_stamp_ns_' in hpp
    assert 'last_emitted_stamp_ns_ == logical_stamp_ns' in cpp
    assert 'last_emitted_stamp_ns_ = stamp_ns' in cpp
