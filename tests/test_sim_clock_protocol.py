import pytest
from visualnav_transformer.eat_nav.sim_clock_protocol import ClockProtocol, STEP_NS


def test_monotonic_120_steps_and_reset():
    p=ClockProtocol()
    p.advance(1_000_000_000, reset=True)
    for k in range(1,121):p.advance(1_000_000_000+k*STEP_NS)
    assert p.stamp_ns==7_000_000_000
    assert p.motion_elapsed_seconds(0,120)==6.0
    assert p.motion_elapsed_seconds(0,120)<20.0
    with pytest.raises(ValueError,match='did not increase'):p.advance(1_000_000_000,reset=True)
    p.advance(11_000_000_000,reset=True)
    with pytest.raises(ValueError,match='jump'):p.advance(11_100_000_000)


def test_stamp_and_same_stamp_command_guards():
    p=ClockProtocol();p.advance(1_000_000_000,reset=True)
    assert not p.accept(0,p.stamp_ns,(.3,0,0))
    p.set_tf(p.stamp_ns);p.set_path(p.stamp_ns)
    assert not p.accept(0,p.stamp_ns-STEP_NS,(.3,0,0))
    assert not p.accept(0,p.stamp_ns+STEP_NS,(.3,0,0))
    assert p.accept(0,p.stamp_ns,(.3,0,0))
    assert p.accept(0,p.stamp_ns,(.3,0,0))
    with pytest.raises(ValueError,match='different commands'):p.accept(0,p.stamp_ns,(.4,0,0))
    p.advance(p.stamp_ns+STEP_NS)
    assert not p.accept(0,p.stamp_ns,(.3,0,0))
    with pytest.raises(ValueError,match='TF stamp'):p.set_tf(p.stamp_ns-STEP_NS)
    p.set_tf(p.stamp_ns)
    assert p.accept(0,p.stamp_ns,(.3,0,0))
