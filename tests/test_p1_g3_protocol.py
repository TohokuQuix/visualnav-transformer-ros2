from visualnav_transformer.eat_nav.p1_longhorizon_protocol import G3Exposure


def test_g3_arrival_then_forty_full_hold_steps():
    trial = G3Exposure()
    assert all(trial.observe(step, step >= 48) for step in range(89))
    assert trial.arrival_step == 48
    assert trial.stop_exclusive == 89
    assert trial.finished_after(88)
    assert not trial.observe(89, True)


def test_g3_arrival_at_cap_permits_640_rows():
    trial = G3Exposure()
    for step in range(640):
        assert trial.observe(step, step >= 599)
    assert trial.arrival_step == 599
    assert trial.stop_exclusive == 640
    assert not trial.censored


def test_g3_unreached_is_right_censored_at_600():
    trial = G3Exposure()
    for step in range(600):
        assert trial.observe(step, False)
    assert trial.stop_exclusive == 600
    assert trial.censored
    assert trial.finished_after(599)
    assert not trial.observe(600, True)
