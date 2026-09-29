from scripts.eat_nav_phase_c_readiness import command_delivery, prepath_ready, clock_progressed


def test_zero_command_is_fresh_when_sequence_advances():
    assert command_delivery([0, 0], [1, 1], [[0, 0, 0], [0, 0, 0]]) == 'fresh'
    assert command_delivery([1, 1], [1, 1], [[0, 0, 0], [0, 0, 0]]) == 'stale'
    assert command_delivery([0, 0], [1, 0], [[0, 0, 0], None]) == 'partial'


def test_prepath_requires_all_processes_subscribers_status_and_tf():
    sample=dict(alive=[True]*8,subscribers=[1]*8,status_counts=[2]*8,tf_ready=[True]*8,
                clock_samples_ns=[100, 200, 300])
    assert prepath_ready(**sample)
    for key,value in [('alive',False),('subscribers',0),('status_counts',0),('tf_ready',False)]:
        altered={k:list(v) for k,v in sample.items()}
        altered[key][3]=value
        assert not prepath_ready(**altered)
    altered={k:list(v) for k,v in sample.items()};altered['clock_samples_ns']=[100,100]
    assert not prepath_ready(**altered)


def test_clock_must_advance_twice():
    assert clock_progressed([1, 2, 3])
    assert not clock_progressed([1, 2, 2])
    assert not clock_progressed([1, 2])


def test_command_stamp_must_follow_new_path_publish():
    from scripts.eat_nav_phase_c_readiness import current_command_stamp
    assert not current_command_stamp(900, None)
    assert not current_command_stamp(900, 1000)
    assert current_command_stamp(1000, 1000)
    assert current_command_stamp(1001, 1000)
