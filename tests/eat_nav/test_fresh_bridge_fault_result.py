from eat_nav_fresh_bridge_run import classify_fault_result


def test_failure_marker_takes_precedence_over_isaac_exit_zero():
    marker={'stage':'failed','step':1,'error':'RuntimeError(TimeoutError("fresh command timeout"))'}
    assert classify_fault_result(0,marker,complete_exists=False)['injected_timeout']


def test_complete_marker_invalidates_fault_run():
    marker={'stage':'failed','step':1,'error':'fresh command timeout'}
    assert not classify_fault_result(0,marker,complete_exists=True)['injected_timeout']
