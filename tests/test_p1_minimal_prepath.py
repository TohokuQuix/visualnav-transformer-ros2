from visualnav_transformer.eat_nav.p1_minimal_prepath import first_divergence, resource_violation


def test_first_divergence_separates_nonzero_and_frozen_threshold():
    a = [
        {'physics_step': 0, 'root_state_w': [0, 0, .5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {'physics_step': 1, 'root_state_w': [0, 0, .5, 1, 0, 0, 0, .00005, 0, 0, 0, 0, 0]},
        {'physics_step': 2, 'root_state_w': [0, 0, .5, 1, 0, 0, 0, .0002, 0, 0, 0, 0, 0]},
    ]
    b = [
        {'physics_step': 0, 'root_state_w': [0, 0, .5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {'physics_step': 1, 'root_state_w': [0, 0, .5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {'physics_step': 2, 'root_state_w': [0, 0, .5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    ]
    got = first_divergence(a, b, 'root_velocity', .0001)
    assert got['first_nonzero_step'] == 1
    assert got['first_threshold_step'] == 2
    assert got['first_threshold_max_abs'] == .0002


def test_first_divergence_reports_unobserved_signal():
    got = first_divergence([{'physics_step': 0}], [{'physics_step': 0}], 'joint_target', .001)
    assert got['status'] == 'unobserved'


def test_resource_violation_uses_frozen_caps():
    cap = {'gpu_device_total_mib': 7680, 'host_available_min_gib': 10,
           'active_storage_gib': 1, 'heartbeat_stale_s': 90, 'process_timeout_s': 600}
    good = {'gpu_mib': 7680, 'host_available_gib': 10, 'active_storage_gib': 1,
            'heartbeat_age_s': 90, 'elapsed_s': 600}
    assert resource_violation(good, cap) is None
    assert resource_violation({**good, 'gpu_mib': 7681}, cap) == 'gpu_device_total'
    assert resource_violation({**good, 'heartbeat_age_s': 91}, cap) == 'heartbeat_stale'
