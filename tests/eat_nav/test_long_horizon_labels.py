import math

from visualnav_transformer.eat_nav.long_horizon_labels import label_trace


def make_row(step, *, tilt_deg=0.0, terminal=None, remaining=0.75,
             commanded=0.0, actual=0.0, commanded_angular=0.0, actual_angular=0.0):
    half_angle = math.radians(tilt_deg) / 2
    return {
        'control_step': step,
        'logical_sim_time_ns': step * 50_000_000,
        'root_state': [0.0, 0.0, 0.0, math.cos(half_angle), math.sin(half_angle),
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'post_root_state': [0.0, 0.0, 0.0, math.cos(half_angle), math.sin(half_angle),
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'candidate_cycle': 0,
        'follower_command': [0.0, 0.0, 0.02 if commanded_angular else 0.0],
        'body_angular_velocity_radps': [0.0, 0.0,
                        0.005 if actual_angular else 0.0],
        'follower_diagnostic': {
            'remaining_path_distance': remaining,
            'commanded_distance': commanded,
            'actual_distance': actual,
            'commanded_angular_distance': commanded_angular,
            'actual_angular_distance': actual_angular,
        },
        'terminal': terminal,
    }


def test_passive_labels_detect_tilt_and_stall_without_marking_safe_horizon():
    rows = [make_row(step, tilt_deg=60.0 if 3 <= step <= 8 else 0.0,
                     remaining=0.75, commanded=0.30, actual=0.03,
                     commanded_angular=0.03, actual_angular=0.01)
            for step in range(40)]
    rows[30] = make_row(30, tilt_deg=85.0, terminal={
        'termination_reason': 'tilt_over',
        'termination_reasons': ['tilt_over'],
        'pre_reset_state': {'root_state': make_row(30, tilt_deg=85.0)['root_state']},
    }, remaining=0.75, commanded=0.30, actual=0.03,
       commanded_angular=0.03, actual_angular=0.01)

    result = label_trace(rows)

    assert result['events']['near_fall'] == 1
    assert result['events']['tip_over'] == 1
    assert result['events']['linear_stall'] == 1
    assert result['events']['turn_stall'] == 1
    assert result['labels'][0]['horizons']['2s']['events']['near_fall'] is True
    assert result['labels'][0]['horizons']['2s']['events']['tip_over'] is True


def test_turn_stall_requires_continuous_command_and_low_body_yaw_response():
    rows = [make_row(step, commanded_angular=0.03, actual_angular=0.01)
        for step in range(40)]

    result = label_trace(rows)

    assert result['events']['turn_stall'] == 1
    rows[10]['follower_command'][2] = 0.0
    assert label_trace(rows)['events']['turn_stall'] == 0
    rows[10]['follower_command'][2] = 0.02
    rows[10]['body_angular_velocity_radps'][2] = 0.011
    assert label_trace(rows)['events']['turn_stall'] == 0


def test_unobserved_future_is_censored_not_safe():
    rows = [make_row(step) for step in range(10)]

    result = label_trace(rows)

    assert result['labels'][0]['horizons']['1s']['observed'] is False
    assert result['labels'][0]['horizons']['1s']['censored'] is True
    assert result['labels'][0]['horizons']['1s']['events']['event_free'] is None
