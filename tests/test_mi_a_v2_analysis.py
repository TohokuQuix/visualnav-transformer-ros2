import pytest


def test_candidate_differences_pair_same_direction_and_keep_reverse_x_sign():
    from visualnav_transformer.eat_nav.mi_a_v2_analysis import summarize_labels
    def labels(x, y=0., yaw=0.):
        return [{'available': True, 'delta_body_xy_m': [x, y],
                 'delta_body_yaw_rad': yaw} for _ in range(4)]
    entries = {'forward_direct': labels(.2), 'forward_+30': labels(.18, .04, .08),
               'forward_-30': labels(.18, -.04, -.08),
               'reverse_direct': labels(-.2), 'reverse_+30': labels(-.18, -.04, .09),
               'reverse_-30': labels(-.18, .04, -.09)}
    result = summarize_labels(entries, labels(.2), seed=7)
    assert result['candidate_differences']['reverse_+30']['per_env'][0]['delta_xy_m'] == pytest.approx([.02, -.04])
    assert result['candidate_differences']['reverse_+30']['per_env'][0]['signed_travel_lateral_m'] == pytest.approx(.04)
    assert result['candidate_differences']['forward_-30']['per_env'][0]['signed_yaw_rad'] == pytest.approx(.08)
    assert result['independent_terrain_seed_count'] == 1


def test_candidate_difference_rejects_incomplete_label():
    from visualnav_transformer.eat_nav.mi_a_v2_analysis import summarize_labels
    entries = {role: [{'available': True, 'delta_body_xy_m': [0., 0.],
                       'delta_body_yaw_rad': 0.} for _ in range(4)]
               for role in ('forward_direct', 'forward_+30', 'forward_-30',
                            'reverse_direct', 'reverse_+30', 'reverse_-30')}
    entries['reverse_+30'][0]['available'] = False
    with pytest.raises(ValueError, match='incomplete'):
        summarize_labels(entries, entries['forward_direct'], seed=7)
