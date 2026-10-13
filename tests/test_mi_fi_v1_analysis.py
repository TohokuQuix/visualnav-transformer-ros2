import pytest


def test_interaction_is_difference_of_arc_minus_direct_between_conditions():
    from visualnav_transformer.eat_nav.mi_fi_v1_analysis import summarize_interactions
    def label(x, y, yaw):
        return {'available': True, 'delta_body_xy_m': [x, y], 'delta_body_yaw_rad': yaw}
    base = {'forward_direct': [label(.5, 0, 0)] * 4,
            'forward_+30': [label(.4, .1, .2)] * 4,
            'forward_-30': [label(.4, -.1, -.2)] * 4,
            'reverse_direct': [label(-.3, 0, 0)] * 4,
            'reverse_+30': [label(-.3, .01, .01)] * 4,
            'reverse_-30': [label(-.3, -.01, -.01)] * 4}
    rear = {key: [dict(value) for value in rows] for key, rows in base.items()}
    rear['reverse_+30'] = [label(-.4, .03, .11)] * 4
    summary = summarize_interactions({'front_folded': base, 'rear_folded': rear})
    effect = summary['interactions']['reverse_+30']['mean_delta_yaw_rad']
    assert effect == pytest.approx(.1)
    assert summary['interactions']['reverse_+30']['mean_delta_xy_m'] == pytest.approx([-.1, .02])
