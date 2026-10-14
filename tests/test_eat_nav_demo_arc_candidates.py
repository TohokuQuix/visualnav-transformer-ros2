import numpy as np

from visualnav_transformer.eat_nav import demo_arc_candidates as demo_arcs


def test_body_arc_bank_fans_out_from_robot_front_without_goal_rotation():
    pose = np.asarray([2., 3., np.pi / 2])
    body, mapped, info = demo_arcs.candidates_from_body_arcs(pose)

    assert tuple(body) == ('direct', 'left_30', 'right_30', 'left_60',
                           'right_60', 'left_90', 'right_90')
    assert all(path.shape == (8, 2) for path in body.values())
    assert np.allclose(body['direct'][-1], [.75, 0.])
    assert np.allclose(mapped['direct'][-1], [2., 3.75])
    assert np.allclose(body['left_90'][:, 1], -body['right_90'][:, 1])
    assert np.allclose(body['left_90'][:, 0], body['right_90'][:, 0])
    assert info['maximum_turn_deg'] == 90

    other_body, _, _ = demo_arcs.candidates_from_body_arcs(pose)
    assert all(np.array_equal(body[role], other_body[role]) for role in body)


def test_demo_arc_roles_match_audited_stage_one_angles():
    from visualnav_transformer.eat_nav.arc_bank import STAGE1_ANGLES, audit_arc

    assert {0, 30, -30, 60, -60, 90, -90} == set(STAGE1_ANGLES)
    assert set(demo_arcs.DEMO_ARC_ANGLES.values()) == set(STAGE1_ANGLES)
    body, _, _ = demo_arcs.candidates_from_body_arcs([0., 0., 0.])
    assert all(audit_arc(body[role], angle)['geometry_valid']
               for role, angle in demo_arcs.DEMO_ARC_ANGLES.items())


def test_all_seven_arcs_are_accepted_by_demo_follower_core(tmp_path):
    from visualnav_transformer.eat_nav.p1_follower_core import PureFollower, compile_core

    body, _, _ = demo_arcs.candidates_from_body_arcs([0., 0., 0.])
    library = compile_core(tmp_path)
    for path in body.values():
        follower = PureFollower(path.tolist(), library=library)
        try:
            command, diagnostic = follower.step_diagnostic(0., 0., 0., 50_000_000)
            assert np.isfinite(command).all()
            assert np.isfinite(diagnostic['goal_distance'])
        finally:
            follower.close()
