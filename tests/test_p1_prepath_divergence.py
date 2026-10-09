import math

from visualnav_transformer.eat_nav.p1_prepath_divergence import pair_metrics, initial_state_contract


def test_pair_metrics_uses_tile_local_position_and_all_dem_frames():
    base = {'root_state_w': [10, 20, .4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'target_tile_origin_xy': [10, 20], 'joint_position': [0],
            'joint_velocity': [0], 'actor_input': [0, 1],
            'dem_frames': [{'transformed': [[[0, 0]], [[1, 1]]]}]}
    other = {**base, 'root_state_w': [30.01, 40, .4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             'target_tile_origin_xy': [30, 40], 'actor_input': [0, 2],
             'dem_frames': [{'transformed': [[[0, .02]], [[1, 0]]]}]}
    got = pair_metrics(base, other)
    assert math.isclose(got['root_local_max_abs_m'], .01, abs_tol=1e-9)
    assert got['actor_max_abs'] == 1
    assert got['dem_frames'][0]['mask_iou'] == .5
    assert math.isclose(got['dem_frames'][0]['height_rms_m'], math.sqrt(.0002))


def test_initial_contract_does_not_claim_unobserved_state_matches():
    fixture = {'root_state_w': [1, 2, .5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'joint_position': [0, 1], 'joint_velocity': [0, 0],
               'flipper_fold_mode_latch': 0}
    got = initial_state_contract(fixture)
    assert got['root_z']['status'] == 'retained_from_batch_reset_not_source_fixture'
    assert got['root_velocity']['status'] == 'retained_from_batch_reset_not_source_fixture'
    assert got['joint_position']['status'] == 'source_fixture_written_not_measured_at_step_0'
    assert got['action_manager']['status'] == 'not_restored_or_recorded'
