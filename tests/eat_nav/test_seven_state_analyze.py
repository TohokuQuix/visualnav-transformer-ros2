from scripts.eat_nav_seven_state_analyze import classify_case, classify_state, command_cluster_count

LIMITS={'tilt_max_deg':85.,'progress_min_m':.02,'stack_displacement_min_m':.02,
        'progress_vs_nominal_min_fraction':.5,'goal_distance_allowance_m':.15,
        'path_tracking_error_max_m':.25,'drawdown_min':-.1,
        'max_ood_fraction_for_classification':.5}


def metric(**override):
    out={'body_tilt_max_rad':0.,'terminal':None,'progress_m':.2,'xy_displacement_m':.2,
         'end_goal_distance_m':.3,'path_tracking_error_m':.1,'drawdown':0.,
         'proposal_ood':{'ood_over_train_p95':False,'in_dem_footprint':True,
                         'self_intersects':False,'speed_over_frozen_max':False}}
    out.update(override);return out


def test_level_criteria_are_nested_and_ood_is_na():
    assert classify_case(metric(),metric(),LIMITS)['level3']
    assert not classify_case(metric(progress_m=.005),metric(),LIMITS)['level2']
    assert not classify_case(metric(body_tilt_max_rad=2.),metric(),LIMITS)['level1']
    assert classify_case(metric(proposal_ood={'ood_over_train_p95':True}),metric(),LIMITS)['classification']=='proposal_ood'


def test_state_classes_do_not_claim_multi_feasible_from_one_path():
    cases={'nominal':{'level1':True,'level2':True,'level3':True,'classification':'supported'},
           'other':{'level1':False,'level2':False,'level3':False,'classification':'supported'}}
    assert classify_state(cases,LIMITS)=='single_solution'
    cases['other']={'level1':True,'level2':True,'level3':True,'classification':'supported'}
    assert classify_state(cases,LIMITS)=='multi_feasible'
    cases['other']['classification']='proposal_ood'
    assert classify_state(cases,LIMITS)=='single_solution'


def test_full_series_cluster_count_uses_frozen_command_thresholds():
    commands={'nominal':[[.2,0,0]]*40,'left':[[.2,0,.1]]*40,'same':[[.2,0,0]]*40}
    assert command_cluster_count(commands,.03,.05)==2
