import numpy as np

from visualnav_transformer.eat_nav.radial_interaction import classify_dem


def dem():
    return np.zeros((20,20),dtype=np.float32),np.ones((20,20),dtype=np.float32)


def test_flat_control_has_no_interaction():
    h,m=dem();x=classify_dem(h,m,.1)
    assert x['class']=='flat-only' and x['interaction_start_m'] is None


def test_early_step_is_transition_within_proxy_horizon():
    h,m=dem();h[:,13:]=.08;x=classify_dem(h,m,.1)
    assert x['class']=='transition' and x['interaction_within_two_second_proxy']
    assert 0 <= x['interaction_start_m'] <= .6


def test_forward_ramp_is_graded():
    h,m=dem();h[:,10:]=np.arange(10,dtype=np.float32)[None,:]*.015
    x=classify_dem(h,m,.1)
    assert x['class']=='graded' and x['max_forward_slope'] >= .1


def test_left_right_height_difference_is_differential():
    h,m=dem();h[11:,:]=.07
    x=classify_dem(h,m,.1)
    assert x['class']=='differential' and x['left_center_right_difference_m'] >= .04


def test_unknown_only_difference_is_excluded_from_shortlist():
    h,m=dem();h[:,13:]=.08;m[:,13:]=0
    x=classify_dem(h,m,.1)
    assert x['class']=='unknown' and not x['supply_candidate']
