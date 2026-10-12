"""Offline three-period analysis keeps command geometry separate from motion."""

import pytest

from eat_nav_p1_g3_rosfree_crossover_complete_analyze import motion_series


def test_motion_series_reports_heading_cap_and_actual_forward_separately():
    row={'control_step':0,'logical_sim_time_ns':2_000_000_000,
         'follower_diagnostic':{'heading_error':0.2,'goal_reached':False,
                                'linear_recovery':False,'angular_recovery':False},
         'follower_command':[0.28,0.,0.1],
         'applied_command':[0.28,0.,0.1],
         'core_evidence':{'pose':[0.,0.,0.]},
         'root_state':[0.,0.,0.,1.,0.,0.,0.,0.15,0.02,0.,0.,0.,0.04]}
    result=motion_series([row],arrival_step=None)
    assert result[0]['heading_error_rad']==pytest.approx(0.2)
    assert result[0]['nominal_heading_forward_cap_mps']==pytest.approx(0.3*(1-0.2/3.141592653589793))
    assert result[0]['follower_forward_mps']==pytest.approx(0.28)
    assert result[0]['actual_forward_mps']==pytest.approx(0.15)
    assert result[0]['follower_yaw_rate_radps']==pytest.approx(0.1)


def test_turn_in_place_has_zero_nominal_forward_cap():
    row={'control_step':0,'logical_sim_time_ns':2_000_000_000,
         'follower_diagnostic':{'heading_error':0.8,'goal_reached':False,
                                'linear_recovery':False,'angular_recovery':False},
         'follower_command':[0.,0.,0.4],
         'applied_command':[0.,0.,0.4],
         'core_evidence':{'pose':[0.,0.,0.]},
         'root_state':[0.,0.,0.,1.,0.,0.,0.,0.01,0.,0.,0.,0.,0.05]}
    assert motion_series([row],arrival_step=None)[0]['nominal_heading_forward_cap_mps']==0.
