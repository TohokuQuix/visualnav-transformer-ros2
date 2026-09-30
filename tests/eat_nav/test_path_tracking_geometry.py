import numpy as np
import pytest
from visualnav_transformer.eat_nav.path_tracking_geometry import (
    map_from_base, base_from_map, to_waypoints, to_increments,
    project_polyline, tracking_metrics,
)


def test_local_map_and_increment_round_trips_catch_rotation_or_first_point_loss():
    base=np.array([[0.,0.],[1.,0.],[1.,1.]])
    mapped=map_from_base(base,[4.,-20.],np.pi/2)
    np.testing.assert_allclose(mapped,[[4.,-20.],[4.,-19.],[3.,-19.]],atol=1e-12)
    np.testing.assert_allclose(base_from_map(mapped,[4.,-20.],np.pi/2),base,atol=1e-12)
    increments=to_increments(base[1:]);np.testing.assert_allclose(increments,[[1.,0.],[0.,1.]])
    np.testing.assert_allclose(to_waypoints(increments),base[1:])


def test_segment_projection_measures_cross_track_and_along_track():
    p=project_polyline([[0.,0.],[1.,0.],[1.,1.]],[[.5,.2],[1.2,.5]])
    np.testing.assert_allclose(p['cross_track_m'],[.2,.2])
    np.testing.assert_allclose(p['arc_m'],[.5,1.5])
    assert p['segment_index'].tolist()==[0,1]


def test_self_intersection_tie_uses_earliest_segment_and_counts_ambiguity():
    p=project_polyline([[-1.,-1.],[1.,1.],[-1.,1.],[1.,-1.]],[[0.,0.]])
    assert p['segment_index'].tolist()==[0]
    assert p['ambiguous_count']==1


def test_zero_length_segment_is_skipped_and_overshoot_is_retained():
    p=project_polyline([[0.,0.],[0.,0.],[1.,0.]],[[2.,0.]])
    assert p['zero_length_segments']==1
    assert p['segment_index'].tolist()==[1]
    assert p['overshoot_m'][0]==pytest.approx(1.)


def test_endpoint_first_arrival_and_escape_use_post_step_time():
    path=[[0.,0.],[1.,0.]]
    xy=np.zeros((121,2));xy[0]=[0.,0.];xy[1:]=[.5,0.];xy[20:]=[.85,0.];xy[60:]=[1.4,0.]
    result=tracking_metrics(path,xy,np.zeros(121),np.tile([.2,0.,0.],(120,1)),dt=.05,tolerance=.2)
    assert result['endpoint']['first_arrival_sec']==pytest.approx(1.)
    assert result['endpoint']['reached'] is True
    assert result['endpoint']['escaped_after_arrival'] is True
    assert result['endpoint']['final_distance_m']==pytest.approx(.4)
    assert result['two_second']['final_progress_m']==pytest.approx(.85)


def test_tracking_rejects_misaligned_steps():
    with pytest.raises(ValueError,match='positions must have one more sample'):
        tracking_metrics([[0,0],[1,0]],np.zeros((120,2)),np.zeros(120),np.zeros((120,3)),dt=.05,tolerance=.2)


def test_waypoint_pass_rate_counts_all_future_waypoints():
    xy=np.zeros((121,2));xy[1:]=[1.,0.]
    result=tracking_metrics([[0.,0.],[1.,0.],[2.,0.]],xy,np.zeros(121),
                            np.tile([.2,0.,0.],(120,1)),dt=.05,tolerance=.2)
    assert result['two_second']['waypoint_pass_rate']==pytest.approx(.5)


def test_descriptive_category_keeps_reached_and_escape_distinct():
    from visualnav_transformer.eat_nav.path_tracking_geometry import classify_run
    row={'endpoint':{'reached':True,'escaped_after_arrival':True},
         'two_second':{'final_progress_m':.8,'cross_track_p95_m':.04},
         'six_second':{'reverse_fraction':0.,'forward_fraction':1.}}
    assert classify_run(row,constraint_valid=True,tracking_failure_threshold=.1)=='overshoot_or_escape'
    row['endpoint']['escaped_after_arrival']=False
    assert classify_run(row,constraint_valid=True,tracking_failure_threshold=.1)=='reached'
    row['endpoint']['reached']=False
    row['two_second']['cross_track_p95_m']=.2
    assert classify_run(row,constraint_valid=True,tracking_failure_threshold=.1)=='tracking_deviation'
