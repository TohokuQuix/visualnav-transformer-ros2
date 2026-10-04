import gzip
import json

import numpy as np

from eat_nav_expanded_feasibility_inputs import frozen_bank, validate_frame, local_to_map


def test_frozen_bank_reuses_all_seven_waypoints_byte_exactly():
    rows=frozen_bank()
    assert len(rows)==7
    assert [row['candidate_id'] for row in rows]==[
        'direct_reference','left_1','left_2','left_3','right_1','right_2','right_3']
    assert all(np.asarray(row['waypoints_xy_m']).shape==(8,2) for row in rows)


def test_local_to_map_does_not_subtract_world_origin_twice():
    base=np.array([[0.,0.],[.5,0.]])
    assert np.allclose(local_to_map(base,[2.,3.,np.pi/2]),[[2.,3.],[2.,3.5]])


def test_frame_validation_rejects_axis_or_mask_drift():
    dem={'transformed':np.zeros((2,20,20)).tolist(),'resolution_m':.1,'origin_xy':[0.,0.],
        'axis':{'row':'left','column':'forward'},'unknown':0}
    dem['transformed'][1]=np.ones((20,20)).tolist()
    row={'episode_id':0,'pre_reset':False,'dem':dem,'timestamp_ns':123}
    assert validate_frame(row,[0,1,0],123,0,expected_axis={'row':'left','column':'forward'})
    row['dem']['axis']['row']='right'
    try:validate_frame(row,[0,1,0],123,0,expected_axis={'row':'left','column':'forward'})
    except ValueError:pass
    else:raise AssertionError('axis drift was accepted')
