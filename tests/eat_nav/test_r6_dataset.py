import numpy as np
import pytest
from eat_nav_r6_dataset import validate_group_roles, validate_input_times


def test_seed_group_cannot_leak_between_roles():
    with pytest.raises(ValueError):
        validate_group_roles([dict(field='boxes',seed=1,role='train'),dict(field='boxes',seed=1,role='validation')])
    validate_group_roles([dict(field='boxes',seed=1,role='train'),dict(field='boxes',seed=1,role='train')])


def test_future_dem_timestamp_rejected():
    with pytest.raises(ValueError):validate_input_times(dict(dem_stamps_ns=[1,2,3,5]),4)
    assert validate_input_times(dict(dem_stamps_ns=[1,2,3,4]),4)=='checked'
    assert validate_input_times({},4)=='unavailable'


def test_pack_group_only_labels_actual_candidate():
    from eat_nav_r6_dataset import pack_group
    initial=dict(dem=np.zeros((4,2,20,20)),actor_state=np.zeros(22),body_state=np.zeros(9))
    packed,contacts=pack_group(initial,[dict(slot=3,path=np.zeros((8,2)),direction=1,rows=[])])
    assert packed['candidate_mask'].sum()==1
    assert packed['candidate_mask'][3]
    assert not packed['event_mask'].any() and not packed['trajectory_mask'].any()
    assert contacts['3']['available'] is False
