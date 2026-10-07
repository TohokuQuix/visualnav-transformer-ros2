"""Regression tests for preserving PhysX raw contact indices before validation."""
import hashlib
import numpy as np


def test_out_of_bounds_pair_is_durably_saved_before_rejection(tmp_path):
    from visualnav_transformer.eat_nav.contact_raw_capture import capture_contact_api
    arrays=(np.arange(4,dtype=np.float32).reshape(4,1),
            np.zeros((4,3),dtype=np.float32), np.zeros((4,3),dtype=np.float32),
            np.zeros((4,1),dtype=np.float32),
            np.array([[0],[5]],dtype=np.uint32), np.array([[0],[0]],dtype=np.uint32))
    result=capture_contact_api(tmp_path,29,'pose-a',arrays,
        ['/World/envs/env_0/onix_chassis/chassis_body','/World/envs/env_0/onix_chassis/chassis_track'],
        ['/World/ground/terrain/mesh'],4)
    assert result['first_invalid']['sensor_index']==1
    assert result['first_invalid']['sensor_path'].endswith('chassis_track')
    assert result['first_invalid']['count']==5
    assert result['first_invalid']['start']==0
    assert result['max_start_plus_count']==5
    assert result['sum_counts']==5
    assert result['array_shapes']==[[4,1],[4,3],[4,3],[4,1],[2,1],[2,1]]
    raw=tmp_path/'raw'/'step-0029'/'pose-a.npz'
    assert raw.exists()
    assert result['raw_sha256']==hashlib.sha256(raw.read_bytes()).hexdigest()
    with np.load(raw,allow_pickle=False) as saved:
        assert saved['counts'].tolist()==[[0],[5]]
        assert saved['starts'].tolist()==[[0],[0]]


def test_zero_count_start_past_buffer_is_not_sliced(tmp_path):
    from visualnav_transformer.eat_nav.contact_raw_capture import capture_contact_api
    arrays=(np.zeros((3,1),dtype=np.float32),np.zeros((3,3),dtype=np.float32),
            np.zeros((3,3),dtype=np.float32),np.zeros((3,1),dtype=np.float32),
            np.array([[0]],dtype=np.uint32),np.array([[4]],dtype=np.uint32))
    result=capture_contact_api(tmp_path,1,'pose-a',arrays,['/sensor'],['/filter'],3)
    assert result['first_invalid']['count']==0
    assert result['first_invalid']['start']==4
    assert result['raw_sha256']


def test_diagnostic_never_advances_past_60_physics_steps():
    from visualnav_transformer.eat_nav.contact_raw_capture import diagnostic_should_stop
    assert diagnostic_should_stop(59,invalid=False) is False
    assert diagnostic_should_stop(60,invalid=False) is True
    assert diagnostic_should_stop(29,invalid=True) is True


def test_capacity_is_smallest_power_of_two_covering_twice_observed_need():
    from visualnav_transformer.eat_nav.contact_raw_capture import choose_contact_capacity
    assert choose_contact_capacity(334)==1024
    assert choose_contact_capacity(128)==256


def test_full_capacity_must_equal_prospective_formula():
    from visualnav_transformer.eat_nav.contact_raw_capture import validate_view_capacity
    assert validate_view_capacity('diagnostic',256,None)==256
    assert validate_view_capacity('full',1024,334)==1024
    import pytest
    with pytest.raises(ValueError):
        validate_view_capacity('full',512,334)


def test_exact_60x1_runtime_shape_yields_one_prospective_capacity(tmp_path):
    from visualnav_transformer.eat_nav.contact_raw_capture import capture_contact_api, capacity_decision_from_raw
    counts=np.zeros((60,1),dtype=np.int32)
    starts=np.zeros((60,1),dtype=np.int32)
    counts[20,0]=75;counts[30,0]=229;counts[40,0]=30
    starts[20,0]=0;starts[30,0]=75;starts[40,0]=243
    arrays=(np.zeros((256,1),dtype=np.float32),np.zeros((256,3),dtype=np.float32),
            np.zeros((256,3),dtype=np.float32),np.zeros((256,1),dtype=np.float32),counts,starts)
    paths=[f'/World/envs/env_96/onix_chassis/body_{i}' for i in range(60)]
    raw=capture_contact_api(tmp_path,29,'rail',arrays,paths,['/World/ground/terrain/mesh'],256)
    decision=capacity_decision_from_raw(raw)
    assert decision['classification']=='B_contact_capacity_insufficient'
    assert decision['observed_required_entries']==334
    assert decision['prospective_capacity']==1024


def test_frozen_target_paths_and_shapes_reject_drift():
    from visualnav_transformer.eat_nav.contact_raw_capture import verify_frozen_view_contract
    paths=['/World/envs/env_96/onix_chassis/chassis_body']
    shapes=[[1024,1],[1024,3],[1024,3],[1024,1],[1,1],[1,1]]
    assert verify_frozen_view_contract(paths,paths,['/World/ground/terrain/mesh'],
                                      ['/World/ground/terrain/mesh'],shapes,shapes) is True
    import pytest
    with pytest.raises(ValueError):
        verify_frozen_view_contract(paths,['/World/envs/env_95/onix_chassis/chassis_body'],
                                    ['/World/ground/terrain/mesh'],['/World/ground/terrain/mesh'],shapes,shapes)
    with pytest.raises(ValueError):
        verify_frozen_view_contract(paths,paths,['/World/ground/terrain/mesh'],
                                    ['/World/ground/terrain/mesh'],[[512,1],*shapes[1:]],shapes)
