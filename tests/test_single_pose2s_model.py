import numpy as np
import pytest
from visualnav_transformer.eat_nav.single_pose2s_model import SinglePose2s,fit_normalization,prepare,predict
from visualnav_transformer.eat_nav.single_pose2s_contract import INPUT_SCHEMA,MODEL_SCHEMA,PROFILE_SHA
from visualnav_transformer.eat_nav.bidirectional_bank import bank
import torch


def fixture(n=4):
    dem=np.zeros((n,3,40,40),np.float32);dem[:,1:]=1
    state=np.zeros((n,27),np.float32);state[:,2]=-1;state[:,25]=1
    return dict(input_schema=INPUT_SCHEMA,profile_sha256=PROFILE_SHA,dem=dem,state=state,state_mask=np.ones((n,27),np.float32),state_age=np.zeros((n,27),np.float32),dem_context=np.tile(np.array([0,1,0,0,0,1],np.float32),(n,1)),paths=np.tile(np.array([c.waypoints for c in bank()],np.float32),(n,1,1,1)),directions=np.tile(np.array([[c.direction] for c in bank()],np.float32),(n,1,1)),position=np.zeros((n,3),np.float32),pose_mask=np.ones(n,bool),input_valid=np.ones(n,bool))


def test_normalization_train_only_and_masks():
    a=fixture();a['position'][2:]=999;a['state'][2:]=999
    norm=fit_normalization(a,[0,1])
    np.testing.assert_allclose(norm['position_mean'],0)
    np.testing.assert_allclose(norm['state_mean'][:2],0)
    a['state'][:,20]=777;a['state_mask'][:,20]=0
    norm=fit_normalization(a,[0,1]);assert norm['state_mean'][20]==0


def test_k14_permutation_and_legacy_refusal():
    torch.manual_seed(1);a=fixture(1);norm=fit_normalization(a,[0]);m=SinglePose2s().eval()
    with torch.inference_mode():first=m(*prepare(a,norm))
    order=np.arange(14)[::-1].copy();a['paths']=a['paths'][:,order];a['directions']=a['directions'][:,order]
    with torch.inference_mode():second=m(*prepare(a,norm))
    np.testing.assert_allclose(first['position_z'].numpy()[:,order],second['position_z'].numpy(),atol=1e-6)
    with pytest.raises(ValueError):predict(dict(schema='legacy'),a)


def test_public_prediction_timestamps_and_current_pose_composition():
    a=fixture(1);a.update(t0_ns=np.array([0]),dem_stamp_ns=np.array([0]),dem_received_ns=np.array([0]),q0_wxyz=np.array([[1.,0,0,0]]))
    m=SinglePose2s();payload=dict(schema=MODEL_SCHEMA,input_schema=INPUT_SCHEMA,profile_sha256=PROFILE_SHA,normalization=fit_normalization(a,[0]),state_dict=m.state_dict())
    r=predict(payload,a)
    assert r['position_body_m'].shape==(1,14,3) and r['future_rpy_deg'].shape==(1,14,3)
    assert r['rotation_valid'].all()
    a['dem_received_ns'][0]=1
    with pytest.raises(ValueError,match='DEM'):predict(payload,a)


def test_inference_rejects_shape_compatible_wheel_odom_input():
    a=fixture(1);a.update(t0_ns=np.array([0]),dem_stamp_ns=np.array([0]),dem_received_ns=np.array([0]))
    m=SinglePose2s();payload=dict(schema=MODEL_SCHEMA,input_schema=INPUT_SCHEMA,profile_sha256=PROFILE_SHA,normalization=fit_normalization(a,[0]),state_dict=m.state_dict())
    a['input_schema']='eatnav_single_dem_pose2s_input_v1'
    with pytest.raises(ValueError,match='input provenance'):predict(payload,a)
