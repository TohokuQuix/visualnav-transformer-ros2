"""Behavioral contract for the separate two-second endpoint model."""
import importlib
import numpy as np
import pytest
import torch


def module():
    return importlib.import_module('visualnav_transformer.eat_nav.moving_pose2s_model')


def arrays(n=4,k=1):
    a=dict(dem=np.zeros((n,4,2,20,20),np.float32),actor_state=np.zeros((n,22),np.float32),body_state=np.zeros((n,9),np.float32),paths=np.zeros((n,k,8,2),np.float32),directions=np.ones((n,k,1),np.float32),candidate_mask=np.ones((n,k),bool),position_targets=np.zeros((n,k,3),np.float32),rotation_targets=np.tile([1,0,0,0,1,0],(n,k,1)).astype(np.float32),pose_mask=np.ones((n,k),bool),event_targets=np.zeros((n,k,4),np.float32),event_mask=np.ones((n,k,4),bool),current_quat_wxyz=np.tile([1,0,0,0],(n,1)).astype(np.float32))
    a['dem'][:,:,1]=1;a['body_state'][:,2]=-1;a['actor_state'][:,5]=-1
    a['paths'][:,:,:,0]=np.linspace(0,.75,8)
    a['position_targets'][:,:,0]=np.arange(n)[:,None]*.1
    a['event_targets'][0,:,0]=1
    return a


def test_shapes_candidate_equivariance_and_identity_rotation():
    m=module();torch.manual_seed(1);a=arrays(k=14);s=m.fit_normalization(a,[0,1,2]);net=m.MovingPose2sEvaluator().eval()
    x=m.prepare_inputs(a,[3],s)
    with torch.no_grad():
        y=net(**x)
        one=net(**{key:(value[:,:1] if key in ('paths','directions','candidate_mask') else value) for key,value in x.items()})
    assert y['position_z'].shape==(1,14,3)
    assert y['rotation_6d'].shape==(1,14,6)
    assert y['event_logits'].shape==(1,14,4)
    torch.testing.assert_close(y['rotation_6d'],torch.tensor([1.,0,0,0,1,0]).expand(1,14,6))
    torch.testing.assert_close(y['position_z'][:,:1],one['position_z'])


def test_train_only_statistics_commands_excluded():
    m=module();a=arrays();a['body_state'][3,3]=1e6;s=m.fit_normalization(a,[0,1,2])
    assert s['state_mean'][22]==0
    x=m.prepare_inputs(a,[0],s)
    a['actor_state'][0,6:9]=[999,-999,12]
    y=m.prepare_inputs(a,[0],s)
    torch.testing.assert_close(x['state'],y['state'],rtol=0,atol=0)
    assert s['position_std'][2]>=.02


def test_pose_mask_does_not_drop_early_event_and_loss_finite():
    m=module();a=arrays();a['pose_mask'][0]=False;a['position_targets'][0]=np.nan
    s=m.fit_normalization(a,[0,1,2]);support=m.training_support(a,[0,1,2]);net=m.MovingPose2sEvaluator()
    out=net(**m.prepare_inputs(a,[0,1,2],s));loss=m.objective(out,a,[0,1,2],s,support)
    assert all(torch.isfinite(x) for x in loss.values())
    loss['total'].backward()
    assert torch.isfinite(net.event_head.weight.grad).all()
    assert support['counts'][0]['positive']==1


def test_checkpoint_horizon_strict_cpu_and_prediction(tmp_path):
    m=module();a=arrays(k=14);s=m.fit_normalization(a,[0,1,2]);support=m.training_support(a,[0,1,2]);net=m.MovingPose2sEvaluator()
    p=tmp_path/'new.pt';m.save_checkpoint(p,net,s,support,{'seed':123})
    restored,payload=m.load_checkpoint(p)
    pred=m.predict(restored,s,a,[3],support)
    assert pred['schema']=='eatnav_moving_pose2s_prediction_v1'
    assert pred['horizon_sec']==2.
    assert pred['delta_position_body_m'].shape==(1,14,3)
    np.testing.assert_allclose(pred['future_tilt_rad'],0,atol=1e-6)
    assert np.isfinite(pred['event_probability']).all()
    payload['horizon_sec']=10.;torch.save(payload,p)
    with pytest.raises(ValueError,match='schema|horizon'):m.load_checkpoint(p)


def test_only_feature_weights_transferred():
    m=module();net=m.MovingPose2sEvaluator();old={}
    for key,value in net.state_dict().items():
        if key.startswith(('dem_encoder.','state_encoder.','path_encoder.','fusion.')):old[key]=torch.full_like(value,.123)
    old['risk_head.weight']=torch.full((4,32),99.)
    report=m.transfer_features(net,{'schema':'eatnav_fall_stall_checkpoint_v2_command_free','state_dict':old})
    assert report['excluded_heads']==['risk_head','progress_head']
    torch.testing.assert_close(net.state_encoder[0].weight,torch.full_like(net.state_encoder[0].weight,.123))
    assert torch.max(torch.abs(net.event_head.weight))<99.


@pytest.mark.parametrize('key,value',[('quaternion_order','xyzw'),('pose_frame','yaw_only'),('rotation_6d','first_two_rows'),('prediction_schema','legacy')])
def test_loader_rejects_contradictory_coordinate_contract(tmp_path,key,value):
    m=module();a=arrays();s=m.fit_normalization(a,[0,1,2]);sup=m.training_support(a,[0,1,2]);net=m.MovingPose2sEvaluator();p=tmp_path/'x.pt'
    m.save_checkpoint(p,net,s,sup,{})
    payload=torch.load(p,weights_only=True);payload[key]=value;torch.save(payload,p)
    with pytest.raises(ValueError,match='contract|schema'):m.load_checkpoint(p)


def test_nonfinite_network_rotation_is_rejected_not_identity_success():
    m=module();a=arrays();s=m.fit_normalization(a,[0,1,2]);net=m.MovingPose2sEvaluator()
    with torch.no_grad():net.rotation_head.bias[0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):m.predict(net,s,a,[0])
