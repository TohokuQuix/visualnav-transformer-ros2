import copy
import numpy as np
import pytest
import torch
from visualnav_transformer.eat_nav.r6_model import (PoseRiskEvaluator,CONTRACT,fit_normalization,prepare_inputs,training_support,objective,save_checkpoint,load_checkpoint,predict)

def arrays():
 rng=np.random.default_rng(42);n=3
 a=dict(dem=np.zeros((n,4,2,20,20),np.float32),actor_state=rng.normal(size=(n,22)).astype(np.float32),body_state=rng.normal(size=(n,9)).astype(np.float32),paths=rng.normal(size=(n,14,8,2)).astype(np.float32),directions=np.ones((n,14,1),np.float32),candidate_mask=np.ones((n,14),bool),event_targets=np.zeros((n,14,4),np.float32),event_mask=np.ones((n,14,4),bool),trajectory=np.zeros((n,14,20,9),np.float32),trajectory_mask=np.ones((n,14,20),bool))
 a['dem'][:,:,1]=1;a['event_targets'][0,0]=1;a['trajectory'][...,4]=1;a['trajectory'][...,6]=1;a['trajectory'][...,8]=1
 return a

def test_new_contract_is_two_seconds_with_20_future_samples_and_no_10s_progress():
 assert CONTRACT['event_horizon_sec']==2
 assert np.allclose(CONTRACT['times_sec'],np.arange(1,21)/10)
 a=arrays();stats=fit_normalization(a,[0,1]);net=PoseRiskEvaluator();o=net(**prepare_inputs(a,[0,1],stats))
 assert o['event_logits'].shape==(2,14,4)
 assert o['trajectory_z'].shape==(2,14,20,9)
 assert 'progress_z' not in o
 assert all(torch.isfinite(v).all() for v in o.values())

def test_normalization_is_train_only_and_command_invariant():
 a=arrays();s=fit_normalization(a,[0,1]);b=copy.deepcopy(a);b['dem'][2,:,0]=1e6;b['trajectory'][2,...,:3]=1e6
 assert fit_normalization(b,[0,1])==s
 b['actor_state'][:,6:9]=np.nan
 x=prepare_inputs(a,[0,1],s);y=prepare_inputs(b,[0,1],s)
 for k in x:assert torch.equal(x[k],y[k])

def test_loss_masks_unknown_events_and_future_pose_before_arithmetic():
 a=arrays();a['event_mask'][0,0,0]=False;a['event_targets'][0,0,0]=np.nan
 a['trajectory_mask'][0,1,4:]=False;a['trajectory'][0,1,4:]=np.nan
 s=fit_normalization(a,[0,1]);net=PoseRiskEvaluator();o=net(**prepare_inputs(a,[0,1],s))
 losses=objective(o,a,[0,1],s,training_support(a,[0,1]))
 assert all(torch.isfinite(v) for v in losses.values())
 losses['total'].backward();assert torch.isfinite(net.event_head.weight.grad).all()

def test_all_head_support_uses_two_second_events_and_candidate_mask():
 a=arrays();a['candidate_mask'][0,0]=False
 support=training_support(a,[0,1]);assert support['active']==[False]*4
 assert all(v['positive']==0 for v in support['counts'])

def test_checkpoint_roundtrip_and_strict_old_schema_rejection(tmp_path):
 a=arrays();s=fit_normalization(a,[0,1]);net=PoseRiskEvaluator();p=tmp_path/'new.pt'
 save_checkpoint(p,net,s,training_support(a,[0,1]),{'seed':123})
 restored,payload=load_checkpoint(p);assert payload['contract']==CONTRACT
 before=predict(net,s,a,[0]);after=predict(restored,s,a,[0])
 assert np.allclose(before['trajectory'],after['trajectory'])
 assert before['events'].shape==(1,14,4)
 assert before['trajectory'].shape==(1,14,20,9)
 orient=after['trajectory'][...,3:].reshape(1,14,20,3,2)
 assert np.allclose(np.linalg.norm(orient,axis=-1),1.,atol=1e-5)
 torch.save({'schema':'eatnav_fall_stall_checkpoint_v2_command_free'},p)
 with pytest.raises(ValueError,match='schema'):load_checkpoint(p)

def test_wrong_candidate_count_and_nonfinite_state_rejected():
 a=arrays();s=fit_normalization(a,[0,1]);x=prepare_inputs(a,[0],s)
 x['paths']=x['paths'][:,:6]
 with pytest.raises(ValueError):PoseRiskEvaluator()(**x)
 a['body_state'][0,0]=np.nan
 with pytest.raises(ValueError):prepare_inputs(a,[0],s)
