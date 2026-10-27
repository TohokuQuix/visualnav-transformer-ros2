"""Hand-checked normalization, architecture and balanced-loss contracts."""
import copy
import importlib
import importlib.util
import math

import numpy as np
import pytest
import torch

from test_fall_stall_dataset_tensors import group, IDS
from visualnav_transformer.eat_nav.fall_stall_dataset_tensors import pack_groups


def api():
    name='visualnav_transformer.eat_nav.fall_stall_learning'
    assert importlib.util.find_spec(name) is not None,'baseline learning implementation is missing'
    return importlib.import_module(name)


def learning_arrays(count=12):
    groups=[]
    for i in range(count):
        g=group();g['initial']['actor_pre_noise']+=i
        g['initial']['dem'][:,0]=i*.01
        for j,name in enumerate(IDS):
            target=g['candidates'][name]['targets']
            for h in (1,2,5,10):
                for k,event in enumerate(('tip_over','near_fall','linear_stall','turn_stall')):
                    target['horizons'][f'{h}s']['events'][event]['label']=int(j==(k%3)+1) if k<3 else 0
                target['horizons'][f'{h}s']['progress']['value_m']=h*.1+i*.01+j*.02
            target['event_free_10s']['label']=int(j==0)
        groups.append(g)
    return pack_groups(groups)


def test_normalization_depends_only_on_train_and_preserves_mask_planes():
    arrays=learning_arrays();arrays['dem'][0,1,1,0,0]=0;arrays['dem'][0,1,0,0,0]=0
    before=copy.deepcopy(arrays);stats=api().fit_normalization(arrays,[0,1,2,3])
    assert stats['state_mean'][0]==pytest.approx(1.5)
    changed=copy.deepcopy(arrays);changed['actor_state'][4:]+=999.;changed['dem'][4:,:,0]+=500.
    assert api().fit_normalization(changed,[0,1,2,3])==stats
    inputs=api().prepare_inputs(arrays,[0,1,2,3],stats)
    assert inputs['state'].shape==(4,31) and inputs['dem'].shape==(4,8,20,20)
    np.testing.assert_array_equal(inputs['dem'].numpy().reshape(4,4,2,20,20)[:,:,1],arrays['dem'][:4,:,1])
    assert inputs['dem'][0,2,0,0]==0
    for name in arrays:np.testing.assert_array_equal(arrays[name],before[name])


def test_train_support_ignores_unknowns_and_caps_positive_weight():
    arrays=learning_arrays(8)
    arrays['event_targets'][:,:,3,0]=0;arrays['event_targets'][0,0,3,0]=1
    arrays['event_targets'][:,:,3,1]=-1;arrays['event_mask'][:,:,3,1]=False
    support=api().training_support(arrays,list(range(8)))
    assert support['active']==[True,False,True,False]
    assert support['pos_weight'][0]==20.
    assert support['counts'][1]=={'positive':0,'negative':0,'unknown':32}


@pytest.mark.parametrize('use_dem',[False,True])
def test_model_shapes_candidate_permutation_and_finite_gradients(use_dem):
    torch.set_num_threads(1);torch.manual_seed(123)
    arrays=learning_arrays();stats=api().fit_normalization(arrays,[0,1,2,3])
    inputs=api().prepare_inputs(arrays,[0,1,2,3],stats);model=api().RiskEvaluator(use_dem=use_dem)
    result=model(**inputs)
    assert result['risk_logits'].shape==result['progress_z'].shape==(4,4,4)
    permuted={k:v[:,[3,1,0,2]] if k in ('paths','directions','candidate_mask') else v for k,v in inputs.items()}
    other=model(**permuted)
    torch.testing.assert_close(other['risk_logits'],result['risk_logits'][:,[3,1,0,2]],atol=1e-6,rtol=1e-6)
    (result['risk_logits'].sum()+result['progress_z'].sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    inputs['candidate_mask'][:]=False
    with pytest.raises(ValueError,match='candidate'):model(**inputs)


def blank_batch(count):
    return {'events':torch.full((count,4,4),-1.),'event_mask':torch.zeros(count,4,4,dtype=torch.bool),
            'progress':torch.zeros(count,4,4),'progress_mask':torch.zeros(count,4,4,dtype=torch.bool),
            'candidate_mask':torch.ones(count,4,dtype=torch.bool)}


def progress_stats():
    return {'progress_mean':[0.]*4,'progress_std':[1.]*4,'progress_count':[0]*4}


def test_event_loss_balances_candidates_groups_seeds_then_heads():
    batch=blank_batch(3)
    batch['events'][0,0,0]=1;batch['event_mask'][0,0,0]=True
    batch['events'][1:,:,0]=0;batch['event_mask'][1:,:,0]=True
    logits=torch.zeros(3,4,4,requires_grad=True);progress=torch.zeros_like(logits,requires_grad=True)
    loss=api().objective(logits,progress,batch,[11,11,22],[True,False,False,False],[2.,1.,1.,1.],progress_stats())
    # Seed11: (2 ln2 + ln2)/2; seed22: ln2; average=1.25 ln2.
    assert loss['event'].item()==pytest.approx(1.25*math.log(2))
    assert loss['pair'].item()==0 and loss['progress'].item()==0
    loss['total'].backward()
    assert logits.grad[0,1,0]==0 and torch.all(logits.grad[:,:,1:]==0)


def test_pair_loss_orders_safe_below_risky_and_counts_no_pair_group_as_zero():
    batch=blank_batch(2);batch['events'][0,:2,0]=torch.tensor([0.,1.]);batch['event_mask'][0,:2,0]=True
    logits=torch.zeros(2,4,4,requires_grad=True)
    with torch.no_grad():logits[0,1,0]=2.
    loss=api().objective(logits,torch.zeros_like(logits),batch,[11,22],[True,False,False,False],[1.]*4,progress_stats())
    assert loss['pair'].item()==pytest.approx(math.log1p(math.exp(-2))/2)


def test_progress_loss_standardizes_only_known_targets():
    batch=blank_batch(1);batch['progress'][0,0,0]=3.;batch['progress_mask'][0,0,0]=True
    stats={'progress_mean':[1.,0.,0.,0.],'progress_std':[2.,1.,1.,1.],'progress_count':[1,0,0,0]}
    logits=torch.zeros(1,4,4,requires_grad=True);pred=torch.zeros_like(logits,requires_grad=True)
    loss=api().objective(logits,pred,batch,[11],[False]*4,[1.]*4,stats)
    assert loss['progress'].item()==pytest.approx(.5)
    assert loss['total'].item()==pytest.approx(.05)
    loss['total'].backward();assert torch.isfinite(pred.grad).all()


def test_empty_masks_return_differentiable_zero_without_nan():
    logits=torch.zeros(2,4,4,requires_grad=True);pred=torch.zeros_like(logits,requires_grad=True)
    result=api().objective(logits,pred,blank_batch(2),[11,22],[False]*4,[1.]*4,progress_stats())
    assert result['total'].item()==0
    result['total'].backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(pred.grad).all()
