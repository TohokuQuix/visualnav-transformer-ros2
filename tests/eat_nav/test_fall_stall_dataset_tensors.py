"""Input/target contracts with independent numerical fixtures."""
import copy
import importlib
import importlib.util
import math

import numpy as np
import pytest


IDS=('forward_direct','forward_-30','forward_+30','reverse_direct')
HEADS=('tip_over','near_fall','linear_stall','turn_stall')


def api():
    name='visualnav_transformer.eat_nav.fall_stall_dataset_tensors'
    assert importlib.util.find_spec(name) is not None,'dataset tensor implementation is missing'
    return importlib.import_module(name)


def teacher(label=0):
    return {'schema':'eatnav_fall_stall_targets_v2',
            'horizons':{f'{h}s':{'events':{name:{'label':label,'target_mask':label is not None,'reason':'fixture'}
                                          for name in HEADS},
                                 'progress':{'value_m':float(h)/10,'target_mask':True,'reason':'fixture'}}
                        for h in (1,2,5,10)},
            'event_free_10s':{'label':1 if label==0 else 0 if label==1 else None,
                              'target_mask':label is not None,'reason':'fixture'}}


def group():
    dem=np.zeros((4,2,20,20),dtype=np.float32);dem[:,1]=1
    initial={'dem':dem,'actor_pre_noise':np.arange(22,dtype=np.float32),
             'actor_post_noise':np.full(22,900.),'actor_input':np.full(22,800.),
             'root':[123.,456.,3.,math.sqrt(.5),0.,0.,math.sqrt(.5),1.,0.,0.,0.,2.,0.]}
    candidates={}
    for name in IDS:
        direction=-1 if name=='reverse_direct' else 1
        candidates[name]={'path':np.array([[direction*.75*i/7,0.] for i in range(8)]),
                          'direction':direction,'targets':teacher()}
    return {'initial':initial,'candidates':candidates,'sample_id':'terrain-level1-rep0/env0',
            'seed':123,'env':0,'terrain':'terrain','level':1,'replicate':0,
            'future_state':np.full(13,999.),'reward':999.}


def test_body_rotation_and_pre_noise_state_are_initial_only():
    g=group();arrays=api().pack_groups([g])
    assert arrays['dem'].shape==(1,4,2,20,20)
    assert arrays['actor_state'].shape==(1,22)
    np.testing.assert_array_equal(arrays['actor_state'][0],np.arange(22))
    np.testing.assert_allclose(arrays['body_state'][0],[0.,0.,-1.,0.,-1.,0.,2.,0.,0.],atol=1e-6)
    assert arrays['paths'].shape==(1,4,8,2)
    np.testing.assert_array_equal(arrays['directions'][0,:,0],[1,1,1,-1])
    assert set(arrays)=={'dem','actor_state','body_state','paths','directions','candidate_mask',
                         'event_targets','event_mask','progress_targets','progress_mask',
                         'event_free_10s','event_free_mask'}


def test_unknown_event_and_progress_are_masked_not_negative():
    g=group();targets=g['candidates']['forward_direct']['targets']
    targets['horizons']['10s']['events']['tip_over']={'label':None,'target_mask':False,'reason':'right_censored'}
    targets['horizons']['10s']['progress']={'value_m':None,'target_mask':False,'reason':'right_censored'}
    arrays=api().pack_groups([g])
    assert arrays['event_targets'].shape==(1,4,4,4)
    assert arrays['event_targets'][0,0,3,0]==-1
    assert not arrays['event_mask'][0,0,3,0]
    assert arrays['progress_targets'][0,0,3]==0 and not arrays['progress_mask'][0,0,3]
    assert arrays['event_targets'].dtype==np.int8 and arrays['event_mask'].dtype==np.bool_


def test_missing_candidate_stays_masked_and_all_invalid_sample_is_rejected():
    g=group();del g['candidates']['reverse_direct']
    arrays=api().pack_groups([g])
    assert arrays['candidate_mask'].tolist()==[[True,True,True,False]]
    assert np.all(arrays['event_targets'][0,3]==-1)
    assert not arrays['event_mask'][0,3].any()
    g['candidates']={}
    with pytest.raises(ValueError,match='candidate'):api().pack_groups([g])


def test_candidate_permutation_reorders_paths_and_targets_together():
    g=group();g['candidates']['reverse_direct']['targets']=teacher(1)
    standard=api().pack_groups([g]);reverse=api().pack_groups([g],candidate_ids=IDS[::-1])
    for key in ('paths','directions','candidate_mask','event_targets','event_mask','progress_targets','progress_mask'):
        np.testing.assert_array_equal(reverse[key],standard[key][:,::-1])


def test_invalid_masked_heights_are_zeroed_without_changing_valid_mask_or_source():
    g=group();g['initial']['dem'][0,1,0,0]=0;g['initial']['dem'][0,0,0,0]=np.nan
    original=copy.deepcopy(g)
    arrays=api().pack_groups([g])
    assert arrays['dem'][0,0,0,0,0]==0
    assert arrays['dem'][0,0,1,0,0]==0
    assert np.isnan(g['initial']['dem'][0,0,0,0])
    np.testing.assert_array_equal(g['initial']['actor_pre_noise'],original['initial']['actor_pre_noise'])


@pytest.mark.parametrize('mutation',['nonbinary_mask','nonfinite_valid_height','low_coverage','nonfinite_state',
                                     'zero_quaternion','wrong_direction','missing_pre_noise','bad_target_mask'])
def test_malformed_input_or_target_is_rejected(mutation):
    g=group()
    if mutation=='nonbinary_mask':g['initial']['dem'][0,1,0,0]=.5
    elif mutation=='nonfinite_valid_height':g['initial']['dem'][0,0,0,0]=np.nan
    elif mutation=='low_coverage':g['initial']['dem'][0,1]=0
    elif mutation=='nonfinite_state':g['initial']['actor_pre_noise'][0]=np.inf
    elif mutation=='zero_quaternion':g['initial']['root'][3:7]=[0.]*4
    elif mutation=='wrong_direction':g['candidates']['reverse_direct']['direction']=1
    elif mutation=='missing_pre_noise':del g['initial']['actor_pre_noise']
    else:g['candidates']['forward_direct']['targets']['horizons']['10s']['events']['tip_over']['target_mask']=False
    with pytest.raises(ValueError):api().pack_groups([g])


def test_validation_rejects_unmasked_unknown_and_unknown_arrays():
    arrays=api().pack_groups([group()])
    arrays['event_targets'][0,0,3,0]=-1
    with pytest.raises(ValueError):api().validate_arrays(arrays)
    arrays=api().pack_groups([group()]);arrays['future_reward']=np.ones(1)
    with pytest.raises(ValueError):api().validate_arrays(arrays)
