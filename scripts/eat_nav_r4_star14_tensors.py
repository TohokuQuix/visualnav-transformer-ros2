"""Unnormalized, initial-decision tensors with explicit unknown target masks."""
from __future__ import annotations

import numpy as np

from visualnav_transformer.eat_nav.fall_stall_targets import EVENT_NAMES, HORIZONS, SCHEMA as TARGET_SCHEMA, body_vector


INPUT_SCHEMA='eatnav_fall_stall_input_v1'
from eat_nav_r4_star14 import CANDIDATE_IDS
FLOAT_SHAPES={'dem':(4,2,20,20),'actor_state':(22,),'body_state':(9,),
              'paths':(14,8,2),'directions':(14,1),'progress_targets':(14,4)}
MASK_SHAPES={'candidate_mask':(14,),'event_mask':(14,4,4),'progress_mask':(14,4),'event_free_mask':(14,)}
LABEL_SHAPES={'event_targets':(14,4,4),'event_free_10s':(14,)}


def check(condition,message):
    if not condition:
        raise ValueError(message)


def candidate_order(candidate_ids):
    result=tuple(candidate_ids)
    check(len(result)==14 and set(result)==set(CANDIDATE_IDS),'candidate order must contain the frozen fourteen IDs')
    return result


def float_array(value,shape,name):
    try:
        array=np.asarray(value,dtype=np.float32)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError(f'{name} must be a numeric array') from exc
    check(array.shape==shape,f'{name} shape mismatch')
    return array.copy()


def target_value(target,value_key='label'):
    check(isinstance(target,dict) and value_key in target and 'target_mask' in target,'missing target value/mask')
    value=target[value_key];mask=target['target_mask']
    check(isinstance(mask,(bool,np.bool_)) and bool(mask)==(value is not None),'target value/mask mismatch')
    if value_key=='label':
        check(value is None or value in (0,1),'event target must be 0/1/null')
    else:
        check(value is None or isinstance(value,(int,float)) and np.isfinite(value),'progress must be finite/null')
    return value


def pack_groups(groups,*,candidate_ids=CANDIDATE_IDS):
    """Pack each shared initial StateGroup with a fixed candidate axis.

    Extra metadata/future fields are ignored deliberately: only the three
    allowlisted initial arrays and initial candidate Paths enter model inputs.
    """
    ids=candidate_order(candidate_ids)
    check(bool(groups),'nonempty StateGroup dataset required')
    count=len(groups)
    arrays={name:np.zeros((count,*shape),dtype=np.float32) for name,shape in FLOAT_SHAPES.items()}
    arrays.update({name:np.zeros((count,*shape),dtype=bool) for name,shape in MASK_SHAPES.items()})
    arrays.update({name:np.full((count,*shape),-1,dtype=np.int8) for name,shape in LABEL_SHAPES.items()})
    for index,group in enumerate(groups):
        initial=group.get('initial') or {}
        check(all(key in initial for key in ('dem','actor_pre_noise','root')),'missing initial pre-noise/DEM/root input')
        dem=float_array(initial['dem'],(4,2,20,20),'initial DEM')
        mask=dem[:,1];height=dem[:,0]
        check(np.isfinite(mask).all() and np.logical_or(mask==0,mask==1).all(),'DEM mask must be finite binary')
        check((mask.mean(axis=(-2,-1))>=.25).all(),'DEM frame coverage below 25%')
        check(np.isfinite(height[mask==1]).all(),'DEM valid height is nonfinite')
        dem[:,0]=np.where(mask==1,height,0)
        arrays['dem'][index]=dem
        state=float_array(initial['actor_pre_noise'],(22,),'initial Actor pre-noise state')
        check(np.isfinite(state).all(),'initial Actor state is nonfinite')
        arrays['actor_state'][index]=state
        root=float_array(initial['root'],(13,),'initial root').astype(np.float64).tolist()
        check(np.isfinite(root).all(),'initial root is nonfinite')
        body=[body_vector(root,[0.,0.,-1.]),body_vector(root,root[7:10]),body_vector(root,root[10:13])]
        check(all(vector is not None for vector in body),'initial quaternion is invalid')
        arrays['body_state'][index]=np.concatenate(body)
        entries=group.get('candidates') or {}
        check(set(entries)<=set(ids),'unknown candidate ID')
        check(bool(entries),'all candidates invalid in StateGroup')
        for slot,name in enumerate(ids):
            if name not in entries:
                continue
            entry=entries[name];path=float_array(entry.get('path'),(8,2),'candidate Path')
            direction=entry.get('direction');expected=-1 if name.startswith('reverse_') else 1
            check(direction==expected,'candidate direction mismatch')
            check(np.isfinite(path).all() and np.allclose(path[0],0,atol=1e-7,rtol=0)
                  and path[1,0]*direction>0,'candidate Path frame/direction invalid')
            targets=entry.get('targets') or {}
            check(targets.get('schema')==TARGET_SCHEMA,'target schema mismatch')
            arrays['paths'][index,slot]=path;arrays['directions'][index,slot,0]=direction
            arrays['candidate_mask'][index,slot]=True
            for horizon_index,horizon in enumerate(HORIZONS):
                target=(targets.get('horizons') or {}).get(f'{horizon}s') or {}
                events=target.get('events') or {}
                check(set(events)==set(EVENT_NAMES),'event head keys mismatch')
                for head,name in enumerate(EVENT_NAMES):
                    value=target_value(events[name])
                    if value is not None:
                        arrays['event_targets'][index,slot,horizon_index,head]=value
                        arrays['event_mask'][index,slot,horizon_index,head]=True
                value=target_value(target.get('progress'),'value_m')
                if value is not None:
                    arrays['progress_targets'][index,slot,horizon_index]=value
                    arrays['progress_mask'][index,slot,horizon_index]=True
            value=target_value(targets.get('event_free_10s'))
            if value is not None:
                arrays['event_free_10s'][index,slot]=value
                arrays['event_free_mask'][index,slot]=True
    validate_arrays(arrays,candidate_ids=ids)
    return arrays


def validate_arrays(arrays,*,candidate_ids=CANDIDATE_IDS):
    ids=candidate_order(candidate_ids)
    check(set(arrays)==set(FLOAT_SHAPES)|set(MASK_SHAPES)|set(LABEL_SHAPES),'unexpected tensor keys; possible future/metadata input')
    count=len(arrays['dem'])
    check(count>0,'empty dataset')
    for shapes,dtype in ((FLOAT_SHAPES,np.float32),(MASK_SHAPES,np.bool_),(LABEL_SHAPES,np.int8)):
        for name,shape in shapes.items():
            value=arrays[name]
            check(isinstance(value,np.ndarray) and value.shape==(count,*shape) and value.dtype==dtype,
                  f'{name} tensor shape/dtype mismatch')
            check(np.isfinite(value).all(),f'{name} tensor is nonfinite')
    mask=arrays['dem'][:,:,1]
    check(np.logical_or(mask==0,mask==1).all() and (mask.mean(axis=(-2,-1))>=.25).all(),'DEM mask/coverage invalid')
    check(np.all(arrays['dem'][:,:,0][mask==0]==0),'invalid DEM height must be zero-filled')
    check(np.allclose(np.linalg.norm(arrays['body_state'][:,:3],axis=1),1,atol=1e-5,rtol=0),'body gravity is not unit length')
    candidate_mask=arrays['candidate_mask']
    check(candidate_mask.any(axis=1).all(),'all candidates invalid in StateGroup')
    for label_key,mask_key in (('event_targets','event_mask'),('event_free_10s','event_free_mask')):
        labels=arrays[label_key];known=arrays[mask_key]
        check(np.isin(labels,[-1,0,1]).all() and np.array_equal(labels!=-1,known),f'{label_key} unknown/mask mismatch')
    check(not arrays['event_mask'][~candidate_mask].any() and not arrays['progress_mask'][~candidate_mask].any()
          and not arrays['event_free_mask'][~candidate_mask].any(),'invalid candidate carries known targets')
    check(np.all(arrays['progress_targets'][~arrays['progress_mask']]==0),'masked progress placeholder must be zero')
    check(np.all(arrays['paths'][~candidate_mask]==0) and np.all(arrays['directions'][~candidate_mask]==0),
          'invalid candidate input placeholders must be zero')
    for slot,name in enumerate(ids):
        active=candidate_mask[:,slot];expected=-1 if name.startswith('reverse_') else 1
        check(np.all(arrays['directions'][active,slot,0]==expected),'candidate direction mismatch')
        paths=arrays['paths'][active,slot]
        check(np.allclose(paths[:,0],0,atol=1e-7,rtol=0) and np.all(paths[:,1,0]*expected>0),'candidate Path frame/direction mismatch')
