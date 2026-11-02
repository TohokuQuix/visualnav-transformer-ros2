"""Report frozen-train input distributions using checkpoint statistics; never refit.

Only normalization.train_indices are materialized from samples.npz. Hashing the
bound dataset archive reads bytes for identity verification, not holdout values.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from visualnav_transformer.eat_nav.moving_pose2s_model import (
    INPUT_SCHEMA, NORMALIZATION_SCHEMA, KEEP, load_checkpoint, validate_raw_inputs,
)

FILES=('samples.npz','sample-index.json','split.json','manifest.json')
QUANTILES=(0.,.01,.05,.25,.5,.75,.95,.99,1.)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_selected_rows(path,key,ids,expected=None):
    """Read NPY rows in ascending file order, without loading unselected rows."""
    with zipfile.ZipFile(path) as archive, archive.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
        else:raise ValueError('unsupported NPY version')
        if fortran or not shape or dtype.hasobject or dtype.kind not in 'biuf':
            raise ValueError('numeric C-order row arrays required')
        if expected is not None and (list(shape)!=expected.get('shape') or str(dtype)!=expected.get('dtype')):
            raise ValueError('manifest array schema mismatch: '+key)
        if any(i<0 or i>=shape[0] for i in ids):raise ValueError('train index out of range')
        offset=f.tell();size=int(np.prod(shape[1:]))*dtype.itemsize
        result=np.empty((len(ids),*shape[1:]),dtype=dtype)
        for destination in sorted(range(len(ids)),key=lambda j:ids[j]):
            f.seek(offset+ids[destination]*size)
            raw=f.read(size)
            if len(raw)!=size:raise ValueError('truncated dataset array')
            result[destination]=np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
    return result


def summary(values):
    values=np.asarray(values,np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():raise ValueError('empty/nonfinite diagnostic channel')
    return dict(count=len(values),min=float(values.min()),max=float(values.max()),
        mean=float(values.mean()),std=float(values.std()),constant=bool(values.min()==values.max()),
        quantiles={str(q):float(v) for q,v in zip(QUANTILES,np.quantile(values,QUANTILES,method='linear'))})


def channel(label,values,mean,std,**metadata):
    return dict(label=label,**metadata,saved_mean=float(mean),saved_std=float(std),
        raw=summary(values),standardized=summary((np.asarray(values,np.float64)-mean)/std))


def diagnose(dataset,checkpoint,output):
    dataset=Path(dataset);checkpoint=Path(checkpoint);output=Path(output)
    if output.exists() or output.suffix!='.json':raise ValueError('fresh .json output required')
    _,payload=load_checkpoint(checkpoint)
    binding=payload.get('metadata',{}).get('dataset_binding',{})
    actual={name:sha(dataset/name) for name in FILES}
    if not all(binding.get(name)==actual[name] for name in FILES):
        raise ValueError('checkpoint dataset SHA binding mismatch')
    manifest=json.loads((dataset/'manifest.json').read_text())
    if manifest.get('schema')!='eatnav_moving_pose2s_dataset_v1' or manifest.get('input_schema')!=INPUT_SCHEMA or manifest.get('horizon_sec')!=2.:
        raise ValueError('dataset schema mismatch')
    index=json.loads((dataset/'sample-index.json').read_text())
    roles=json.loads((dataset/'split.json').read_text())['roles']
    stats=payload['normalization'];ids=stats.get('train_indices')
    if stats.get('schema')!=NORMALIZATION_SCHEMA or stats.get('source')!='train_split_only' or stats.get('state_keep_indices')!=list(KEEP):
        raise ValueError('normalization schema/source mismatch')
    if not isinstance(ids,list) or not ids or any(type(i) is not int for i in ids) or len(ids)!=len(set(ids)):
        raise ValueError('nonempty unique integer frozen train IDs required')
    train=roles.get('train',[])
    if not set(ids)<=set(train) or any(i<0 or i>=len(index) for i in ids):
        raise ValueError('normalization IDs must be in bound train split')
    if any(set(ids)&set(members) for role,members in roles.items() if role!='train'):
        raise ValueError('normalization IDs overlap evaluation roles')
    for name,width in (('state',28),('path',2),('position',3),('dem',None)):
        for suffix in ('mean','std'):
            value=np.asarray(stats.get(name+'_'+suffix),np.float64)
            expected=() if width is None else (width,)
            if value.shape!=expected or not np.isfinite(value).all() or (suffix=='std' and np.any(value<=0)):
                raise ValueError('invalid saved normalization: '+name+'_'+suffix)
    keys=('dem','actor_state','body_state','paths','directions','candidate_mask','pose_mask','position_targets')
    descriptions=manifest.get('arrays',{})
    if any(k not in descriptions or descriptions[k]['shape'][0]!=len(index) for k in keys):
        raise ValueError('manifest/index batch mismatch')
    arrays={k:read_selected_rows(dataset/'samples.npz',k,ids,descriptions[k]) for k in keys}
    validate_raw_inputs(arrays)
    cm=arrays['candidate_mask'].astype(bool);pm=arrays['pose_mask']
    if pm.shape!=cm.shape or not np.isin(pm,[0,1]).all() or arrays['position_targets'].shape!=(*cm.shape,3):
        raise ValueError('target shape/mask mismatch')
    state=np.concatenate([arrays['actor_state'],arrays['body_state']],axis=1)[:,KEEP]
    heights=arrays['dem'][:,:,0][arrays['dem'][:,:,1]==1]
    paths=arrays['paths'][cm].reshape(-1,2)
    positions=arrays['position_targets'][pm.astype(bool)&cm]
    body_names=('gravity_x','gravity_y','gravity_z','linear_velocity_x','linear_velocity_y','linear_velocity_z','angular_velocity_x','angular_velocity_y','angular_velocity_z')
    state_channels=[]
    for j,source in enumerate(KEEP):
        actor=source<22;source_index=source if actor else source-22
        label=f'actor_pre_noise[{source}]' if actor else 'body_'+body_names[source_index]
        state_channels.append(channel(label,state[:,j],stats['state_mean'][j],stats['state_std'][j],
            source='actor_state' if actor else 'body_state',source_index=source_index,model_channel=j))
    channels=dict(dem_height=channel('DEM height where valid_mask == 1',heights,stats['dem_mean'],stats['dem_std']),
        dem_valid_mask=dict(label='DEM valid mask (unchanged, not standardized)',raw=summary(arrays['dem'][:,:,1])),
        state=state_channels,path=[channel('path_'+axis,paths[:,j],stats['path_mean'][j],stats['path_std'][j]) for j,axis in enumerate(('x','y'))],
        position_target=[channel('position_target_'+axis,positions[:,j],stats['position_mean'][j],stats['position_std'][j]) for j,axis in enumerate(('x','y','z'))])
    report=dict(schema='eatnav_moving_pose2s_input_diagnostics_v1',checkpoint_sha256=sha(checkpoint),
        dataset_files_sha256=actual,normalization_train_indices=ids,normalization_source='checkpoint_frozen_train_indices_only',
        selected_rows=len(ids),split_train_rows=len(train),holdout_rows_materialized=0,
        quantile_method='numpy linear',quantile_probabilities=list(QUANTILES),
        state_keep_indices=list(KEEP),channels=channels,normalization_refitted=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=diagnose(a.dataset,a.checkpoint,a.output)
    print(json.dumps(dict(output=str(a.output),selected_train_rows=report['selected_rows'],checkpoint_sha256=report['checkpoint_sha256'])))

if __name__=='__main__':main()
