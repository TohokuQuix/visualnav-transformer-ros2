"""CPU-capable 14-candidate inference only; never sends robot commands."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from visualnav_transformer.eat_nav.moving_pose2s_model import INPUT_SCHEMA,load_checkpoint,predict

REQUIRED={'dem','actor_state','body_state','paths','directions','candidate_mask'}
OPTIONAL={'current_quat_wxyz','dem_timestamps_ns','current_time_ns','input_schema'}


def load_inputs(path):
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    unexpected=set(a)-REQUIRED-OPTIONAL
    if unexpected:raise ValueError('unexpected input keys (teachers/future metadata forbidden): '+','.join(sorted(unexpected)))
    if not REQUIRED<=set(a):raise ValueError('missing required input tensors')
    if str(a.get('input_schema',''))!=INPUT_SCHEMA:raise ValueError('input schema mismatch')
    a.pop('input_schema')
    if ('current_time_ns' in a)!=('dem_timestamps_ns' in a):raise ValueError('history/current time must be provided together')
    if 'current_time_ns' in a:
        t=np.asarray(a.pop('current_time_ns'));history=np.asarray(a.pop('dem_timestamps_ns'))
        if t.shape!=(len(a['dem']),) or history.shape!=(len(t),4):raise ValueError('history/current time shape mismatch')
        if not np.issubdtype(t.dtype,np.integer) or not np.issubdtype(history.dtype,np.integer):raise ValueError('history time must be integer nanoseconds')
        if np.any(history>t[:,None]) or np.any(t-history[:,-1]>250_000_000) or np.any(t-history[:,-1]<0) or np.any(np.diff(history,axis=1)!=250_000_000):raise ValueError('noncausal future/stale/malformed DEM history')
    return a


def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(a.threads);model,payload=load_checkpoint(a.checkpoint,a.device);inputs=load_inputs(a.input)
    result=predict(model,payload['normalization'],inputs,None,payload['support']);result['model_path']=str(a.checkpoint);result['units']={'position':'m','rotation_rpy':'rad','quaternion':'wxyz','horizon':'s'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output),schema=result['schema'],horizon_sec=result['horizon_sec'],batch=inputs['paths'].shape[0],candidate_count=inputs['paths'].shape[1])))
if __name__=='__main__':main()
