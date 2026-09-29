"""Freeze original collection actions for other envs during the 40-step suffix."""
from __future__ import annotations
import argparse,gzip,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--first-step',type=int,required=True);p.add_argument('--count',type=int,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
m=json.loads(a.manifest.read_text());rows={}
for chunk in m['chunks']:
 with gzip.open(a.manifest.parent/chunk['file'],'rt') as stream:
  for line in stream:
   r=json.loads(line)
   if a.first_step<=r['step']<a.first_step+a.count:
    rows.setdefault(r['step'],{})[r['env_id']]=r['action']
 if len(rows.get(a.first_step+a.count-1,{}))==m['metadata']['num_envs']:break
array=np.asarray([[rows[step][env] for env in range(m['metadata']['num_envs'])]
                  for step in range(a.first_step,a.first_step+a.count)],dtype=np.float32)
if array.shape!=(a.count,100,6):raise ValueError('incomplete source actions')
a.output.parent.mkdir(parents=True,exist_ok=True)
temp=a.output.with_name(a.output.stem+'.tmp.npz');np.savez_compressed(temp,actions=array);temp.replace(a.output)
print('SUFFIX_ACTIONS_FROZEN',array.shape,flush=True)
