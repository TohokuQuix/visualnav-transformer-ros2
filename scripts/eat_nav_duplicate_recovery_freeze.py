"""Preserve the exact logged prefix and selected raw anchor for failed duplicate case."""
from __future__ import annotations
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def atomic(path,value):
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--selection',type=Path,required=True);p.add_argument('--state-index',type=int,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 selection=json.loads(a.selection.read_text())['train'][a.state_index]
 manifest=json.loads(a.manifest.read_text());step=selection['step'];env=selection['episode_key'][1]
 actions={};command={};anchor=None
 for chunk in manifest['chunks']:
  with gzip.open(a.manifest.parent/chunk['file'],'rt') as stream:
   for line in stream:
    row=json.loads(line)
    if row['step']>step:continue
    actions.setdefault(row['step'],{})[row['env_id']]=row['action']
    if row['env_id']==env:command[row['step']]=row['command']
    if row['step']==step and row['env_id']==env:anchor=row
  if anchor is not None and len(actions.get(step,{}))==manifest['metadata']['num_envs']:break
 if anchor is None or list((anchor['session_id'],anchor['env_id'],anchor['episode_id']))!=selection['episode_key']:
  raise ValueError('selected raw anchor mismatch')
 array=np.asarray([[actions[i][j] for j in range(manifest['metadata']['num_envs'])] for i in range(step+1)],dtype=np.float32)
 commands=np.asarray([command[i] for i in range(step+1)],dtype=np.float32)
 target=a.output/'prefix-actions.npz'
 if target.exists():raise FileExistsError(target)
 temp=target.with_name('prefix-actions.tmp.npz');np.savez_compressed(temp,actions=array,selected_env_commands=commands)
 temp.replace(target)
 raw=a.output/'logged-anchor.json'
 if raw.exists():raise FileExistsError(raw)
 atomic(raw,anchor)
 print(json.dumps({'prefix_shape':array.shape,'selected_env':env,'anchor_step':step,
                   'prefix_sha256':sha(target),'logged_anchor_sha256':sha(raw)},sort_keys=True),flush=True)
if __name__=='__main__':main()
