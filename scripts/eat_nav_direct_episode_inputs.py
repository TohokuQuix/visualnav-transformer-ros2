"""Freeze episode-local action slices without copying recorded start state into the simulator."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def episode_slice(actions,commands,*,anchor_step,local_step,env_id):
    actions=np.asarray(actions);commands=np.asarray(commands)
    if (actions.ndim!=3 or actions.shape[2]!=6 or commands.ndim!=2 or commands.shape[1]!=3
            or actions.shape[0]!=anchor_step+1 or commands.shape[0]!=anchor_step+1
            or env_id<0 or env_id>=actions.shape[1] or local_step<0 or local_step>anchor_step):
        raise ValueError('invalid frozen prefix shape or local step')
    start=anchor_step-local_step
    return {'episode_start_step':start,'actions':actions[start:anchor_step+1].copy(),
            'commands':commands[start:anchor_step+1].copy()}


def validate_episode_boundary(row,start_step,episode_id,env_id):
    if (int(row['step'])!=start_step or int(row['episode_id'])!=episode_id
            or int(row['env_id'])!=env_id or bool(row['pre_reset'])):
        raise ValueError('recorded episode-start identity mismatch')
    return True


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def boundary_rows(manifest,start,env_id):
    previous=None;current=None
    source=json.loads(Path(manifest).read_text())
    for chunk in source['chunks']:
        with gzip.open(Path(manifest).parent/chunk['file'],'rt') as stream:
            for line in stream:
                row=json.loads(line)
                if row['env_id']==env_id and row['step'] in (start-1,start):
                    if row['step']==start-1:previous=row
                    else:current=row
                if row['step']>start and current is not None:return previous,current
    return previous,current


def write_once(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=obj:raise FileExistsError('frozen input differs: '+str(path))
        return
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--state-index',type=int,required=True)
    p.add_argument('--prefix',type=Path,required=True);p.add_argument('--logged-anchor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    state=json.loads(a.selection.read_text())['train'][a.state_index]
    logged=json.loads(a.logged_anchor.read_text())
    env_id=int(state['episode_key'][1]);episode_id=int(state['episode_key'][2])
    if [logged['session_id'],logged['env_id'],logged['episode_id']]!=state['episode_key'] or logged['step']!=state['step']:
        raise ValueError('selection/logged anchor mismatch')
    with np.load(a.prefix) as data:
        part=episode_slice(data['actions'],data['selected_env_commands'],anchor_step=int(state['step']),
                           local_step=int(state['local_step']),env_id=env_id)
    start=part['episode_start_step'];previous,current=boundary_rows(a.manifest,start,env_id)
    if current is None:raise ValueError('missing recorded episode start')
    validate_episode_boundary(current,start,episode_id,env_id)
    if start and (previous is None or previous['episode_id']!=episode_id-1
                  or not (previous['terminated'] or previous['truncated']) or not previous['pre_reset']):
        raise ValueError('previous episode terminal boundary mismatch')
    a.output.mkdir(parents=True,exist_ok=True)
    array_path=a.output/'episode-local-actions.npz'
    if not array_path.exists():np.savez_compressed(array_path,actions=part['actions'],selected_env_commands=part['commands'])
    else:
        with np.load(array_path) as data:
            if not np.array_equal(data['actions'],part['actions']) or not np.array_equal(data['selected_env_commands'],part['commands']):
                raise FileExistsError('episode-local action freeze differs')
    metadata={'state_index':a.state_index,'episode_key':state['episode_key'],'anchor_global_step':state['step'],
              'episode_local_step':state['local_step'],'episode_start_global_step':start,
              'terrain_seed':state['terrain_seed'],'terrain_type':state['terrain_type'],
              'boundary':{'start_row_episode_id':current['episode_id'],
                          'start_row_initial_seed':current['initial_seed'],
                          'start_row_command_seed':current['command_seed'],
                          'previous_terminal':None if previous is None else {'step':previous['step'],
                          'episode_id':previous['episode_id'],'terminated':previous['terminated'],
                          'truncated':previous['truncated']}},
              'sources_sha256':{key:sha(path) for key,path in {'manifest':a.manifest,'selection':a.selection,
                     'prefix':a.prefix,'logged_anchor':a.logged_anchor}.items()},
              'episode_actions_sha256':sha(array_path),
              'recorded_start_state_used_by_runner':False}
    write_once(a.output/'episode-input-freeze.json',metadata)
    print('DIRECT_INPUT_FROZEN',a.state_index,start,state['step'],flush=True)

if __name__=='__main__':main()
