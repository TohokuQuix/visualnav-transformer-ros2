"""Offline CPU Critic replay and conditional terminal-command diagnostics.

Run inside the existing Isaac image with read-only mounts. No simulator is loaded.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic

parser=argparse.ArgumentParser()
parser.add_argument('--work',type=Path,required=True)
parser.add_argument('--old',type=Path,required=True)
parser.add_argument('--new',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--stage',choices=['equivalence','terminal'],required=True)
args=parser.parse_args()

torch.set_num_threads(1)
base=args.work/'docs/quality-identifiability-gate'
config=__import__('yaml').safe_load((base/'g40-boundary-freeze/boundary-analysis-config.yaml').read_text())
schema=json.loads((base/'g40-boundary-freeze/critic-observation-schema.json').read_text())
prior=[json.loads(line) for line in gzip.open(base/'g40-teacher-review/g40-per-candidate.jsonl.gz','rt')]
if len(prior)!=63 or schema['total_dim']!=267 or config['critic_replay']['absolute_tolerance']!=1e-5:
    raise ValueError('frozen input/config mismatch')

weights=torch.load(args.checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
dummy=TensorDict({'policy':torch.zeros(1,22),'critic':torch.zeros(1,267)},batch_size=[1])
with contextlib.redirect_stdout(sys.stderr):
    model=ActorCritic(dummy,{'policy':['policy'],'critic':['critic']},6,
                      actor_obs_normalization=False,critic_obs_normalization=False,
                      actor_hidden_dims=[1024,512,256],critic_hidden_dims=[1024,512,256],activation='elu',
                      init_noise_std=1.0,noise_std_type='scalar',state_dependent_std=False)
model.load_state_dict(weights,strict=True)
model.eval()

def critic(inputs):
    batch=torch.as_tensor(inputs,dtype=torch.float32)
    obs=TensorDict({'policy':torch.zeros((len(batch),22),dtype=torch.float32),
                    'critic':batch},batch_size=[len(batch)])
    with torch.inference_mode():return model.evaluate(obs).flatten().tolist()

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def load_rows():
    result=[]
    for row in prior:
        root=args.old if row['state_group_id'].startswith('g0e-') else args.new
        path=root/'runs'/row['process_id']/'live-0.trace.json.gz'
        if sha(path)!=row['trace_sha256']:raise ValueError(f"trace SHA differs {row['process_id']}")
        trace=json.load(gzip.open(path,'rt'))
        if len(trace)!=120:raise ValueError('trace length mismatch')
        result.append((row,trace))
    return result

data=load_rows()
idx=schema['command_field']['start'];end=schema['command_field']['end']
if (idx,end)!=(9,12):raise ValueError('command slice differed from frozen schema')
observations=[];saved=[];keys=[];command_error=[]
for row,trace in data:
    for k in (0,39,40):
        rec=trace[k]
        tensor=torch.tensor(rec['critic_observation'],dtype=torch.float32)
        command=torch.tensor(rec['command'],dtype=torch.float32)
        if tensor.shape!=(267,) or command.shape!=(3,):raise ValueError('observation/command shape')
        command_error.append(float((tensor[idx:end]-command).abs().max()))
        observations.append(tensor)
        saved.append(float(rec['critic_value']))
        keys.append((row['process_id'],k))
if max(command_error)>1e-6:raise ValueError(f'command term mismatch: {max(command_error)}')
predicted=critic(torch.stack(observations))
diff=[abs(a-b) for a,b in zip(predicted,saved)]
equivalence={'status':'pass' if max(diff)<=config['critic_replay']['absolute_tolerance'] else 'fail',
             'checked_values':len(diff),'trace_count':63,
             'maximum_absolute_difference':max(diff),
             'mean_absolute_difference':sum(diff)/len(diff),
             'tolerance':config['critic_replay']['absolute_tolerance'],
             'command_slice':[idx,end],
             'maximum_command_observation_difference':max(command_error),
             'per_value':[{'run_id':p,'step':k,'saved':s,'replayed':v,'absolute_difference':d}
                          for (p,k),s,v,d in zip(keys,saved,predicted,diff)],
             'model':'rsl_rl.ActorCritic evaluate, CPU float32, strict checkpoint state_dict'}
if args.stage=='equivalence':
    print(json.dumps(equivalence,allow_nan=False))
    raise SystemExit(0 if equivalence['status']=='pass' else 2)
if equivalence['status']!='pass':
    print(json.dumps(equivalence,allow_nan=False))
    raise SystemExit(2)

# Conditional value queries do not step the robot. Every non-command input
# is the exact recorded float32 terminal observation.
bygroup={}
for row,trace in data:
    if row['candidate_id']=='direct_reference' and row['repeat_id']==0:
        bygroup[row['state_group_id']]=trace[39]['command']
if len(bygroup)!=7:raise ValueError('direct-reference command support differs')
canonical=[config['terminal_commands'][key] for key in
           ('canonical_vx_mps','canonical_vy_mps','canonical_wz_radps')]
all_inputs=[];meta=[]
for row,trace in data:
    original=torch.tensor(trace[40]['critic_observation'],dtype=torch.float32)
    commands={'zero_recorded':trace[40]['command'],'hold_candidate':trace[39]['command'],
              'canonical':canonical,'direct_reference':bygroup[row['state_group_id']]}
    for name,cmd in commands.items():
        cmd=torch.tensor(cmd,dtype=torch.float32)
        if cmd.shape!=(3,) or not torch.isfinite(cmd).all():raise ValueError('invalid command')
        modified=original.clone();modified[idx:end]=cmd
        if not torch.equal(modified[:idx],original[:idx]) or not torch.equal(modified[end:],original[end:]):
            raise ValueError('non-command input changed')
        all_inputs.append(modified);meta.append((row,name,cmd.tolist()))
values=critic(torch.stack(all_inputs))
output=[]
for (row,name,cmd),value in zip(meta,values):
    output.append({'process_id':row['process_id'],'state_group_id':row['state_group_id'],
                   'terrain':row['terrain'],'candidate_id':row['candidate_id'],
                   'repeat_id':row['repeat_id'],'condition':name,'command':cmd,
                   'terminal_value':value,'R_gamma_40':row['R_gamma_40'],
                   'G40':row['R_gamma_40']+config['gamma']**40*value,
                   'V0':row['v0'],'A40':row['R_gamma_40']+config['gamma']**40*value-row['v0'],
                   'deltaV40':value-row['v0'],
                   'map_metrics':row['map_metrics'],'critic_drawdown':row['critic_drawdown'],
                   'body_metrics':row['body_metrics'],
                   'termination_mask':not row['body_metrics']['termination']})
print(json.dumps({'equivalence':equivalence,'terminal_values':output},allow_nan=False))
