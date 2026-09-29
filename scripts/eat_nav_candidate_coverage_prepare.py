"""Freeze all-env full-prefix actions and 13 Path inputs for selected episode-0 states."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from scripts.eat_nav_candidate_coverage_select import atomic_once,digest


def save_npz_once(path, **arrays):
    if path.exists():
        with np.load(path) as old:
            if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                raise FileExistsError(f'frozen npz differs: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    temp.replace(path)


def run(args):
    selection=json.loads(args.selection.read_text())
    schedule=json.loads(args.schedule.read_text())
    config=yaml.safe_load(args.pilot_config.read_text())
    if digest(args.selection)!=schedule['selection_sha256'] or digest(args.pilot_config)!=schedule['config_sha256']:
        raise ValueError('frozen selection/schedule/config SHA mismatch')
    selected=selection['selected']
    if len(selected)!=7 or any(r['episode_key'][2]!=0 or r['full_prefix_steps']>40 for r in selected):
        raise ValueError('pilot preparer requires seven frozen early episode-0 states')
    max_step=max(r['step'] for r in selected)+config['control_steps']
    actions={};records={};commands={i:{} for i in range(len(selected))}
    manifest=json.loads(args.manifest.read_text())
    for chunk in manifest['chunks']:
        with gzip.open(args.manifest.parent/chunk['file'],'rt') as stream:
            for line in stream:
                row=json.loads(line);step=int(row['step'])
                if step>max_step:continue
                env=int(row['env_id']);actions.setdefault(step,{})[env]=row['action']
                for i,state in enumerate(selected):
                    if env==state['episode_key'][1] and step<=state['step']:
                        if [row['session_id'],env,row['episode_id']]!=state['episode_key']:
                            raise ValueError('selected episode changed during prefix')
                        commands[i][step]=row['command']
                    if step==state['step'] and env==state['episode_key'][1]:
                        if row['pre_reset'] or [row['session_id'],env,row['episode_id']]!=state['episode_key']:
                            raise ValueError('logged anchor identity mismatch')
                        records[i]=row
        if len(records)==len(selected) and all(len(actions.get(step,{}))==100 for step in range(max_step+1)):
            break
    if len(records)!=len(selected) or not all(len(actions.get(step,{}))==100 for step in range(max_step+1)):
        raise ValueError('incomplete all-env full-prefix/suffix recording')
    all_actions=np.asarray([[actions[step][env] for env in range(100)] for step in range(max_step+1)],dtype=np.float32)
    if all_actions.shape!=(max_step+1,100,6):raise ValueError('recorded action shape mismatch')
    state_file=args.output/'selected-states.json'
    atomic_once(state_file,{'train':selected,'heldout_used':False})
    result={'selection_sha256':digest(args.selection),'schedule_sha256':digest(args.schedule),
            'manifest_sha256':digest(args.manifest),'state_file_sha256':digest(state_file),'states':{}}
    for i,state in enumerate(selected):
        folder=args.output/f'state-{i:03d}';folder.mkdir(parents=True,exist_ok=True)
        step=state['step']
        prefix=folder/'prefix-actions.npz';suffix=folder/'suffix-actions.npz'
        cmd=np.asarray([commands[i][t] for t in range(step+1)],dtype=np.float32)
        save_npz_once(prefix,actions=all_actions[:step+1],selected_env_commands=cmd)
        save_npz_once(suffix,actions=all_actions[step+1:step+1+config['control_steps']])
        anchor=folder/'logged-anchor.json';atomic_once(anchor,records[i])
        cfg={'schema_version':1,'control_steps':config['control_steps'],
             'anchor_limits':{'position_m':.02,'orientation_rad':.03,'dem_rms_m':.03,'mask_iou_min':.95},
             'source_state':{'split':'train','state_index':i,'env_index':state['episode_key'][1],
                             'episode_id':0,'anchor_step':step}}
        cfg_path=folder/'config.yaml'
        if cfg_path.exists():
            if yaml.safe_load(cfg_path.read_text())!=cfg:raise FileExistsError('state config changed')
        else:cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=True))
        offline=args.offline/f"{state['state_id']}.json"
        candidates=json.loads(offline.read_text())
        if candidates['state']!=state:raise ValueError('offline candidate/state mismatch')
        ids={r['candidate_id']:r for r in candidates['candidates']}
        names=[f'nomad-seed-{seed}' for seed in config['nomad_seeds_pilot']]
        names += [f'bank-{name}' for name in config['bank_names']]
        path_sha={}
        for name in names:
            candidate=ids[name]
            if not candidate['finite'] or 'ros_base_path' not in candidate:
                raise ValueError(f'nonfinite scheduled candidate {name}')
            path=folder/f'{name}-base-path.json'
            atomic_once(path,candidate['ros_base_path'])
            path_sha[name]=digest(path)
        result['states'][state['state_id']]={'view_index':state['view_index'],'episode_key':state['episode_key'],
            'anchor_step':step,'full_prefix_steps':step+1,'prefix_sha256':digest(prefix),
            'suffix_sha256':digest(suffix),'logged_anchor_sha256':digest(anchor),
            'config_sha256':digest(cfg_path),'offline_sha256':digest(offline),'base_path_sha256':path_sha}
    atomic_once(args.output/'input-freeze.json',result)
    atomic_once(args.output/'status.json',{'stage':'pilot_inputs_frozen','state_count':7,
                'input_freeze_sha256':digest(args.output/'input-freeze.json'),'rollout_started':False})
    print('PILOT_INPUTS_FROZEN',digest(args.output/'input-freeze.json'),flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','selection','schedule','pilot-config','offline','output'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
