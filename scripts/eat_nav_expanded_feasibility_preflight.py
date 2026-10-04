"""Outcome-blind Stage X0/X1 preparation for the expanded E1 feasibility pilot."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView

CONFIG=Path(__file__).resolve().parents[1]/'docs/quality-identifiability-gate/expanded-feasibility/expanded-pilot-config.yaml'
EXPECTED={'manifest':'55d9e2d89631c2afa19dabfb02b2753616bc2b71907843f5570bd3a8e41e1475',
    'reference_config':'caa88735e765fc0358930dfa41579ecaac9eb96dc4052dc01a81390377279992',
    'complete_ids':'904b22816531c4bf026181de5b70aaad386f0557ea44d9266223c89d48cddf4e',
    'g0e_candidates':'aa0ec83560db08af59e5b27b9d5dcd6cbb0fc9311b08ecb1db838f5de37e4aed'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def rank_candidate(row,salt):
    fields=[list(map(int,row['episode_key'])),int(row['global_step']),
            int(row['terrain_seed']),str(row['terrain'])]
    return hashlib.sha256((salt+'\n'+json.dumps(fields,separators=(',',':'))).encode()).hexdigest()


def select_states(rows,*,salt,terrains,excluded_env_ids):
    selected=[];used_env=set(excluded_env_ids);used_episode=set();used_seed=set()
    for terrain in terrains:
        eligible=[r for r in rows if r['terrain']==terrain and r['source_episode']==0
            and r['episode_key'][2]==0 and r['global_step']==r['local_step']
            and r['prefix_steps']<=80 and r['env_id'] not in used_env
            and tuple(r['episode_key']) not in used_episode and r['terrain_seed'] not in used_seed
            and not any(reason!='previous_pilot_anchor_excluded' for reason in r.get('exclusion_reasons',[]))]
        if not eligible:raise ValueError('insufficient eligible support for '+terrain)
        chosen=min(eligible,key=lambda r:(rank_candidate(r,salt),r['source_window_id']))
        selected.append({k:v for k,v in chosen.items() if k not in ('critic_value','reward','termination','stack','tilt')})
        selected[-1]['selection_rank']=rank_candidate(chosen,salt)
        used_env.add(chosen['env_id']);used_episode.add(tuple(chosen['episode_key']));used_seed.add(chosen['terrain_seed'])
    return selected


def forecast_runtime(measured_total_sec,measured_runs,new_runs,*,margin,cap_sec):
    if measured_total_sec<=0 or measured_runs<=0 or new_runs<=0 or margin<1:
        raise ValueError('positive measured runtime and margin required')
    point=measured_total_sec*new_runs/measured_runs
    extended=point*margin
    return {'measured_total_sec':measured_total_sec,'measured_runs':measured_runs,
        'new_runs':new_runs,'point_sec':point,'with_margin_sec':extended,
        'cap_sec':cap_sec,'status':'pass' if extended<=cap_sec else 'stop'}


def complete_window_lookup(cfg,selected):
    dataset=EATDataset(cfg['source_manifest'],'train',load_eat_config(cfg['source_reference_config']),split_seed=123)
    view=CompleteOnlyView(dataset,8)
    if len(view)!=49598:raise ValueError('complete-only train count differs')
    split_hash=hashlib.sha256(json.dumps(dataset.split_map,sort_keys=True).encode()).hexdigest()
    if split_hash!='ae584ee5813be6a09f16baa4fae8271ed20c457ef40d9dd44005636a71e9234f':
        raise ValueError('split SHA differs')
    wanted={(tuple(r['episode_key']),r['global_step']):r for r in selected}
    found={}
    for i in view.indices:
        w=dataset.windows[i];key=(tuple(w['episode_key']),int(w['step']))
        if key not in wanted:continue
        row=wanted[key];dem=np.asarray(w['dem'],dtype=np.float32)
        if (w['terrain_type']!=row['terrain'] or w['terrain_seed']!=row['terrain_seed']
            or int(w['local_step'])!=row['local_step'] or dem.shape!=(4,2,20,20)
            or np.asarray(w['action']).shape!=(8,2) or not np.asarray(w['action_mask']).all()
            or not np.isfinite(dem).all() or not np.isfinite(w['goal']).all()
            or not np.isfinite(w['action']).all()
            or not np.logical_or(dem[:,1]==0,dem[:,1]==1).all()):
            raise ValueError('selected complete-only window invalid '+str(key))
        found[key]=w
    if set(found)!=set(wanted):raise ValueError('selected window missing from complete-only train')
    return found,split_hash


def main():
    cfg=yaml.safe_load(CONFIG.read_text())
    output=Path(cfg['output_root'])
    if output.exists():raise FileExistsError('expanded pilot output root already exists')
    inputs={'manifest':cfg['source_manifest'],'reference_config':cfg['source_reference_config'],
        'complete_ids':cfg['source_complete_ids'],'g0e_candidates':cfg['source_g0e_candidate_list']}
    digests={name:sha(path) for name,path in inputs.items()}
    if digests!=EXPECTED:raise ValueError('source SHA differs '+str(digests))
    old=Path(cfg['source_e1_root']);addendum=Path(cfg['source_coordinate_addendum'])
    if json.loads((addendum/'original-vs-corrected-decision.json').read_text())['original_frozen_gate']!='inconclusive':
        raise ValueError('corrected addendum decision differs')
    with gzip.open(cfg['source_g0e_candidate_list'],'rt') as stream:
        rows=[json.loads(line) for line in stream]
    if len(rows)!=49598:raise ValueError('candidate complete-only count differs')
    old_states=json.loads((old/'inputs/selected-states.json').read_text())['train']
    excluded={s['env_id'] for s in old_states}
    if excluded!=set(cfg['selection']['exclude_existing_env_ids']):raise ValueError('old E1 env identity differs')
    selected=select_states(rows,salt=cfg['selection']['salt'],terrains=cfg['selection']['terrains'],
        excluded_env_ids=excluded)
    windows,split_hash=complete_window_lookup(cfg,selected)
    output.mkdir(parents=True)
    candidate_path=output/'state-selection-candidates.jsonl.gz'
    with candidate_path.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
        for row in rows:
            item={k:v for k,v in row.items() if k not in ('selection_rank',)}
            item['expanded_selection_rank']=rank_candidate(row,cfg['selection']['salt'])
            stream.write((json.dumps(item,sort_keys=True,allow_nan=False)+'\n').encode())
    prepared=[];arrays=[]
    for i,row in enumerate(selected):
        w=windows[(tuple(row['episode_key']),row['global_step'])]
        dem=np.asarray(w['dem'],dtype=np.float32)
        teacher=np.asarray(w['action'],dtype=np.float32)
        prepared.append({**row,'state_id':f'expanded-train-{i:03d}','state_index':i,'split':'train',
            'step':row['global_step'],'terrain_type':row['terrain'],
            'logged_goal':np.asarray(w['goal'],dtype=float).tolist(),
            'dem_sha256':hashlib.sha256(dem.tobytes()).hexdigest(),
            'teacher_sha256':hashlib.sha256(teacher.tobytes()).hexdigest(),
            'dem_history_identity':[[row['episode_key'],s] for s in
                (row['global_step']-15,row['global_step']-10,row['global_step']-5,row['global_step'])]})
        arrays.append((dem,np.asarray(w['goal'],dtype=np.float32),teacher))
    write_once(output/'selected-states.json',{'train':prepared,'heldout_used':False,'outcomes_used':False,
        'complete_only_train_count':49598,'split_hash':split_hash})
    with (output/'selected-states.npz').open('wb') as stream:
        np.savez_compressed(stream,dem=np.stack([x[0] for x in arrays]),
            goal=np.stack([x[1] for x in arrays]),action=np.stack([x[2] for x in arrays]))
    provenance={'stage':'X0_X1_selected_no_rollout','config_sha256':sha(CONFIG),
        'source_input_sha256':digests,'split_hash':split_hash,
        'candidate_selection_sha256':sha(candidate_path),
        'selected_states_sha256':sha(output/'selected-states.json'),
        'selected_arrays_sha256':sha(output/'selected-states.npz'),
        'e1_source_commit':json.loads((old/'source-freeze.json').read_text())['HEAD'],
        'e1_source_freeze_sha256':sha(old/'source-freeze.json'),
        'e1_artifact_ledger_sha256':sha(old/'artifact-sha-ledger.json'),
        'coordinate_addendum_analysis_source_sha256':sha(addendum/'source-provenance.json'),
        'coordinate_addendum_decision_sha256':sha(addendum/'original-vs-corrected-decision.json'),
        'old_18_run_shas':json.loads((addendum/'source-provenance.json').read_text())['run_artifacts'],
        'source_selection_rows':len(rows),'terrain_eligible_counts':dict(Counter(r['terrain'] for r in rows
            if r['source_episode']==0 and r['prefix_steps']<=80 and
            not any(x!='previous_pilot_anchor_excluded' for x in r['exclusion_reasons']))),
        'no_new_candidate_outcomes_seen':True}
    write_once(output/'source-provenance.json',provenance)
    write_once(output/'status.json',{'stage':'X1_SELECTED_NO_ROLLOUT','selected':len(prepared),'candidate_outcomes':0})
    print('X1_SELECTED',[(r['terrain'],r['env_id'],r['step'],r['terrain_seed']) for r in prepared],flush=True)


if __name__=='__main__':main()
