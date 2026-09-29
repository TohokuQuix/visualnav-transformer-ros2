"""Freeze outcome-blind train pilot states under an approved full-prefix budget."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.candidate_coverage_selection import freeze_pilot_selection
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView

EXPECTED_SHA = {
    'manifest':'55d9e2d89631c2afa19dabfb02b2753616bc2b71907843f5570bd3a8e41e1475',
    'reference_config':'caa88735e765fc0358930dfa41579ecaac9eb96dc4052dc01a81390377279992',
    'reference_checkpoint':'191d6a03266bca2ce8d43afac44b04828ae0d2571632be4f47a5f0dc3564b4e0',
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def atomic_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=data:raise FileExistsError(f'frozen artifact differs: {path}')
        return
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def run(args):
    if args.approved_budget_hours is None or args.max_full_prefix_steps is None:
        raise ValueError('approved runtime budget and full-prefix cap must be fixed before selection')
    inputs={'manifest':args.manifest,'reference_config':args.reference_config,
            'reference_checkpoint':args.reference_checkpoint}
    sha={name:digest(path) for name,path in inputs.items()}
    if sha!=EXPECTED_SHA:raise ValueError(f'input SHA mismatch: {sha}')
    cfg=load_eat_config(args.reference_config)
    if cfg.context_size!=3 or cfg.len_traj_pred!=8:raise ValueError('reference checkpoint input shape mismatch')
    view=CompleteOnlyView(EATDataset(args.manifest,'train',cfg,split_seed=123),cfg.len_traj_pred)
    if len(view)!=49598:raise ValueError('complete-only train count mismatch')
    rows=[]
    for vi,di in enumerate(view.indices):
        w=view.dataset.windows[di];dem=np.asarray(w['dem'],dtype=np.float32)
        if dem.shape[0]!=4 or not np.asarray(w['action_mask'],dtype=bool).all():
            raise ValueError('natural history or complete future missing')
        rows.append({'split':'train','complete':True,'natural_history':True,
            'terrain_type':w['terrain_type'],'episode_key':list(w['episode_key']),
            'view_index':vi,'step':int(w['step']),'local_step':int(w['local_step']),
            'terrain_seed':int(w['terrain_seed']),'goal':np.asarray(w['goal'],dtype=float).tolist(),
            'dem_sha256':hashlib.sha256(dem.tobytes()).hexdigest()})
    result=freeze_pilot_selection(rows,terrains=sorted({r['terrain_type'] for r in rows}),
        seed=20260930,min_goal_m=.30,max_full_prefix_steps=args.max_full_prefix_steps,
        approved_budget_sec=args.approved_budget_hours*3600/1.2,run_count_per_state=17,
        cost_intercept_sec=46.10692857959084,cost_slope_sec=.8296659776629849)
    result.update({'source_sha256':sha,'complete_train_windows':len(view),'dem_history_frames':4,
        'heldout_used':False,'candidate_outcomes_used':False,
        'approved_budget_hours':args.approved_budget_hours,
        'predicted_with_20pct_reserve_sec':result['predicted_pilot_sec']*1.2})
    target=args.output/'pilot-selection.json'
    atomic_once(target,result)
    atomic_once(args.output/'status.json',{'stage':'pilot_selection_frozen',
        'selected_count':len(result['selected']),'na_terrains':result['na_terrains'],
        'rollout_started':False,'pilot_selection_sha256':digest(target)})
    print('PILOT_SELECTION_FROZEN',[(r['terrain_type'],r['view_index'],r['full_prefix_steps'])
                                    for r in result['selected']],flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--reference-config',type=Path,required=True)
    p.add_argument('--reference-checkpoint',type=Path,required=True)
    p.add_argument('--approved-budget-hours',type=float,required=True)
    p.add_argument('--max-full-prefix-steps',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
