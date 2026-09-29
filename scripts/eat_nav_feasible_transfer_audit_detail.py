"""Sample-level input equivalence detail without modifying frozen Evaluator results."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from visualnav_transformer.eat_nav.feasible_transfer import to_increments,to_waypoints
from eat_nav_forward_evaluator import load_split

ROOT=Path('/tmp/eatnav-candidate-coverage');OUT=ROOT/'feasible-transfer';BASE=ROOT/'baseline-worktree-8e8f4fd'

def sha_bytes(x):return hashlib.sha256(np.asarray(x,dtype=np.float32).tobytes()).hexdigest()

def run():
    selection=json.loads((ROOT/'stage0/frozen/pilot-selection.json').read_text())['selected']
    details=json.loads((OUT/'feasibility-by-candidate.json').read_text())['rows']
    ref=BASE/'docs/quality-identifiability-gate/phase-c/reference-eat.yaml'
    cfg=load_eat_config(ref)
    view=CompleteOnlyView(EATDataset(Path('/tmp/eatnav-extended/output/manifest.json'),'train',cfg,split_seed=123),8)
    phase=yaml.safe_load(Path('/tmp/eatnav-train/forward-evaluator-config.yaml').read_text())
    test=load_split('test',phase)
    held_action=np.asarray(test['action'][0],dtype=np.float32)
    held_dem=np.asarray(test['dem'][0],dtype=np.float32)
    held_goal=np.asarray(test['goal'][0],dtype=np.float32)
    checks=[]
    for state in selection:
        sid=state['state_id'];w=view.dataset.windows[view.indices[state['view_index']]]
        dem=np.asarray(w['dem'],dtype=np.float32);goal=np.asarray(w['goal'],dtype=np.float32)
        action=np.asarray(w['action'],dtype=np.float32)
        if sha_bytes(dem)!=state['dem_sha256']:raise ValueError(f'DEM mismatch: {sid}')
        selected=[r for r in details if r['state_id']==sid]
        converted={}
        for variant,key in [('P','planned_path'),('C','command_derived_path'),('R','realized_path')]:
            paths=np.asarray([r[key] for r in selected],dtype=np.float32)
            reconstruction=np.stack([to_waypoints(to_increments(path)) for path in paths])
            converted[variant]={'shape':list(paths.shape),'increment_shape':[13,8,2],
                'increment_cumulative_roundtrip_max_abs_m':float(np.max(np.abs(reconstruction-paths))),
                'endpoint_distance_to_fixed_goal_m':[float(np.linalg.norm(path[-1]-goal)) for path in paths]}
        checks.append({'state_id':sid,'terrain':state['terrain_type'],
            'candidate_ids':[r['candidate_id'] for r in selected],
            'generators':[r['generator'] for r in selected],
            'seeds':[r['seed'] for r in selected],
            'dem_sha256':sha_bytes(dem),'dem_shape':list(dem.shape),
            'mask_fraction':float((dem[:,1]>.5).mean()),
            'goal':goal.tolist(),'selected_goal_max_abs_diff_m':float(np.max(np.abs(goal-state['goal']))),
            'logged_teacher_increment_shape':list(action.shape),
            'logged_teacher_increment_cumulative_roundtrip_max_abs_m':float(np.max(np.abs(to_increments(to_waypoints(action))-action))),
            'logged_teacher_endpoint_distance_to_goal_m':float(np.linalg.norm(to_waypoints(action)[-1]-goal)),
            'candidate_conversion':converted})
    result={'source':'frozen Phase B split and complete-only train pilot view',
        'phase_b_heldout_example':{'index':int(test['index'][0]),'dem_sha256':sha_bytes(held_dem),
            'dem_shape':list(held_dem.shape),'mask_fraction':float((held_dem[:,1]>.5).mean()),
            'goal':held_goal.tolist(),'action_increment_shape':list(held_action.shape),
            'increment_cumulative_roundtrip_max_abs_m':float(np.max(np.abs(to_increments(to_waypoints(held_action))-held_action))),
            'endpoint_distance_to_goal_m':float(np.linalg.norm(to_waypoints(held_action)[-1]-held_goal))},
        'pilot_states':checks,'comparison':'same EATDataset window representation; P/C/R differ in semantic source, not coordinate/unit conversion',
        'validation_used_for_selection':False,'heldout_used_for_selection':False}
    path=OUT/'evaluator-input-audit-detail.json';content=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=content:raise FileExistsError('audit detail differs')
    else:
        tmp=path.with_name(path.name+'.tmp');tmp.write_text(content);tmp.replace(path)
    print('FEASIBLE_TRANSFER_AUDIT_DETAIL_COMPLETE',len(checks),flush=True)

if __name__=='__main__':run()
