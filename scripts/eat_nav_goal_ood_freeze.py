"""Freeze outcome-blind 64 states, 13 goals/state and eight paired seeds."""
from __future__ import annotations
import gzip,hashlib,json,math,time
import numpy as np,yaml
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from eat_nav_goal_ood_prepare import DOC,RAW,REF,MANIFEST,CHECKPOINT,sha,read,write,classify

def choose_states(view,count,seed,clearance):
    maxstep={}
    for w in view.dataset.windows:
        key=tuple(w['episode_key']);maxstep[key]=max(maxstep.get(key,0),int(w['local_step']))
    options=[]
    for vi,di in enumerate(view.indices):
        w=view.dataset.windows[di];key=tuple(w['episode_key'])
        if int(w['local_step'])<20 or maxstep[key]-int(w['local_step'])<clearance or w['termination_reason']:
            continue
        if not np.isfinite(w['dem']).all() or not np.isfinite(w['goal']).all() or not np.isfinite(w['action']).all():continue
        options.append((vi,w))
    rng=np.random.default_rng(seed);ordered=rng.permutation(len(options))
    byterrain={}
    for index in ordered:
        vi,w=options[int(index)];byterrain.setdefault(w['terrain_type'],[]).append((vi,w))
    terrains=sorted(byterrain);chosen=[];episodes=set();used=set()
    while len(chosen)<count:
        progress=False
        for terrain in terrains:
            pool=byterrain[terrain]
            match=next(((vi,w) for vi,w in pool if vi not in used and tuple(w['episode_key']) not in episodes),None)
            if match is None:continue
            vi,w=match;chosen.append((vi,w));used.add(vi);episodes.add(tuple(w['episode_key']));progress=True
            if len(chosen)==count:break
        if not progress:break
    if len(chosen)<count:
        raise ValueError(f'only {len(chosen)} independent-episode states eligible of {count}')
    return chosen

def geometry_screen(goal,dem,setting):
    # Offline map support test only. It does not establish physical feasibility.
    g=np.asarray(goal);height,mask=np.asarray(dem)[-1];h,w=height.shape;resolution=.1
    points=np.linspace([0.,0.],g,9)
    ix=np.rint(points[:,0]/resolution+(w-1)/2).astype(int);iy=np.rint(points[:,1]/resolution+(h-1)/2).astype(int)
    inside=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
    reasons=[]
    if not inside.all():reasons.append('outside_dem')
    if inside.any():
        observed=mask[iy[inside],ix[inside]]>.5
        if not observed.all():reasons.append('unknown_dem')
        hh=height[iy[inside],ix[inside]]
        if len(hh)>1 and np.max(np.abs(np.diff(hh)))>setting['max_local_height_jump_m']:reasons.append('height_jump')
    gx,gy=int(np.rint(g[0]/resolution+(w-1)/2)),int(np.rint(g[1]/resolution+(h-1)/2))
    cells=int(math.ceil(setting['goal_footprint_radius_m']/resolution))
    footprint=[]
    for di in range(-cells,cells+1):
        for dj in range(-cells,cells+1):
            if math.hypot(di,dj)*resolution>setting['goal_footprint_radius_m']+1e-9:continue
            x,y=gx+di,gy+dj
            footprint.append(0<=x<w and 0<=y<h and bool(mask[y,x]>.5) if 0<=x<w and 0<=y<h else False)
    if not all(footprint):reasons.append('footprint_unknown_or_outside')
    return {'geometry_screen_valid':not reasons,'reasons':reasons,'observed_fraction':float(np.mean(footprint)) if footprint else 0.,
            'physically_feasible':None}

def main():
    started=time.monotonic();g1=read(RAW/'g1.done.json')
    if sha(DOC/'source-provenance.json')!=g1['source_sha256'] or sha(DOC/'goal-support-thresholds.json')!=g1['support_sha256'] or sha(RAW/'train-derived.npz')!=g1['train_derived_sha256']:
        raise ValueError('G1 SHA drift')
    setting=yaml.safe_load((DOC/'goal-ood-config.yaml').read_text());config_sha=sha(DOC/'goal-ood-config.yaml')
    train_derived=np.load(RAW/'train-derived.npz');train_goals=train_derived['goals'];support=read(DOC/'goal-support-thresholds.json')
    fixed=[]
    for slot in setting['goal_slots']:
        a=math.radians(slot['angle_deg']);goal=[float(slot['radius_m']*math.cos(a)),float(slot['radius_m']*math.sin(a))]
        label,distance=classify([goal],train_goals,support)
        if label[0]!=slot['class']:raise ValueError('predeclared goal support differs '+slot['id']+' '+label[0])
        fixed.append({**slot,'goal_xy':goal,'knn_distance_m':distance[0]})
    if len(fixed)!=12 or {x['class'] for x in fixed}!={'ID','near_ood','far_ood'}:raise ValueError('goal class support')
    cfg=load_eat_config(REF/'reference-eat.yaml')
    arrays=[];states=[]
    for split,count,offset in [('train',setting['train_state_count'],0),('val',setting['validation_state_count'],1000)]:
        view=CompleteOnlyView(EATDataset(MANIFEST,split,cfg,split_seed=123),8)
        chosen=choose_states(view,count,setting['selection_seed']+offset,setting['state_future_clearance_steps'])
        for vi,w in chosen:
            dem=np.asarray(w['dem'],dtype=np.float32);goal=np.asarray(w['goal'],dtype=np.float32)
            sid=f'{split}-{len(states):03d}'
            states.append({'state_id':sid,'split':split,'view_index':vi,'episode_key':list(w['episode_key']),
                'terrain':w['terrain_type'],'terrain_seed':int(w['terrain_seed']),'local_step':int(w['local_step']),
                'step':int(w['step']),'dem_sha256':hashlib.sha256(dem.tobytes()).hexdigest(),
                'logged_goal':goal.tolist(),'action_sha256':hashlib.sha256(np.asarray(w['action'],dtype=np.float32).tobytes()).hexdigest()})
            arrays.append((dem,goal,np.asarray(w['action'],dtype=np.float32)))
    if len(states)!=64:raise ValueError('64 state support')
    selection={'source_config_sha256':config_sha,'manifest_sha256':sha(MANIFEST),'checkpoint_sha256':sha(CHECKPOINT),
        'complete_ids_sha256':sha(REF/'complete-window-ids.jsonl.gz'),'train_support_sha256':sha(DOC/'goal-support-thresholds.json'),
        'selection_outcome_blind':True,'state_count':64,'states':states}
    write(DOC/'state-selection.json',selection)
    np.savez_compressed(RAW/'selected-states.npz',dem=np.stack([x[0] for x in arrays]),logged_goals=np.stack([x[1] for x in arrays]),actions=np.stack([x[2] for x in arrays]))
    inputs=[];identities=[]
    for idx,state in enumerate(states):
        goals=[{'id':'logged','class':'logged','goal_xy':state['logged_goal']}]+fixed
        for order,goal in enumerate(goals):
            label,dist=classify([goal['goal_xy']],train_goals,support)
            geometry=geometry_screen(goal['goal_xy'],arrays[idx][0],setting)
            row={'state_id':state['state_id'],'split':state['split'],'terrain':state['terrain'],'goal_id':goal['id'],
                'goal_order':order,'goal_xy':goal['goal_xy'],'declared_class':goal['class'],'actual_train_support_class':label[0],
                'knn_distance_m':dist[0],**geometry,'dem_sha256':state['dem_sha256']}
            inputs.append(row)
            for seed in setting['diffusion_seeds']:
                identities.append({**row,'seed':int(seed),'identity':f"{state['state_id']}:{goal['id']}:{seed}"})
    if len(inputs)!=832 or len(identities)!=6656:raise ValueError('sweep size mismatch')
    write(DOC/'goal-sweep-inputs.json',{'config_sha256':config_sha,'state_selection_sha256':sha(DOC/'state-selection.json'),
        'selected_states_npz_sha256':sha(RAW/'selected-states.npz'),'support_sha256':sha(DOC/'goal-support-thresholds.json'),
        'seeds':setting['diffusion_seeds'],'input_count':len(inputs),'path_count':len(identities),'inputs':inputs})
    content=''.join(json.dumps(x,sort_keys=True,allow_nan=False)+'\n' for x in identities).encode();path=DOC/'goal-sweep-identities.jsonl.gz'
    if path.exists():
        with gzip.open(path,'rb') as f:
            if f.read()!=content:raise FileExistsError('identity changed')
    else:
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(content)
        tmp.replace(path)
    write(RAW/'g2.done.json',{'stage':'G2_FROZEN','config_sha256':config_sha,'state_sha256':sha(DOC/'state-selection.json'),
        'inputs_sha256':sha(DOC/'goal-sweep-inputs.json'),'identities_sha256':sha(path),'elapsed_sec':time.monotonic()-started})
    write(RAW/'status.json',{'stage':'G2_FROZEN','elapsed_sec':time.monotonic()-started},mutable=True)
    print('G2_FROZEN',len(states),len(inputs),len(identities),flush=True)
if __name__=='__main__':main()
