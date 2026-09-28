"""Split distributions, confounding audits, and train-thresholded held-out pairs."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from visualnav_transformer.eat_nav.quality_gate_evaluation import pair_cluster_ci
from visualnav_transformer.eat_nav.quality_gate_data import trajectory_shape
from visualnav_transformer.eat_nav.relative_quality_matching import geometric_candidates,label_thresholds,select_disjoint_pairs,episode_id_list

ROOT=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality')
OUT=ROOT/'output'
SPEC=Path('/tmp/eatnav-train/relative-quality-config.yaml')
METRICS=('quality_abs','v0','delta_v','drawdown','recovery','return_h','advantage_h','critic_slope_per_sec')
PAIRED=('quality_abs','delta_v','drawdown','recovery','return_h','advantage_h')


def spec():
    import yaml
    return yaml.safe_load(SPEC.read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)
def records():
    result=defaultdict(list)
    with gzip.open(OUT/'relative-quality-values.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);result[row['split']].append(row)
    return result

def stats(a):
    x=np.asarray(a,dtype=float)
    if not len(x):return {'status':'insufficient support','n':0,'mean':None,'median':None,'p10':None,'p90':None}
    return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
            'p10':float(np.quantile(x,.1)),'p90':float(np.quantile(x,.9))}
def corr(a,b):
    x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float)
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    z=float(spearmanr(x,y).statistic)
    return z if np.isfinite(z) else None

def eta_squared(values,groups):
    y=np.asarray(values,dtype=float);g=np.asarray(groups)
    if len(y)<2 or np.var(y)==0:return None
    avg=y.mean();between=sum((g==key).sum()*(y[g==key].mean()-avg)**2 for key in np.unique(g))
    return float(between/np.sum((y-avg)**2))

def proxy_fit(train,heldout,metric):
    """Train-only two-variable linear proxy for goal distance and DEM roughness."""
    def xy(rows):
        return np.column_stack((np.ones(len(rows)),[r['goal_distance_m'] for r in rows],
                                [r['dem_height_std_m'] for r in rows]))
    xt=xy(train);yt=np.asarray([r[metric] for r in train]);beta=np.linalg.lstsq(xt,yt,rcond=None)[0]
    x=xy(heldout);y=np.asarray([r[metric] for r in heldout]);pred=x@beta
    denom=np.sum((y-y.mean())**2)
    return {'heldout_r2':float(1-np.sum((y-pred)**2)/denom) if denom>0 else None,
            'train_coefficients':beta.tolist(),'features':['intercept','goal_distance_m','dem_height_std_m']}

def distributions():
    data=records();result={};correlations={}
    for split in ('train','val','test'):
        source=data[split];rows=[r for r in source if r['primary_available']]
        result[split]={'complete':len(source),'primary':len(rows),'by_metric':{},'by_terrain':{},'by_q_bin':{},
                       'by_terminal_reason':{},'terminal_episode_counts':dict(Counter('+'.join(r['episode_terminal_reason']) or 'none'
                                          for r in {tuple(x['episode_key']):x for x in source}.values()))}
        correlations[split]={}
        covariates={'goal_distance_m':'goal_distance_m','dem_height_std_m':'dem_height_std_m',
                    'path_length_m':'path_length_m','forward_progress_m':'forward_progress_m',
                    'turning_rad':'turning_rad','terminal_seconds':'terminal_seconds',
                    'quality_p10_raw':'quality_p10_raw','quality_abs':'quality_abs'}
        for name in METRICS:
            result[split]['by_metric'][name]=stats([r[name] for r in rows])
            correlations[split][name]={key:corr([r[name] for r in rows],[r[field] for r in rows])
                                       for key,field in covariates.items() if key!=name}
            correlations[split][name]['terrain_eta_squared']=eta_squared([r[name] for r in rows],[r['terrain_type'] for r in rows])
        for dimension,keyfun,target in [('by_terrain',lambda r:r['terrain_type'],None),
                                        ('by_q_bin',lambda r:str(min(3,int(r['quality_abs']*4))),None),
                                        ('by_terminal_reason',lambda r:'+'.join(r['episode_terminal_reason']) or 'none',None)]:
            for key in sorted({keyfun(r) for r in rows}):
                subset=[r for r in rows if keyfun(r)==key]
                result[split][dimension][key]={'n':len(subset),'episodes':len({tuple(r['episode_key']) for r in subset}),
                    'metrics':{name:stats([r[name] for r in subset]) for name in METRICS}}
    result['train_only_scalers']=json.loads((OUT/'relative-quality-extraction.json').read_text())['train_only_scalers']
    result['reward_decomposition']='not recorded; total reward only'
    train=[r for r in data['train'] if r['primary_available']]
    held=[r for r in data['test'] if r['primary_available']]
    correlations['goal_dem_proxy_heldout']={name:proxy_fit(train,held,name) for name in METRICS}
    write(OUT/'relative-quality-summary.json',result)
    write(OUT/'relative-quality-correlations.json',correlations)
    print('RELATIVE_DISTRIBUTIONS',json.dumps({s:result[s]['primary'] for s in ('train','val','test')}),flush=True)


def load(split):return np.load(ROOT/'splits'/f'{split}.npz')
def label_array(data,name):return np.asarray(data['quality_abs'] if name=='quality_abs' else data['z_'+name])
def pair_geometry(data,i,j):
    ai=np.asarray(data['action'][i]);aj=np.asarray(data['action'][j]);xi,fi,li,ti=trajectory_shape(ai);xj,fj,lj,tj=trajectory_shape(aj)
    return {'trajectory_rms_m':float(np.sqrt(np.mean(np.sum((xi-xj)**2,axis=1)))),
            'endpoint_m':float(np.linalg.norm(xi[-1]-xj[-1])),
            'forward_progress_m':abs(fi-fj),'path_length_m':abs(li-lj),'turning_rad':abs(ti-tj)}

def paired():
    s=spec();train=load('train');test=load('test')
    kwargs={'goal_distance_m':s['match_goal_distance_m'],'goal_direction_deg':s['match_goal_direction_deg'],
            'min_mask_iou':s['match_min_mask_iou'],'max_dem_rmse_m':s['match_max_dem_rmse_m']}
    tc=geometric_candidates(train,seed=s['match_seed'],limit=s['match_train_candidate_cap'],**kwargs)
    if not tc:raise ValueError('no train geometric candidates')
    threshold={name:label_thresholds([abs(float(label_array(train,name)[i]-label_array(train,name)[j]))
                                       for i,j,*_ in tc],s['match_small_difference_train_quantile'],
                                       s['match_large_difference_train_quantile']) for name in PAIRED}
    threshold_doc={'source_split':'train','source_candidate_count':len(tc),'thresholds':threshold,
                   'geometry':kwargs,'seed':s['match_seed'],'heldout_used_for_threshold':False}
    write(OUT/'relative-quality-match-thresholds.json',threshold_doc)
    hc=geometric_candidates(test,seed=s['match_seed']+1,limit=s['match_heldout_candidate_cap'],**kwargs)
    episodes=[tuple(row) for row in test['episode']];results={}
    for name in PAIRED:
        a=label_array(test,name);t=threshold[name]
        cross=[(i,j,abs(float(a[i]-a[j])),g,angle,iou,rmse) for i,j,g,angle,iou,rmse in hc if abs(float(a[i]-a[j]))>=t['large']]
        same=[(i,j,abs(float(a[i]-a[j])),g,angle,iou,rmse) for i,j,g,angle,iou,rmse in hc if abs(float(a[i]-a[j]))<=t['small']]
        selected_cross=select_disjoint_pairs(cross,episodes,max_pairs=s['match_pairs_per_kind'],episode_cap=s['match_episode_cap'],descending=True)
        used={i for row in selected_cross for i in row[:2]}
        same=[row for row in same if row[0] not in used and row[1] not in used]
        selected_same=select_disjoint_pairs(same,episodes,max_pairs=s['match_pairs_per_kind'],episode_cap=s['match_episode_cap'],descending=False)
        output={};for_ci={}
        for label,selected in [('cross',selected_cross),('same',selected_same)]:
            rows=[]
            for i,j,d,g,angle,iou,rmse in selected:
                row={'left_index':int(test['index'][i]),'right_index':int(test['index'][j]),
                     'left_episode':episode_id_list(episodes[i]),'right_episode':episode_id_list(episodes[j]),
                     'left_terrain':str(test['terrain'][i]),'label_difference':d,
                     'goal_difference_m':g,'goal_angle_deg':angle,'mask_iou':iou,'dem_rmse_m':rmse,
                     **pair_geometry(test,i,j)}
                rows.append(row)
            output[label]={'count':len(rows),'pairs':rows}
            for_ci[label]=rows
        comparison={}
        for metric in ('trajectory_rms_m','endpoint_m','forward_progress_m','path_length_m','turning_rad'):
            cr=for_ci['cross'];sr=for_ci['same']
            if len(cr)>=10 and len(sr)>=10:
                comparison[metric]=pair_cluster_ci([r[metric] for r in cr],[(tuple(r['left_episode']),tuple(r['right_episode'])) for r in cr],
                    [r[metric] for r in sr],[(tuple(r['left_episode']),tuple(r['right_episode'])) for r in sr],
                    seed=s['bootstrap_seed'],repeats=s['bootstrap_repeats'])
            else:comparison[metric]={'status':'insufficient support','cross_count':len(cr),'same_count':len(sr)}
        results[name]={'threshold':t,'cross':output['cross'],'same':output['same'],'cross_minus_same':comparison}
        print('RELATIVE_PAIRS',name,len(selected_cross),len(selected_same),flush=True)
    write(OUT/'relative-quality-matched-pairs.json',{'source_split':'test','train_threshold_file':'relative-quality-match-thresholds.json',
           'candidate_counts':{'train':len(tc),'test':len(hc)},'metrics':results})


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['distributions','pairs'],required=True)
    a=p.parse_args();{'distributions':distributions,'pairs':paired}[a.stage]()
if __name__=='__main__':main()
