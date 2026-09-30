"""State-unit offline goal adherence, sensitivity, OOD, and diversity analysis."""
from __future__ import annotations
import collections,gzip,json,math,time
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from visualnav_transformer.eat_nav.candidate_coverage_offline import path_metrics,candidate_constraints
from visualnav_transformer.eat_nav.feasible_transfer import cluster_labels
from eat_nav_goal_ood_prepare import DOC,RAW,REPO,sha,read,write

def distribution(values):
    x=np.asarray(values,dtype=float)
    return {'n':len(x),'median':float(np.median(x)) if len(x) else None,
        'p75':float(np.quantile(x,.75)) if len(x) else None,'p95':float(np.quantile(x,.95)) if len(x) else None}
def rho(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    out=float(spearmanr(x,y).statistic)
    return out if np.isfinite(out) else None
def path_rms(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def mode_count(paths,threshold=.05):
    if len(paths)<2:return 1
    return len(set(cluster_labels(np.asarray(paths),threshold=threshold)))
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi

def main():
    started=time.monotonic();g2=read(RAW/'g2.done.json')
    source=read(DOC/'goal-sweep-inputs.json')
    if sha(DOC/'goal-sweep-inputs.json')!=g2['inputs_sha256'] or sha(RAW/'selected-states.npz')!=source['selected_states_npz_sha256']:
        raise ValueError('G2 input drift')
    status=read(RAW/'status.json')
    if status['stage']!='G3_RAW_COMPLETE' or status['completed_states']!=64:raise ValueError('G3 raw incomplete')
    arrays=np.load(RAW/'selected-states.npz');states=read(DOC/'state-selection.json')['states']
    train=np.load(RAW/'train-derived.npz');stats=read(REPO/'docs/quality-identifiability-gate/phase-c/reference-data-audit.json')
    mean=np.asarray(stats['train_action_mean']);std=np.asarray(stats['train_action_std']);train_norm=((train['actions']-mean)/std).reshape(len(train['actions']),-1)
    tree=cKDTree(train_norm);limits=read('/tmp/eatnav-candidate-coverage/stage1/train-constraint-limits.json')
    ood=read(REPO/'docs/quality-identifiability-gate/phase-c-remediation/candidate-thresholds.json')
    if limits['source_split']!='complete_only_train' or ood['source_split']!='complete_only_train':raise ValueError('trajectory thresholds not train-only')
    if not np.allclose(mean,ood['train_action_mean'],atol=1e-8) or not np.allclose(std,ood['train_action_std'],atol=1e-8):raise ValueError('normalization drift')
    identities=[];paths=[];runtimes=[]
    for index,state in enumerate(states):
        file=RAW/'inference'/f"{state['state_id']}.npz";marker=read(RAW/'inference'/f"{state['state_id']}.done.json")
        if sha(file)!=marker['output_sha256'] or marker['input_sha256']!=sha(DOC/'goal-sweep-inputs.json') or marker['path_count']!=104:
            raise ValueError('inference marker/SHA mismatch '+state['state_id'])
        data=np.load(file);p=data['paths'];goal=data['goals']
        if p.shape!=(13,8,8,2) or goal.shape!=(13,2) or list(data['seeds'])!=source['seeds']:
            raise ValueError('output shape/seed drift')
        for j in range(13):
            row=source['inputs'][index*13+j]
            if row['state_id']!=state['state_id'] or not np.allclose(goal[j],row['goal_xy'],atol=1e-7):raise ValueError('output goal identity drift')
        identities.append((state,p,goal));paths.append(p);runtimes.append(marker)
    paths=np.asarray(paths);flat=paths.reshape(-1,8,2)
    delta=np.diff(np.concatenate((np.zeros((len(flat),1,2),dtype=flat.dtype),flat),axis=1),axis=1)
    normalized=((delta-mean)/std).reshape(len(flat),-1)
    nearest=tree.query(normalized,k=1,workers=-1)[0]/math.sqrt(normalized.shape[1])
    output=[];state_summary=[];sensitivity=[];diversity=[]
    for index,(state,p,goals) in enumerate(identities):
        state_rows=[];dem=arrays['dem'][index]
        for j in range(13):
            source_row=source['inputs'][index*13+j];g=goals[j]
            for k,seed in enumerate(source['seeds']):
                path=p[j,k];feature=path_metrics(path);constraint=candidate_constraints(path,dem,resolution_m=.1,limits=limits)
                endpoint=np.asarray(feature['endpoint_xy_m']);gangle=math.atan2(g[1],g[0]);pangle=math.atan2(endpoint[1],endpoint[0]);angle_error=abs(float(wrap(pangle-gangle)))
                initial=math.atan2(path[0,1],path[0,0]);idx=(index*13+j)*8+k
                item={'state_id':state['state_id'],'split':state['split'],'terrain':state['terrain'],
                    'goal_id':source_row['goal_id'],'goal_class':source_row['declared_class'],
                    'actual_train_support_class':source_row['actual_train_support_class'],
                    'goal_xy':g.tolist(),'goal_order':j,'seed':int(seed),'endpoint_goal_error_m':float(np.linalg.norm(endpoint-g)),
                    'endpoint_goal_angle_error_rad':angle_error,'goal_radius_m':float(np.linalg.norm(g)),
                    'endpoint_radius_m':float(np.linalg.norm(endpoint)),
                    'initial_heading_error_rad':abs(float(wrap(initial-gangle))),
                    'initial_turn_direction_match':bool(np.sign(initial)==np.sign(g[1])) if abs(g[1])>1e-5 else None,
                    'path_metrics':feature,'lateral_excursion_m':float(np.max(np.abs(path[:,1]))),
                    'finite':bool(np.isfinite(path).all()),'trajectory_ood_distance':float(nearest[idx]),
                    'trajectory_ood':bool(nearest[idx]>ood['train_ood_nearest_l2_per_dimension_p95']),
                    'constraint':constraint,'geometry_screen_valid':source_row['geometry_screen_valid']}
                state_rows.append(item);output.append(item)
        if len(state_rows)!=104:raise ValueError('state output count')
        classes={}
        for label in ('logged','ID','near_ood','far_ood'):
            indices=[j for j in range(13) if source['inputs'][index*13+j]['declared_class']==label]
            subset=[r for r in state_rows if r['goal_class']==label]
            errors=[r['endpoint_goal_error_m'] for r in subset]
            classes[label]={'goals':len(indices),'paths':len(subset),'error':distribution(errors),
                'within_0p2m_fraction':float(np.mean(np.asarray(errors)<=.2)),
                'angle_error_median_rad':float(np.median([r['endpoint_goal_angle_error_rad'] for r in subset])),
                'radial_spearman':rho([r['goal_radius_m'] for r in subset],[r['endpoint_radius_m'] for r in subset]),
                'initial_direction_match_fraction':float(np.mean([r['initial_turn_direction_match'] for r in subset if r['initial_turn_direction_match'] is not None])) if any(r['initial_turn_direction_match'] is not None for r in subset) else None,
                'trajectory_ood_rate':float(np.mean([r['trajectory_ood'] for r in subset])),
                'constraint_invalid_rate':float(np.mean([not r['constraint']['valid'] for r in subset])),
                'geometry_screen_valid_rate':float(np.mean([r['geometry_screen_valid'] for r in subset]))}
        state_summary.append({'state_id':state['state_id'],'split':state['split'],'terrain':state['terrain'],'classes':classes})
        perclass={};modes={}
        for label in ('ID','near_ood','far_ood'):
            ids=[j for j in range(13) if source['inputs'][index*13+j]['declared_class']==label]
            goal_change=[];seed_change=[];endpoint_change=[];heading_change=[];curvature_change=[];left_consistency=[]
            for seed_index in range(8):
                for u,ja in enumerate(ids):
                    for jb in ids[u+1:]:
                        a,b=p[ja,seed_index],p[jb,seed_index]
                        goal_change.append(path_rms(a,b));endpoint_change.append(float(np.linalg.norm(a[-1]-b[-1])))
                        heading_change.append(abs(float(wrap(math.atan2(a[0,1],a[0,0])-math.atan2(b[0,1],b[0,0])))))
                        curvature_change.append(abs(path_metrics(a)['curvature_rad']-path_metrics(b)['curvature_rad']))
                        if goals[ja,1]!=goals[jb,1]:left_consistency.append(bool((goals[ja,1]-goals[jb,1])*(a[-1,1]-b[-1,1])>0))
            for j in ids:
                for u in range(8):
                    for v in range(u+1,8):seed_change.append(path_rms(p[j,u],p[j,v]))
            median_goal=float(np.median(goal_change));median_seed=float(np.median(seed_change))
            perclass[label]={'goal_effect_rms_m':distribution(goal_change),'sampling_effect_rms_m':distribution(seed_change),
                'goal_over_sampling_ratio':median_goal/median_seed if median_seed>1e-9 else None,
                'denominator_near_zero':median_seed<=1e-9,'endpoint_change_m':distribution(endpoint_change),
                'initial_heading_change_rad':distribution(heading_change),'curvature_change_rad':distribution(curvature_change),
                'left_right_pair_consistency':float(np.mean(left_consistency)) if left_consistency else None,
                'goal_conditioning_collapse':bool(all(mode_count(p[ids,seed])==1 for seed in range(8))),
                'sampling_collapse':bool(all(mode_count(p[j])==1 for j in ids))}
            modes[label]={'same_goal_different_seed':[mode_count(p[j]) for j in ids],
                'different_goal_same_seed':[mode_count(p[ids,seed]) for seed in range(8)],
                'all_goal_seed':mode_count(p[ids].reshape(-1,8,2))}
        # Explicit predeclared same-angle distance pair, no post-hoc pair selection.
        short,long=p[1],p[2]
        radial=float(np.mean([np.linalg.norm(long[s,-1])>np.linalg.norm(short[s,-1]) for s in range(8)]))
        sensitivity.append({'state_id':state['state_id'],'split':state['split'],'terrain':state['terrain'],
            'classes':perclass,'id_forward_radial_monotonic_fraction':radial})
        diversity.append({'state_id':state['state_id'],'split':state['split'],'terrain':state['terrain'],
            'classes':modes,'ID_plus_near_all_goal_seed':mode_count(p[1:9].reshape(-1,8,2)),
            'all_sweep_goal_seed':mode_count(p[1:13].reshape(-1,8,2))})
    if len(output)!=6656:raise ValueError('6656 candidate support')
    # Raw per-path rows stay in /tmp; Git records state-level summaries and source hashes.
    raw=RAW/'per-path-results.jsonl.gz';temp=raw.with_suffix('.tmp')
    with temp.open('wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
            for row in output:z.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode())
    temp.replace(raw)
    def aggregate(records,key):
        result={}
        for split in ('train','val'):
            for label in ('logged','ID','near_ood','far_ood'):
                subset=[s for s in records if s['split']==split]
                if key=='adherence':
                    values=[s['classes'][label] for s in subset];result[f'{split}:{label}']={
                        'states':len(values),'state_macro_within_0p2m':float(np.mean([x['within_0p2m_fraction'] for x in values])),
                        'state_macro_error_median_m':float(np.mean([x['error']['median'] for x in values])),
                        'state_macro_angle_error_median_rad':float(np.mean([x['angle_error_median_rad'] for x in values])),
                        'state_macro_trajectory_ood_rate':float(np.mean([x['trajectory_ood_rate'] for x in values])),
                        'state_macro_constraint_invalid_rate':float(np.mean([x['constraint_invalid_rate'] for x in values]))}
                elif label!='logged':
                    values=[s['classes'][label] for s in subset];result[f'{split}:{label}']={
                        'states':len(values),'goal_effect_state_median_m':float(np.median([x['goal_effect_rms_m']['median'] for x in values])),
                        'sampling_effect_state_median_m':float(np.median([x['sampling_effect_rms_m']['median'] for x in values])),
                        'ratio_state_median':float(np.median([x['goal_over_sampling_ratio'] for x in values if x['goal_over_sampling_ratio'] is not None])) if any(x['goal_over_sampling_ratio'] is not None for x in values) else None,
                        'goal_collapse_states':sum(x['goal_conditioning_collapse'] for x in values),'sampling_collapse_states':sum(x['sampling_collapse'] for x in values)}
        return result
    write(DOC/'goal-adherence-results.json',{'state_count':64,'paths':6656,'state_unit':True,'states':state_summary,'macro':aggregate(state_summary,'adherence')})
    write(DOC/'goal-sensitivity-results.json',{'state_unit':True,'states':sensitivity,'macro':aggregate(sensitivity,'sensitivity')})
    write(DOC/'goal-vs-sampling-results.json',{'ratio_rule':'no epsilon added; null if sampling median <=1e-9',
        'state_unit':True,'macro':aggregate(sensitivity,'sensitivity'),
        'id_goal_effect_exceeds_sampling_states':sum(s['classes']['ID']['goal_effect_rms_m']['median']>s['classes']['ID']['sampling_effect_rms_m']['median'] for s in sensitivity)})
    write(DOC/'goal-ood-results.json',{'source_trajectory_ood_threshold_sha256':sha(REPO/'docs/quality-identifiability-gate/phase-c-remediation/candidate-thresholds.json'),
        'trajectory_ood_threshold':ood['train_ood_nearest_l2_per_dimension_p95'],'macro':aggregate(state_summary,'adherence'),
        'raw_per_path_sha256':sha(raw),'nonfinite_count':sum(not x['finite'] for x in output)})
    write(DOC/'goal-diversity-results.json',{'existing_path_cluster_rms_m':.05,'state_unit':True,'states':diversity,
        'ID_multi_mode_states':sum(s['classes']['ID']['all_goal_seed']>1 for s in diversity),
        'ID_plus_near_multi_mode_states':sum(s['ID_plus_near_all_goal_seed']>1 for s in diversity),
        'all_multi_mode_states':sum(s['all_sweep_goal_seed']>1 for s in diversity)})
    write(RAW/'g3.done.json',{'stage':'G3_ANALYSIS_COMPLETE','per_path_sha256':sha(raw),
        'adherence_sha256':sha(DOC/'goal-adherence-results.json'),'sensitivity_sha256':sha(DOC/'goal-sensitivity-results.json'),
        'diversity_sha256':sha(DOC/'goal-diversity-results.json'),'elapsed_sec':time.monotonic()-started})
    write(RAW/'status.json',{'stage':'G3_ANALYSIS_COMPLETE','elapsed_sec':time.monotonic()-started},mutable=True)
    print('G3_ANALYSIS_COMPLETE',aggregate(state_summary,'adherence'),flush=True)
if __name__=='__main__':main()
