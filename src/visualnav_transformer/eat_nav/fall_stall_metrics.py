"""Descriptive held-out metrics; unknown labels are never converted to negatives."""
from __future__ import annotations
import numpy as np

EVENTS=('tip_over','near_fall','linear_stall','turn_stall')
HORIZONS=(1,2,5,10)


def binary_metrics(labels,probabilities,risk_scores=None):
    y=np.asarray(labels);p=np.asarray(probabilities,dtype=float)
    if y.shape!=p.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('binary metrics require known labels and finite probabilities')
    scores=p if risk_scores is None else np.asarray(risk_scores,dtype=float)
    if scores.shape!=p.shape or not np.isfinite(scores).all():raise ValueError('invalid risk ordering scores')
    if not len(y):return None
    predicted=p>=.5;positive=y==1
    tp=int((predicted&positive).sum());fp=int((predicted&~positive).sum())
    fn=int((~predicted&positive).sum());tn=int((~predicted&~positive).sum())
    ap=None
    if positive.any() and (~positive).any():
        order=np.argsort(-scores,kind='stable');ys=positive[order];ps=scores[order]
        ends=np.r_[np.flatnonzero(ps[:-1]!=ps[1:]),len(ps)-1]
        cumulative=np.cumsum(ys)[ends];increments=np.diff(np.r_[0,cumulative])
        ap=float(np.sum((increments/positive.sum())*(cumulative/(ends+1))))
    bins=[]
    for i in range(10):
        mask=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
        bins.append({'lower':i/10,'upper':(i+1)/10,'count':int(mask.sum()),
                     'mean_probability':float(p[mask].mean()) if mask.any() else None,
                     'observed_frequency':float(y[mask].mean()) if mask.any() else None})
    return {'threshold':.5,'tp':tp,'fp':fp,'tn':tn,'fn':fn,
            'recall':tp/(tp+fn) if tp+fn else None,'precision':tp/(tp+fp) if tp+fp else None,
            'false_positive_rate':fp/(fp+tn) if fp+tn else None,'brier':float(np.mean((p-y)**2)),
            'average_precision':ap,'pr_auc_definition':'tie_aware_noninterpolated_average_precision',
            'ap_unavailable_reason':None if ap is not None else 'both_classes_required',
            'reliability':bins}


def evaluate_predictions(arrays,indices,seeds,probabilities,progress,active,risk_scores=None):
    ids=np.asarray(indices);seeds=np.asarray(seeds);p=np.asarray(probabilities,dtype=float)
    scores=p if risk_scores is None else np.asarray(risk_scores,dtype=float)
    if scores.shape!=p.shape:raise ValueError('risk score shape mismatch')
    pred_progress=np.asarray(progress,dtype=float)
    y=arrays['event_targets'][ids,:,3,:];candidate=arrays['candidate_mask'][ids]
    mask=arrays['event_mask'][ids,:,3,:]&candidate[:,:,None]
    if p.shape!=y.shape or pred_progress.shape!=arrays['progress_targets'][ids].shape or len(seeds)!=len(ids):
        raise ValueError('evaluation shape mismatch')
    result={'base_seeds':len(set(seeds.tolist())),'state_groups':len(ids),
            'candidate_outcomes':int(candidate.sum()),'events':{},'progress':{}}
    for h,name in enumerate(EVENTS):
        valid=mask[:,:,h];pos=valid&(y[:,:,h]==1);neg=valid&(y[:,:,h]==0)
        support={'positive':int(pos.sum()),'negative':int(neg.sum()),'unknown':int((candidate&~valid).sum()),
                 'positive_state_groups':int(pos.any(1).sum()),'negative_state_groups':int(neg.any(1).sum()),
                 'positive_base_seeds':len(set(seeds[pos.any(1)].tolist())),
                 'negative_base_seeds':len(set(seeds[neg.any(1)].tolist()))}
        ranking=None
        if active[h]:
            if not np.isfinite(p[:,:,h][candidate]).all() or not np.isfinite(scores[:,:,h][candidate]).all():raise ValueError('supported probabilities/scores missing')
            agreements=[];group_scores=[];group_seeds=[]
            for g in range(len(ids)):
                safe=np.flatnonzero(neg[g]);risky=np.flatnonzero(pos[g])
                if len(safe) and len(risky):
                    a=scores[g,safe,h][:,None];b=scores[g,risky,h][None,:]
                    pair_scores=(a<b).astype(float)+.5*(a==b)
                    agreements.extend(pair_scores.ravel().tolist());group_scores.append(float(pair_scores.mean()));group_seeds.append(int(seeds[g]))
            per_seed=[np.mean([v for v,s in zip(group_scores,group_seeds) if s==seed]) for seed in sorted(set(group_seeds))]
            ranking={'pair_count':len(agreements),'state_group_count':len(group_scores),'base_seed_count':len(per_seed),
                     'agreement':float(np.mean(per_seed)) if per_seed else None,
                     'reduction':'pair_mean_within_group_then_group_mean_within_seed_then_seed_mean'}
        result['events'][name]={'train_supported':bool(active[h]),'support':support,
                               'metrics':binary_metrics(y[:,:,h][valid],p[:,:,h][valid],scores[:,:,h][valid]) if active[h] else None,
                               'pair_ranking':ranking}
    for h,horizon in enumerate(HORIZONS):
        valid=arrays['progress_mask'][ids,:,h]&candidate
        target=arrays['progress_targets'][ids,:,h]
        finite=np.isfinite(pred_progress[:,:,h])
        used=valid&finite
        result['progress'][f'{horizon}s']={'known_count':int(valid.sum()),'evaluated_count':int(used.sum()),
            'unknown_count':int((candidate&~valid).sum()),
            'mae_m':float(np.abs(pred_progress[:,:,h][used]-target[used]).mean()) if used.any() else None}
    selection={'train_tip_supported':bool(active[0]),'eligible_groups':0,'excluded_unknown_tip_groups':0,
               'observed_tip_overs':0,'selected_candidates':[],'progress_known_count':0,'mean_progress_10s_m':None,
               'tie_rule':'first_in_frozen_candidate_order'}
    values=[]
    if active[0]:
        for g in range(len(ids)):
            if not candidate[g].all() or not mask[g,:,0].all():
                selection['excluded_unknown_tip_groups']+=1;continue
            k=int(np.argmin(scores[g,:,0]));selection['eligible_groups']+=1
            selection['observed_tip_overs']+=int(y[g,k,0]);selection['selected_candidates'].append({'sample_index':int(ids[g]),'candidate_index':k})
            if arrays['progress_mask'][ids[g],k,3]:values.append(float(arrays['progress_targets'][ids[g],k,3]))
        selection['progress_known_count']=len(values)
        selection['mean_progress_10s_m']=float(np.mean(values)) if values else None
    result['selection']=selection
    return result
