"""CPU-only metric and diversity stages for frozen readiness traces."""
from __future__ import annotations
import gzip,json,pathlib,time
import numpy as np
from visualnav_transformer.eat_nav.path_tracking_audit import audit_run,select_live_entries,recovery_from_log_lines,_yaw_from_root,_sha
from visualnav_transformer.eat_nav.path_tracking_geometry import tracking_metrics,classify_run
from visualnav_transformer.eat_nav.feasible_transfer import cluster_labels,command_cluster_labels

REPO=pathlib.Path('/tmp/eatnav-vnt-candidate-coverage')
PRIOR=REPO/'docs/quality-identifiability-gate/candidate-coverage/evaluator-readiness-pilot'
DOC=REPO/'docs/quality-identifiability-gate/candidate-coverage/path-tracking-audit'
ROOT=pathlib.Path('/tmp/eatnav-candidate-coverage/path-tracking-audit')
UNIFIED=pathlib.Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol/unified')
INPUT=pathlib.Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-inputs')
def read(path):return json.loads(pathlib.Path(path).read_text())
def atomic(path,data,once=True):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if path.exists() and once:
        if path.read_text()!=content:raise FileExistsError('frozen output changed '+str(path))
        return
    temp=path.with_name(path.name+'.tmp');temp.write_text(content);temp.replace(path)
def gzlines(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:yield json.loads(line)
def atomic_gz(path,rows):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=''.join(json.dumps(x,sort_keys=True,allow_nan=False)+'\n' for x in rows).encode()
    if path.exists():
        with gzip.open(path,'rb') as stream:
            if stream.read()!=content:raise FileExistsError('frozen raw changed '+str(path))
        return _sha(path)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as stream:stream.write(content)
    tmp.replace(path);return _sha(path)
def log_recovery(entry,summary):
    reset=int(summary['follower_startup']['reset_number']);name=f'reset-{reset:04d}-branch-0.log'
    old=pathlib.Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-pilot/follower-logs')
    paths=[UNIFIED/'runs'/entry['run_id']/'follower-logs'/(name+'.gz'),old/(name+'.gz'),old/name]
    if paths[0].exists():
        path=paths[0]
    else:
        found=[p for p in paths[1:] if p.exists()]
        if len(found)!=1:raise ValueError('recovery log missing/ambiguous '+entry['run_id'])
        path=found[0]
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as f:result=recovery_from_log_lines(f,entry['logical_t0_ns'],121)
    return {**result,'log_path':str(path),'log_sha256':_sha(path)}
def stage_a2():
    started=time.monotonic()
    if read(ROOT/'a0.done.json')['source_provenance_sha256']!=_sha(DOC/'source-provenance.json'):raise ValueError('A0 drift')
    live,_=select_live_entries(read(PRIOR/'schedule.json')['runs'])
    ledger={x['run_id']:x for x in read(PRIOR/'artifact-sha-ledger.json')['runs']}
    outcomes={x['run_id']:x for x in read(UNIFIED/'analysis/per-candidate-outcomes.json')}
    thresholds=read(PRIOR/'analysis-thresholds.json')
    records=[];series=[]
    for entry in live:
        run=audit_run(entry,UNIFIED/'runs'/entry['run_id'],INPUT/f"state-{entry['state_index']:03d}",ledger[entry['run_id']])
        trace=run['trace'];path=run['map_path']+run['env_origin']
        roots=[trace[0]['root_state']]+[r['post_root_state'] for r in trace]
        pos=np.asarray([r[:2] for r in roots]);yaws=np.asarray([_yaw_from_root(r) for r in roots]);cmd=np.asarray([r['command'] for r in trace])
        metric=tracking_metrics(path,pos,yaws,cmd,dt=.05,tolerance=.2)
        recovery=log_recovery(entry,run['summary'])
        outcome=outcomes.get(entry['run_id'])
        source='matched_run'
        if entry['kind']=='nominal' and entry['repeat']!=0:
            source='nominal_repeat_0_path_only; repeat outcome unavailable'
            outcome=next(x for x in outcomes.values() if x['state_id']==entry['state_id'] and x['kind']=='nominal')
        if outcome is None or outcome['base_path_sha256']!=entry['base_path_sha256']:raise ValueError('outcome/path identity mismatch')
        valid=bool(outcome['constraint_valid'])
        records.append({'run_id':entry['run_id'],'state_id':entry['state_id'],'terrain':entry['terrain_type'],
            'generator':entry['kind'],'candidate_id':entry['candidate_id'],'repeat':entry['repeat'],
            'path_sha256':entry['base_path_sha256'],'trace_sha256':run['identity']['trace_sha256'],
            'two_second':metric['two_second'],'six_second':metric['six_second'],'endpoint':metric['endpoint'],
            'path_length_m':metric['path_length_m'],'projection_ambiguous_count':metric['projection_ambiguous_count'],
            'projection_segment_jump_count':metric['projection_segment_jump_count'],'zero_length_segments':metric['zero_length_segments'],
            'recovery':recovery,'constraint_valid':valid,'outcome_source':source,
            'stack':outcome['six_second']['stack'] if source=='matched_run' else None,
            'excessive_tilt':outcome['six_second']['maximum_tilt_rad']>thresholds['excessive_tilt_rad'] if source=='matched_run' else None,
            'termination':outcome['six_second']['termination'] if source=='matched_run' else None,
            'category':classify_run(metric,constraint_valid=valid,tracking_failure_threshold=thresholds['tracking_failure_rms_m'])})
        series.append({'run_id':entry['run_id'],'state_id':entry['state_id'],'kind':entry['kind'],
            'candidate_id':entry['candidate_id'],'repeat':entry['repeat'],'base_path':run['base_path'].tolist(),
            'world_path':path.tolist(),'world_positions':pos.tolist(),'yaws':yaws.tolist(),
            'commands':cmd.tolist(),'anchor_world_xy':pos[0].tolist(),'anchor_yaw_rad':run['coordinate']['anchor_yaw_rad']})
    if len(records)!=112 or len(series)!=112:raise ValueError('112 support required')
    rawsha=atomic_gz(ROOT/'tracking-series.jsonl.gz',series)
    atomic(DOC/'path-tracking-metrics.json',{'scope':'112 complete live; 2 s Path and 6 s execution',
        'geometry':'nearest nonzero segment; tie earliest; waypoint pass = maximum projected arclength past waypoint arclength',
        'thresholds':{'goal_tolerance_m':.2,'tracking_failure_rms_m':thresholds['tracking_failure_rms_m']},
        'statistical_unit':'state','raw_series_sha256':rawsha,'runs':records})
    groups={}
    for kind in ('nominal','nomad','bank'):
        selected=[r for r in records if r['generator']==kind]
        groups[kind]={'runs':len(selected),'reached':sum(r['endpoint']['reached'] for r in selected),
            'escaped_after_arrival':sum(r['endpoint']['escaped_after_arrival'] for r in selected),
            'median_min_distance_m':float(np.median([r['endpoint']['min_distance_m'] for r in selected])),
            'median_two_second_progress_ratio':float(np.median([r['two_second']['progress_ratio'] for r in selected]))}
    states={sid:{kind:sum(r['endpoint']['reached'] for r in records if r['state_id']==sid and r['generator']==kind)
        for kind in ('nominal','nomad','bank')} for sid in sorted({r['state_id'] for r in records})}
    atomic(DOC/'endpoint-reach-summary.json',{'goal_tolerance_m':.2,'groups':groups,'states':states,
        'path_horizon_sec':2.,'execution_duration_sec':6.,'state_count':7})
    marker={'stage':'A2_PASS','metrics_sha256':_sha(DOC/'path-tracking-metrics.json'),
        'series_sha256':rawsha,'endpoint_sha256':_sha(DOC/'endpoint-reach-summary.json')}
    if (ROOT/'a2.done.json').exists():
        previous=read(ROOT/'a2.done.json')
        if any(previous.get(k)!=v for k,v in marker.items()):raise ValueError('A2 marker drift')
    else:atomic(ROOT/'a2.done.json',{**marker,'elapsed_sec':time.monotonic()-started})
    atomic(ROOT/'status.json',{'stage':'A2_PASS','included_live_runs':112,'elapsed_sec':time.monotonic()-started,'next':'A3'},once=False)
    return groups

def _rms(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def _distribution(values):
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'min':float(np.min(a)) if len(a) else None,'median':float(np.median(a)) if len(a) else None,
        'p95':float(np.percentile(a,95)) if len(a) else None,'max':float(np.max(a)) if len(a) else None}
def _set_indices(rows,name):
    if name=='nomad':return [i for i,r in enumerate(rows) if r['kind']=='nomad']
    if name=='bank':return [i for i,r in enumerate(rows) if r['kind']=='bank']
    if name=='nominal_repeat':return [i for i,r in enumerate(rows) if r['kind']=='nominal']
    if name=='all_candidate':return [i for i,r in enumerate(rows) if r['kind']!='nominal' or r['repeat']==0]
    raise ValueError(name)
def _state_cluster(rows,thresholds):
    base=_set_indices(rows,'all_candidate')
    p=np.asarray([rows[i]['base_path'][1:] for i in base]);c=np.asarray([rows[i]['commands'] for i in base]);r=np.asarray([rows[i]['world_positions'][1:] for i in base])
    labels={'P':cluster_labels(p,threshold=thresholds['planned_path_cluster_rms_m']),
        'C':command_cluster_labels(c,linear_threshold=thresholds['command_linear_rms_mps'],angular_threshold=thresholds['command_angular_rms_radps']),
        'R':cluster_labels(r,threshold=thresholds['realized_trajectory_cluster_rms_m'])}
    return base,labels

def pair_transition_counts(labels,indices):
    """Pair counts are within-state diagnostics; caller must aggregate states, not pairs."""
    counts={'pair_count':0,'P_different_C_same':0,'C_different_R_same':0,'P_C_R_all_different':0}
    for x,i in enumerate(indices):
        for j in indices[x+1:]:
            counts['pair_count']+=1
            pd=labels['P'][i]!=labels['P'][j];cd=labels['C'][i]!=labels['C'][j];rd=labels['R'][i]!=labels['R'][j]
            counts['P_different_C_same']+=int(pd and not cd)
            counts['C_different_R_same']+=int(cd and not rd)
            counts['P_C_R_all_different']+=int(pd and cd and rd)
    return counts

def stage_a3():
    started=time.monotonic();a2=read(ROOT/'a2.done.json')
    if a2['series_sha256']!=_sha(ROOT/'tracking-series.jsonl.gz') or a2['metrics_sha256']!=_sha(DOC/'path-tracking-metrics.json'):
        raise ValueError('A2 SHA drift')
    series=list(gzlines(ROOT/'tracking-series.jsonl.gz'));metrics={r['run_id']:r for r in read(DOC/'path-tracking-metrics.json')['runs']}
    th=read(PRIOR/'analysis-thresholds.json');states=sorted({r['state_id'] for r in series})
    distances=[];transitions=[];classifications=[]
    for sid in states:
        rows=[x for x in series if x['state_id']==sid]
        if len(rows)!=16:raise ValueError('state support mismatch '+sid)
        base,lab=_state_cluster(rows,th);full={key:[None]*len(rows) for key in lab}
        for k,i in enumerate(base):
            for name in lab:full[name][i]=lab[name][k]
        # Nominal repeats are clustered with the same fixed thresholds to quantify duplicate variation.
        nominal=_set_indices(rows,'nominal_repeat')
        for name,data,threshold in [('P',[r['base_path'][1:] for r in rows],th['planned_path_cluster_rms_m']),
                                    ('R',[r['world_positions'][1:] for r in rows],th['realized_trajectory_cluster_rms_m'])]:
            labels=cluster_labels([data[i] for i in nominal],threshold=threshold)
            for i,label in zip(nominal,labels):
                if full[name][i] is None:full[name][i]=label
        clabels=command_cluster_labels([rows[i]['commands'] for i in nominal],linear_threshold=th['command_linear_rms_mps'],angular_threshold=th['command_angular_rms_radps'])
        for i,label in zip(nominal,clabels):
            if full['C'][i] is None:full['C'][i]=label
        for name in ('nomad','bank','nominal_repeat','all_candidate'):
            indices=_set_indices(rows,name)
            subset_lab={
                'P':cluster_labels([rows[i]['base_path'][1:] for i in indices],threshold=th['planned_path_cluster_rms_m']),
                'C':command_cluster_labels([rows[i]['commands'] for i in indices],linear_threshold=th['command_linear_rms_mps'],angular_threshold=th['command_angular_rms_radps']),
                'R':cluster_labels([rows[i]['world_positions'][1:] for i in indices],threshold=th['realized_trajectory_cluster_rms_m'])}
            pairdist={k:[] for k in ('P','C_linear_2s','C_angular_2s','C_linear_6s','C_angular_6s','R_2s','R_6s')}
            for x,i in enumerate(indices):
                for j in indices[x+1:]:
                    a,b=rows[i],rows[j];ac=np.asarray(a['commands']);bc=np.asarray(b['commands']);ar=np.asarray(a['world_positions']);br=np.asarray(b['world_positions'])
                    pairdist['P'].append(_rms(np.asarray(a['base_path'])[1:],np.asarray(b['base_path'])[1:]))
                    for sec,n in [('2s',40),('6s',120)]:
                        pairdist[f'C_linear_{sec}'].append(_rms(ac[:n,:2],bc[:n,:2]));pairdist[f'C_angular_{sec}'].append(_rms(ac[:n,2],bc[:n,2]))
                        pairdist[f'R_{sec}'].append(_rms(ar[1:n+1],br[1:n+1]))
            distances.append({'state_id':sid,'set':name,'candidate_count':len(indices),
                'cluster_count':{key:len(set(subset_lab[key])) for key in subset_lab},
                'pairwise_distance':{k:_distribution(v) for k,v in pairdist.items()},
                'pair_transitions':pair_transition_counts(subset_lab,list(range(len(indices))))})
        cross={k:[] for k in ('P','C_2s','C_6s','R_2s','R_6s')}
        for i in _set_indices(rows,'nomad'):
            for j in _set_indices(rows,'bank'):
                a,b=rows[i],rows[j]
                cross['P'].append(_rms(np.asarray(a['base_path'])[1:],np.asarray(b['base_path'])[1:]))
                for sec,n in [('2s',40),('6s',120)]:
                    cross[f'C_{sec}'].append(_rms(np.asarray(a['commands'])[:n],np.asarray(b['commands'])[:n]))
                    cross[f'R_{sec}'].append(_rms(np.asarray(a['world_positions'])[1:n+1],np.asarray(b['world_positions'])[1:n+1]))
        distances.append({'state_id':sid,'set':'nomad_bank_cross','pairwise_distance':{k:_distribution(v) for k,v in cross.items()}})
        candidate=[]
        for i,row in enumerate(rows):
            metric=metrics[row['run_id']]
            candidate.append({'run_id':row['run_id'],'generator':row['kind'],'candidate_id':row['candidate_id'],
                'P_cluster':full['P'][i],'C_cluster':full['C'][i],'R_cluster':full['R'][i],
                'path_length_m':metric['path_length_m'],'cross_track_p95_m':metric['two_second']['cross_track_p95_m'],
                'endpoint_reached':metric['endpoint']['reached']})
        transitions.append({'state_id':sid,'terrain':metrics[rows[0]['run_id']]['terrain'],'candidates':candidate,
            'nomad_pair_counts':next(x['pair_transitions'] for x in distances if x['state_id']==sid and x['set']=='nomad'),
            'all_candidate_pair_counts':next(x['pair_transitions'] for x in distances if x['state_id']==sid and x['set']=='all_candidate')})
        nom=next(x for x in distances if x['state_id']==sid and x['set']=='nomad');n=nom['cluster_count']
        if n['P']==1:classification='generator_collapse'
        elif n['C']==1:classification='follower_collapse'
        elif n['R']==1:classification='realization_collapse'
        elif n['P']>1 and n['C']>1 and n['R']>1:classification='diversity_preserved'
        else:classification='mixed'
        nommetrics=[metrics[rows[i]['run_id']] for i in _set_indices(rows,'nomad')]
        classifications.append({'state_id':sid,'terrain':metrics[rows[0]['run_id']]['terrain'],'classification':classification,
            'scope':'NoMaD K=8 within state','cluster_count':n,'pairwise_distance':nom['pairwise_distance'],
            'median_cross_track_p95_m':float(np.median([x['two_second']['cross_track_p95_m'] for x in nommetrics])),
            'endpoint_reached':sum(x['endpoint']['reached'] for x in nommetrics),'candidate_count':8})
    atomic(DOC/'path-command-realized-distances.json',{'state_count':7,'thresholds':{k:th[k] for k in ('planned_path_cluster_rms_m','command_linear_rms_mps','command_angular_rms_radps','realized_trajectory_cluster_rms_m')},'states_and_sets':distances},once=False)
    atomic(DOC/'mode-transition-table.json',{'state_count':7,'states':transitions},once=False)
    atomic(DOC/'collapse-classification.json',{'state_count':7,'states':classifications,'counts':{name:sum(x['classification']==name for x in classifications) for name in ('generator_collapse','follower_collapse','realization_collapse','diversity_preserved','mixed','inconclusive')}},once=False)
    marker={'stage':'A3_PASS','distance_sha256':_sha(DOC/'path-command-realized-distances.json'),
        'transition_sha256':_sha(DOC/'mode-transition-table.json'),'classification_sha256':_sha(DOC/'collapse-classification.json')}
    atomic(ROOT/'a3.done.json',{**marker,'elapsed_sec':time.monotonic()-started},once=False)
    atomic(ROOT/'status.json',{'stage':'A3_PASS','elapsed_sec':time.monotonic()-started,'next':'A4'},once=False)
    return marker

def _rank(a):
    a=np.asarray(a,dtype=float);order=np.argsort(a,kind='stable');out=np.empty(len(a),dtype=float);i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2;i=j
    return out
def _rho(a,b):
    if len(a)<3:return None
    x,y=_rank(a),_rank(b)
    if np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])
def _macro_loso(state_values):
    vals=[float(v) for v in state_values.values() if v is not None]
    if not vals:return {'state_support':0,'macro_mean':None,'leave_one_state_out':{}}
    return {'state_support':len(vals),'macro_mean':float(np.mean(vals)),
        'leave_one_state_out':{sid:float(np.mean([v for key,v in state_values.items() if key!=sid and v is not None])) if any(key!=sid and v is not None for key,v in state_values.items()) else None for sid in state_values}}
def stage_a4():
    if read(ROOT/'a3.done.json')['transition_sha256']!=_sha(DOC/'mode-transition-table.json'):raise ValueError('A3 drift')
    metrics={r['run_id']:r for r in read(DOC/'path-tracking-metrics.json')['runs']}
    outcomes=read(UNIFIED/'analysis/per-candidate-outcomes.json')
    transitions={s['state_id']:s for s in read(DOC/'mode-transition-table.json')['states']}
    records=[]
    fields=('delta_v','advantage_h','drawdown','recovery','forward_goal_progress_m','maximum_tilt_rad','maximum_contact_n')
    for sid in sorted({r['state_id'] for r in outcomes}):
        rows=[r for r in outcomes if r['state_id']==sid]
        if len(rows)!=14:raise ValueError('oracle state support')
        dv=[r['six_second']['delta_v'] for r in rows]
        cross=[metrics[r['run_id']]['two_second']['cross_track_p95_m'] for r in rows]
        progress=[metrics[r['run_id']]['two_second']['progress_ratio'] for r in rows]
        lookup={c['run_id']:c for c in transitions[sid]['candidates']}
        modes=[lookup[r['run_id']]['R_cluster'] for r in rows]
        within=[];between=[]
        for i,a in enumerate(rows):
            for j in range(i+1,len(rows)):
                delta=abs(dv[i]-dv[j]);(within if modes[i]==modes[j] else between).append(delta)
        correlations={field:{'cross_track_spearman':_rho(cross,[r['six_second'][field] for r in rows]),
            'progress_spearman':_rho(progress,[r['six_second'][field] for r in rows])} for field in fields}
        nom=[r for r in rows if r['kind']=='nomad'];bank=[r for r in rows if r['kind']=='bank']
        record={'state_id':sid,'terrain':rows[0]['terrain_type'],'n':14,
            'reached_count':sum(metrics[r['run_id']]['endpoint']['reached'] for r in rows),
            'reached_vs_not_delta_v':'NA: all reached' if all(metrics[r['run_id']]['endpoint']['reached'] for r in rows) else 'mixed support',
            'correlations':correlations,'within_R_cluster_abs_delta_v':_distribution(within),
            'between_R_cluster_abs_delta_v':_distribution(between),
            'nomad_minus_bank_mean_delta_v':float(np.mean([r['six_second']['delta_v'] for r in nom])-np.mean([r['six_second']['delta_v'] for r in bank])),
            'nomad_minus_bank_mean_cross_track_p95_m':float(np.mean([metrics[r['run_id']]['two_second']['cross_track_p95_m'] for r in nom])-np.mean([metrics[r['run_id']]['two_second']['cross_track_p95_m'] for r in bank])),
            'termination_count':sum(r['six_second']['termination'] is not None for r in rows),
            'stack_count':sum(r['six_second']['stack'] for r in rows)}
        records.append(record)
    summary={field:{metric:_macro_loso({r['state_id']:r['correlations'][field][metric] for r in records}) for metric in ('cross_track_spearman','progress_spearman')} for field in fields}
    for key in ('nomad_minus_bank_mean_delta_v','nomad_minus_bank_mean_cross_track_p95_m'):
        summary[key]=_macro_loso({r['state_id']:r[key] for r in records})
    atomic(DOC/'tracking-outcome-analysis.json',{'state_count':7,'candidate_count':98,'statistical_unit':'state',
        'source_oracle_sha256':_sha(UNIFIED/'analysis/per-candidate-outcomes.json'),
        'nominal_repeats_excluded_from_oracle_association':14,'states':records,'state_macro_loso':summary,
        'pairwise_R_cluster_differences_are_descriptive_not_independent':True})
    marker={'stage':'A4_PASS','outcome_sha256':_sha(DOC/'tracking-outcome-analysis.json')};atomic(ROOT/'a4.done.json',marker)
    atomic(ROOT/'status.json',{'stage':'A4_PASS','next':'A5'},once=False)
    return summary

def _subset_rank(rows):
    if len(rows)<2:return {'status':'insufficient support','candidate_count':len(rows)}
    score=np.asarray([r['score'] for r in rows]);actual=np.asarray([r['delta_v'] for r in rows]);
    top=np.argsort(-score,kind='stable');oracle=np.max(actual);best=top[0]
    comparable=[(i,j) for i in range(len(rows)) for j in range(i+1,len(rows)) if score[i]!=score[j] and actual[i]!=actual[j]]
    return {'status':'observed','candidate_count':len(rows),'top1_recall':bool(actual[best]==oracle),
        'top3_recall':bool(any(actual[i]==oracle for i in top[:min(3,len(rows))])),
        'pairwise_accuracy':float(np.mean([(score[i]-score[j])*(actual[i]-actual[j])>0 for i,j in comparable])) if comparable else None,
        'pairwise_support':len(comparable),'oracle_regret':float(oracle-actual[best]),
        'score_spread':float(np.max(score)-np.min(score)),
        'mean_ensemble_disagreement':float(np.mean([r['disagreement'] for r in rows])),
        'tracking_rank_spearman':_rho([r['cross_track'] for r in rows],actual)}
def stage_a5():
    if read(ROOT/'a4.done.json')['outcome_sha256']!=_sha(DOC/'tracking-outcome-analysis.json'):raise ValueError('A4 drift')
    metrics={r['run_id']:r for r in read(DOC/'path-tracking-metrics.json')['runs']}
    outcomes=read(UNIFIED/'analysis/per-candidate-outcomes.json');evaluator=read(UNIFIED/'analysis/frozen-evaluator.json')
    transition={s['state_id']:s for s in read(DOC/'mode-transition-table.json')['states']}
    if evaluator['outcomes_sha256']!=_sha(UNIFIED/'analysis/per-candidate-outcomes.json'):raise ValueError('evaluator outcome SHA mismatch')
    by_state=[]
    for sid in sorted({r['state_id'] for r in outcomes}):
        rows=[r for r in outcomes if r['state_id']==sid];e=next(x for x in evaluator['states'] if x['state_id']==sid)
        if [r['candidate_id'] for r in rows]!=e['candidate_ids']:raise ValueError('Evaluator positional candidate mismatch '+sid)
        if len(e['delta_v_prediction_mean'])!=14:raise ValueError('Evaluator score length')
        tr={c['run_id']:c for c in transition[sid]['candidates']}
        enriched=[]
        for i,r in enumerate(rows):
            m=metrics[r['run_id']];mode=tr[r['run_id']]
            enriched.append({'run_id':r['run_id'],'kind':r['kind'],'score':e['delta_v_prediction_mean'][i],
                'disagreement':e['ensemble_disagreement'][i],'delta_v':r['six_second']['delta_v'],
                'cross_track':m['two_second']['cross_track_p95_m'],'reached':m['endpoint']['reached'],
                'valid':r['constraint_valid'],'P':mode['P_cluster'],'C':mode['C_cluster'],'R':mode['R_cluster']})
        cutoff=float(np.median([r['cross_track'] for r in enriched]))
        pc={r['run_id']:any(r['P']!=s['P'] and r['C']==s['C'] for s in enriched if s is not r) for r in enriched}
        cr={r['run_id']:any(r['C']!=s['C'] and r['R']==s['R'] for s in enriched if s is not r) for r in enriched}
        strata={'all':enriched,'reached':[r for r in enriched if r['reached']],
            'not_reached':[r for r in enriched if not r['reached']],
            'low_cross_track':[r for r in enriched if r['cross_track']<=cutoff],
            'high_cross_track':[r for r in enriched if r['cross_track']>cutoff],
            'P_to_C_collapse':[r for r in enriched if pc[r['run_id']]],
            'no_P_to_C_collapse':[r for r in enriched if not pc[r['run_id']]],
            'C_to_R_collapse':[r for r in enriched if cr[r['run_id']]],
            'no_C_to_R_collapse':[r for r in enriched if not cr[r['run_id']]],
            'nomad':[r for r in enriched if r['kind']=='nomad'],
            'bank':[r for r in enriched if r['kind']=='bank'],
            'constraint_valid':[r for r in enriched if r['valid']],
            'constraint_invalid':[r for r in enriched if not r['valid']],
            'realized_mode_multiple':enriched if len({r['R'] for r in enriched})>1 else [],
            'realized_mode_single':enriched if len({r['R'] for r in enriched})==1 else []}
        by_state.append({'state_id':sid,'terrain':rows[0]['terrain_type'],'cross_track_median_split_m':cutoff,
            'strata':{name:_subset_rank(subset) for name,subset in strata.items()},
            'all_candidates':enriched})
    names=by_state[0]['strata'].keys();macro={}
    for name in names:
        macro[name]={metric:_macro_loso({s['state_id']:s['strata'][name].get(metric) for s in by_state}) for metric in
            ('top1_recall','top3_recall','pairwise_accuracy','oracle_regret','score_spread','mean_ensemble_disagreement','tracking_rank_spearman')}
    atomic(DOC/'evaluator-tracking-strata.json',{'state_count':7,'oracle_candidate_count':98,
        'score_source_sha256':_sha(UNIFIED/'analysis/frozen-evaluator.json'),
        'score_identity':'frozen GT(P); positional candidate ID verified including duplicate bank-nominal',
        'cross_track_split':'within-state median, descriptive stratification; no model selection',
        'top_k_for_small_strata':'min(k,n)','statistical_unit':'state','states':by_state,'state_macro_loso':macro,
        'online_selector_approved':False})
    marker={'stage':'A5_PASS','strata_sha256':_sha(DOC/'evaluator-tracking-strata.json')};atomic(ROOT/'a5.done.json',marker)
    atomic(ROOT/'status.json',{'stage':'A5_PASS','next':'A6'},once=False)
    return macro

def stage_a6():
    if read(ROOT/'a5.done.json')['strata_sha256']!=_sha(DOC/'evaluator-tracking-strata.json'):raise ValueError('A5 drift')
    live,_=select_live_entries(read(PRIOR/'schedule.json')['runs']);counts={k:0 for k in ('dem_history','relative_goal','actor_observation','actor_latent','body_velocity','roll_pitch_quaternion','joint_position','joint_velocity','joint_effort_raw','previous_action','planned_path','first_follower_command','path_geometry')}
    shapes={};missing={}
    for entry in live:
        folder=UNIFIED/'runs'/entry['run_id'];summary=read(folder/'live-0.json');anchor=summary['post_follower_anchor']
        with gzip.open(folder/'live-0.trace.json.gz','rt') as f:step=json.load(f)[0]
        available={'dem_history':anchor.get('dem_history'),'relative_goal':summary['state'].get('goal'),
            'actor_observation':step.get('actor_input'),'actor_latent':step.get('actor_latent'),
            'body_velocity':step.get('root_state',[None]*13)[7:13],
            'roll_pitch_quaternion':step.get('root_state',[None]*7)[3:7],
            'joint_position':step.get('joint_position'),'joint_velocity':step.get('joint_velocity'),
            'joint_effort_raw':step.get('joint_effort'),'previous_action':anchor.get('action_manager_previous_action'),
            'planned_path':summary.get('map_path'),'first_follower_command':step.get('follower_command'),
            'path_geometry':summary.get('map_path')}
        for name,value in available.items():
            if value is not None:
                counts[name]+=1
                array=np.asarray(value,dtype=float)
                if not np.isfinite(array).all():raise ValueError('nonfinite schema '+name+' '+entry['run_id'])
                shapes.setdefault(name,set()).add(tuple(array.shape))
            else:missing.setdefault(name,[]).append(entry['run_id'])
    if counts['joint_effort_raw']!=0:raise ValueError('unexpected raw effort support')
    if any(v!=112 for k,v in counts.items() if k!='joint_effort_raw'):raise ValueError('online schema other than effort incomplete')
    provenance=read(REPO/'docs/quality-identifiability-gate/forward-evaluator/actor-input-provenance.json')
    atomic(DOC/'evaluator-input-schema-audit.json',{'live_rollouts':112,'availability_count':counts,
        'shapes':{k:[list(x) for x in sorted(v)] for k,v in shapes.items()},'missing_run_ids':missing,
        'actor_observation_provenance_sha256':_sha(REPO/'docs/quality-identifiability-gate/forward-evaluator/actor-input-provenance.json'),
        'saved_training_toggles':provenance['saved_training_toggles'],
        'joint_effort':{'raw_separate_field':False,'actor_observation_contains':'joint_effort_with_track_sum',
            'field_order':provenance['fields'],'actor_dim':provenance['actor_observation_dim'],
            'exact_raw_recovery_from_actor':'unavailable: scaled/clipped/noisy and track-summed observation is not invertible',
            'scale_and_clip':'checkpoint/config chain potentially defines scale and clip; raw effort was not serialized and cannot be reconstructed exactly',
            'same_step_online_capture':'requires explicit robot.data.applied_torque or measured joint effort capture at Actor observation step; not demonstrated for this dataset'},
        'actor_observation':'post-delay actual Actor input; pre-noise and post-noise/pre-delay also in every trace',
        'body_velocity':'root_state[7:13] at same pre-step time; rotate from world to body for a future evaluator',
        'roll_pitch':'root_state[3:7] quaternion at same pre-step time; Euler/projected gravity derivable',
        'relative_goal':'frozen state goal at anchor; same-step online goal capture requires explicit interface validation',
        'first_follower_command':'trace[0] real response; candidate-conditioned preview available only if follower invoked; no new preview computed here',
        'online_schema_complete':False,'missing_training_teacher':'candidate-specific counterfactual failure-positive labels and verified same-step joint effort stream'})
    marker={'stage':'A6_PASS','schema_sha256':_sha(DOC/'evaluator-input-schema-audit.json')};atomic(ROOT/'a6.done.json',marker)
    atomic(ROOT/'status.json',{'stage':'A6_PASS','next':'FIGURES_REPORT'},once=False)
    return counts

def stage_candidate_features():
    from visualnav_transformer.eat_nav.feasible_transfer import integrate_commands
    from visualnav_transformer.eat_nav.path_tracking_geometry import base_from_map,wrap_angle
    if read(ROOT/'a3.done.json')['transition_sha256']!=_sha(DOC/'mode-transition-table.json'):raise ValueError('A3 drift')
    rows=[]
    for run in gzlines(ROOT/'tracking-series.jsonl.gz'):
        path=np.asarray(run['base_path'],dtype=float);cmd=np.asarray(run['commands'],dtype=float)
        world=np.asarray(run['world_positions'],dtype=float);yaw=np.asarray(run['yaws'],dtype=float)
        local=base_from_map(world,np.asarray(run['anchor_world_xy']),run['anchor_yaw_rad'])
        headings=np.arctan2(np.diff(path,axis=0)[:,1],np.diff(path,axis=0)[:,0])
        command_xy=integrate_commands(cmd,dt_sec=.05)
        record={'run_id':run['run_id'],'state_id':run['state_id'],'generator':run['kind'],'candidate_id':run['candidate_id'],
            'planned':{'cumulative_waypoints_base':path[1:].tolist(),'endpoint_base_xy':path[-1].tolist(),
                'path_length_m':float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                'lateral_excursion_m':float(np.max(np.abs(path[:,1]))),
                'curvature_abs_total_rad':float(np.abs(wrap_angle(np.diff(headings))).sum()),
                'initial_heading_rad':float(headings[0])},
            'command':{},'realized':{}}
        for name,n in (('two_second',40),('six_second',120)):
            sub=cmd[:n];rpos=local[:n+1]
            record['command'][name]={'integrated_xy_m':command_xy[n-1].tolist(),
                'integrated_yaw_rad':float(sub[:,2].sum()*.05),
                'linear_x_mean_mps':float(np.mean(sub[:,0])),'angular_z_mean_radps':float(np.mean(sub[:,2])),
                'turn_in_place_fraction':float(np.mean((sub[:,0]==0)&(sub[:,1]==0)&(sub[:,2]!=0))),
                'reverse_fraction':float(np.mean(sub[:,0]<0)),
                'stop_fraction':float(np.mean(np.all(sub==0,axis=1)))}
            record['realized'][name]={'endpoint_base_xy_m':rpos[-1].tolist(),
                'path_length_m':float(np.linalg.norm(np.diff(rpos,axis=0),axis=1).sum()),
                'lateral_excursion_m':float(np.max(np.abs(rpos[:,1]))),
                'heading_change_rad':float(wrap_angle(yaw[n]-yaw[0]))}
        rows.append(record)
    if len(rows)!=112:raise ValueError('feature support')
    atomic(DOC/'candidate-geometry-kinematics.json',{'live_runs':112,'features':rows})
    marker={'stage':'FEATURES_PASS','sha256':_sha(DOC/'candidate-geometry-kinematics.json')};atomic(ROOT/'features.done.json',marker)
    return marker

def stage_endpoint_timing():
    """Supplement 2 s and 6 s arrival without redefining goal tolerance."""
    if read(ROOT/'a2.done.json')['series_sha256']!=_sha(ROOT/'tracking-series.jsonl.gz'):raise ValueError('A2 drift')
    records=[]
    for row in gzlines(ROOT/'tracking-series.jsonl.gz'):
        pos=np.asarray(row['world_positions']);end=np.asarray(row['world_path'][-1]);distance=np.linalg.norm(pos-end,axis=1)
        within=distance<=.2;first=np.flatnonzero(within)
        records.append({'run_id':row['run_id'],'state_id':row['state_id'],'generator':row['kind'],
            'reached_by_2s':bool(np.any(within[:41])),'reached_by_6s':bool(np.any(within)),
            'first_arrival_step':int(first[0]) if len(first) else None,
            'first_arrival_sec':float(first[0]*.05) if len(first) else None,
            'min_distance_2s_m':float(np.min(distance[:41])),'min_distance_6s_m':float(np.min(distance))})
    groups={g:{'runs':sum(x['generator']==g for x in records),
        'reached_by_2s':sum(x['reached_by_2s'] for x in records if x['generator']==g),
        'reached_by_6s':sum(x['reached_by_6s'] for x in records if x['generator']==g)} for g in ('nominal','nomad','bank')}
    atomic(DOC/'endpoint-reach-timing.json',{'goal_tolerance_m':.2,'state_count':7,'groups':groups,'runs':records})
    return groups
