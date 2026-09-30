"""Summarize frozen four-state oracle outcomes; never select candidates online."""
from __future__ import annotations
import gzip, hashlib, json, math, time
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.command_preview_long_horizon import endpoint_metrics, post_root

ROOT=Path('/tmp/eatnav-candidate-coverage/exact-stamp-stage-b')
PREFLIGHT=Path('/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon')
BASE=Path('/tmp/eatnav-candidate-coverage')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def metrics_at(summary,trace,steps):
    if len(trace)<steps:
        return {'status':'terminated_before_horizon','available_steps':len(trace),
                'termination':trace[-1].get('terminal') if trace else None}
    if len(trace)>steps:end=trace[steps]['critic_value']
    elif summary['suffix_end_critic_value'] is not None:end=summary['suffix_end_critic_value']
    else:
        termination=trace[-1].get('terminal')
        end=0. if termination and termination.get('terminated') else trace[-1]['critic_value']
    path=summary['map_path']
    return {'status':'observed',**endpoint_metrics(trace,steps,end,path[0],path[-1],path)}

def first_differences(a,b):
    signals=('command','actor_input','actor_latent','actor_action','critic_observation',
             'critic_value','root_state','contact_force','raycast_hits','reward','dem')
    result={}
    for signal in signals:
        for k,(left,right) in enumerate(zip(a,b)):
            x=left[signal]['transformed_sha256'] if signal=='dem' else left[signal]
            y=right[signal]['transformed_sha256'] if signal=='dem' else right[signal]
            if x!=y:
                try:error=float(np.max(np.abs(np.asarray(x,dtype=float)-np.asarray(y,dtype=float))))
                except (ValueError,TypeError):error=None
                result[signal]={'step':k,'max_abs':error}
                break
    return result

def main():
    started=time.monotonic()
    complete=read(ROOT/'stage-b-runs-complete.json')
    if complete['run_count']!=20:raise ValueError('20 completed runs required')
    frozen=read(BASE/'sim-clock-long-horizon/clock-corrected-run-identities.json')['runs']
    previous=read(BASE/'feasible-transfer/feasibility-by-candidate.json')
    feasibility={(x['state_id'],x['candidate_id']):x for x in previous['rows']}
    rows=[];traces={};summaries={}
    for identity in frozen:
        j=identity['order'];run_id=f'long-{j:03d}';folder=ROOT/'stage-b/runs'/run_id
        marker=read(folder/'run-complete.json')
        if marker['frozen_identity']!=identity:raise ValueError('frozen identity changed '+run_id)
        for name,digest in marker['files_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('output SHA mismatch '+run_id+' '+name)
        summary=read(folder/'live-0.json')
        with gzip.open(folder/'live-0.trace.json.gz','rt') as stream:trace=json.load(stream)
        if len(trace)!=120:raise ValueError('not 120 steps '+run_id)
        traces[run_id]=trace;summaries[run_id]=summary
        root=np.asarray([post_root(x)[:2] for x in trace],dtype=float)
        command=np.asarray([x['command'] for x in trace],dtype=float)
        constraint=feasibility[(identity['state_id'],identity['candidate_id'])]['condition']
        rows.append({'run_id':run_id,'state_id':identity['state_id'],'terrain':identity['terrain'],
          'candidate_id':identity['candidate_id'],'kind':identity['kind'],'seed':identity['seed'],
          'repeat':identity['repeat'],'constraint_valid':constraint['constraint_valid'],
          'constraint':constraint,'trace_sha256':sha(folder/'live-0.trace.json.gz'),
          'map_path_sha256':identity['map_path_sha256'],'two_second':metrics_at(summary,trace,40),
          'six_second':metrics_at(summary,trace,120),'root_xy':root.tolist(),
          'command':command.tolist(),'critic_value':[x['critic_value'] for x in trace],
          'reward':[x['reward'] for x in trace],'termination_reason':trace[-1].get('terminal')})
    states=[]
    for offset in range(0,20,5):
        group=rows[offset:offset+5];sid=group[0]['state_id']
        duplicate=read(ROOT/'duplicate-gates'/f'{sid}.json')
        nominal=group[:3];nomad=group[3];bank=group[4]
        ranks={}
        for horizon,key in (('2s','two_second'),('6s','six_second')):
            ordered=sorted([x for x in group if x[key]['status']=='observed'],
                 key=lambda x:x[key]['delta_v'],reverse=True)
            ranks[horizon]=[{'run_id':x['run_id'],'candidate_id':x['candidate_id'],
                'kind':x['kind'],'delta_v':x[key]['delta_v']} for x in ordered]
        first=first_differences(traces[nominal[0]['run_id']],traces[nominal[1]['run_id']])
        nroot=np.asarray(nomad['root_xy'])-np.asarray(nomad['root_xy'][0])
        broot=np.asarray(bank['root_xy'])-np.asarray(bank['root_xy'][0])
        ncmd=np.asarray(nomad['command']);bcmd=np.asarray(bank['command'])
        separation={'realized_xy_rms_m':float(np.sqrt(np.mean((nroot-broot)**2))),
            'command_linear_rms_mps':float(np.sqrt(np.mean((ncmd[:,0]-bcmd[:,0])**2))),
            'command_angular_rms_radps':float(np.sqrt(np.mean((ncmd[:,2]-bcmd[:,2])**2)))}
        separation['distinct']=bool(separation['realized_xy_rms_m']>.05 or
            separation['command_linear_rms_mps']>.03 or separation['command_angular_rms_radps']>.05)
        comparison={}
        for key in ('two_second','six_second'):
            n,b=nomad[key],bank[key]
            comparison[key]={'nominal_delta_v':[x[key].get('delta_v') for x in nominal],
                'nomad':n,'bank':b,
                'bank_minus_nomad_delta_v':b['delta_v']-n['delta_v'] if n['status']==b['status']=='observed' else None,
                'bank_minus_nomad_progress_m':b['forward_goal_progress_m']-n['forward_goal_progress_m'] if n['status']==b['status']=='observed' else None}
        states.append({'state_id':sid,'terrain':group[0]['terrain'],'nominal_duplicate':duplicate,
            'nominal_first_differences':first,'candidate_run_ids':[nomad['run_id'],bank['run_id']],
            'candidate_separation':separation,'rankings':ranks,'ranking_changed':
            [x['run_id'] for x in ranks['2s']]!=[x['run_id'] for x in ranks['6s']],
            'outcomes':comparison,'nomad_constraint_valid':nomad['constraint_valid'],
            'bank_constraint_valid':bank['constraint_valid']})
    compact=[{k:v for k,v in row.items() if k not in ('root_xy','command','critic_value','reward')}
             for row in rows]
    result={'run_count':20,'state_count':4,'scope':'four frozen train states; 6 seconds',
        'statistical_unit':'state','all_terrain_generalization_claim':False,
        'online_evaluator_selection_used':False,'states':states,'runs':compact,
        'analysis_elapsed_sec':time.monotonic()-started}
    atomic(ROOT/'stage-b-results.json',result)
    decision={'clock_tf_protocol':'pass','duplicate_reproducibility':'pass',
        'long_horizon_candidate_coverage':'four-state descriptive oracle only',
        'gate_n':'inconclusive','formal_pass':False,
        'reason':'Four train states cannot establish long-horizon feasible coverage across terrains.',
        'recommended_next':'Review per-state oracle evidence before considering a separate formal validation.',
        'results_sha256':sha(ROOT/'stage-b-results.json')}
    atomic(ROOT/'final-decision.json',decision)
    atomic(ROOT/'stage-b-analysis.done.json',{'status':'complete','results_sha256':sha(ROOT/'stage-b-results.json'),
        'decision_sha256':sha(ROOT/'final-decision.json')})
    print('EXACT_STAGE_B_ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':main()
