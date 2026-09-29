"""Summarize the frozen three-state fresh-process maneuver smoke."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

NAMES=('nominal','slight_left','slight_right','strong_left','strong_right')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def atomic(path,obj):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)


def load(root,stage,index):
    summary=root/f'{stage}-{index}.json';trace=root/f'{stage}-{index}.trace.json.gz';marker=root/f'{stage}-{index}.marker.json'
    for path in (summary,trace,marker):
        if not path.exists():raise FileNotFoundError(path)
    s=json.loads(summary.read_text());m=json.loads(marker.read_text())
    if s['trace_sha256']!=sha(trace):raise ValueError(f'trace SHA mismatch {trace}')
    for path in (summary,trace):
        if m['outputs_sha256'].get(str(path))!=sha(path) and m['outputs_sha256'].get(next((k for k in m['outputs_sha256'] if k.endswith('/'+path.name)),''))!=sha(path):
            raise ValueError(f'marker SHA mismatch {path}')
    with gzip.open(trace,'rt') as f:t=json.load(f)
    return s,t


def post_root(row):
    terminal=row.get('terminal')
    return terminal['root_state'] if terminal and 'root_state' in terminal else row['post_root_state']


def xy_rms(traces):
    n=min(map(len,traces))
    if n!=40:raise ValueError(f'duplicate trace length {n}, expected 40')
    origin=[s[0]['root_state'][:2] for s in traces]
    diffs=[post_root(traces[i][k])[j]-origin[i][j]-(post_root(traces[0][k])[j]-origin[0][j])
           for i in range(len(traces)) for k in range(n) for j in range(2)]
    return math.sqrt(sum(x*x for x in diffs)/len(diffs))


def quat_roll_pitch(q):
    w,x,y,z=q
    roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch=math.asin(max(-1.,min(1.,2*(w*y-z*x))))
    return roll,pitch


def metrics(summary,trace,goal,map_path,ood):
    vals=[r['critic_value'] for r in trace]
    rewards=[r['reward'] for r in trace]
    gamma=.99
    terminal=trace[-1]['terminal']
    end=summary['suffix_end_critic_value']
    if terminal and terminal.get('terminated'):
        boot=0.;kind='true_terminal_zero'
    elif terminal and terminal.get('truncated'):
        boot=vals[-1];kind='timeout_preterminal'
    else:
        boot=end;kind='endpoint_value'
    return_h=sum(gamma**k*r for k,r in enumerate(rewards))+(gamma**len(rewards))*boot
    series=vals+([] if end is None else [end])
    anchor=trace[0]['root_state'][:2]
    origin=[anchor[j]-map_path[0][j] for j in range(2)]
    final=[post_root(trace[-1])[j]-origin[j] for j in range(2)]
    goal_xy=map_path[-1]
    distance=lambda a,b:math.hypot(a[0]-b[0],a[1]-b[1])
    tilt=[max(abs(v) for v in quat_roll_pitch(post_root(r)[3:7])) for r in trace]
    contact=[max(math.sqrt(sum(v*v for v in f)) for f in r['contact_force']) for r in trace]
    return {'delta_v':None if end is None else end-vals[0],
            'advantage_h':return_h-vals[0],
            'drawdown':min(series)-vals[0],
            'recovery':None if end is None else end-min(series),
            'return_h':return_h,'bootstrap_kind':kind,
            'terminal':terminal,'step_count':len(trace),
            'progress_m':distance(map_path[0],goal_xy)-distance(final,goal_xy),
            'body_tilt_max_rad':max(tilt),'contact_max_n':max(contact),
            'proposal_ood':ood,'elapsed_sec':summary['elapsed_sec'],
            'root_xy':[post_root(r)[:2] for r in trace],
            'command_series':[r['command'] for r in trace],
            'critic_series':vals,'reward_series':rewards}


def candidate_effect(a,b):
    n=min(len(a),len(b))
    diffs=[post_root(a[k])[j]-post_root(b[k])[j] for k in range(n) for j in range(2)]
    return math.sqrt(sum(x*x for x in diffs)/len(diffs))


def command_divergence(a,b,threshold=1e-4):
    n=min(len(a),len(b))
    distances=[math.sqrt(sum((a[k]['command'][j]-b[k]['command'][j])**2 for j in range(3))) for k in range(n)]
    return {'first_step':next((i for i,x in enumerate(distances) if x>threshold),None),
            'max_l2':max(distances),'rms_l2':math.sqrt(sum(x*x for x in distances)/n),
            'threshold_l2':threshold}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--partial',action='store_true');a=p.parse_args()
    root=a.root;bank=json.loads(a.bank.read_text())
    selected=json.loads((root/'smoke/smoke-input-freeze.json').read_text())['state_indices']
    rows=[]
    for index in selected:
        folder=root/'smoke'/f'state-{index:03d}'
        if index==19:
            duplicates=[load(root,'live',i) for i in range(3)]
        else:
            try:duplicates=[load(folder/'nominal','live',i) for i in range(3)]
            except FileNotFoundError:
                if a.partial:continue
                raise
        dup_rms=xy_rms([t for _,t in duplicates])
        if index==19 and abs(dup_rms)>1e-12:raise ValueError('state 19 duplicate changed')
        if len({s['fixed_map_path_sha256'] for s,_ in duplicates})!=1:raise ValueError('nominal Path mismatch')
        critical=('root_state','joint_position','joint_velocity','command_manager','policy_pre_noise',
                  'policy_post_noise_pre_delay','policy_actor_input','actor_latent','actor_action',
                  'critic_observation','critic_value','action_manager_action','latency_substeps',
                  'latency_seconds','latency_policy_obs_prev','action_delay_buffers','dem_history_sha256')
        anchor_exact=all(all(s['anchor'][field]==duplicates[0][0]['anchor'][field] for s,_ in duplicates[1:]) for field in critical)
        command_exact=all([r['command'] for r in t]==[r['command'] for r in duplicates[0][1]] for _,t in duplicates[1:])
        if not anchor_exact or not command_exact:raise ValueError(f'state {index} nominal anchor/command mismatch')
        nominal_summary,nominal_trace=duplicates[0]
        cases={}
        for name in NAMES:
            if name=='nominal':summary,trace=nominal_summary,nominal_trace
            else:
                try:summary,trace=load(folder/name,'live',0)
                except FileNotFoundError:
                    if a.partial:continue
                    raise
            fixed=folder/f'{name}-map-path.json'
            if sha(fixed)!=summary['fixed_map_path_sha256']:raise ValueError('candidate Path mismatch')
            ood=next(m for m in bank['states'][index]['metrics'] if m['name']==name)
            metric=metrics(summary,trace,bank['states'][index]['state']['goal'],json.loads(fixed.read_text()),ood)
            effect=candidate_effect(nominal_trace,trace)
            metric.update({'name':name,'xy_effect_rms_m':effect,'effect_comparable_steps':min(len(nominal_trace),len(trace)),
                           'duplicate_xy_rms_m':dup_rms,
                           'rho':None if effect==0 else dup_rms/effect,
                           'command_divergence':command_divergence(nominal_trace,trace),
                           'result_sha256':sha((folder/name/f'live-0.json') if name!='nominal' or index!=19 else root/'live-0.json'),
                           'trace_sha256':summary['trace_sha256']})
            cases[name]=metric
        if not a.partial and len(cases)!=5:raise ValueError('incomplete 5-case state')
        rows.append({'state_index':index,'state':bank['states'][index]['state'],
                     'duplicate_xy_rms_m':dup_rms,'anchor_critical_exact_equal':anchor_exact,
                     'nominal_command_exact_equal':command_exact,'duplicate_trace_sha256':[s['trace_sha256'] for s,_ in duplicates],
                     'cases':cases,
                     'all_commands_substantially_identical':all(c['command_divergence']['first_step'] is None for n,c in cases.items() if n!='nominal'),
                     'all_candidate_effect_at_or_below_duplicate':all(c['xy_effect_rms_m']<=dup_rms for n,c in cases.items() if n!='nominal')})
    final={'state_count':len(rows),'selected_state_indices':selected,'completed_state_indices':[r['state_index'] for r in rows],
           'formal_rollout_started':False,'bootstrap_generalization_claim':False,'rows':rows}
    if not a.partial and len(rows)!=3:raise ValueError('missing smoke state')
    atomic(root/('three-state-maneuver-smoke.partial.json' if a.partial else 'three-state-maneuver-smoke.json'),final)
    print('SMOKE_ANALYSIS',[(r['state_index'],r['duplicate_xy_rms_m'],[(n,c['xy_effect_rms_m'],c['command_divergence']['first_step']) for n,c in r['cases'].items()]) for r in rows])

if __name__=='__main__':main()
