"""Aggregate only completed seven-terrain pilot cases using frozen feasibility limits."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import yaml

from scripts.eat_nav_duplicate_recovery_smoke_analyze import load, metrics as prior_metrics, candidate_effect, command_divergence
from scripts.eat_nav_seven_state_pilot import atomic, xy_rms

NAMES=('nominal','slight_left','slight_right','strong_left','strong_right')


def classify_case(case, nominal, limits):
    ood=case.get('proposal_ood') or {}
    if ood.get('ood_over_train_p95') or ood.get('in_dem_footprint') is False or ood.get('self_intersects') or ood.get('speed_over_frozen_max'):
        return {'classification':'proposal_ood','level1':False,'level2':False,'level3':False}
    terminal=case.get('terminal') or {}
    l1=(not terminal.get('terminated',False) and
        math.degrees(case['body_tilt_max_rad'])<=limits['tilt_max_deg'])
    l2=l1 and case['progress_m']>=limits['progress_min_m'] and case['xy_displacement_m']>=limits['stack_displacement_min_m']
    threshold=max(0.,nominal['progress_m']*limits['progress_vs_nominal_min_fraction'])
    l3=(l2 and case['progress_m']>=threshold and
        case['end_goal_distance_m']<=nominal['end_goal_distance_m']+limits['goal_distance_allowance_m'] and
        case['path_tracking_error_m']<=limits['path_tracking_error_max_m'] and
        case['drawdown'] is not None and case['drawdown']>=limits['drawdown_min'])
    return {'classification':'supported','level1':bool(l1),'level2':bool(l2),'level3':bool(l3)}


def classify_state(cases, limits):
    if not cases:return 'measurement_invalid'
    unsupported=sum(v['classification']=='proposal_ood' for v in cases.values())
    if unsupported/len(cases)>limits['max_ood_fraction_for_classification']:
        return 'proposal_inconclusive'
    supported=[v for v in cases.values() if v['classification']=='supported']
    if sum(v['level3'] for v in supported)>=2:return 'multi_feasible'
    if sum(v['level1'] for v in supported)>=2:return 'physically_multi_feasible'
    if sum(v['level2'] for v in supported)>=1:return 'single_solution'
    return 'no_solution'


def command_cluster_count(commands, linear_min, angular_min):
    names=list(commands);parent=list(range(len(names)))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            a,b=commands[names[i]],commands[names[j]]
            if len(a)!=len(b):continue
            linear=math.sqrt(sum((x[0]-y[0])**2 for x,y in zip(a,b))/len(a))
            angular=math.sqrt(sum((x[2]-y[2])**2 for x,y in zip(a,b))/len(a))
            if linear<linear_min and angular<angular_min:parent[root(j)]=root(i)
    return len({root(i) for i in range(len(names))})


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_pose(row):
    terminal=row.get('terminal')
    return terminal['root_state'] if terminal and 'root_state' in terminal else row['post_root_state']


def path_tracking(trace,map_path):
    origin=[trace[0]['root_state'][j]-map_path[0][j] for j in range(2)]
    path=[tuple(p) for p in map_path]
    def distance(point):
        nearest=float('inf')
        for a,b in zip(path,path[1:]):
            dx,dy=b[0]-a[0],b[1]-a[1]
            t=max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/max(dx*dx+dy*dy,1e-9)))
            nearest=min(nearest,math.hypot(point[0]-(a[0]+t*dx),point[1]-(a[1]+t*dy)))
        return nearest
    return statistics.mean(distance((root_pose(row)[0]-origin[0],root_pose(row)[1]-origin[1])) for row in trace)


def add_metrics(metric,trace,map_path):
    initial=math.hypot(map_path[-1][0]-map_path[0][0],map_path[-1][1]-map_path[0][1])
    final_distance=initial-metric['progress_m']
    first=trace[0]['root_state'][:2];last=root_pose(trace[-1])[:2]
    metric.update(initial_goal_distance_m=initial,end_goal_distance_m=final_distance,
                  xy_displacement_m=math.hypot(last[0]-first[0],last[1]-first[1]),
                  path_tracking_error_m=path_tracking(trace,map_path))
    return metric


def analyze_state(root,index,bank,spec):
    folder=root/'pilot-runs'/f'state-{index:03d}'
    if (folder/'state-invalid.json').exists():
        return {'state_index':index,'classification':'measurement_invalid',
                'failure':json.loads((folder/'state-invalid.json').read_text())}
    if not (folder/'state-complete.json').exists():
        return {'state_index':index,'classification':'measurement_invalid','failure':'missing state-complete marker'}
    duplicates=[load(folder/'nominal','live',i) for i in range(3)]
    dup_rms=xy_rms([t for _,t in duplicates])
    if dup_rms>spec['replay']['duplicate_rms_max_m']:raise ValueError('duplicate gate mismatch')
    nominal_s,nominal_t=load(folder/'nominal','live',3)
    base_s,base_t=duplicates[0]
    cases={};raw_rows=[]
    for name in NAMES:
        summary,trace=(nominal_s,nominal_t) if name=='nominal' else load(folder/name,'live',0)
        map_path=json.loads((root/'pilot-inputs'/f'state-{index:03d}'/f'{name}-map-path.json').read_text())
        if sha(root/'pilot-inputs'/f'state-{index:03d}'/f'{name}-map-path.json')!=summary['fixed_map_path_sha256']:
            raise ValueError('candidate Path SHA mismatch')
        ood=next(row for row in bank['states'][index]['metrics'] if row['name']==name)
        metric=prior_metrics(summary,trace,bank['states'][index]['state']['goal'],map_path,ood)
        metric=add_metrics(metric,trace,map_path)
        effect=candidate_effect(base_t,trace)
        metric.update(name=name,xy_effect_rms_m=effect,duplicate_xy_rms_m=dup_rms,
                      rho=None if effect==0 else dup_rms/effect,
                      command_divergence=command_divergence(base_t,trace),
                      summary_sha256=sha(folder/name/f'live-{3 if name=="nominal" else 0}.json'),
                      trace_sha256=summary['trace_sha256'])
        cases[name]=metric
    nominal=cases['nominal']
    labels={name:classify_case(metric,nominal,spec['feasibility']) for name,metric in cases.items()}
    classification=classify_state(labels,spec['feasibility'])
    commands={name:metric['command_series'] for name,metric in cases.items()}
    distinct=spec['command_distinctness']
    clusters=command_cluster_count(commands,distinct['initial_linear_min_mps'],distinct['initial_angular_min_radps'])
    candidate_effects=[m['xy_effect_rms_m'] for n,m in cases.items() if n!='nominal']
    row={'state_index':index,'terrain_type':bank['states'][index]['state']['terrain_type'],
         'episode_key':bank['states'][index]['state']['episode_key'],
         'duplicate_xy_rms_m':dup_rms,'max_candidate_xy_effect_rms_m':max(candidate_effects),
         'rho_vs_max_effect':None if max(candidate_effects)==0 else dup_rms/max(candidate_effects),
         'candidate_effect_exceeds_duplicate':any(v>dup_rms for v in candidate_effects),
         'command_cluster_count':clusters,'classification':classification,
         'case_levels':labels,'cases':cases}
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--partial',action='store_true');a=p.parse_args()
    selection=json.loads((a.root/'pilot-state-selection.json').read_text())
    bank=json.loads(a.bank.read_text());spec=yaml.safe_load(a.spec.read_text())
    rows=[analyze_state(a.root,s['state_index'],bank,spec) for s in selection['selected']]
    complete=[r for r in rows if r['classification']!='measurement_invalid']
    invalid=len(rows)-len(complete)
    if not a.partial and not (a.root/'pilot-run-status.json').exists():
        raise ValueError('raw pilot still running')
    feasible={'selected_state_count':len(rows),'measurement_valid_count':len(complete),
              'measurement_invalid_count':invalid,'rows':rows,
              'terrain_generalization_claim':False,'formal_56_train_started':False,
              'validation_20_started':False}
    atomic(a.root/'pilot-feasibility.json',feasible)
    with gzip.open(a.root/'maneuver-rollouts.jsonl.gz.tmp','wt') as stream:
        for row in complete:
            for name,metric in row['cases'].items():
                stream.write(json.dumps({'state_index':row['state_index'],'terrain_type':row['terrain_type'],
                    'maneuver':name,'metrics':metric,'levels':row['case_levels'][name]},separators=(',',':'))+'\n')
    (a.root/'maneuver-rollouts.jsonl.gz.tmp').replace(a.root/'maneuver-rollouts.jsonl.gz')
    multi_effect=sum(r['candidate_effect_exceeds_duplicate'] for r in complete)
    cluster_states=sum(r['command_cluster_count']>=2 for r in complete)
    rho_low=sum(r['rho_vs_max_effect'] is not None and r['rho_vs_max_effect']<.25 for r in complete)
    continuation=(invalid<=1 and all(r['duplicate_xy_rms_m']<=.0105 for r in complete)
                  and multi_effect>=2 and cluster_states>=1 and len(complete)>=6)
    decision={'pilot_complete':len(complete)+invalid==7,'measurement_valid_count':len(complete),
              'measurement_invalid_count':invalid,'candidate_effect_gt_duplicate_state_count':multi_effect,
              'rho_below_0_25_state_count':rho_low,'command_cluster_ge_2_state_count':cluster_states,
              'continuation_conditions_met':bool(continuation),'formal_rollout_approved':False,
              'formal_56_train_started':False,'validation_20_started':False,
              'requires_user_approval_for_56_train':True}
    atomic(a.root/'pilot-decision.json',decision)
    print('PILOT_ANALYSIS',len(complete),'valid',invalid,'invalid','continuation',continuation)

if __name__=='__main__':main()
