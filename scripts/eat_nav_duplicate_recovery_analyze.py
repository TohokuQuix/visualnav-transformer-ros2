"""State-unit duplicate and first-divergence audit for fresh-prefix repeats."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def numbers(value):
    if isinstance(value,bool) or value is None:return []
    if isinstance(value,(int,float)):
        return [float(value)] if np.isfinite(value) else []
    if isinstance(value,list):
        result=[]
        for item in value:result+=numbers(item)
        return result
    if isinstance(value,dict):
        result=[]
        for key in sorted(value):result+=numbers(value[key])
        return result
    return []


def compare(values):
    digests=[hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=True).encode()).hexdigest() for v in values]
    numeric=[np.asarray(numbers(v),dtype=np.float64) for v in values]
    sizes=[len(x) for x in numeric]
    result={'exact_equal':len(set(digests))==1,'sha256':digests,'numeric_lengths':sizes}
    if len(set(sizes))==1 and sizes[0]:
        diffs=np.stack([x-numeric[0] for x in numeric])
        result.update(max_abs=float(np.max(np.abs(diffs))),rms=float(np.sqrt(np.mean(diffs*diffs))))
    else:result.update(max_abs=None,rms=None)
    return result


def first_divergence(traces,keys,tolerance=1e-7):
    length=min(len(t) for t in traces)
    series=[]
    for step in range(length):
        objects=[{key:run[step].get(key) for key in keys} for run in traces]
        series.append(compare(objects))
    max_abs=[row['max_abs'] for row in series]
    rms=[row['rms'] for row in series]
    first=next((i for i,v in enumerate(max_abs) if not series[i]['exact_equal'] or v is None or v>tolerance),None)
    return {'keys':keys,'first_divergence_step':first,'tolerance':tolerance,
            'max_abs_by_step':max_abs,'rms_by_step':rms,
            'equal_all_steps':first is None,'common_step_count':length,
            'trace_lengths':[len(t) for t in traces]}


def xy_rms(traces,anchors):
    length=min(len(t) for t in traces)
    if length!=40:raise ValueError(f'expected 40-step duplicate, got {length}')
    xy=np.asarray([[row['post_root_state'][:2] for row in run[:length]] for run in traces],dtype=np.float64)
    origins=np.asarray([a['root_state'][:2] for a in anchors],dtype=np.float64)
    relative=xy-origins[:,None]
    delta=relative-relative[:1]
    return {'rms_m':float(np.sqrt(np.mean(delta*delta))),
            'max_abs_m':float(np.max(np.abs(delta))),
            'per_run_rms_to_first_m':[float(np.sqrt(np.mean((relative[i]-relative[0])**2))) for i in range(len(relative))],
            'step_rms_to_first_m':np.sqrt(np.mean(delta*delta,axis=(0,2))).astype(float).tolist()}


def read_runs(root,stage,count=3):
    summaries=[];traces=[]
    for index in range(count):
        result=root/f'{stage}-{index}.json';trace_path=root/f'{stage}-{index}.trace.json.gz';marker=root/f'{stage}-{index}.marker.json'
        if not result.exists() or not trace_path.exists() or not marker.exists():raise FileNotFoundError(f'incomplete {stage} run {index}')
        marker_data=json.loads(marker.read_text())
        if marker_data['outputs_sha256'][str(result)]!=sha(result) or marker_data['outputs_sha256'][str(trace_path)]!=sha(trace_path):
            raise ValueError(f'{stage} run {index} marker SHA mismatch')
        row=json.loads(result.read_text())
        if row['trace_sha256']!=sha(trace_path):raise ValueError('trace SHA mismatch')
        summaries.append(row)
        with gzip.open(trace_path,'rt') as stream:traces.append(json.load(stream))
    return summaries,traces


def analyze_live(root):
    summaries,traces=read_runs(root,'live')
    if len({row['fixed_map_path_sha256'] for row in summaries})!=1:
        raise ValueError('duplicate Paths differ')
    if len({json.dumps(row['source_sha256'],sort_keys=True) for row in summaries})!=1:
        raise ValueError('duplicate source hashes differ')
    anchors=[row['anchor'] for row in summaries]
    fields=sorted(set.intersection(*[set(x) for x in anchors]))
    field_comparison={field:compare([anchor[field] for anchor in anchors]) for field in fields}
    post=[row['post_follower_anchor'] for row in summaries]
    post_comparison={field:compare([anchor[field] for anchor in post]) for field in sorted(set.intersection(*[set(x) for x in post]))}
    critical=['root_state','joint_position','joint_velocity','command_manager','policy_pre_noise',
              'policy_post_noise_pre_delay','policy_actor_input','actor_latent','actor_action',
              'critic_observation','critic_value','action_manager_action','latency_substeps',
              'latency_seconds','latency_policy_obs_prev','action_delay_buffers','dem_history_sha256']
    anchor_critical_equal=all(field_comparison[field]['exact_equal'] for field in critical)
    anchor_report={'source_sha256':summaries[0]['source_sha256'],'field_comparison':field_comparison,
        'post_follower_field_comparison':post_comparison,
        'logged_anchor_errors':[row['anchor_error_vs_logged'] for row in summaries],
        'critical_fields':critical,'critical_exact_equal':anchor_critical_equal,
        'first_unequal_field':next((field for field in fields if not field_comparison[field]['exact_equal']),None),
        'path_sha256':[row['fixed_map_path_sha256'] for row in summaries],
        'snapshot_restore_used':False}
    atomic(root/'prefix-replay-anchor-comparison.json',anchor_report)
    signals={
        'follower_command_from_bridge':['command'],
        'command_manager':['command'],
        'pre_noise_observation':['pre_noise'],
        'post_noise_pre_delay':['post_noise'],
        'post_delay_actor_input':['actor_input'],
        'actor_latent':['actor_latent'],
        'actor_action':['actor_action'],
        'processed_action':['processed_action'],
        'joint_target':['joint_target'],
        'joint_state':['joint_position','joint_velocity'],
        'root_state':['root_state','post_root_state'],
        'contact':['contact_force'],
        'raycast_dem':['raycast_hits','dem'],
        'critic':['critic_observation','critic_value'],
        'reward':['reward']}
    divergence={name:first_divergence(traces,keys) for name,keys in signals.items()}
    ordered=sorted(((v['first_divergence_step'],k) for k,v in divergence.items() if v['first_divergence_step'] is not None))
    atomic(root/'first-divergence-report.json',{'source':'fresh_prefix_live_nominal_3_runs',
        'signals':divergence,'earliest':ordered[:8],
        'follower_command_note':'runner records command-manager value immediately after each fresh bridge command; raw ROS response series was not separately serialized',
        'confirmed_cause_region':('none_detected' if not ordered else 'ROS/command boundary' if ordered[0][1] in ('follower_command_from_bridge','command_manager') else 'observation/physics boundary'),
        'specific_mechanism_unproven':True})
    xy=xy_rms(traces,anchors)
    command=compare([[row['command'] for row in run] for run in traces])
    result={'state_count':1,'run_count':3,'independence_unit':'fresh_simulator_process',
            'path_sha256':summaries[0]['fixed_map_path_sha256'],'source_sha256':summaries[0]['source_sha256'],
            'anchor_critical_exact_equal':anchor_critical_equal,'xy':xy,'command':command,
            'failures_included':True,'snapshot_restore_used':False,
            'follower_startup':[row['follower_startup'] for row in summaries],
            'full_trace_sha256':[row['trace_sha256'] for row in summaries]}
    atomic(root/'prefix-replay-duplicates.json',result)
    limit=.0105
    gate_pass=(xy['rms_m']<=limit and anchor_critical_equal and all(row['anchor_error_vs_logged']['dem_history_sha_match'] for row in summaries))
    preliminary={'stage':'live_duplicate','decision':'pass' if gate_pass else 'fail',
                 'xy_rms_m':xy['rms_m'],'limit_m':limit,'anchor_critical_exact_equal':anchor_critical_equal,
                 'recorded_command_required':not gate_pass,'formal_rollout_allowed':False}
    atomic(root/'duplicate-gate-preliminary.json',preliminary)
    if not gate_pass:
        commands=np.asarray([row['command'] for row in traces[0]],dtype=np.float32)
        if commands.shape!=(40,3):raise ValueError('live follower command sequence shape mismatch')
        path=root/'recorded-command.npz'
        if path.exists():
            with np.load(path) as data:
                if not np.array_equal(data['command'],commands):raise ValueError('recorded command source changed')
        else:
            tmp=path.with_name('recorded-command.tmp.npz');np.savez_compressed(tmp,command=commands);tmp.replace(path)
        atomic(root/'recorded-command-source.json',{'live_run':0,'live_trace_sha256':summaries[0]['trace_sha256'],
            'command_sha256':sha(path),'shape':list(commands.shape),'prefix_and_path_unchanged':True})
    else:
        atomic(root/'recorded-command-control.json',{'status':'not_required','reason':'live duplicate gate passed'})
    print('RECOVERY_LIVE_ANALYSIS',json.dumps(preliminary),flush=True)
    return preliminary


def analyze_recorded(root):
    summaries,traces=read_runs(root,'recorded')
    anchors=[row['anchor'] for row in summaries]
    anchor_cmp={field:compare([a[field] for a in anchors]) for field in anchors[0]}
    xy=xy_rms(traces,anchors)
    command=compare([[row['command'] for row in run] for run in traces])
    result={'status':'complete','source_sha256':summaries[0]['source_sha256'],
            'path_sha256':[r['fixed_map_path_sha256'] for r in summaries],
            'anchor_exact_equal':all(v['exact_equal'] for v in anchor_cmp.values()),
            'xy':xy,'command':command,'first_divergence':{key:first_divergence(traces,[key]) for key in
                ('pre_noise','post_noise','actor_input','actor_action','processed_action','joint_position','root_state','contact_force','raycast_hits','critic_value','reward')},
            'interpretation':'ROS timing region if XY variation disappears; anchor/observation/physics region if it remains'}
    atomic(root/'recorded-command-control.json',result)
    live=json.loads((root/'duplicate-gate-preliminary.json').read_text())
    final={'decision':'measurement_invalid' if live['decision']!='pass' else 'pass',
           'live_xy_rms_m':live['xy_rms_m'],'recorded_xy_rms_m':xy['rms_m'],
           'limit_m':live['limit_m'],'cause_region':('ROS follower/timing' if xy['rms_m']<=live['limit_m'] else 'anchor/observation/latency/physics'),
           'formal_56_train_20_validation_allowed':False,
           'next_step':'stop_and_report' if live['decision']!='pass' else 'at_most_3_state_smoke'}
    atomic(root/'duplicate-gate.json',final)
    print('RECOVERY_RECORDED_ANALYSIS',json.dumps(final),flush=True)
    return final


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['live','recorded']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.stage=='live':analyze_live(a.output)
    else:analyze_recorded(a.output)
if __name__=='__main__':main()
