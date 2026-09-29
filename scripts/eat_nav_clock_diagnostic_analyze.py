"""Align frozen nominal replay, follower callbacks, and bridge clocks."""
from __future__ import annotations
import bisect,gzip,json,math,re
from pathlib import Path
import numpy as np
from eat_nav_clock_diagnostic_stage0 import OUT,OLD,atomic_json,sha

ROOT=OUT/'clock-diagnostic-traces'
FOLLOWER_LOGS=OUT/'bridge-follower-logs'
NUMERIC=re.compile(r'^-?(?:\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?$')


def parse_clock_line(line):
    if 'CLOCK_DIAG ' not in line:return None
    payload=line.split('CLOCK_DIAG ',1)[1].strip().split()
    if not payload:return None
    row={'event':payload[0]}
    for token in payload[1:]:
        if '=' not in token:continue
        key,val=token.split('=',1)
        if NUMERIC.match(val):
            row[key]=float(val) if any(c in val for c in '.eE') else int(val)
        else:row[key]=val
    stamp=re.search(r'^\[[A-Z]+\] \[(\d+\.\d+)\]',line)
    row['log_system_ns']=int(float(stamp.group(1))*1e9) if stamp else None
    return row


def first_difference(a,b,key,tolerance=1e-6):
    for index,(left,right) in enumerate(zip(a,b)):
        x=left.get(key);y=right.get(key)
        if x is None or y is None:
            if x!=y:return index
            continue
        try:
            xx=np.asarray(x,dtype=float);yy=np.asarray(y,dtype=float)
            if xx.shape!=yy.shape or not np.allclose(xx,yy,rtol=0,atol=tolerance,equal_nan=True):return index
        except (TypeError,ValueError):
            if x!=y:return index
    if len(a)!=len(b):return min(len(a),len(b))
    return None


def nearest_previous(events,stamp,key):
    if stamp is None or not events:return None
    values=[x[key] for x in events]
    index=bisect.bisect_right(values,stamp)-1
    return events[index] if index>=0 else None


def load_trace(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def follower_rows(log_path):
    lines=log_path.read_text(errors='replace').splitlines()
    parsed=[x for line in lines if (x:=parse_clock_line(line))]
    events={kind:sorted([x for x in parsed if x['event']==kind],key=lambda x:x.get('node_ns',x.get('steady_ns',0)))
            for kind in ('path','tf','pre','post')}
    warnings=[]
    for line in lines:
        if 'Linear stuck detected' in line or 'Path progress stalled' in line:
            match=re.search(r'^\[WARN\] \[(\d+\.\d+)\]',line)
            warnings.append({'line':line,'log_system_ns':int(float(match.group(1))*1e9) if match else None,
                             'kind':'linear_stuck' if 'Linear stuck detected' in line else 'progress_stall'})
    return events,warnings


def align_run(index,trace,events):
    posts=events['post'];pres={x['seq']:x for x in events['pre']}
    start_mono=trace[0]['clock_diag']['step_start_monotonic_ns']
    first_post_ns=posts[0]['node_ns'] if posts else None
    rows=[]
    for step,record in enumerate(trace):
        clock=record['clock_diag'];response=clock.get('bridge_response') or {}
        diagnostics=response.get('command_diagnostics') or [None]
        diag=diagnostics[0] if diagnostics else None
        follower=nearest_previous(posts,diag.get('stamp_ns') if diag else None,'node_ns')
        if follower is not None and diag['stamp_ns']-follower['node_ns']>150_000_000:
            follower=None  # Early-return callbacks have no post snapshot.
        pre=pres.get(follower['seq']) if follower else None
        tf=nearest_previous(events['tf'],follower['steady_ns'] if follower else None,'steady_ns')
        rows.append({'run':index,'sim_step':step,'command_applied_step':step+1,'sim_nominal_elapsed_sec':clock['sim_nominal_elapsed_sec'],
            'step_wall_elapsed_sec':(clock['step_start_monotonic_ns']-start_mono)/1e9,
            'step_wall_duration_sec':(clock['step_end_monotonic_ns']-clock['step_start_monotonic_ns'])/1e9,
            'sim_context_time_before':clock.get('sim_context_time_before'),
            'sim_context_time_after':clock.get('sim_context_time_after'),
            'physics_step_count_before':clock.get('physics_step_count_before'),
            'physics_step_count_after':clock.get('physics_step_count_after'),
            'bridge_step_started_monotonic_ns':response.get('step_started_monotonic_ns'),
            'bridge_command_sequence':diag.get('sequence') if diag else None,
            'bridge_command_stamp_ns':diag.get('stamp_ns') if diag else None,
            'bridge_command_receive_monotonic_ns':diag.get('receive_monotonic_ns') if diag else None,
            'bridge_node_clock_ns':response.get('bridge_node_clock_ns'),
            'follower_callback_sequence':follower.get('seq') if follower else None,
            'follower_node_elapsed_sec':(follower['node_ns']-first_post_ns)/1e9 if follower and first_post_ns else None,
            'follower_node_clock_ns':follower.get('node_ns') if follower else None,
            'motion_window_elapsed_sec':pre.get('window_elapsed') if pre else None,
            'motion_window_ready':pre.get('ready') if pre else None,
            'motion_window_sample_count':pre.get('count') if pre else None,
            'commanded_distance_m':pre.get('cmd_dist') if pre else None,
            'actual_distance_m':pre.get('actual_dist') if pre else None,
            'actual_to_commanded_ratio':pre.get('ratio') if pre else None,
            'recovery_active':follower.get('recovery') if follower else None,
            'follower_command':record['command'],
            'tf_stamp_ns':int(tf['stamp_sec'])*1_000_000_000+int(tf['stamp_nsec']) if tf else None,
            'tf_age_at_follower_sec':(follower['node_ns']-(int(tf['stamp_sec'])*1_000_000_000+int(tf['stamp_nsec'])))/1e9 if tf and follower else None,
            'root_xy':record['root_state'][:2],
            'post_root_xy':record['post_root_state'][:2]})
    return rows


def compare_traces(records):
    keys=('command','actor_input','actor_latent','actor_action','root_state','post_root_state',
          'contact_force','raycast_hits','critic_observation','critic_value','reward')
    answer={}
    for i in range(1,len(records)):
        answer[f'run-{i:03d}_vs_run-000']={key:first_difference(records[0],records[i],key) for key in keys}
    return answer


def main():
    complete=json.loads((OUT/'stage-d2-complete.json').read_text())
    assert complete['status']=='complete' and complete['run_count']==3
    traces=[];events_list=[];warnings_list=[];aligned=[];motion=[];source=[]
    for index in range(3):
        folder=ROOT/f'run-{index:03d}'
        marker=json.loads((folder/'run-complete.json').read_text())
        assert sha(folder/'live-0.trace.json.gz')==marker['outputs_sha256']['live-0.trace.json.gz']
        trace=load_trace(folder/'live-0.trace.json.gz');assert len(trace)==120
        events,warnings=follower_rows(FOLLOWER_LOGS/f'reset-{index+1:04d}-branch-0.log')
        assert events['pre'] and events['post'] and events['tf'] and events['path']
        rows=align_run(index,trace,events)
        aligned.extend(rows)
        first_node=events['post'][0]['node_ns']
        by_post={r['seq']:r for r in events['post']}
        motion.extend([{'run':index,'callback_sequence':r['seq'],
            'node_elapsed_sec':(r['node_ns']-first_node)/1e9,
            'node_ns':r['node_ns'],'steady_ns':r['steady_ns'],
            'window_elapsed_sec':r['window_elapsed'],'ready':r['ready'],
            'sample_count':r['count'],'cmd_dist':r['cmd_dist'],'actual_dist':r['actual_dist'],
            'ratio':r['ratio'],'recovery_before':r['recovery'],
            'vx_after':by_post.get(r['seq'],{}).get('vx'),
            'recovery_after':by_post.get(r['seq'],{}).get('recovery')}
            for r in events['pre']])
        traces.append(trace);events_list.append(events);warnings_list.append(warnings)
        source.append({'run':index,'trace_sha256':sha(folder/'live-0.trace.json.gz'),
            'follower_log_sha256':sha(FOLLOWER_LOGS/f'reset-{index+1:04d}-branch-0.log'),
            'warning_events':warnings,'callback_count':len(events['post']),
            'first_ready_callback':next((x['seq'] for x in events['pre'] if x['ready']),None),
            'first_recovery_callback':next((x['seq'] for x in events['post'] if x['recovery']),None),
            'bridge_response_for_command_step_39_wall_elapsed_sec':rows[38]['step_wall_elapsed_sec'],
            'bridge_response_for_command_step_39_follower_node_elapsed_sec':rows[38]['follower_node_elapsed_sec'],
            'bridge_response_for_command_step_39_follower_callback_sequence':rows[38]['follower_callback_sequence']})
    old=[load_trace(OLD/'long-horizon/runs'/f'long-{i:03d}'/'live-0.trace.json.gz') for i in (5,6,7)]
    old_diff=compare_traces(old);new_diff=compare_traces(traces)
    identities=json.loads((OUT/'clock-diagnostic-run-identities.json').read_text())['runs']
    pilot=[load_trace(Path('/tmp/eatnav-candidate-coverage/pilot/runs')/r['source_run_id']/'live-0.trace.json.gz') for r in identities]
    parity=[]
    for i in range(3):
        comparison={}
        for reference_name,reference in (('pilot',pilot[i]),('original_long',old[i])):
            pair={key:first_difference(reference[:40],traces[i][:40],key) for key in
                  ('command','actor_input','actor_latent','actor_action','root_state','post_root_state','contact_force','raycast_hits','critic_value','reward')}
            xy_ref=np.asarray([x['post_root_state'][:2] for x in reference[:40]],dtype=float)
            xy_new=np.asarray([x['post_root_state'][:2] for x in traces[i][:40]],dtype=float)
            pair['xy_rms_40_m']=float(np.sqrt(np.mean(np.sum((xy_ref-xy_new)**2,axis=1))))
            comparison[reference_name]=pair
        parity.append({'run':i,'comparison':comparison})
    atomic_json(OUT/'first40-parity-clock.json',{'source_run_ids':[r['source_run_id'] for r in identities],
        'rows':parity,'note':'Diagnostic logging may change wall scheduling; differences are reported, not filtered.'})
    per_run=[aligned[i*120:(i+1)*120] for i in range(3)]
    clock_diff={}
    for i in (1,2):
        left,right=per_run[0],per_run[i]
        clock_diff[f'run-{i:03d}_vs_run-000']={
            'sim_nominal_time':first_difference(left,right,'sim_nominal_elapsed_sec',1e-12),
            'wall_elapsed_gt_0_5_sec':first_difference(left,right,'step_wall_elapsed_sec',.5),
            'follower_callback_sequence':first_difference(left,right,'follower_callback_sequence',0),
            'motion_window_ready':first_difference(left,right,'motion_window_ready',0),
            'commanded_distance_gt_0_1_m':first_difference(left,right,'commanded_distance_m',.1),
            'actual_distance_gt_0_05_m':first_difference(left,right,'actual_distance_m',.05),
            'recovery_active':first_difference(left,right,'recovery_active',0),
            'tf_age_gt_0_05_sec':first_difference(left,right,'tf_age_at_follower_sec',.05),
        }
    atomic_json(OUT/'clock-alignment.json',{'status':'complete','source':source,'rows':aligned,
        'join_method':'nearest follower post with node_ns <= bridge TwistStamped stamp; response during env.step(k) supplies the trace command in step(k+1)',
        'command_offset_verified':'For sampled k=0,1,38,39,40, bridge response command equals trace[k+1] after float32 rounding, not trace[k]',
        'uncertainty':'original failure has no internal follower callback trace; new callback timing is direct diagnostic evidence only'})
    atomic_json(OUT/'follower-motion-window-trace.json',{'source':source,'rows':motion,
        'missing_fields':'steady/system time unavailable for original production follower internal state'})
    atomic_json(OUT/'first-divergence-clock.json',{
        'original_frozen_120_step_pairs':old_diff,'new_diagnostic_120_step_pairs':new_diff,
        'new_clock_signal_divergence':clock_diff,'warnings_by_run':warnings_list,
        'first_observed_vs_root_cause':'First observed divergence is a measured signal; causal classification requires source timing and pre-event state evidence.',
        'original_internal_clock_fields':None,
        'source_sha256':source})
    print('D3_ANALYZED',json.dumps({'new_diff':new_diff,'clock_diff':clock_diff,
                                  'warnings':[len(x) for x in warnings_list]},sort_keys=True))

if __name__=='__main__':main()
