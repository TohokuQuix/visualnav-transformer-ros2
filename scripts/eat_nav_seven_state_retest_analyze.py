"""Open the seven-state pilot only after three clean State 42 replays."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from scripts.eat_nav_seven_state_pilot import xy_rms, atomic


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_tree(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,list):return all(finite_tree(x) for x in value)
    if isinstance(value,dict):return all(finite_tree(x) for x in value.values())
    return True


def analyze(root, fixed_path):
    summaries=[];traces=[];checks=[]
    for repeat in range(3):
        summary=root/f'live-{repeat}.json'
        trace=root/f'live-{repeat}.trace.json.gz'
        marker=root/f'live-{repeat}.marker.json'
        for path in (summary,trace,marker):
            if not path.exists():
                return {'classification':'infrastructure_invalid','reason':f'missing clean repeat {repeat}: {path.name}',
                        'pilot_allowed':False}
        s=json.loads(summary.read_text());m=json.loads(marker.read_text())
        if s['trace_sha256']!=sha(trace):raise ValueError('trace hash mismatch')
        for source,digest in m['outputs_sha256'].items():
            host=root/Path(source).name
            if sha(host)!=digest:raise ValueError('marker output hash mismatch')
        with gzip.open(trace,'rt') as stream:t=json.load(stream)
        readiness=s['follower_startup']['readiness']
        pre=readiness['prepath'];post=readiness['postpath']
        valid=(s['suffix_steps']==40 and s['fresh_command_response_count']==41 and len(t)==40
               and s['fixed_map_path_sha256']==sha(fixed_path)
               and all(v==0 for v in pre['command_sequence'])
               and all(pre['followers_alive']) and all(pre['tf_ready'])
               and all(x>=2 for x in pre['status_counts'])
               and all(x>=1 for x in post['command_sequence'])
               and readiness['path_resend_count']<=1
               and all(x>=readiness['path_first_wall_time'] for x in post['first_command_wall_time'])
               and s['anchor_error_vs_logged']['position_m']<=.02
               and s['anchor_error_vs_logged']['orientation_rad']<=.03
               and s['anchor_error_vs_logged']['dem_rms_m']<=.03
               and s['anchor_error_vs_logged']['mask_iou']>=.95
               and s['anchor_error_vs_logged']['dem_history_sha_match']
               and s['anchor_error_vs_logged']['prefix_command_max_abs']<=1e-5
               and finite_tree([row['command'] for row in t])
               and finite_tree([row['root_state'] for row in t])
               and finite_tree([row['critic_value'] for row in t]))
        checks.append({'repeat_id':repeat,'valid':valid,'summary_sha256':sha(summary),
                       'trace_sha256':sha(trace),'fresh_command_responses':s['fresh_command_response_count'],
                       'path_resend_count':readiness['path_resend_count'],
                       'prepath_command_sequence':pre['command_sequence'],
                       'postpath_command_sequence':post['command_sequence'],
                       'first_command_delay_ms':[(x-readiness['path_first_wall_time'])*1000 for x in post['first_command_wall_time']],
                       'anchor_error':s['anchor_error_vs_logged'],
                       'suffix_steps':len(t),'termination':t[-1]['terminal']})
        summaries.append(s);traces.append(t)
    rms=xy_rms(traces)
    path_same=len({s['fixed_map_path_sha256'] for s in summaries})==1
    critical=('root_state','joint_position','joint_velocity','command_manager','policy_pre_noise',
              'policy_post_noise_pre_delay','policy_actor_input','actor_latent','actor_action',
              'critic_observation','critic_value','action_manager_action','latency_substeps',
              'latency_seconds','latency_policy_obs_prev','action_delay_buffers','dem_history_sha256')
    anchor_same=all(all(s['anchor'][key]==summaries[0]['anchor'][key] for key in critical) for s in summaries[1:])
    command_same=all([row['command'] for row in t]==[row['command'] for row in traces[0]] for t in traces[1:])
    valid=all(c['valid'] for c in checks) and path_same and anchor_same and command_same and rms<=.0105
    return {'classification':'valid' if valid else 'infrastructure_invalid',
            'pilot_allowed':bool(valid),'state_index':42,'repeat_count':3,
            'duplicate_xy_rms_m':rms,'duplicate_limit_m':.0105,'path_same':path_same,
            'anchor_critical_same':anchor_same,'command_series_same':command_same,
            'checks':checks,'formal_56_train_started':False,'validation_20_started':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--fixed-path',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.root,a.fixed_path);atomic(a.output,result)
    print('STATE42_RETEST_GATE',result['classification'],result.get('duplicate_xy_rms_m'),flush=True)

if __name__=='__main__':main()
