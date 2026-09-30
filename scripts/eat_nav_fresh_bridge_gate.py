"""Outcome-blind three-run old/fresh measurement equivalence gate."""
from __future__ import annotations
import argparse,gzip,json,math,pathlib,subprocess,time
from eat_nav_fresh_bridge_run import ROOT,OLD,DOC,atomic,read,run_one,sha
from visualnav_transformer.eat_nav.fresh_bridge_equivalence import compare,difference


def load_trace(folder):
 with gzip.open(folder/'live-0.trace.json.gz','rt') as stream:return json.load(stream)


def check_entry(entry,target):
 old=OLD/'runs'/entry['run_id']
 old_summary=read(old/'live-0.json');new_summary=read(target/'live-0.json')
 source_old=old_summary['source_sha256'];source_new=new_summary['source_sha256']
 scientific=[key for key in source_old if key!='runner_source']
 source_mismatch=[key for key in scientific if source_old[key]!=source_new.get(key)]
 anchor_fields=('root_state','joint_position','joint_velocity','command_manager','policy_pre_noise',
                'policy_post_noise_pre_delay','policy_actor_input','actor_latent','actor_action',
                'critic_observation','critic_value','dem_history_sha256')
 anchor_diff={key:difference(old_summary['anchor'].get(key),new_summary['anchor'].get(key)) for key in anchor_fields}
 path_equal=old_summary['fixed_map_path_sha256']==new_summary['fixed_map_path_sha256']
 trace=compare(load_trace(old),load_trace(target),xy_rms_limit_m=.0105,numeric_limit=1e-5)
 old_end=old_summary['suffix_end_critic_value'];new_end=new_summary['suffix_end_critic_value']
 end_diff=difference(old_end,new_end)
 logs=[json.loads(x) for x in (target/'follower-logs/bridge-events.jsonl').read_text().splitlines()]
 resources=[x for x in logs if x['event']=='resource']
 event_counts={name:sum(x['event']==name for x in logs) for name in ('executor_heartbeat','shutdown_entry','shutdown_return','follower_sigkill','prepath_check')}
 prepath=[x for x in logs if x['event']=='prepath_check']
 lifecycle={'reset_entry_count':sum(x['event']=='reset_entry' for x in logs),
            'fd_min_max':[min(x['fd_count'] for x in resources),max(x['fd_count'] for x in resources)],
            'threads_min_max':[min(x['thread_count'] for x in resources),max(x['thread_count'] for x in resources)],
            'sockets_min_max':[min(x['socket_count'] for x in resources),max(x['socket_count'] for x in resources)],
            'rss_pages_min_max':[min(x['rss_pages'] for x in resources),max(x['rss_pages'] for x in resources)],
            'events':event_counts,'prepath_ready':len(prepath)==1 and prepath[0]['command_count']==[0]*8 and
              prepath[0]['sequence']==[0]*8 and prepath[0]['subscribers']==[1]*8 and
              all(prepath[0]['tf_ready']) and prepath[0]['clock_publishers']==1}
 final=read(target/'final-status.json')
 checks={'input_sha_equal':not source_mismatch,'path_sha_equal':path_equal,
         'anchor_equal':all(v<=1e-5 for v in anchor_diff.values()),'trace_equal':trace['pass'],
         'end_value_equal':end_diff<=1e-5,'120_steps':trace['step_count']==120,
         'lifecycle':lifecycle['reset_entry_count']==1 and lifecycle['prepath_ready'] and
          event_counts['shutdown_entry']==event_counts['shutdown_return']==1 and
          event_counts['follower_sigkill']==0 and event_counts['executor_heartbeat']>0 and
          not final['residual_processes']}
 result={'entry':entry,'pass':all(checks.values()),'checks':checks,'source_mismatch':source_mismatch,
         'anchor_max_abs':anchor_diff,'trace':trace,'end_value_abs_diff':end_diff,
         'lifecycle':lifecycle,'old_complete_sha256':sha(old/'run-complete.json'),
         'fresh_complete_sha256':sha(target/'run-complete.json')}
 atomic(target/'equivalence-report.json',result)
 return result


def run_gate():
 freeze=read(ROOT/'r0-freeze.json');r3=read(ROOT/'r3-fault-gate.json')
 if not r3['operational_cleanup_pass']:raise RuntimeError('R3 operational cleanup gate failed')
 results=[];started=time.time()
 for record in freeze['selected']:
  entry=record['entry'];name='equivalence-'+entry['run_id'];target=ROOT/'runs'/name
  atomic(ROOT/'status.json',{'stage':'R4_RUNNING','current_run':name,'completed':100,'remaining':19,
        'resume_decision':'blocked','elapsed_sec':time.time()-started,'storage_cap_bytes':1500000000})
  if not target.exists():run_one(entry,name)
  elif not (target/'run-complete.json').exists():raise RuntimeError('partial equivalence run; do not retry')
  result=check_entry(entry,target);results.append(result)
  if not result['pass']:
   atomic(ROOT/'r4-equivalence-gate.json',{'stage':'FAILED','results':results})
   raise RuntimeError('equivalence mismatch '+entry['run_id']+' '+repr(result['checks']))
  print('EQUIVALENCE_PASS',entry['run_id'],result['trace']['xy_rms_m'],flush=True)
 # Fresh process lifetime must return to the same process/ROS baseline after every run.
 source='/opt/ros/jazzy/setup.bash'
 graph=subprocess.run(['bash','-lc',f'source {source}; ros2 node list'],capture_output=True,text=True,timeout=10)
 stray=[x for x in graph.stdout.splitlines() if 'eat_nav' in x or 'follower' in x]
 trend={name:[r['lifecycle'][name][1] for r in results] for name in
        ('fd_min_max','threads_min_max','sockets_min_max','rss_pages_min_max')}
 monotonic_increase={name:all(a<b for a,b in zip(values,values[1:])) for name,values in trend.items()}
 stress={'pass':len(results)==3 and not stray and not any(monotonic_increase.values()),
         'consecutive_runs':3,'residual_ros_nodes':stray,
         'reset_counts':[r['lifecycle']['reset_entry_count'] for r in results],
         'resource_maxima':trend,'strictly_increasing':monotonic_increase,
         'fd_ranges':[r['lifecycle']['fd_min_max'] for r in results],
         'thread_ranges':[r['lifecycle']['threads_min_max'] for r in results],
         'socket_ranges':[r['lifecycle']['sockets_min_max'] for r in results],
         'rss_page_ranges':[r['lifecycle']['rss_pages_min_max'] for r in results]}
 atomic(ROOT/'r4-equivalence-gate.json',{'stage':'PASS','results':results})
 atomic(ROOT/'r5-stress-gate.json',stress)
 if not stress['pass']:raise RuntimeError('R5 residual ROS nodes')
 atomic(ROOT/'status.json',{'stage':'R5_PASS','completed':100,'remaining':19,'resume_decision':'blocked',
        'equivalence_run_ids':[r['entry']['run_id'] for r in results],
        'elapsed_sec':time.time()-started,'storage_cap_bytes':1500000000})
 return results

if __name__=='__main__':
 try:run_gate()
 except BaseException as exc:
  atomic(ROOT/'status.json',{'stage':'STOPPED','current_stage':'R4_OR_R5','stop_reason':repr(exc),
         'completed':100,'remaining':19,'resume_decision':'blocked'})
  raise
