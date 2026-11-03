from pathlib import Path
import yaml,json,hashlib,subprocess
N=Path('/home/ubuntu/eatnav-moving-state-pose2s-20261103');O=N/'docs/eat-nav/single-dem-pose2s-20261103/left-right-turn-audit'
saved=yaml.load((N/'checkpoints/eat_nav/rl_actor/env.yaml').read_text(),Loader=yaml.BaseLoader);configs=json.loads(Path('/tmp/eatnav_lr_current_cfg.json').read_text())
def norm(x,key=''):
 if isinstance(x,dict):return {k:norm(v,k) for k,v in x.items()}
 if isinstance(x,list):
  if key in ['joint_ids','body_ids','fixed_tendon_ids','object_collection_ids'] and x==['null','null','null']:return 'slice(None, None, None)'
  return [norm(v) for v in x]
 if x is None:return 'null'
 if isinstance(x,bool):return str(x).lower()
 return str(x)
def diff(a,b,p=''):
 if isinstance(a,dict) and isinstance(b,dict):
  return sum([diff(a.get(k,'MISSING'),b.get(k,'MISSING'),p+'/'+str(k)) for k in sorted(set(a)|set(b))],[])
 return [] if a==b else [dict(path=p,saved=a,current=b)]
results={}
for version in ['base','collection']:
 results[version]={section:diff(norm(saved[section]),norm(configs[version][section])) for section in ['observations','actions']}
 results[version]['policy_order_saved']=[k for k,v in saved['observations']['policy'].items() if isinstance(v,dict) and 'func' in v]
 results[version]['policy_order_current']=[k for k,v in configs[version]['observations']['policy'].items() if isinstance(v,dict) and 'func' in v]
 results[version]['action_order_saved']=[k for k,v in saved['actions'].items() if isinstance(v,dict)]
 results[version]['action_order_current']=[k for k,v in configs[version]['actions'].items() if isinstance(v,dict)]
 # Include expected physical and initialization changes rather than silently calling whole environment identical.
 results[version]['robot_config_differences']=diff(norm(saved['scene']['robot']),norm(configs[version]['scene']['robot']))
 results[version]['reset_events_differences']=diff(norm(saved['events']),norm(configs[version]['events']))
results['saved_config_sha256']=hashlib.sha256((N/'checkpoints/eat_nav/rl_actor/env.yaml').read_bytes()).hexdigest()
results['shared_config_sha256']=hashlib.sha256(Path('/home/ubuntu/eatnav-student-share-20261018/rl_actor_env.yaml').read_bytes()).hexdigest()
results['historical_training_run_available']=False
results['historical_training_run_note']='2026-07-24_18-20-16 directory not found in inspected host/container roots; paired env.yaml cannot prove the historical executable state.'
(O/'actor-contract-comparison.json').write_text(json.dumps(results,indent=2)+'\n')
for v in ['base','collection']:print(v,'observation diffs',len(results[v]['observations']),'action diffs',len(results[v]['actions']),'robot diffs',results[v]['robot_config_differences'])
print('OUTPUT',O/'actor-contract-comparison.json')
