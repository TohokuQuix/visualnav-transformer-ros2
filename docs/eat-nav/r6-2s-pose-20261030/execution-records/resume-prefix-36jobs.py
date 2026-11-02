from pathlib import Path
import json,hashlib,subprocess,time
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030');state=json.loads((r/'operations/latest-poweroff-save.json').read_text());old=Path(state['partial_snapshot'])
a=[x.decode() for x in Path('/proc/91264/cmdline').read_bytes().split(b'\0') if x];remote=a[a.index('--output')+1]
keys=['control_step','root_state','post_root_state','joint_position','joint_velocity','applied_torque','actor_command','follower_command','protective_stop']
def digest_rows(data,n):
 lines=data.splitlines();lines=lines if data.endswith(b'\n') else lines[:-1];rows=[json.loads(x) for x in lines[:n]]
 assert len(rows)==n
 return hashlib.sha256(json.dumps([{k:x[k] for k in keys} for x in rows],sort_keys=True,separators=(',',':')).encode()).hexdigest()
counts={}
for p in sorted(old.glob('group-*/trace.jsonl')):
 data=p.read_bytes();lines=data.splitlines();lines=lines if data.endswith(b'\n') else lines[:-1];counts[p.parent.name]=len(lines)
code="from pathlib import Path; import json,hashlib,sys; p=Path(sys.argv[1]); counts=json.loads(sys.argv[2]); keys=json.loads(sys.argv[3]); result={}\nfor group,cap in counts.items():\n data=(p/group/'trace.jsonl').read_bytes();lines=data.splitlines();lines=lines if data.endswith(b'\\n') else lines[:-1]; n=min(cap,len(lines)); rows=[json.loads(x) for x in lines[:n]]; digest=hashlib.sha256(json.dumps([{k:x[k] for k in keys} for x in rows],sort_keys=True,separators=(',',':')).encode()).hexdigest(); result[group]={'rows':n,'sha256':digest}\nprint(json.dumps(result))"
new=json.loads(subprocess.check_output(['docker','exec','isaac-lab-onix','/isaac-sim/kit/python/bin/python3','-c',code,remote,json.dumps(counts),json.dumps(keys)],text=True));assert len(new)==len(counts)==30
comparison={}
for group,x in new.items():
 assert x['rows']>0
 prior=digest_rows((old/group/'trace.jsonl').read_bytes(),x['rows'])
 comparison[group]=dict(compared_rows=x['rows'],previous_sha256=prior,current_sha256=x['sha256'],identical=prior==x['sha256'])
report=dict(schema='eatnav_r6_restart_observed_prefix_comparison_v1',verified_unix=time.time(),job=Path(remote).name,previous_snapshot=state['snapshot'],current_stage=remote,keys=keys,groups=comparison,compared_rows=sum(x['compared_rows'] for x in comparison.values()),all_compared_physical_command_rows_identical=all(x['identical'] for x in comparison.values()),scope='Only common fully recorded prefixes; not proof of unobserved future equality or independent safety performance. Paused partial results remain excluded from completed trials.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
p=r/'operations/resumed-36jobs-prefix-comparison.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(groups=len(comparison),rows=report['compared_rows'],min_steps=min(x['compared_rows'] for x in comparison.values()),all_identical=report['all_compared_physical_command_rows_identical'],mismatched_groups=[g for g,x in comparison.items() if not x['identical']])))
