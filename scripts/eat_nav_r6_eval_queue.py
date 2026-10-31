"""Resumable actual Isaac evaluation queue; frozen cohorts and strict completion gates.

No ROS publishers. Validation and test are separate commands; test requires the
frozen selection digest. Exit status alone is never accepted as completion.
"""
import argparse, collections, hashlib, json, subprocess, time, shutil
from pathlib import Path
import numpy as np
from eat_nav_r6_run import read,write,sha,stage_runtime,REPO
from visualnav_transformer.eat_nav.r6_evaluation import aggregate_trials
from visualnav_transformer.eat_nav.r6_model import load_checkpoint
from eat_nav_r6_shadow import choice_equivalence
from eat_nav_r6_storage import verify_copied_tree,compress_traces,remove_verified_scratch
from visualnav_transformer.eat_nav.r6_provenance import runtime_source_hashes,verify_execution_freeze


def evaluation_jobs(groups,conditions,n=10):
 cells={}
 for g in groups:cells.setdefault((g['field'],g['level']),[]).append(g)
 jobs=[]
 for (field,level),members in cells.items():
  for condition in conditions:
   for offset in range(0,len(members),n):
    chunk=[dict(g,env_index=i) for i,g in enumerate(members[offset:offset+n])]
    jobs.append(dict(job_id=f"{condition['id']}-{field}-l{level}-batch{offset//n:02d}",condition=condition,field=field,level=level,groups=chunk,physical_batch_id=f'{field}-l{level}-batch{offset//n:02d}'))
 return jobs


def validate_completion(complete,n):
 if not complete or complete.get('state')!='completed' or complete.get('smoke',True) or complete.get('num_envs')!=n or len(complete.get('results',[]))!=n:raise ValueError('missing/full protocol completion gate')
 for r in complete['results']:
  if r.get('initial_valid',False) and not r.get('full_evaluation_valid',False):raise ValueError('not full evaluation')
 return complete['results']


def conditions_for(profiles,role,frozen=None):
 if role=='validation':
  # All12 numeric combinations remain in derivation; ten are chosen before any
  # rollout outcomes, retaining both low/high endpoints of the quantile grid.
  ids=(0,2,3,4,5,6,8,9,10,11)
  chosen=[profiles['profiles'][i] for i in ids]
 else:
  if not frozen or frozen.get('source')!='validation_only' or len(frozen['profiles'])!=3:raise ValueError('frozen validation top3 required for test')
  chosen=frozen['profiles']
 permissive=chosen[-1] if role=='validation' else frozen['comparison_profiles']['event_2s']
 hybrid=permissive if role=='validation' else frozen['comparison_profiles']['combined']
 return [dict(id='old_reference',mode='old_reference'),dict(id='event_2s',mode='event',profile=permissive)]+[dict(id=p['id'],mode='pose',profile=p) for p in reversed(chosen)]+[dict(id='combined',mode='hybrid',profile=hybrid)]


def old_checkpoint(field):
 index=read(REPO/'checkpoints/eat_nav/fall_stall_r4star_r5_index.json')
 candidates=[m for m in index['checkpoints'] if m['field']==field and m['model_seed']==123]
 if len(candidates)!=1:raise ValueError('unique old reference seed123 required')
 path=REPO/candidates[0]['path']
 if sha(path)!=candidates[0]['sha256']:raise ValueError('old frozen checkpoint mismatch')
 return path


def verify_initial_cohort(source,destination):
 for old in source.glob('group-*/initial-state.npz'):
  other=destination/old.relative_to(source)
  with np.load(old,allow_pickle=False) as a,np.load(other,allow_pickle=False) as b:
   if set(a.files)!=set(b.files) or any(a[k].shape!=b[k].shape or not np.allclose(a[k],b[k],atol=1e-5,rtol=0) for k in a.files):raise ValueError('paired initialization/input identity mismatch: '+str(old))
 if sha(source/'actual-physics-properties.npz')!=sha(destination/'actual-physics-properties.npz'):raise ValueError('paired physics properties mismatch')


def summarize_queue(jobs,ledger,output,role):
 conditions={j['condition']['id'] for j in jobs};bycondition={k:{} for k in conditions}
 for job in jobs:
  state=ledger.get(job['job_id'],{})
  if state.get('state')!='completed':continue
  for result in state['results']:bycondition[job['condition']['id']][result['group_id']]=result
 completed=all(ledger.get(j['job_id'],{}).get('state')=='completed' for j in jobs)
 expected={g['group_id'] for j in jobs for g in j['groups']}
 matched=set.intersection(*(set(rows) for rows in bycondition.values())) if bycondition else set()
 matched={gid for gid in matched if all(rows[gid].get('initial_valid',False) for rows in bycondition.values())}
 rows=[];aggregates={}
 for name,results in sorted(bycondition.items()):
  actual=[results[k] for k in sorted(matched)]
  if not actual:continue
  a=aggregate_trials(actual);aggregates[name]=a
  rows.append(dict(id=name,split=role,trials=a['trials'],fall_count=a['tip_count'],ranking_fall_count_upper_bound=a['ranking_fall_count_upper_bound'],unknown_fall_trials=a['fall_outcome_unknown_trials'],mean_stop_sec=a['mean_ranking_STOP_upper_bound_sec'],mean_progress_m=a['mean_route_progress_m'],linear_windows=a['mean_linear_windows'],turn_windows=a['mean_turn_windows'],linear_stall_sec=a['mean_linear_stall_sec'],turn_stall_sec=a['mean_turn_stall_sec']))
 members={g['group_id']:g for j in jobs for g in j['groups']}
 cell_counts=collections.Counter((members[gid]['field'],members[gid]['level']) for gid in matched)
 planned_cell_counts=collections.Counter((g['field'],g['level']) for g in members.values())
 result=dict(schema='eatnav_r6_queue_summary_v1',split=role,planned_cell_counts={f'{field}/level{level}':count for (field,level),count in planned_cell_counts.items()},matched_cell_counts={f'{field}/level{level}':count for (field,level),count in cell_counts.items()},all_jobs_completed=completed,planned_jobs=len(jobs),completed_jobs=sum(r.get('state')=='completed' for r in ledger.values()),planned_cohort_trials=len(expected),matched_initial_valid_trials=len(matched),excluded_initial_invalid_or_missing=sorted(expected-matched),aggregates=aggregates,selection_rows=rows)
 write(output/'summary.json',result);write(output/'selection-rows.json',rows);return result


def run_queue(plan,checkpoint,profile_index,output,role,*,frozen=None,max_jobs=None,container='isaac-lab-onix',timeout_sec=2400,num_envs=10):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 conditions=conditions_for(profile_index,role,frozen);jobs=evaluation_jobs(plan[role+'_rollouts'],conditions,n=num_envs)
 sources=runtime_source_hashes(REPO)
 if role=='test':
  frozen=verify_execution_freeze(frozen,sha(checkpoint),sources)
  if frozen.get('execution_num_envs')!=num_envs:raise ValueError('frozen evaluation capacity mismatch')
 identity=dict(schema='eatnav_r6_evaluation_binding_v1',source_sha256=sources,execution_num_envs=num_envs,cohorts_sha256=hashlib.sha256(json.dumps(plan[role+'_rollouts'],sort_keys=True).encode()).hexdigest(),model_sha256=sha(checkpoint),role=role,conditions=conditions,frozen_selection=frozen['freeze_sha256'] if frozen else None)
 if (output/'frozen-binding.json').exists() and read(output/'frozen-binding.json')!=identity:raise ValueError('frozen queue binding changed')
 write(output/'frozen-binding.json',identity);write(output/'jobs.json',jobs)
 progress=output/'progress.json';ledger=read(progress) if progress.exists() else {}
 if any(v.get('state')=='running' for v in ledger.values()):raise ValueError('running attempt must be investigated before resume')
 pending=[j for j in jobs if j['job_id'] not in ledger]
 if max_jobs is not None:pending=pending[:max_jobs]
 if not pending:return summarize_queue(jobs,ledger,output,role)
 binding=stage_runtime(container,output);stage=Path(binding['stage']);model,payload=load_checkpoint(checkpoint)
 subprocess.run(['docker','cp',str(checkpoint),container+':'+str(stage/'new.pt')],check=True)
 envpath=':'.join([str(stage),str(stage/'src'),'/workspace/isaaclab_onix/source/onix_isaac_lab','/workspace/isaaclab/source/isaaclab','/workspace/isaaclab/source/isaaclab_rl','/workspace/isaaclab/source/isaaclab_tasks'])
 references={};shadow_cache={}
 for j in jobs:
  record=ledger.get(j['job_id'],{})
  if record.get('state')=='completed' and record.get('execution')=='simulated':references.setdefault(j['physical_batch_id'],[]).append(output/j['job_id']/'sim-output')
 previous_cell=None
 for job in pending:
  current_cell=(job['field'],job['level'])
  if current_cell!=previous_cell:shadow_cache.clear();previous_cell=current_cell
  directory=output/job['job_id']
  if shutil.disk_usage(output).free<4*1024**3:raise OSError('less than4GiBfree; preserve data and stop before new trial')
  directory.mkdir();write(directory/'assignment.json',job)
  condition=job['condition'];mode=condition['mode'];groups=job['groups'];n=len(groups)
  # Whole-batch reuse only: retain original physical randomization composition.
  reused=False
  if mode!='old_reference':
   for source in references.get(job['physical_batch_id'],[]):
    manifest=read(source/'manifest.json')
    if manifest['mode']=='old_reference' or manifest['model_sha256']!=sha(checkpoint):continue
    audits=[];results=[]
    for i in range(n):
     folder=source/f'group-{i:03d}'
     if not read(folder/'group.json')['initial_valid']:
      audit=dict(equivalent=True,reason='same initialization-invalid environment; no policy action or outcome credited')
     else:
      cache_key=str(folder)
      if cache_key not in shadow_cache:
       decisions=[read(p) for p in sorted(folder.glob('decision-*.json'))];dems=[];trimmed=[]
       for d in decisions:
        if 'policy_features' in d:
         trimmed.append({key:d[key] for key in ('decision_step','selected_index','policy_features','footprint')});dems.append(None)
        else:
         trimmed.append(d)
         with np.load(folder/f"input-{d['decision_step']:04d}.npz",allow_pickle=False) as inp:dems.append(inp['dem'][0,-1])
       shadow_cache[cache_key]=(trimmed,dems)
      decisions,dems=shadow_cache[cache_key]
      audit=choice_equivalence(decisions,dems,condition['profile'],mode,payload['support']['active'])
     audits.append(audit)
     if not audit['equivalent']:break
     results.append(dict(read(folder/'result.json'),mode=mode,execution='verified_policy_equivalent_reuse',timing_origin='source actual full inference; not independently timed',source=str(folder)))
    if len(audits)==n and all(a['equivalent'] for a in audits):
     write(directory/'equivalence-audit.json',dict(source=str(source),source_manifest_sha256=sha(source/'manifest.json'),checks=audits))
     ledger[job['job_id']]=dict(state='completed',execution='verified_policy_equivalent_reuse',source=str(source),results=results);write(progress,ledger);reused=True;break
  if reused:
   print('R6_EVAL_EQUIVALENT '+job['job_id'],flush=True);continue
  groupfile=stage/(job['job_id']+'-groups.json');write(groupfile,dict(groups=groups));subprocess.run(['docker','cp',str(groupfile),container+':'+str(groupfile)],check=True)
  old=old_checkpoint(job['field']);subprocess.run(['docker','cp',str(old),container+':'+str(stage/'old.pt')],check=True)
  profilefile=stage/(job['job_id']+'-profile.json');write(profilefile,condition.get('profile',{}));subprocess.run(['docker','cp',str(profilefile),container+':'+str(profilefile)],check=True)
  simout=stage/'evaluations'/job['job_id']
  command=['docker','exec','-e','PYTHONPATH='+envpath,container,'timeout','--signal=INT','--kill-after=20s',str(timeout_sec)+'s','/isaac-sim/python.sh',str(stage/'eat_nav_r6_eval_sim.py'),'--headless','--output',str(simout),'--terrain-type',job['field'],'--level',str(job['level']),'--num-envs',str(n),'--seed',str(groups[0]['seed']),'--parent-seeds',','.join(str(g['seed']) for g in groups),'--terrain-cols',str(n),'--groups',str(groupfile),'--mode',mode,'--checkpoint',str(stage/'new.pt'),'--old-checkpoint',str(stage/'old.pt'),'--profile',str(profilefile),'--rl-checkpoint',str(stage/'actor.pt'),'--actor-env-yaml',str(stage/'env.yaml'),'--eat-config',str(stage/'eat_nav.yaml')]
  write(directory/'command.json',command);ledger[job['job_id']]=dict(state='running',started_unix=time.time());write(progress,ledger)
  try:
   with (directory/'simulator.log').open('x') as log:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout_sec+30).returncode
   copied=subprocess.run(['docker','cp',container+':'+str(simout),str(directory/'sim-output')],capture_output=True,text=True)
   actual=directory/'sim-output';complete=read(actual/'complete.json') if (actual/'complete.json').exists() else None
   if code or copied.returncode or (actual/'failure.json').exists():raise ValueError('simulator execution/output failure')
   results=validate_completion(complete,n)
   verify_copied_tree(container,simout,actual)
   if references.get(job['physical_batch_id']):verify_initial_cohort(references[job['physical_batch_id']][0],actual)
   ledger[job['job_id']]=dict(state='completed',execution='simulated',results=results,elapsed_seconds=complete['elapsed_seconds'],ended_unix=time.time())
   compress_traces(actual)
   try:remove_verified_scratch(container,simout)
   except subprocess.CalledProcessError as cleanup_error:ledger[job['job_id']]['container_cleanup_warning']=str(cleanup_error)
   references.setdefault(job['physical_batch_id'],[]).append(actual)
  except BaseException as exc:
   ledger[job['job_id']]=dict(state='failed',reason=repr(exc),ended_unix=time.time());write(progress,ledger);summarize_queue(jobs,ledger,output,role);raise
  write(progress,ledger);summarize_queue(jobs,ledger,output,role)
  print('R6_EVAL_ACTUAL '+job['job_id'],flush=True)
 return summarize_queue(jobs,ledger,output,role)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('plan','checkpoint','profiles','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--role',choices=('validation','test'),required=True);p.add_argument('--frozen',type=Path);p.add_argument('--max-jobs',type=int);p.add_argument('--num-envs',type=int,default=10);a=p.parse_args()
 print(json.dumps(run_queue(read(a.plan),a.checkpoint,read(a.profiles),a.output,a.role,frozen=read(a.frozen) if a.frozen else None,max_jobs=a.max_jobs,num_envs=a.num_envs),indent=2))
if __name__=='__main__':main()
