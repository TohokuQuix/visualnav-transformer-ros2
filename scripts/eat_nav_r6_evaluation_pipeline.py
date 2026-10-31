"""Run R6 benchmark, validation, freeze, unusedtest and offlinebag in fixed order.
No Git push or real robot commands. Every stage is journaled and independently
checkable; the scientific completion marker requires all downstream gates.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from eat_nav_r6_run import read,write,sha,stage_runtime,REPO
from eat_nav_r6_eval_queue import run_queue


def benchmark(root):
 output=root/'evaluation-runtime-benchmark';output.mkdir();binding=stage_runtime('isaac-lab-onix',output);stage=Path(binding['stage'])
 model=root/'final-model/checkpoint.pt';profiles=read(root/'production-profiles/index.json');profile=profiles['profiles'][-1]
 selected=[g for g in read(root/'collection-plan.json')['validation_rollouts'] if g['field']=='k_rail_field' and g['level']==4]
 envpath=':'.join([str(stage),str(stage/'src'),'/workspace/isaaclab_onix/source/onix_isaac_lab','/workspace/isaaclab/source/isaaclab','/workspace/isaaclab/source/isaaclab_rl','/workspace/isaaclab/source/isaaclab_tasks'])
 write(stage/'benchmark-profile.json',profile)
 for src,name in [(model,'new.pt'),(stage/'benchmark-profile.json','benchmark-profile.json')]:subprocess.run(['docker','cp',str(src),'isaac-lab-onix:'+str(stage/name)],check=True)
 reports=[];geometry={}
 for n in (10,30):
  directory=output/f'envs{n:02d}';directory.mkdir();groups=[dict(g,group_id='benchmark/'+g['group_id'],role='benchmark',env_index=i) for i,g in enumerate(selected[:n])]
  groupfile=stage/f'benchmark-{n}-groups.json';write(groupfile,dict(groups=groups));subprocess.run(['docker','cp',str(groupfile),'isaac-lab-onix:'+str(groupfile)],check=True)
  simout=stage/f'benchmark-output-{n}'
  cmd=['docker','exec','-e','PYTHONPATH='+envpath,'isaac-lab-onix','timeout','--signal=INT','--kill-after=20s','600s','/isaac-sim/python.sh',str(stage/'eat_nav_r6_eval_sim.py'),'--headless','--output',str(simout),'--terrain-type','k_rail_field','--level','4','--num-envs',str(n),'--seed',str(groups[0]['seed']),'--parent-seeds',','.join(str(g['seed']) for g in groups),'--terrain-cols',str(n),'--groups',str(groupfile),'--max-steps','40','--mode','event','--checkpoint',str(stage/'new.pt'),'--profile',str(stage/'benchmark-profile.json'),'--rl-checkpoint',str(stage/'actor.pt'),'--actor-env-yaml',str(stage/'env.yaml'),'--eat-config',str(stage/'eat_nav.yaml')]
  write(directory/'command.json',cmd);start=time.perf_counter()
  with (directory/'simulator.log').open('x') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=650).returncode
  copied=subprocess.run(['docker','cp','isaac-lab-onix:'+str(simout),str(directory/'sim-output')],capture_output=True,text=True)
  host_elapsed=time.perf_counter()-start;marker=directory/'sim-output/complete.json';failure=directory/'sim-output/failure.json'
  if code or copied.returncode or not marker.exists() or failure.exists():
   report=dict(num_envs=n,passed=False,returncode=code,reason=read(failure) if failure.exists() else 'missing_output');reports.append(report)
   if n==10:raise ValueError('required10env evaluation integration gate failed')
   continue
  data=read(marker);physics=read(directory/'sim-output/physics-audit.json');manifest=read(directory/'sim-output/manifest.json')
  if not data['smoke'] or not physics['valid'] or data['recorded_rows']<=0:raise ValueError('runtime benchmark is invalid')
  free,total=data['gpu_memory_free_total_bytes'];used=total-free
  report=dict(num_envs=n,passed=True,recorded_rows=data['recorded_rows'],simulator_loop_sec=data['elapsed_seconds'],host_total_sec=host_elapsed,loop_rows_per_second=data['recorded_rows']/data['elapsed_seconds'],host_rows_per_second=data['recorded_rows']/host_elapsed,driver_memory_used_bytes=used,driver_memory_total_bytes=total,memory_fraction=used/total,initialization_valid=sum(r['initial_valid'] for r in data['results']),scope='2simulatedseconds integration/performance probe, not a60s outcome trial')
  reports.append(report);geometry[n]={(m['parent_seed'],m['tile_index']%5):(m['geometry_sha256'],m['difficulty']) for m in manifest['geometry_metadata']}
  write(directory/'benchmark-report.json',report)
 if 30 in geometry:
  if any(geometry[30].get(k)!=v for k,v in geometry[10].items()):raise ValueError('parent geometry/difficulty changed with batch capacity')
 first=reports[0];larger=reports[-1]
 n=30 if larger['passed'] and larger['memory_fraction']<.85 and larger['host_rows_per_second']>first['host_rows_per_second']*1.05 else 10
 result=dict(schema='eatnav_r6_evaluation_runtime_benchmark_v1',passed=True,selected_num_envs=n,measurements=reports,parent_geometry_and_difficulty_capacity_invariant=30 in geometry,model_sha256=sha(model),source_binding_sha256=sha(output/'runtime-binding.json'),gui=False,selection='actual host throughput improvement>5% and driverVRAM<85%; fixed seed cohort unchanged')
 write(output/'benchmark.json',result);return result


def run(root,python,benchmark_only=False):
 root=Path(root);ledger_file=root/'evaluation-pipeline.json';ledger=read(ledger_file) if ledger_file.exists() else {}
 def phase(name,action,marker):
  if name in ledger:
   old=ledger[name]
   if old['state']!='completed':raise ValueError('previous failed/running stage requires explicit investigation: '+name)
   if not Path(marker).exists() or sha(marker)!=old['marker_sha256']:raise ValueError('stage artifact changed: '+name)
   return old['result']
  ledger[name]=dict(state='running',started_unix=time.time());write(ledger_file,ledger);print(name+' started',flush=True)
  try:
   result=action();ledger[name].update(state='completed',ended_unix=time.time(),result=result,marker_sha256=sha(marker))
  except BaseException as exc:
   ledger[name].update(state='failed',ended_unix=time.time(),reason=repr(exc));write(ledger_file,ledger);raise
  write(ledger_file,ledger);print(name+' completed',flush=True);return result
 def command(name,args,marker):
  cmd=[python,*map(str,args)];folder=root/'pipeline-logs';folder.mkdir(exist_ok=True);write(folder/(name+'-command.json'),cmd)
  def execute():
   with (folder/(name+'.log')).open('x') as log:subprocess.run(cmd,cwd=REPO,env=dict(os.environ,PYTHONPATH='src:scripts:tests/eat_nav:.'),stdout=log,stderr=subprocess.STDOUT,check=True)
   if not Path(marker).exists():raise ValueError('missing stage marker')
   return dict(marker=str(marker))
  return phase(name,execute,marker)
 ready=root/'collection-training-complete.json'
 if not ready.exists() or read(ready)['state']!='completed':raise ValueError('collection/audit/training must complete first')
 runtime=phase('evaluation_runtime_benchmark',lambda:benchmark(root),root/'evaluation-runtime-benchmark/benchmark.json')
 if benchmark_only:return
 n=runtime['selected_num_envs'];model=root/'final-model/checkpoint.pt';profile_file=root/'production-profiles/index.json';profiles=read(profile_file);plan=read(root/'collection-plan.json')
 addendum=read(root/'evaluation-initialization-addendum.json')
 if addendum['base_plan_sha256']!=sha(root/'collection-plan.json'):raise ValueError('evaluation addendum source mismatch')
 for role in ('validation','test'):plan[role+'_rollouts']=plan[role+'_rollouts']+addendum[role+'_rollouts_additional']
 plan['validation_condition_runs']=len(plan['validation_rollouts'])*13;plan['test_condition_runs']=len(plan['test_rollouts'])*6
 write(root/'active-evaluation-plan.json',plan)
 calibration=root/'flipper-geometry-calibration/sim-output/flipper-geometry-calibration.json'
 command('anchor_leadtime',['scripts/eat_nav_r6_leadtime.py','--dataset',root/'final-merged-dataset','--checkpoint',model,'--records',root/'final-dataset-audit/validation-records.json','--profiles',profile_file,'--calibration',calibration,'--output',root/'anchor-leadtime'],root/'anchor-leadtime/report.json')
 def validation():
  summary=run_queue(plan,model,profiles,root/'validation-evaluation','validation',num_envs=n)
  if not summary['all_jobs_completed']:raise ValueError('validation queue incomplete')
  return dict(planned_trials_per_condition=summary['planned_cohort_trials'],matched_trials=summary['matched_initial_valid_trials'])
 phase('validation_rollouts',validation,root/'validation-evaluation/summary.json')
 command('freeze_top3',['scripts/eat_nav_r6_thresholds.py','freeze','--profiles',profile_file,'--evaluations',root/'validation-evaluation/summary.json','--evaluation-binding',root/'validation-evaluation/frozen-binding.json','--output',root/'frozen-top3.json'],root/'frozen-top3.json')
 frozen=read(root/'frozen-top3.json')
 def test():
  summary=run_queue(plan,model,profiles,root/'test-evaluation','test',frozen=frozen,num_envs=n)
  if not summary['all_jobs_completed']:raise ValueError('test queue incomplete')
  return dict(planned_trials_per_condition=summary['planned_cohort_trials'],matched_trials=summary['matched_initial_valid_trials'],freeze_sha256=frozen['freeze_sha256'])
 phase('unused_test_rollouts',test,root/'test-evaluation/summary.json')
 for role in ('validation','test'):
  command(role+'_forecast_metrics',['scripts/eat_nav_r6_rollout_metrics.py','--queue-root',root/(role+'-evaluation'),'--checkpoint',model,'--output',root/(role+'-forecast-metrics')],root/(role+'-forecast-metrics/report.json'))
 command('offline_bag_top3',['scripts/eat_nav_r6_bag_evaluate.py','--intermediate',root/'bag-intermediate-observed','--checkpoint',model,'--frozen',root/'frozen-top3.json','--calibration',calibration,'--output',root/'top3-bag-evaluation'],root/'top3-bag-evaluation/report.json')
 command('offline_bag_inpainted_sensitivity',['scripts/eat_nav_r6_bag_evaluate.py','--intermediate',root/'bag-intermediate-portable','--checkpoint',model,'--frozen',root/'frozen-top3.json','--calibration',calibration,'--output',root/'top3-bag-inpainted-sensitivity','--input-interpretation','inpainted_reference'],root/'top3-bag-inpainted-sensitivity/report.json')
 write(root/'scientific-evaluation-complete.json',dict(state='completed',model_sha256=sha(model),freeze_sha256=frozen['freeze_sha256'],gui=False,evaluation_num_envs=n,real_robot_commands_sent=False,remaining=['final independent code/result audit','package reproducible checkpoints/configuration/smallresults','commit and Git push, verify remote SHA']))
 print('R6_SCIENTIFIC_EVALUATION_COMPLETE',flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--benchmark-only',action='store_true');a=p.parse_args();run(a.output.resolve(),a.python,a.benchmark_only)
if __name__=='__main__':main()
