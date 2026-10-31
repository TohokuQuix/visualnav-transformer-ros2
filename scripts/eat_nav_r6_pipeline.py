"""Continuous R6 collection→audit→GPU training orchestration with an explicit ledger.

Runs only local simulation/training; no ROS commands or automatic Git publication.
Later evaluation/freeze/test/bag/review gates must finish before the final push.
"""
import argparse,collections,json,os,subprocess,sys,time
from pathlib import Path
from eat_nav_r6_run import read,write,sha,run_collection,stage_runtime
from eat_nav_r6_collection import audit_fixture_rays
import numpy as np

REPO=Path(__file__).resolve().parents[1]


def wait_collection(root,timeout_sec=7200):
 start=time.monotonic()
 while True:
  records=read(root/'collection/progress.json');counts=collections.Counter(r['state'] for r in records.values())
  if counts.get('failed'):raise ValueError('preserved collection failure requires investigation')
  if counts.get('completed',0)==80 and not counts.get('running'):return counts
  if time.monotonic()-start>timeout_sec:raise TimeoutError('collection wait timed out; do not duplicate running jobs')
  time.sleep(30)


def flipper_calibration(root):
 output=root/'flipper-geometry-calibration'
 if output.exists():
  marker=output/'sim-output/complete.json'
  if marker.exists() and not (output/'sim-output/failure.json').exists():return read(marker)
  raise ValueError('preserved incomplete flipper calibration requires investigation')
 output.mkdir()
 binding=stage_runtime('isaac-lab-onix',output);stage=Path(binding['stage']);simout=stage/'flipper-calibration-output'
 envpath=':'.join([str(stage),str(stage/'src'),'/workspace/isaaclab_onix/source/onix_isaac_lab','/workspace/isaaclab/source/isaaclab','/workspace/isaaclab/source/isaaclab_rl','/workspace/isaaclab/source/isaaclab_tasks'])
 cmd=['docker','exec','-e','PYTHONPATH='+envpath,'isaac-lab-onix','timeout','--signal=INT','--kill-after=20s','600s','/isaac-sim/python.sh',str(stage/'eat_nav_r6_collect_sim.py'),'--headless','--output',str(simout),'--terrain-type','boxes','--level','0','--scene','field','--flipper-calibration-probe','--smoke-steps','1','--seed','123','--num-envs','1','--terrain-cols','10','--rl-checkpoint',str(stage/'actor.pt'),'--actor-env-yaml',str(stage/'env.yaml'),'--eat-config',str(stage/'eat_nav.yaml')]
 write(output/'command.json',cmd)
 with (output/'simulator.log').open('x') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=630).returncode
 subprocess.run(['docker','cp','isaac-lab-onix:'+str(simout),str(output/'sim-output')],check=True)
 complete=output/'sim-output/complete.json'
 if code or not complete.exists() or (output/'sim-output/failure.json').exists():raise ValueError('flipper geometry calibration failed')
 return read(complete)


def run(root,python):
 root=Path(root);ledger_file=root/'continuous-pipeline.json';ledger=read(ledger_file) if ledger_file.exists() else {}
 def phase(name,action,marker=None):
  if name in ledger:
   old=ledger[name]
   if old['state']!='completed':raise ValueError('previous stage requires explicit investigation: '+name)
   if marker and (not Path(marker).exists() or sha(marker)!=old['marker_sha256']):raise ValueError('completed stage artifact changed: '+name)
   print(name+' verified completed',flush=True);return
  ledger[name]=dict(state='running',started_unix=time.time());write(ledger_file,ledger);print(name+' started',flush=True)
  try:
   result=action();ledger[name].update(state='completed',ended_unix=time.time(),result=result)
   if marker:ledger[name]['marker_sha256']=sha(marker)
  except BaseException as exc:
   ledger[name].update(state='failed',ended_unix=time.time(),reason=repr(exc));write(ledger_file,ledger);raise
  write(ledger_file,ledger);print(name+' completed',flush=True)
 def command(name,args,marker):
  cmd=[python,*args];folder=root/'pipeline-logs';folder.mkdir(exist_ok=True);write(folder/(name+'-command.json'),cmd)
  env=dict(os.environ,PYTHONPATH='src:scripts:tests/eat_nav:.')
  def execute():
   with (folder/(name+'.log')).open('x') as log:subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   if not Path(marker).exists():raise ValueError('missing stage result marker')
   return dict(marker=str(marker))
  phase(name,execute,marker)
 phase('wait_original80',lambda:dict(wait_collection(root)))
 def addon():
  progress=run_collection(read(root/'wall60x100-addendum-plan.json'),root/'wall60x100-collection',read(root/'benchmark.json'),read(root/'internal-production-approval.json'))
  if len(progress)!=8 or any(r['state']!='completed' for r in progress.values()):raise ValueError('addendum not fully completed')
  return dict(jobs=len(progress))
 phase('collect_wall60x100',addon,root/'wall60x100-collection/progress.json')
 phase('flipper_geometry_calibration',lambda:flipper_calibration(root),root/'flipper-geometry-calibration/sim-output/complete.json')
 def source_audit():
  reports={};destination=root/'all-collection-sources';destination.mkdir()
  for source in (root/'collection',root/'wall60x100-collection'):
   for job in source.iterdir():
    if not job.is_dir() or not (job/'assignment.json').exists():continue
    (destination/job.name).symlink_to(job,target_is_directory=True)
    sim=job/'sim-output';manifest=read(sim/'manifest.json')
    if manifest.get('diagnostic_forced_contact'):raise ValueError('diagnostic overlap cannot enter dataset')
    if manifest['scene']!='field':
     with np.load(sim/'fixture-raycast-evidence.npz',allow_pickle=False) as data:r=audit_fixture_rays(data['ray_hits_w'],data['origins'],manifest['scene'])
     if not r['passed']:raise ValueError('fixture DEM/physics disagreement: '+job.name)
     reports[job.name]=r
  write(root/'all-supplement-fixture-dem-audit.json',reports);return dict(fixture_jobs=len(reports),passed=True)
 phase('audit_source_geometry',source_audit,root/'all-supplement-fixture-dem-audit.json')
 dataset=root/'final-merged-dataset';audit=root/'final-dataset-audit'
 command('pack_final',['scripts/eat_nav_r6_run.py','--plan',str(root/'all-collection-pack-plan.json'),'--output',str(root/'all-collection-sources'),'--pack-output',str(dataset),'--reuse-dataset',str(root/'reused-dataset')],dataset/'manifest.json')
 command('audit_final',['scripts/eat_nav_r6_dataset_audit.py','--dataset',str(dataset),'--output',str(audit)],audit/'field-level-statistics.json')
 if not read(audit/'audit.json')['passed']:raise ValueError('dataset audit failed')
 for name,extra in [('final-model',[]),('state-only-diagnostic',['--without-dem'])]:
  command('train_'+name,['scripts/eat_nav_r6_train.py','--dataset',str(dataset),'--output',str(root/name),'--epochs','300','--batch-size','16','--seed','123','--device','cuda',*extra],root/name/'training.json')
  if not all(read(root/name/'training.json')['support']['active']):raise ValueError('missing actual training support for four event heads')
 command('dependence_diagnostics',['scripts/eat_nav_r6_diagnostics.py','--dataset',str(dataset),'--checkpoint',str(root/'final-model/checkpoint.pt'),'--output',str(root/'final-model-diagnostics')],root/'final-model-diagnostics/diagnostics.json')
 command('calibrate_profiles',['scripts/eat_nav_r6_calibrate_profiles.py','--dataset',str(dataset),'--checkpoint',str(root/'final-model/checkpoint.pt'),'--records',str(audit/'validation-records.json'),'--output',str(root/'final-derived-profiles')],root/'final-derived-profiles/index.json')
 write(root/'collection-training-complete.json',dict(state='completed',scope='collection_audit_training_only',model_sha256=sha(root/'final-model/checkpoint.pt'),dataset_sha256=sha(dataset/'samples.npz'),next='actual validation rollouts, freeze top3, unusedtest, offlinebag, independentreview, thenGitpush',real_robot_commands_sent=False,git_push_completed=False))
 print('R6_COLLECTION_TRAINING_COMPLETE',flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--python',default=sys.executable);a=p.parse_args();run(a.output.resolve(),a.python)
if __name__=='__main__':main()
