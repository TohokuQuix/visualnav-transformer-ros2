"""Approved30s/no-history collection, FAST-LIO, training and offline comparison only."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,subprocess,time,shutil
import numpy as np
from visualnav_transformer.eat_nav.single_pose2s_dataset import build_dataset
from visualnav_transformer.eat_nav.single_pose2s_collection import pending_parent_batches
from eat_nav_single_pose2s_pipeline import stage_source,run_batch,write
REPO=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();prod=root/'production'
 auth=json.loads((root/'approval.json').read_text())
 if not auth['user_approved'] or hashlib.sha256((root/'approved-spec.md').read_bytes()).hexdigest()!=auth['spec_sha256']:raise ValueError('approved spec mismatch')
 lock=(root/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status=dict(state='running',pid=os.getpid(),started_unix=time.time(),scope=auth['scope'],dataset_complete=False,training_complete=False,analysis_complete=False,push_done=False)
 def phase(name,**extra):
  status.update(phase=name,updated_unix=time.time(),**extra);write(root/'status.json',status);print('PHASE '+name,flush=True)
 def run(name,*args):
  subprocess.run([os.sys.executable,str(REPO/'scripts'/name),*map(str,args)],cwd=REPO,check=True,env=dict(os.environ,PYTHONPATH=str(REPO/'src')+':'+str(REPO/'scripts')))
 def completed():
  found=set();expected={p['parent_id']:p for p in manifest['parents']}
  for seal in (prod/'raw').glob('*/complete.json'):
   for meta in seal.parent.glob('*/metadata.json'):
    m=json.loads(meta.read_text());pid=m['parent']['parent_id']
    if pid in found or m['parent']!=expected[pid] or m.get('trial_seconds')!=30:raise ValueError('parent/duration/duplicate mismatch')
    if hashlib.sha256(meta.with_name('raw.npz').read_bytes()).hexdigest()!=m['raw_sha256']:raise ValueError('raw changed')
    found.add(pid)
  return found
 try:
  phase('smoke_audit')
  smoke=root/'smoke';seal=smoke/'raw/n2-30s/complete.json'
  if not seal.exists():raise ValueError('30s smoke must finish first')
  run('eat_nav_single_pose2s_fastlio_batch.py','--raw',smoke/'raw','--workers',2)
  sm=json.loads((smoke/'manifest.json').read_text())
  if not (smoke/'dataset/audit.json').exists():build_dataset(smoke/'raw',sm,smoke/'dataset')
  sa=json.loads((smoke/'dataset/audit.json').read_text())
  if sa['valid_pose_windows']<28 or sa['observed_windows']!=30:raise ValueError('smoke30s temporal/FAST-LIO coverage failed; inspect')
  manifest=json.loads((prod/'manifest.json').read_text())
  if manifest['trial_seconds']!=30 or manifest['planned_windows']!=19950:raise ValueError('30s frozen manifest mismatch')
  phase('capacity_smoke_30env')
  capacity=root/'capacity30';capacity.mkdir(exist_ok=True)
  ids=[p for p in manifest['parents'] if p['field']=='random_rough' and p['level']==0][:30]
  ids=[dict(p,parent_id='capacity-'+p['parent_id']) for p in ids]
  stage=stage_source(prod)
  if not (capacity/'raw/n30-30s/complete.json').exists():
   run_batch(capacity,stage,ids,'n30-30s',record_lidar=True,trial_seconds=30)
  cap=json.loads((capacity/'raw/n30-30s/complete.json').read_text())
  if cap['parents']!=30 or cap['planned_windows']!=450:raise ValueError('capacity incomplete')
  size=sum(p.stat().st_size for p in (capacity/'raw').rglob('*') if p.is_file());estimate=int(size/30*1330)
  capacity_audit=dict(parents=30,raw_bytes=size,projected_host_raw_bytes=estimate,estimated_host_and_container_bytes=estimate*2,free_bytes=shutil.disk_usage(root).free,capacity=cap)
  write(root/'capacity-audit.json',capacity_audit)
  if shutil.disk_usage(root).free < estimate*2+15*1024**3:raise ValueError('insufficient space for host/container raw plus15GiB reserve')
  phase('production_collection')
  found=completed()
  for batch in pending_parent_batches(manifest,found,30):
   if shutil.disk_usage(root).free<15*1024**3:raise ValueError('disk reserve reached; partial data preserved')
   digest=hashlib.sha256(json.dumps([p['parent_id'] for p in batch]).encode()).hexdigest()[:12]
   tag=f"mixed-{batch[0]['field']}-l{batch[0]['level']}-n{len(batch)}-{digest}"
   run_batch(prod,stage,batch,tag,record_lidar=True,trial_seconds=30)
   found.update(p['parent_id'] for p in batch);phase('production_collection',completed_parents=len(found),planned_parents=1330)
  found=completed()
  if len(found)!=1330:raise ValueError('production parent coverage incomplete')
  write(prod/'collection-complete.json',dict(state='completed',planned_parents=1330,planned_windows=19950,observed_parents=len(found),finished_unix=time.time()))
  phase('actual_fastlio_processing')
  run('eat_nav_single_pose2s_fastlio_batch.py','--raw',prod/'raw','--workers',4)
  phase('dataset_audit')
  dataset=prod/'dataset'
  if not (dataset/'audit.json').exists():build_dataset(prod/'raw',manifest,dataset)
  audit=json.loads((dataset/'audit.json').read_text())
  if not audit['complete_parent_coverage'] or audit['observed_windows']!=19950:raise ValueError('dataset coverage incomplete')
  write(root/'dataset-completed-notification.json',dict(state='dataset_completed_and_audited',audit_path=str(dataset/'audit.json'),valid_windows=audit['valid_pose_windows'],parents=audit['observed_parents'],completed_unix=time.time(),user_notification_pending=True))
  phase('training',dataset_complete=True,valid_windows=audit['valid_pose_windows'])
  model=prod/'model-seed123'
  if not (model/'training-complete.json').exists():
   if model.exists():raise ValueError('partial training retained; inspect before resume')
   run('eat_nav_single_pose2s_train.py','--dataset',dataset,'--output',model,'--seed',123,'--device','cuda')
  phase('final_accuracy',training_complete=True)
  out=prod/'final-evaluation'
  if not (out/'accuracy-attestation.json').exists():
   if out.exists():raise ValueError('partial final evaluation retained; inspect before resume')
   run('eat_nav_single_pose2s_evaluate.py','--dataset',dataset,'--checkpoint',model/'checkpoint.pt','--output',out,'--device','cpu')
  phase('matched_model_and_duration_comparison')
  run('eat_nav_long_pose2s_compare.py','--root',root)
  phase('completed_audit_pending',analysis_complete=True,state='compute_completed',ended_unix=time.time(),checkpoint_sha256=hashlib.sha256((model/'checkpoint.pt').read_bytes()).hexdigest())
 except BaseException as e:
  phase('failed_preserved',state='failed',error=repr(e));write(root/'workflow-failure.json',dict(error=repr(e),time=time.time()));raise

if __name__=='__main__':main()
