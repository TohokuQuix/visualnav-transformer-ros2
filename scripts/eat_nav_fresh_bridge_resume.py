"""Resume frozen readiness pilot after fresh-bridge equivalence and cleanup gates."""
from __future__ import annotations
import gzip,hashlib,json,pathlib,subprocess,time
from eat_nav_fresh_bridge_run import ROOT,OLD,DOC,INPUT,atomic,read,run_one,sha,frozen
from visualnav_transformer.eat_nav.candidate_coverage_pilot import map_paths_from_anchor
from visualnav_transformer.eat_nav.command_preview_long_horizon import duplicate_120

CAP=1_500_000_000

def used_bytes():
 dirs=[OLD,pathlib.Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-recovery'),ROOT]
 host=sum(sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in dirs)
 remote=subprocess.check_output(['docker','exec','isaac-lab-onix','du','-sb',
    '/tmp/eatnav-candidate-coverage/evaluator-readiness',
    '/tmp/eatnav-candidate-coverage/evaluator-readiness-recovery',
    '/tmp/eatnav-candidate-coverage/evaluator-readiness-fresh-protocol'],text=True)
 container=sum(int(line.split()[0]) for line in remote.splitlines())
 return {'host':host,'container':container,'total':host+container}

def storage_gate():
 sizes=used_bytes();sample=[]
 for record in read(ROOT/'r0-freeze.json')['selected']:
  p=ROOT/'runs'/('equivalence-'+record['entry']['run_id'])
  sample.append(sum(f.stat().st_size for f in p.rglob('*') if f.is_file()))
 # Reserve 150 MB for all postprocessing outputs and safety margin.
 per_run=max(sample,default=8_000_000)+2_000_000
 forecast=sizes['total']+19*per_run+150_000_000
 result={'current_bytes':sizes,'sample_run_bytes':sample,'forecast_per_run_bytes':per_run,
         'postprocessing_reserve_bytes':150_000_000,'forecast_total_bytes':forecast,
         'cap_bytes':CAP,'pass':forecast<=CAP}
 atomic(ROOT/'r6-storage-gate.json',result)
 if not result['pass']:raise RuntimeError('1.5 GB storage forecast exceeded')
 return result

def verify_old_hundred(schedule):
 for entry in schedule[:100]:
  folder=OLD/'runs'/entry['run_id'];complete=read(folder/'run-complete.json')
  marker=read(OLD/'markers'/(entry['run_id']+'.done.json'))
  if complete['identity']!=entry or marker['run_complete_sha256']!=sha(folder/'run-complete.json'):
   raise ValueError('old marker identity/SHA mismatch '+entry['run_id'])
  for name,digest in complete['output_sha256'].items():
   if sha(folder/name)!=digest:raise ValueError('old output SHA mismatch '+entry['run_id']+'/'+name)
 return 100

def prepare_state_six(anchor_folder):
 state=INPUT/'state-006';table=read(state/'candidate-table.json')
 paths={x['candidate_id']:read(state/(x['candidate_id']+'-base-path.json')) for x in table}
 for x in table:
  if sha(state/(x['candidate_id']+'-base-path.json'))!=x['base_path_sha256']:
   raise ValueError('state6 base Path SHA mismatch')
 anchor=read(anchor_folder/'anchor-0.json')
 mapped=map_paths_from_anchor(anchor,paths)
 freeze={'anchor_sha256':sha(anchor_folder/'anchor-0.json'),'map_path_sha256':{}}
 for candidate,points in mapped.items():
  target=state/(candidate+'-map-path.json')
  if target.exists():
   if read(target)!=points:raise ValueError('preexisting state6 map Path differs')
  else:atomic(target,points)
  freeze['map_path_sha256'][candidate]=sha(target)
  subprocess.run(['docker','cp',str(target),
     'isaac-lab-onix:/tmp/eatnav-candidate-coverage/evaluator-readiness/inputs/state-006/'+target.name],check=True)
 atomic(ROOT/'map-path-freeze/readiness-006.json',freeze)
 return freeze

def duplicate_six():
 paths=[ROOT/'runs'/f'measurement-readiness-{n:03d}' for n in (103,104,105)]
 summaries=[read(p/'live-0.json') for p in paths]
 traces=[]
 for p in paths:
  with gzip.open(p/'live-0.trace.json.gz','rt') as f:traces.append(json.load(f))
 result=duplicate_120(summaries,traces,.0105)
 if not result['pass'] or not result['dem_sha_equal'] or any(v not in (0,None) for v in result['signal_rms'].values()):
  raise ValueError('state6 duplicate gate failed')
 atomic(ROOT/'duplicate-gates/readiness-006.json',result)
 return result

def lifecycle_gate(target,baseline):
 rows=[json.loads(x) for x in (target/'follower-logs/bridge-events.jsonl').read_text().splitlines()]
 resources=[x for x in rows if x['event']=='resource']
 value={'fd_max':max(x['fd_count'] for x in resources),'thread_max':max(x['thread_count'] for x in resources),
        'socket_max':max(x['socket_count'] for x in resources),'rss_pages_max':max(x['rss_pages'] for x in resources),
        'reset_count':sum(x['event']=='reset_entry' for x in rows),
        'heartbeat_count':sum(x['event']=='executor_heartbeat' for x in rows),
        'shutdown_return':sum(x['event']=='shutdown_return' for x in rows),
        'sigkill_count':sum(x['event']=='follower_sigkill' for x in rows)}
 allowed={'fd_max':baseline['fd_max']+16,'thread_max':baseline['thread_max']+4,
          'socket_max':baseline['socket_max']+8,'rss_pages_max':int(baseline['rss_pages_max']*1.5)}
 passed=all(value[k]<=allowed[k] for k in allowed) and value['reset_count']==1 and value['heartbeat_count']>0 and value['shutdown_return']==1 and value['sigkill_count']==0
 value.update({'allowed':allowed,'pass':passed})
 atomic(target/'lifecycle-gate.json',value)
 if not passed:raise RuntimeError('resource/lifecycle gate failed '+target.name)
 return value

def run_resume():
 if read(ROOT/'r4-equivalence-gate.json')['stage']!='PASS' or not read(ROOT/'r5-stress-gate.json')['pass'] or not read(ROOT/'r3-fault-gate.json')['operational_cleanup_pass']:
  raise RuntimeError('R3/R4/R5 gate missing')
 storage_gate();schedule=read(DOC/'schedule.json')['runs'];verify_old_hundred(schedule)
 baseline_rows=read(ROOT/'r4-equivalence-gate.json')['results']
 baseline={key:max(r['lifecycle'][name][1] for r in baseline_rows) for key,name in
           [('fd_max','fd_min_max'),('thread_max','threads_min_max'),('socket_max','sockets_min_max'),('rss_pages_max','rss_pages_min_max')]}
 started=time.time();durations=[]
 for index in range(100,119):
  entry=schedule[index];name='measurement-'+entry['run_id'];target=ROOT/'runs'/name
  if index>100 and not (ROOT/'runs/measurement-readiness-100/run-complete.json').exists():
   raise RuntimeError('measurement-100 gate incomplete')
  if index==102 and not (ROOT/'map-path-freeze/readiness-006.json').exists():
   if target.exists():
    if not (target/'run-complete.json').exists():raise RuntimeError('partial anchor; no retry')
    prepare_state_six(target)
  if index>=103 and not (ROOT/'map-path-freeze/readiness-006.json').exists():raise RuntimeError('state6 Path not frozen')
  remaining=119-index;observed=max(durations[-4:],default=180)
  if time.time()+remaining*observed>started+7200:raise TimeoutError('two-hour resume forecast')
  if used_bytes()['total']+remaining*10_000_000+150_000_000>CAP:raise RuntimeError('storage forecast')
  atomic(ROOT/'status.json',{'stage':'R7_RUNNING' if index==100 else 'R8_RUNNING','current_run':name,
     'completed':index,'remaining':remaining,'resume_decision':'running','elapsed_sec':time.time()-started,
     'runtime_forecast_sec':remaining*observed,'storage':used_bytes(),'storage_cap_bytes':CAP})
  if not target.exists():
   before=time.monotonic();run_one(entry,name);durations.append(time.monotonic()-before)
  elif not (target/'run-complete.json').exists():raise RuntimeError('partial measured run; no retry '+name)
  else:durations.append(read(target/'run-complete.json')['elapsed_sec'])
  complete=read(target/'run-complete.json')
  if complete['identity']!=entry:raise ValueError('new run identity mismatch')
  if entry['kind']!='anchor':lifecycle_gate(target,baseline)
  atomic(ROOT/'markers'/(entry['run_id']+'.done.json'),{'run_id':entry['run_id'],
     'run_complete_sha256':sha(target/'run-complete.json'),'schedule_sha256':sha(DOC/'schedule.json')})
  if index==100:
   atomic(ROOT/'r7-measurement-gate.json',{'pass':True,'entry':entry,'marker_sha256':sha(target/'run-complete.json')})
  if index==102:prepare_state_six(target)
  if index==105:duplicate_six()
  print('FRESH_MEASUREMENT_COMPLETE',entry['run_id'],entry['kind'],round(durations[-1],2),flush=True)
 verify_old_hundred(schedule)
 if len(list((ROOT/'markers').glob('*.done.json')))!=19:raise ValueError('19 new marker count')
 atomic(ROOT/'status.json',{'stage':'RAW_COMPLETE','completed':119,'remaining':0,'elapsed_sec':time.time()-started,
    'storage':used_bytes(),'storage_cap_bytes':CAP,'protocol':'fresh_bridge_per_run_after_100'})
 return 119

if __name__=='__main__':
 try:run_resume()
 except BaseException as exc:
  current=read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
  stage='MEASUREMENT_INVALID' if current.get('stage')=='R7_RUNNING' else 'STOPPED'
  atomic(ROOT/'status.json',{'stage':stage,'current_run':current.get('current_run'),
      'completed':current.get('completed',100),'remaining':current.get('remaining',19),
      'stop_reason':repr(exc),'resume_decision':'blocked','storage':used_bytes()})
  raise
