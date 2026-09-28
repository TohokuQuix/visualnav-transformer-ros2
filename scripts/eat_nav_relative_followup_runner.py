"""Detached SHA-marked relative-Quality follow-up runner."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality-followup')
DOCS=Path(__file__).resolve().parents[1]/'docs/quality-identifiability-gate/relative-quality-followup'
REMOTE=ROOT/'output';CONFIG=DOCS/'followup-config.yaml'
STAGES=[
 ('01_audit',['relative-label-audit.json'],None,'eat_nav_relative_followup_audit.py'),
 ('02_smoke',['probe-ablation-smoke.json'],None,'eat_nav_relative_followup_probe_train.py'),
 ('03_train',['probe-ablation-training-summary.json'],'train','eat_nav_relative_followup_probe_train.py'),
 ('04_evaluate',['probe-ablation-results.json'],'evaluate','eat_nav_relative_followup_probe_evaluate.py'),
 ('05_matching',['probe-ablation-matched-pairs.json','probe-ablation-match-thresholds.json'],'matching','eat_nav_relative_followup_probe_matching.py'),
 ('06_permutation',['probe-ablation-permutation.json'],'permutation','eat_nav_relative_followup_probe_permutation.py'),
 ('07_actor_audit',['actor-latent-provenance.json'],'actor','eat_nav_relative_followup_actor_audit.py'),
 ('08_report',['probe-ablation-report.md','forward-evaluator-config.yaml','forward-evaluator-report.md',
               'relative-quality-followup.md','relative-quality-followup-decision.json']+
               [f'plots/{x}.png' for x in ('probe-ablation','scalar-vs-multi','reward-free-vs-advantage')],
               'report','eat_nav_relative_followup_report.py'),
]


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()


def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
 os.replace(temp,path)


def run(*args):
 print('RUN',*args,flush=True);subprocess.run(args,check=True)


def copy_remote(name,source=None):
 target=DOCS/name;target.parent.mkdir(parents=True,exist_ok=True)
 temp=target.with_name(target.name+'.tmp')
 run('docker','cp',f'isaac-lab-onix:{source or REMOTE/name}',str(temp));os.replace(temp,target)


def inputs(stage,previous,source):
 files=[CONFIG,Path(__file__).resolve(),Path(__file__).with_name(source)]
 if stage=='08_report':files.append(DOCS/'probe-ablation-results.json')
 for p in files:
  if not p.is_file():raise FileNotFoundError(p)
 return {'file_sha256':{str(p):sha(p) for p in files},'previous_marker_sha256':previous}


def check_marker(name,files,expected):
 marker=ROOT/'markers'/f'{name}.json'
 if not marker.exists():return None
 data=json.loads(marker.read_text())
 if data['inputs']!=expected or data['config_sha256']!=sha(CONFIG):
  raise ValueError(f'{name} input/config SHA changed; refusing stale resume')
 if data['outputs']!={p:sha(DOCS/p) for p in files}:
  raise ValueError(f'{name} output SHA changed; refusing stale resume')
 return sha(marker)


def docker_stage(script,*args):
 run('docker','exec','-e','PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src',
     'isaac-lab-onix','/isaac-sim/python.sh',f'/tmp/eatnav-train/{script}',*args)


def main():
 ROOT.mkdir(parents=True,exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True)
 (ROOT/'markers').mkdir(exist_ok=True)
 status={'state':'running','completed':[],'current_stage':None,'started_unix':time.time(),'pid':os.getpid()}
 write(ROOT/'status.json',status);previous=None
 try:
  for name,files,action,source in STAGES:
   status['current_stage']=name;write(ROOT/'status.json',status)
   before=inputs(name,previous,source);marker_sha=check_marker(name,files,before)
   if marker_sha is None:
    if action=='train':docker_stage(source)
    elif action=='evaluate':docker_stage(source)
    elif action=='matching':docker_stage(source)
    elif action=='permutation':docker_stage(source)
    elif action=='actor':docker_stage(source)
    elif action=='report':run(sys.executable,str(Path(__file__).with_name(source)))
    if name=='02_smoke':copy_remote(files[0],ROOT/'smoke-probes/training-summary.json')
    elif action!='report':
     for p in files:copy_remote(p)
    outputs={p:sha(DOCS/p) for p in files}
    marker=ROOT/'markers'/f'{name}.json'
    write(marker,{'stage':name,'inputs':before,'config_sha256':sha(CONFIG),
                  'seeds':[11,29,47] if name in ('03_train','04_evaluate','06_permutation') else [],
                  'outputs':outputs,'completed_unix':time.time()})
    marker_sha=sha(marker)
   previous=marker_sha;status['completed'].append(name);write(ROOT/'status.json',status)
   print('STAGE_COMPLETE',name,flush=True)
  artifacts={p:sha(DOCS/p) for _,files,_,_ in STAGES for p in files}
  artifacts['followup-config.yaml']=sha(CONFIG)
  write(DOCS/'artifact-hashes.json',artifacts)
  actor=json.loads((DOCS/'actor-latent-provenance.json').read_text())
  status.update(state='complete_phase_a_forward_blocked' if actor['status']=='actor_observation_missing' else 'complete',
                current_stage=None,finished_unix=time.time(),artifact_hashes=artifacts)
  write(ROOT/'status.json',status);write(DOCS/'status.json',status)
  print('FOLLOWUP_COMPLETE',status['state'],flush=True)
 except Exception as exc:
  status.update(state='failed',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time())
  write(ROOT/'status.json',status);print('FOLLOWUP_FAILED',status['error'],flush=True)
  raise


if __name__=='__main__':main()
