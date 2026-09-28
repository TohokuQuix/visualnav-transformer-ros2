"""Host-side detached, SHA-checked stage runner for the EAT Quality follow-up."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from visualnav_transformer.eat_nav.quality_followup_io import (
    atomic_json, make_stage_marker, sha256_file, verify_stage_marker,
)

ROOT=Path('/tmp/eatnav-quality-identifiability-gate/followup')
DOCS=REPO/'docs/quality-identifiability-gate/followup'
CONTAINER='isaac-lab-onix'
CONTAINER_ROOT='/tmp/eatnav-train/followup'
SESSION='eatnav-quality-followup'
STAGES=[('01_audit','audit'),('02_gate_a_training','gate_a_training'),
        ('03_gate_a_evaluation','gate_a_evaluation'),('04_gate_b1','gate_b1'),
        ('05_gate_b8','gate_b8'),('06_gate_b64','gate_b64'),('07_report','report')]
REQUIRED={
    '01_audit':['complete-window-counts.json','complete-window-identities.jsonl.gz','overfit-ladder-pairs.json','complete-support.json'],
    '02_gate_a_training':['summary.json'],
    '03_gate_a_evaluation':['gate-a-complete-results.json'],
    '04_gate_b1':['summary.json','results.json'],
    '05_gate_b8':['summary.json','results.json'],
    '06_gate_b64':['summary.json','results.json'],
    '07_report':['quality-identifiability-followup.json','quality-identifiability-followup.md',
                 'gate-a-comparison.md','gate-a-comparison.json',
                 'overfit-ladder-results.json','overfit-ladder-training-summary.json',
                 'plots/complete-probe-learning-curves.png','plots/overfit-ladder-loss.png',
                 'plots/overfit-ladder-q-sweep.png','plots/overfit-ladder-effects.png'],
}


def now():return datetime.now(timezone.utc).isoformat()


def status_write(state,current,completed,started,error=None):
    atomic_json(ROOT/'status.json',{'state':state,'current_stage':current,'completed_stages':completed,
      'started_at':started,'updated_at':now(),'pid':os.getpid(),'tmux_session':SESSION,'error':error})


def input_hashes(name,previous):
    paths={
      'manifest':Path('/tmp/eatnav-pilot-100-extended/manifest.json'),
      'quality_config':REPO/'docs/pilot-quality-sensitivity/quality.yaml',
      'followup_config':DOCS/'followup-config.yaml',
      'frozen_pairs':REPO/'docs/quality-identifiability-gate/matched-pairs.json',
      'prior_gate_a':REPO/'docs/quality-identifiability-gate/probe-results.json',
      'prior_gate_b':REPO/'docs/quality-identifiability-gate/overfit-results.json',
      'probe_config':REPO/'docs/quality-identifiability-gate/probe-config.yaml',
      'overfit_config':REPO/'docs/quality-identifiability-gate/overfit-config.yaml',
      'worker':REPO/'scripts/eat_nav_quality_gate_followup_worker.py',
      'probe_evaluator':REPO/'scripts/eat_nav_quality_gate_evaluate.py',
      'overfit_evaluator':REPO/'scripts/eat_nav_quality_gate_overfit_evaluate.py',
      'followup_primitives':REPO/'src/visualnav_transformer/eat_nav/quality_followup.py',
      'followup_io':REPO/'src/visualnav_transformer/eat_nav/quality_followup_io.py',
      'followup_training':REPO/'src/visualnav_transformer/eat_nav/quality_followup_training.py',
    }
    result={key:sha256_file(path) for key,path in paths.items()}
    result['stage']=name
    if previous:
        marker=ROOT/'stages'/f'{previous}.done.json'
        result['previous_marker']=sha256_file(marker)
    return result


def docker_stage(name,stage):
    run=['docker','exec',CONTAINER,'bash','-lc',
         f'cd /tmp/eatnav-train && PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src /isaac-sim/python.sh eat_nav_quality_gate_followup_worker.py --stage {stage}']
    subprocess.run(run,check=True)
    target=ROOT/'stages'/name
    target.mkdir(parents=True,exist_ok=True)
    subprocess.run(['docker','cp',f'{CONTAINER}:{CONTAINER_ROOT}/stages/{name}/.',str(target)],check=True)
    for relative in REQUIRED[name]:
        if not (target/relative).is_file():raise ValueError(f'missing required stage artifact: {name}/{relative}')
    if name=='07_report':
        publish_docs()
    return {str(p.relative_to(target)):p for p in sorted(target.rglob('*')) if p.is_file()}


def publish_docs():
    DOCS.mkdir(parents=True,exist_ok=True)
    source=ROOT/'stages'
    files={
      '01_audit/complete-window-counts.json':'complete-window-counts.json',
      '01_audit/complete-window-identities.jsonl.gz':'complete-window-identities.jsonl.gz',
      '01_audit/overfit-ladder-pairs.json':'overfit-ladder-pairs.json',
      '01_audit/complete-support.json':'complete-support.json',
      '02_gate_a_training/summary.json':'gate-a-complete-training-summary.json',
      '03_gate_a_evaluation/gate-a-complete-results.json':'gate-a-complete-results.json',
      '04_gate_b1/summary.json':'gate-b1-training-summary.json',
      '04_gate_b1/results.json':'gate-b1-results.json',
      '05_gate_b8/summary.json':'gate-b8-training-summary.json',
      '05_gate_b8/results.json':'gate-b8-results.json',
      '06_gate_b64/summary.json':'gate-b64-training-summary.json',
      '06_gate_b64/results.json':'gate-b64-results.json',
      '07_report/overfit-ladder-results.json':'overfit-ladder-results.json',
      '07_report/overfit-ladder-training-summary.json':'overfit-ladder-training-summary.json',
      '07_report/gate-a-comparison.md':'gate-a-comparison.md',
      '07_report/gate-a-comparison.json':'gate-a-comparison.json',
      '07_report/quality-identifiability-followup.md':'quality-identifiability-followup.md',
      '07_report/quality-identifiability-followup.json':'quality-identifiability-followup.json',
    }
    for src,dst in files.items():shutil.copy2(source/src,DOCS/dst)
    plots=DOCS/'plots';plots.mkdir(exist_ok=True)
    for path in (source/'07_report'/'plots').glob('*.png'):shutil.copy2(path,plots/path.name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args()
    if args.root!=ROOT:raise ValueError('follow-up output root is frozen')
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'stages').mkdir(exist_ok=True)
    started=now();completed=[];current=None
    status_write('running',None,completed,started)
    try:
        previous=None
        for name,stage in STAGES:
            current=stage;status_write('running',current,completed,started)
            print('STAGE_START',name,now(),flush=True)
            inputs=input_hashes(name,previous)
            config_sha=inputs['followup_config']
            marker_path=ROOT/'stages'/f'{name}.done.json'
            marker=json.loads(marker_path.read_text()) if marker_path.exists() else None
            if marker and verify_stage_marker(marker,inputs,config_sha):
                print('STAGE_REUSED_VERIFIED',name,flush=True)
                if name=='07_report':publish_docs()
            else:
                if marker:
                    print('STAGE_STALE_RERUN',name,flush=True)
                    marker_path.unlink()
                    shutil.rmtree(ROOT/'stages'/name,ignore_errors=True)
                    subprocess.run(['docker','exec',CONTAINER,'rm','-rf',f'{CONTAINER_ROOT}/stages/{name}'],check=True)
                outputs=docker_stage(name,stage)
                seeds=[11,29,47] if name!='01_audit' and name!='07_report' else []
                marker=make_stage_marker(name,inputs,config_sha,outputs,seeds)
                atomic_json(marker_path,marker)
                print('STAGE_DONE',name,len(outputs),'files',flush=True)
            completed.append(stage);previous=name
            status_write('running',None,completed,started)
        status_write('completed',None,completed,started)
        shutil.copy2(ROOT/'status.json',DOCS/'status.json')
        print('FOLLOWUP_RUNNER_COMPLETED',now(),flush=True)
    except BaseException as exc:
        status_write('failed',current,completed,started,f'{type(exc).__name__}: {exc}')
        print('FOLLOWUP_RUNNER_FAILED',traceback.format_exc(),flush=True)
        raise

if __name__=='__main__':main()
