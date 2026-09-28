"""Detached resumable host orchestrator for the held-out casebook."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from visualnav_transformer.eat_nav.quality_followup_io import atomic_json,make_stage_marker,sha256_file,verify_stage_marker

ROOT=Path('/tmp/eatnav-quality-identifiability-gate/casebook')
DOCS=REPO/'docs/quality-identifiability-gate/casebook'
CONTAINER='isaac-lab-onix'
CREMOTE='/tmp/eatnav-train/casebook'
SESSION='eatnav-quality-casebook'
STAGES=('audit','cohorts','infer','metrics','draw','report','sha')

def now():return datetime.now(timezone.utc).isoformat()

def status(state,current,completed,started,error=None):
    payload={'state':state,'current_stage':current,'completed_stages':completed,'started_at':started,
             'updated_at':now(),'pid':os.getpid(),'tmux_session':SESSION,'error':error}
    atomic_json(ROOT/'status.json',payload);atomic_json(DOCS/'status.json',payload)

def run(cmd):subprocess.run(cmd,check=True)

def copy_from(remote,local):
    local=Path(local);local.parent.mkdir(parents=True,exist_ok=True)
    run(['docker','cp',f'{CONTAINER}:{remote}',str(local)])

def input_hashes(previous):
    fixed={'casebook_config':DOCS/'casebook-config.yaml','worker':REPO/'scripts/eat_nav_quality_casebook_worker.py',
           'host_runner':REPO/'scripts/eat_nav_quality_casebook.py',
           'quality_config':REPO/'docs/pilot-quality-sensitivity/quality.yaml',
           'baseline_config':REPO/'docs/pilot-quality-sensitivity/baseline.yaml',
           'prior_pairs':REPO/'docs/quality-identifiability-gate/heldout-pair-controls.json',
           'prior_audit':REPO/'docs/quality-identifiability-gate/data-audit.json'}
    result={k:sha256_file(v) for k,v in fixed.items()}
    if previous:result['previous_stage_marker']=sha256_file(ROOT/'stages'/f'{previous}.done.json')
    return result

def stage_outputs(stage):
    if stage=='audit':return ['goal-provenance.json','goal-provenance.md']
    if stage=='cohorts':return ['casebook-cohorts.json','casebook-window-identities.jsonl.gz']
    if stage=='metrics':return ['casebook-metrics.json','casebook-metrics.csv','casebook-summary.json']
    if stage=='draw':
        return ['casebook.pdf','casebook.html']+[str(p.relative_to(DOCS)) for p in sorted((DOCS/'cases').glob('*.png'))]+[str(p.relative_to(DOCS)) for p in sorted((DOCS/'plots').glob('*.png'))]
    if stage=='report':return ['casebook-report.md']
    return []

def execute(stage):
    if stage=='sha':
        hashes={str(p.relative_to(DOCS)):sha256_file(p) for p in sorted(DOCS.rglob('*')) if p.is_file() and p.name not in ('artifact-hashes.json','status.json','runner.log')}
        atomic_json(DOCS/'artifact-hashes.json',hashes)
        return {'artifact-hashes.json':DOCS/'artifact-hashes.json'}
    run(['docker','exec',CONTAINER,'bash','-lc',
         f'cd /tmp/eatnav-train && PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src /isaac-sim/python.sh eat_nav_quality_casebook_worker.py --stage {stage}'])
    if stage=='infer':
        copy_from(CREMOTE+'/inference.json',ROOT/'inference.json')
        copy_from(CREMOTE+'/predictions.npz',ROOT/'predictions.npz')
        return {'inference.json':ROOT/'inference.json','predictions.npz':ROOT/'predictions.npz'}
    if stage=='draw':
        # Copy figures after full worker completion to avoid partial artifacts.
        for directory in ('cases','plots'):
            (DOCS/directory).mkdir(exist_ok=True)
            run(['docker','cp',f'{CONTAINER}:{CREMOTE}/output/{directory}/.',str(DOCS/directory)])
    outputs={}
    for name in stage_outputs(stage):
        path=DOCS/name
        if not path.exists():copy_from(CREMOTE+'/output/'+name,path)
        outputs[name]=path
    if stage=='draw':
        for name in ('casebook.pdf','casebook.html'):
            if not (DOCS/name).exists():copy_from(CREMOTE+'/output/'+name,DOCS/name)
            outputs[name]=DOCS/name
        for directory in ('cases','plots'):
            outputs.update({str(p.relative_to(DOCS)):p for p in (DOCS/directory).glob('*.png')})
    return outputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'stages').mkdir(exist_ok=True);DOCS.mkdir(parents=True,exist_ok=True)
    started=now();completed=[];previous=None;current=None
    try:
        for stage in STAGES:
            current=stage;status('running',stage,completed,started)
            inputs=input_hashes(previous);conf=inputs['casebook_config'];marker=ROOT/'stages'/f'{stage}.done.json'
            if a.resume and marker.exists():
                data=json.loads(marker.read_text())
                if not verify_stage_marker(data,inputs,conf):raise ValueError(f'resume SHA mismatch: {stage}')
                print('STAGE_RESUME',stage,flush=True)
            else:
                if marker.exists():raise FileExistsError(f'existing stage marker: {stage}; use --resume')
                print('STAGE_START',stage,now(),flush=True)
                outputs=execute(stage)
                if not outputs:raise ValueError(f'no stage outputs: {stage}')
                data=make_stage_marker(stage,inputs,conf,outputs,[])
                atomic_json(marker,data)
                print('STAGE_DONE',stage,now(),flush=True)
            completed.append(stage);previous=stage
        status('completed',None,completed,started)
    except Exception:
        error=traceback.format_exc();print(error,flush=True);status('failed',current,completed,started,error)
        raise
    finally:
        shutil.copy2(ROOT/'status.json',DOCS/'status.json')

if __name__=='__main__':main()
