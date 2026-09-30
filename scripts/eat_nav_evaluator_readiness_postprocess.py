"""Detached postprocessor: wait for complete raw pilot, then run fixed diagnostics."""
from __future__ import annotations
import hashlib, json, os, subprocess, time
from pathlib import Path

ROOT=Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-pilot')
REPO=Path('/tmp/eatnav-vnt-candidate-coverage')
CONTAINER=Path('/tmp/eatnav-candidate-coverage/evaluator-readiness/postprocess')
CODE=Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-code')
DOCKER_ENV=('PYTHONPATH=/tmp/eatnav-candidate-coverage/evaluator-readiness-code/src:'
            '/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/scripts:'
            '/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def atomic(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    temp.replace(path)

def call(name,cmd):
    with (ROOT/f'postprocess-{name}.log').open('a') as log:
        log.write('BEGIN '+json.dumps(cmd)+'\n');log.flush()
        env=os.environ.copy();env['PYTHONPATH']='src:scripts'
        result=subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        log.write(f'EXIT {result.returncode}\n')
    if result.returncode:raise RuntimeError(f'{name} exit {result.returncode}')

def stage(name,output,task):
    marker=ROOT/f'postprocess-{name}.done.json'
    if marker.exists():
        if json.loads(marker.read_text())['sha256']!=sha(output):
            raise ValueError(f'{name} resume SHA mismatch')
        return
    task()
    atomic(marker,{'stage':name,'path':str(output),'sha256':sha(output)})

def copy_to_container(source,target):
    call('copy',['docker','cp',str(source),f'isaac-lab-onix:{target}'])

def main():
    while True:
        path=ROOT/'status.json'
        if path.exists():
            status=json.loads(path.read_text())
            if status['stage']=='RAW_COMPLETE':break
            if status['stage']=='STOPPED':
                atomic(ROOT/'postprocess-status.json',{'stage':'BLOCKED_BY_RAW_GATE','reason':status.get('reason')})
                return
        time.sleep(60)
    analysis=ROOT/'analysis';analysis.mkdir(parents=True,exist_ok=True)
    atomic(ROOT/'postprocess-status.json',{'stage':'RUNNING','raw_status_sha256':sha(path)})
    try:
        stage('coverage',analysis/'analysis.done.json',lambda:call('coverage',
              ['/usr/bin/python3','scripts/eat_nav_evaluator_readiness_analyze.py']))
        stage('ood-paths',analysis/'ood-paths.json',lambda:call('ood-paths',
              ['/usr/bin/python3','scripts/eat_nav_evaluator_readiness_ood_extract.py']))
        call('prepare-container',['docker','exec','isaac-lab-onix','mkdir','-p',str(CONTAINER)])
        for source,target in (
            (REPO/'scripts/eat_nav_evaluator_readiness_ood.py',CODE/'ood.py'),
            (REPO/'scripts/eat_nav_evaluator_readiness_score.py',CODE/'score.py'),
            (analysis/'ood-paths.json',CONTAINER/'ood-paths.json'),
            (analysis/'per-candidate-outcomes.json',CONTAINER/'per-candidate-outcomes.json'),
            (REPO/'docs/quality-identifiability-gate/candidate-coverage/evaluator-readiness-pilot/selection.json',CONTAINER/'selection.json'),
            (path,CONTAINER/'raw-complete-status.json')):
            copy_to_container(source,target)
        def ood():
            call('ood',['docker','exec','-e',DOCKER_ENV,'isaac-lab-onix','/isaac-sim/python.sh',str(CODE/'ood.py'),
                '--paths',str(CONTAINER/'ood-paths.json'),
                '--manifest','/tmp/eatnav-extended/output/manifest.json',
                '--reference-config','/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
                '--reference-checkpoint','/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt',
                '--output',str(CONTAINER/'ood-results.json')])
            call('copy-ood',['docker','cp',f'isaac-lab-onix:{CONTAINER}/ood-results.json',str(analysis/'ood-results.json')])
        stage('ood',analysis/'ood-results.json',ood)
        def score():
            call('score',['docker','exec','-e',DOCKER_ENV,'isaac-lab-onix','/isaac-sim/python.sh',str(CODE/'score.py'),
                '--selection',str(CONTAINER/'selection.json'),
                '--outcomes',str(CONTAINER/'per-candidate-outcomes.json'),
                '--raw-complete-status',str(CONTAINER/'raw-complete-status.json'),
                '--reference-config','/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
                '--manifest','/tmp/eatnav-extended/output/manifest.json',
                '--output',str(CONTAINER/'frozen-evaluator.json')])
            call('copy-score',['docker','cp',f'isaac-lab-onix:{CONTAINER}/frozen-evaluator.json',str(analysis/'frozen-evaluator.json')])
        stage('frozen-score',analysis/'frozen-evaluator.json',score)
        atomic(ROOT/'postprocess-status.json',{'stage':'COMPLETE','outputs':
            {name:sha(analysis/name) for name in ('analysis.done.json','ood-results.json','frozen-evaluator.json')}})
        print('READINESS_POSTPROCESS_COMPLETE',flush=True)
    except BaseException as exc:
        atomic(ROOT/'postprocess-status.json',{'stage':'STOPPED','reason':repr(exc)})
        raise

if __name__=='__main__':main()
